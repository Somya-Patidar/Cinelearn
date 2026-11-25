import os
import time
import random
import json
import pathlib
import hashlib
import re
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load centralized .env from project root or src directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = Path(__file__).resolve().parent

load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_SRC_DIR / ".env")
load_dotenv()

# Read and cache api_config.json as fallback
_CFG_PATH = _SRC_DIR / "api_config.json"
_CFG = {}
if _CFG_PATH.exists():
    try:
        with _CFG_PATH.open("r", encoding="utf-8") as _f:
            _CFG = json.load(_f)
    except Exception:
        _CFG = {}


def cfg(svc: str, key: str, default=None):
    """Retrieve configuration from environment variable first, then fallback to api_config.json."""
    env_val = os.getenv(f"{svc}_{key}".upper())
    if env_val is not None:
        return env_val
    return _CFG.get(svc, {}).get(key, default)


def generate_log_id():
    """Generate a unique log ID with timestamp."""
    return f"log_{int(time.time() * 1000)}"


# ============================================================================
# Gemini Universal Response Wrapper
# ============================================================================
class GeminiResponseWrapper:
    """
    Universal response wrapper for Gemini that provides compatibility with:
    - response.text
    - response.choices[0].message.content (OpenAI format)
    - response.candidates[0].content.parts[0].text (Google GenAI format)
    - response.usage (token counts)
    - str(response)
    """

    def __init__(self, text: str, usage_info: dict, raw_response=None):
        self._text = text if text is not None else ""
        self._usage_info = usage_info or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._raw_response = raw_response

    @property
    def text(self) -> str:
        return self._text

    @property
    def choices(self):
        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, msg):
                self.message = msg

        return [_Choice(_Msg(self._text))]

    @property
    def candidates(self):
        class _Part:
            def __init__(self, text):
                self.text = text

        class _Content:
            def __init__(self, parts):
                self.parts = parts

        class _Candidate:
            def __init__(self, content):
                self.content = content

        return [_Candidate(_Content([_Part(self._text)]))]

    @property
    def usage(self):
        class _Usage:
            def __init__(self, p, c, t):
                self.prompt_tokens = p
                self.completion_tokens = c
                self.total_tokens = t

        return _Usage(
            self._usage_info.get("prompt_tokens", 0),
            self._usage_info.get("completion_tokens", 0),
            self._usage_info.get("total_tokens", 0),
        )

    def __str__(self):
        return self._text

    def __repr__(self):
        return f"<GeminiResponseWrapper text_length={len(self._text)} tokens={self._usage_info.get('total_tokens', 0)}>"


def _get_gemini_client() -> genai.Client:
    """Initialize and return a google-genai Client using the API key exclusively from .env."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "...", "YOUR_GEMINI_KEY", "your_gemini_api_key_here"):
        raise ValueError(
            "GEMINI_API_KEY is not configured in .env!\n"
            "Please add your Gemini API key to the .env file in the project root:\n"
            "  GEMINI_API_KEY=your_key_here\n"
            "You can get an API key at: https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key.strip())


_CURRENT_GEMINI_MODEL: Optional[str] = None


def _get_gemini_model() -> str:
    """Get the configured Gemini model name, supporting dynamic runtime fallbacks."""
    global _CURRENT_GEMINI_MODEL
    if _CURRENT_GEMINI_MODEL:
        return _CURRENT_GEMINI_MODEL
    model_name = os.getenv("GEMINI_MODEL")
    if not model_name or model_name.strip() in ("", "...", "YOUR_GEMINI_MODEL", "your_gemini_model_here"):
        raise ValueError(
            "GEMINI_MODEL is not configured in .env!\n"
            "Please specify your Gemini model name in the .env file in the project root:\n"
            "  GEMINI_MODEL=gemini-3.6-flash\n"
            "  (or your chosen Gemini model)"
        )
    _CURRENT_GEMINI_MODEL = model_name.strip()
    return _CURRENT_GEMINI_MODEL


def _set_gemini_model(new_model: str):
    global _CURRENT_GEMINI_MODEL
    _CURRENT_GEMINI_MODEL = new_model.strip()


def _trigger_model_fallback(reason: str = "") -> str:
    """Fallback to GEMINI_FALLBACK_MODEL if configured, otherwise retain current active model."""
    current = _get_gemini_model()
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest").strip()
    if fallback and fallback != current:
        _set_gemini_model(fallback)
        print(f"\n🔄 [MODEL FALLBACK] {reason}. Switched active model: {current} -> {fallback}")
        return fallback
    print(f"\nℹ️ [MODEL RETAINED] {reason}. Retaining active model: {current}")
    return current


def _extract_text_and_usage(response):
    """Safely extract generated text and usage tokens from google-genai response."""
    text = ""
    try:
        text = response.text or ""
    except Exception:
        try:
            parts = response.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
        except Exception:
            text = str(response)

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or (prompt_tokens + completion_tokens)

    usage_info = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return GeminiResponseWrapper(text, usage_info, raw_response=response), usage_info


# ============================================================================
# Quota & Cache Management for Gemini API
# ============================================================================

class QuotaErrorType(Enum):
    RPM = "RPM"
    RPD = "RPD"
    BILLING = "BILLING"
    NON_RECOVERABLE = "NON_RECOVERABLE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class DailyQuotaExhaustedError(Exception):
    """Raised when Gemini daily free tier quota (RPD) is exhausted."""
    pass


class ServiceUnavailableError(Exception):
    """Raised when Gemini API returns 503 Service Unavailable / high demand after max retries."""
    pass


def _get_stable_file_stat(file_path: str) -> str:
    """Generate a stable cache signature based on file size and partial content hash, avoiding volatile mtime."""
    try:
        size = os.path.getsize(file_path)
        h = hashlib.sha256()
        h.update(str(size).encode("utf-8"))
        with open(file_path, "rb") as f:
            h.update(f.read(65536))
            if size > 131072:
                f.seek(size - 65536)
                h.update(f.read(65536))
        return f"{file_path}:{size}:{h.hexdigest()[:16]}"
    except Exception:
        try:
            return f"{file_path}:{os.path.getsize(file_path)}"
        except Exception:
            return str(file_path)


class QuotaManager:
    """
    Centralized quota manager and request scheduler for Gemini API calls:
    - Enforces proactive RPM rate limiting using a derived effective RPM safety threshold.
    - Serializes network requests across concurrent section workers via api_execution_lock.
    - Proactively checks and tracks daily RPD requests against effective RPD threshold.
    - Single authoritative retry policy: at most 1 retry for 429/RPM (with 65s+ cooldown) and 1 retry for 503.
    - Daily quota exhaustion halts all Gemini requests immediately, saves persistent flag, and never blindly retries.
    - Thread-safe persistent disk caching (.cache/gemini_cache.json); cache hits never consume API slots.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QuotaManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, rpm: int = 5, cache_dir: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        # 1. RPM Configuration & Safety Margins
        # Provider RPM limit: externally applicable Gemini limit (default: 5)
        rpm_env = os.getenv("GEMINI_RPM_LIMIT") or os.getenv("GEMINI_RPM")
        self.provider_rpm = int(rpm_env) if rpm_env and rpm_env.isdigit() else rpm

        # Internal RPM safety margin (default: 1)
        rpm_margin_env = os.getenv("GEMINI_RPM_SAFETY_MARGIN")
        self.rpm_safety_margin = int(rpm_margin_env) if rpm_margin_env and rpm_margin_env.isdigit() else 1

        # Effective RPM threshold: internal operational limit
        rpm_thresh_env = os.getenv("GEMINI_RPM_THRESHOLD")
        if rpm_thresh_env and rpm_thresh_env.isdigit():
            self.effective_rpm = int(rpm_thresh_env)
        else:
            self.effective_rpm = max(1, self.provider_rpm - self.rpm_safety_margin)

        # Minimum interval derived from effective RPM threshold
        delay_env = os.getenv("GEMINI_REQUEST_DELAY")
        if delay_env and delay_env.replace(".", "", 1).isdigit():
            self.min_interval = float(delay_env)
        else:
            self.min_interval = 60.0 / float(self.effective_rpm)

        # 2. RPD Configuration & Safety Margins
        # Provider RPD limit: None unless explicitly configured
        rpd_env = os.getenv("GEMINI_RPD_LIMIT") or os.getenv("GEMINI_RPD")
        self.provider_rpd = int(rpd_env) if rpd_env and rpd_env.isdigit() else None

        rpd_margin_env = os.getenv("GEMINI_RPD_SAFETY_MARGIN")
        self.rpd_safety_margin = int(rpd_margin_env) if rpd_margin_env and rpd_margin_env.isdigit() else 2

        rpd_thresh_env = os.getenv("GEMINI_RPD_THRESHOLD")
        if rpd_thresh_env and rpd_thresh_env.isdigit():
            self.effective_rpd = int(rpd_thresh_env)
        elif self.provider_rpd is not None:
            self.effective_rpd = max(1, self.provider_rpd - self.rpd_safety_margin)
        else:
            self.effective_rpd = None  # Not configured: operate in fail-safe mode

        self.last_request_time = 0.0
        self.request_lock = threading.Lock()
        self.api_execution_lock = threading.Lock()
        self.cache_lock = threading.Lock()

        self.cache_dir = cache_dir or (_PROJECT_ROOT / ".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.daily_stats_path = self.cache_dir / "quota_daily_stats.json"
        self.cache_path = self.cache_dir / "gemini_cache.json"

        self.cache: Dict[str, Any] = {}
        self._last_cache_mtime = 0.0
        self._load_cache()
        self.actual_calls_count = 0
        self._initialized = True

        self._log_configuration()

    def _log_configuration(self):
        print("\n" + "=" * 60)
        print("Gemini API Central Request Scheduler & Quota Configuration:")
        print(f"  Provider RPM limit:         {self.provider_rpm}")
        print(f"  Internal RPM safety margin:  {self.rpm_safety_margin}")
        print(f"  Effective RPM threshold:    {self.effective_rpm}")
        print(f"  Minimum request interval:   {self.min_interval:.1f}s")
        print(f"  Provider RPD limit:         {self.provider_rpd if self.provider_rpd is not None else 'not configured (fail-safe mode)'}")
        print(f"  Internal RPD safety margin:  {self.rpd_safety_margin if self.provider_rpd is not None else 'not configured'}")
        print(f"  Effective RPD threshold:    {self.effective_rpd if self.effective_rpd is not None else 'not configured'}")
        print("=" * 60 + "\n")

    def _get_pacific_date(self) -> str:
        return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()

    def get_daily_requests_count(self) -> int:
        if not self.daily_stats_path.exists():
            return 0
        try:
            with open(self.daily_stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == self._get_pacific_date():
                    return data.get("count", 0)
        except Exception:
            pass
        return 0

    def increment_daily_requests_count(self) -> int:
        today = self._get_pacific_date()
        current_count = 0
        try:
            if self.daily_stats_path.exists():
                with open(self.daily_stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        current_count = data.get("count", 0)
        except Exception:
            current_count = 0

        new_count = current_count + 1
        try:
            tmp_path = self.daily_stats_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"date": today, "count": new_count, "last_updated": time.time()}, f, indent=2)
            tmp_path.replace(self.daily_stats_path)
        except Exception:
            pass
        return new_count

    def decrement_daily_requests_count(self) -> int:
        """Roll back daily request count when a request fails with 503 so it does not count against RPD quota."""
        today = self._get_pacific_date()
        current_count = 0
        try:
            if self.daily_stats_path.exists():
                with open(self.daily_stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        current_count = data.get("count", 0)
        except Exception:
            current_count = 0

        new_count = max(0, current_count - 1)
        try:
            tmp_path = self.daily_stats_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"date": today, "count": new_count, "last_updated": time.time()}, f, indent=2)
            tmp_path.replace(self.daily_stats_path)
        except Exception:
            pass
        return new_count

    def get_daily_flag_path(self, model_name: Optional[str] = None) -> Path:
        mod = model_name or _get_gemini_model()
        mod_clean = mod.replace("/", "_").replace(":", "_").replace("-", "_")
        return self.cache_dir / f"daily_quota_exhausted_{mod_clean}.flag"

    @property
    def daily_flag_path(self) -> Path:
        return self.get_daily_flag_path()

    def is_daily_quota_exhausted(self, model_name: Optional[str] = None) -> bool:
        flag = self.get_daily_flag_path(model_name)
        if not flag.exists():
            return False

        # If it's a new Pacific day, clear the stale flag
        pacific = datetime.now(ZoneInfo("America/Los_Angeles")).date()
        try:
            mtime = datetime.fromtimestamp(
                flag.stat().st_mtime,
                ZoneInfo("America/Los_Angeles")
            ).date()

            if pacific > mtime:
                self.reset_daily_quota(model_name)
                return False
        except Exception:
            pass

        return True

    def mark_daily_quota_exhausted(self, reason: str = "", model_name: Optional[str] = None):
        try:
            flag = self.get_daily_flag_path(model_name)
            today = self._get_pacific_date()
            with open(flag, "w", encoding="utf-8") as f:
                f.write(f"timestamp: {time.time()}\ndate: {today}\nreason: {reason}\n")
            print(f"🛑 [QUOTA MANAGER] Daily quota exhaustion flag written -> {flag}")
        except Exception:
            pass

    def reset_daily_quota(self, model_name: Optional[str] = None):
        """Only explicitly resets when a new day begins or when manually invoked for tests."""
        flag = self.get_daily_flag_path(model_name)
        if flag.exists():
            try:
                flag.unlink()
            except Exception:
                pass
        # Also clean up legacy flag if exists
        legacy_flag = self.cache_dir / "daily_quota_exhausted.flag"
        if legacy_flag.exists():
            try:
                legacy_flag.unlink()
            except Exception:
                pass

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                self._last_cache_mtime = self.cache_path.stat().st_mtime
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        try:
            temp_path = self.cache_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.cache_path)
            self._last_cache_mtime = self.cache_path.stat().st_mtime
        except Exception:
            pass

    def get_cache_key(self, prompt: str, model: str, extra: str = "") -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(prompt.encode("utf-8"))
        if extra:
            h.update(extra.encode("utf-8"))
        return h.hexdigest()

    def get_cached(self, key: str) -> Optional[Tuple[str, dict]]:
        with self.cache_lock:
            if self.cache_path.exists():
                try:
                    disk_mtime = self.cache_path.stat().st_mtime
                    if disk_mtime > self._last_cache_mtime:
                        self._load_cache()
                except Exception:
                    pass
            if key not in self.cache and self.cache_path.exists():
                self._load_cache()
            val = self.cache.get(key)
            if val and isinstance(val, dict):
                return val.get("text", ""), val.get("usage", {})
        return None

    def set_cached(self, key: str, text: str, usage: dict):
        with self.cache_lock:
            self.cache[key] = {
                "text": text,
                "usage": usage,
                "timestamp": time.time(),
            }
            self._save_cache()

    def classify_error(self, err: Exception) -> Tuple[QuotaErrorType, float]:
        """
        Classifies error into RPD, RPM (429), BILLING, NON_RECOVERABLE, SERVICE_UNAVAILABLE (503), or UNKNOWN.
        Returns (QuotaErrorType, retry_delay_seconds).
        """
        msg = str(err)
        msg_lower = msg.lower()
        err_code = getattr(err, "code", None) or getattr(err, "status_code", None)

        # 1. Daily quota exhaustion (RPD)
        rpd_signals = [
            "generaterequestsperdayperprojectpermodel-freetier",
            "generaterequestsperday",
            "generate_requests_per_day",
            "perday",
            "daily quota",
            "requests per day",
        ]
        if any(sig in msg_lower for sig in rpd_signals):
            return QuotaErrorType.RPD, 0.0

        # 2. 503 Service Unavailable / High Demand / Overloaded
        unavailable_signals = [
            "503",
            "unavailable",
            "service unavailable",
            "overloaded",
            "high demand",
            "server error",
            "backend error",
            "service_unavailable",
        ]
        if any(sig in msg_lower for sig in unavailable_signals) or err_code == 503:
            return QuotaErrorType.SERVICE_UNAVAILABLE, 0.0

        # 3. Billing / Authentication
        billing_signals = [
            "api_key_invalid",
            "api key not valid",
            "billing",
            "permission_denied",
            "forbidden",
            "project has been deleted",
        ]
        if any(sig in msg_lower for sig in billing_signals) or err_code in (401, 403):
            return QuotaErrorType.BILLING, 0.0

        # 4. Non-recoverable client / model errors (404, invalid model, 400 invalid argument)
        non_rec_signals = [
            "not_found",
            "not found",
            "404",
            "is not found for api version",
            "invalid_argument",
            "invalid argument",
            "bad request",
            "400",
            "model not found",
            "publishermodel",
        ]
        if any(sig in msg_lower for sig in non_rec_signals) or err_code in (400, 404):
            return QuotaErrorType.NON_RECOVERABLE, 0.0

        # 5. 429 RESOURCE_EXHAUSTED / Quota limit / Rate limit
        rpm_signals = [
            "resource_exhausted",
            "429",
            "quota",
            "rate limit",
            "generaterequestsperminute",
            "per_minute",
            "too many requests",
        ]
        if any(sig in msg_lower for sig in rpm_signals) or err_code == 429:
            return QuotaErrorType.RPM, 0.0

        return QuotaErrorType.UNKNOWN, 0.0

    def acquire(self):
        """Proactively enforce rate limit and check daily quota."""
        if self.is_daily_quota_exhausted():
            raise DailyQuotaExhaustedError(
                "Daily Gemini API quota exhausted (GenerateRequestsPerDayPerProjectPerModel-FreeTier). "
                "Halting pipeline gracefully."
            )

        if self.effective_rpd is not None:
            count = self.get_daily_requests_count()
            if count >= self.effective_rpd:
                self.mark_daily_quota_exhausted(f"Proactive daily threshold reached: {count}/{self.effective_rpd} RPD")
                raise DailyQuotaExhaustedError(
                    f"Daily request count ({count}) reached effective threshold ({self.effective_rpd} RPD). "
                    "Halting pipeline gracefully."
                )

        with self.request_lock:
            now = time.time()
            if self.last_request_time > 0:
                elapsed = now - self.last_request_time
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    print(f"⏳ Central Scheduler: waiting {sleep_time:.1f}s for next Gemini request slot "
                          f"(effective threshold: {self.effective_rpm} RPM, interval: {self.min_interval:.1f}s)...")
                    time.sleep(sleep_time)
            self.last_request_time = time.time()


# Singleton instance
quota_manager = QuotaManager()


def _extract_retry_after(e: Exception) -> Optional[float]:
    """Extract explicit Retry-After delay from exception attributes, response headers, or error message."""
    # 1. Direct attribute check
    retry_attr = getattr(e, "retry_after", None)
    if retry_attr is not None:
        try:
            return float(retry_attr)
        except (ValueError, TypeError):
            pass

    # 2. Response headers check
    response = getattr(e, "response", None)
    if response is not None and hasattr(response, "headers"):
        header_val = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if header_val:
            try:
                return float(header_val)
            except (ValueError, TypeError):
                pass

    # 3. Regex search in error message
    msg = str(e)
    match = re.search(r"retry(?:[-_\s]+(?:after|in))?\s*[:\s=]\s*(\d+(?:\.\d+)?)s?", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            pass

    return None


class GeminiRetryState:
    """
    Tracks retry attempts per request across different failure modes.
    Enforces strict bounds:
    - 503 Service Unavailable: max 4 retries (5 total attempts max)
    - 429 RPM: max 1 retry
    - Other transient errors: max 1 retry
    - Hard ceiling of 7 total attempts across all failure types
    """
    def __init__(self, max_rpm_retries: int = 1, max_503_retries: int = 4, max_other_retries: int = 1):
        self.rpm_retries = 0
        self.max_rpm_retries = max_rpm_retries
        self.retries_503 = 0
        self.max_503_retries = max_503_retries
        self.other_retries = 0
        self.max_other_retries = max_other_retries
        self.total_attempts = 0

    def can_continue(self) -> bool:
        return (
            self.rpm_retries <= self.max_rpm_retries and
            self.retries_503 <= self.max_503_retries and
            self.other_retries <= self.max_other_retries and
            self.total_attempts < 7
        )


def _handle_gemini_error(
    e: Exception,
    qm: QuotaManager,
    retry_state: Union[GeminiRetryState, int, None] = None,
    max_retries: int = 2,
    op_name: str = "request",
    retry_count: Optional[int] = None,
) -> Union[GeminiRetryState, int]:
    """
    Centralized authoritative error handler for Gemini requests:
    - Fails fast on Daily Quota (RPD): 0 retries, halts pipeline immediately and marks daily flag.
    - 429 / RPM: at most ONE retry with cooldown (~65s or Retry-After); halts on second failure WITHOUT marking daily quota exhausted.
    - 503 Service Unavailable (High Demand): bounded exponential backoff:
        attempt 1 -> wait 15s (retry 1)
        attempt 2 -> wait 30s (retry 2)
        attempt 3 -> wait 60s (retry 3)
        attempt 4 -> wait 120s (retry 4)
        attempt 5 -> stop and checkpoint (raises ServiceUnavailableError)
      Maximum 4 retries after the initial request (5 attempts max).
      Uses provider Retry-After if explicitly provided, capped reasonably (min 5s, max 300s).
      Does NOT count against RPD quota and does NOT call mark_daily_quota_exhausted().
    - Billing / Non-recoverable: 0 retries, halts immediately.
    - Strictly prevents unbounded retry loops.
    """
    effective_state = retry_state if retry_state is not None else (retry_count if retry_count is not None else 0)
    is_legacy_int = isinstance(effective_state, int)
    if is_legacy_int:
        state = GeminiRetryState(max_rpm_retries=1, max_503_retries=4, max_other_retries=1)
        state.rpm_retries = effective_state
        state.retries_503 = effective_state
        state.other_retries = effective_state
        state.total_attempts = effective_state
    else:
        state = effective_state

    error_type, _ = qm.classify_error(e)

    if error_type == QuotaErrorType.RPD:
        qm.mark_daily_quota_exhausted(str(e))
        print(f"\n🛑 Daily Quota Limit Reached (RPD) on {op_name}: {e}. Halting immediately without retry.")
        raise DailyQuotaExhaustedError(f"Daily quota exhausted (RPD): {e}")

    elif error_type == QuotaErrorType.SERVICE_UNAVAILABLE:
        # 1. Roll back daily request count: 503 retries must NOT count as RPD exhaustion
        qm.decrement_daily_requests_count()
        qm.actual_calls_count = max(0, qm.actual_calls_count - 1)

        # 2. Bounded retries: max 4 retries after the initial request (5 attempts total)
        if state.retries_503 >= state.max_503_retries:
            print(
                f"\n🛑 503 Service Unavailable (high demand) repeated on {op_name}: {e}\n"
                f"   Maximum 4 retries exhausted. Halting without marking daily quota exhausted."
            )
            raise ServiceUnavailableError(
                f"503 Service Unavailable (high demand) on {op_name} exhausted after 4 retries: {e}"
            )

        # 3. Exponential backoff policy:
        # attempt 1 -> wait 15s (retry 1)
        # attempt 2 -> wait 30s (retry 2)
        # attempt 3 -> wait 60s (retry 3)
        # attempt 4 -> wait 120s (retry 4)
        delays_503 = [15.0, 30.0, 60.0, 120.0]
        base_delay = delays_503[min(state.retries_503, len(delays_503) - 1)]

        provider_retry_after = _extract_retry_after(e)
        if provider_retry_after is not None:
            wait_time = min(max(provider_retry_after, 5.0), 300.0)
            print(
                f"⏳ 503 Service Unavailable on {op_name} (retry {state.retries_503 + 1}/4). "
                f"Provider specified Retry-After: {provider_retry_after}s -> waiting {wait_time:.1f}s..."
            )
        else:
            test_override = os.getenv("GEMINI_503_TEST_DELAY")
            wait_time = float(test_override) if test_override and test_override.replace(".", "", 1).isdigit() else base_delay
            print(
                f"⏳ 503 Service Unavailable (high demand) on {op_name} (retry {state.retries_503 + 1}/4). "
                f"Waiting {wait_time:.1f}s before retry attempt {state.retries_503 + 2}..."
            )

        time.sleep(wait_time)
        state.retries_503 += 1
        state.total_attempts += 1
        return state.retries_503 if is_legacy_int else state

    elif error_type == QuotaErrorType.RPM:
        # Allow at most ONE retry for 429/RPM error
        if state.rpm_retries >= state.max_rpm_retries:
            print(f"\n🛑 Repeated 429 Rate Limit (RPM) on {op_name}: {e}. Halting after single allowed retry (Daily quota NOT exhausted).")
            raise RuntimeError(f"RPM rate limit (429) exceeded after retry on {op_name}: {e}")

        rpm_delay_env = os.getenv("GEMINI_RPM_RETRY_DELAY", "65.0")
        delay = float(rpm_delay_env) if rpm_delay_env else 65.0
        provider_retry_after = _extract_retry_after(e)
        if provider_retry_after is not None:
            delay = max(delay, provider_retry_after + 2.0)

        print(f"\n⏳ Per-minute rate limit (RPM/429) on {op_name}. Waiting {delay:.1f}s cooldown before single allowed retry attempt...")
        time.sleep(delay)
        state.rpm_retries += 1
        state.total_attempts += 1
        return state.rpm_retries if is_legacy_int else state

    elif error_type == QuotaErrorType.BILLING:
        print(f"\n🛑 Gemini Authentication/Billing Error on {op_name}: {e}. Halting immediately without retry.")
        raise PermissionError(f"Gemini API authentication or billing error: {e}")

    elif error_type == QuotaErrorType.NON_RECOVERABLE:
        print(f"\n🛑 Non-recoverable Gemini API Error on {op_name} ({e}). Cancelling retries immediately.")
        raise e

    else:
        if state.other_retries >= state.max_other_retries:
            raise RuntimeError(f"Gemini {op_name} failed after allowed retry: {str(e)}")
        delay = 2.0
        print(f"Gemini {op_name} retry after error: {e}. Waiting {delay:.1f}s...")
        time.sleep(delay)
        state.other_retries += 1
        state.total_attempts += 1
        return state.other_retries if is_legacy_int else state


# ============================================================================
# Gemini API Implementations (Free API)
# ============================================================================

def request_gemini_token(prompt: str, log_id=None, max_tokens: int = 8000, max_retries: int = 2):
    """Send text prompt to Gemini and return (universal_response_wrapper, usage_info)."""
    qm = QuotaManager()
    client = _get_gemini_client()
    config = types.GenerateContentConfig(max_output_tokens=max_tokens)

    model_name = _get_gemini_model()
    cache_key = qm.get_cache_key(prompt, model_name)

    # 1. Check persistent cache first (0 quota consumed, 0 wait time)
    cached = qm.get_cached(cache_key)
    if cached:
        text, usage_info = cached
        print(f"[CACHE HIT] Returning cached Gemini response ({usage_info.get('total_tokens', 0)} tokens)")
        return GeminiResponseWrapper(text, usage_info), usage_info

    # 2. Only on cache miss: synchronize via central scheduler
    retry_state = GeminiRetryState(max_rpm_retries=1, max_503_retries=4, max_other_retries=1)
    while retry_state.can_continue():
        try:
            with qm.api_execution_lock:
                # Double-checked locking: check cache again inside lock before consuming quota slot
                cached = qm.get_cached(cache_key)
                if cached:
                    text, usage_info = cached
                    print(f"[CACHE HIT] Returning cached Gemini response ({usage_info.get('total_tokens', 0)} tokens)")
                    return GeminiResponseWrapper(text, usage_info), usage_info

                qm.acquire()
                qm.actual_calls_count += 1
                daily_count = qm.increment_daily_requests_count()
                rpd_info = f", daily: {daily_count}/{qm.effective_rpd}" if qm.effective_rpd else f", daily: {daily_count}"
                print(f"📡 [GEMINI API CALL #{qm.actual_calls_count}{rpd_info}] Model: {model_name} (interval >= {qm.min_interval:.1f}s)")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                wrapper, usage_info = _extract_text_and_usage(response)
                qm.set_cached(cache_key, wrapper.text, usage_info)
                return wrapper, usage_info

        except Exception as e:
            _handle_gemini_error(e, qm, retry_state, max_retries, op_name="text request")

    return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def request_gemini(prompt: str, log_id=None, max_tokens: int = 8000, max_retries: int = 2):
    """Send text prompt to Gemini and return universal response wrapper."""
    wrapper, _ = request_gemini_token(prompt, log_id=log_id, max_tokens=max_tokens, max_retries=max_retries)
    return wrapper


def request_gemini_with_video(
    prompt: str, video_path: str, log_id=None, max_tokens: int = 10000, max_retries: int = 2
):
    """Multimodal request to Gemini with video + text prompt."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    qm = QuotaManager()
    video_stat = _get_stable_file_stat(video_path)
    client = _get_gemini_client()
    config = types.GenerateContentConfig(max_output_tokens=max_tokens)

    model_name = _get_gemini_model()
    cache_key = qm.get_cache_key(prompt, model_name, extra=video_stat)

    # 1. Check cache first (0 quota consumed)
    cached = qm.get_cached(cache_key)
    if cached:
        text, usage_info = cached
        print(f"[CACHE HIT] Returning cached Gemini video response ({usage_info.get('total_tokens', 0)} tokens)")
        return GeminiResponseWrapper(text, usage_info)

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    contents = [prompt, video_part]

    # 2. Only on cache miss: execute via central scheduler
    retry_state = GeminiRetryState(max_rpm_retries=1, max_503_retries=4, max_other_retries=1)
    while retry_state.can_continue():
        try:
            with qm.api_execution_lock:
                # Double-checked locking: check cache again inside lock before consuming quota slot
                cached = qm.get_cached(cache_key)
                if cached:
                    text, usage_info = cached
                    print(f"[CACHE HIT] Returning cached Gemini video response ({usage_info.get('total_tokens', 0)} tokens)")
                    return GeminiResponseWrapper(text, usage_info)

                qm.acquire()
                qm.actual_calls_count += 1
                daily_count = qm.increment_daily_requests_count()
                rpd_info = f", daily: {daily_count}/{qm.effective_rpd}" if qm.effective_rpd else f", daily: {daily_count}"
                print(f"📡 [GEMINI API CALL #{qm.actual_calls_count}{rpd_info}] Model: {model_name} (video request, interval >= {qm.min_interval:.1f}s)")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                wrapper, usage_info = _extract_text_and_usage(response)
                qm.set_cached(cache_key, wrapper.text, usage_info)
                return wrapper

        except Exception as e:
            _handle_gemini_error(e, qm, retry_state, max_retries, op_name="video request")
    return None


def request_gemini_video_img_token(
    prompt: str, video_path: str, image_path: str, log_id=None, max_tokens: int = 10000, max_retries: int = 2
):
    """Multimodal request to Gemini with video + reference image + text prompt, returning token usage."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    qm = QuotaManager()
    extra_stat = f"{_get_stable_file_stat(video_path)}:{_get_stable_file_stat(image_path)}"
    client = _get_gemini_client()
    config = types.GenerateContentConfig(max_output_tokens=max_tokens)

    model_name = _get_gemini_model()
    cache_key = qm.get_cache_key(prompt, model_name, extra=extra_stat)

    # 1. Check cache first (0 quota consumed)
    cached = qm.get_cached(cache_key)
    if cached:
        text, usage_info = cached
        print(f"[CACHE HIT] Returning cached Gemini video+img response ({usage_info.get('total_tokens', 0)} tokens)")
        return GeminiResponseWrapper(text, usage_info), usage_info

    with open(video_path, "rb") as f:
        video_bytes = f.read()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    contents = [prompt, video_part, image_part]

    # 2. Only on cache miss: execute via central scheduler
    retry_state = GeminiRetryState(max_rpm_retries=1, max_503_retries=4, max_other_retries=1)
    while retry_state.can_continue():
        try:
            with qm.api_execution_lock:
                # Double-checked locking: check cache again inside lock before consuming quota slot
                cached = qm.get_cached(cache_key)
                if cached:
                    text, usage_info = cached
                    print(f"[CACHE HIT] Returning cached Gemini video+img response ({usage_info.get('total_tokens', 0)} tokens)")
                    return GeminiResponseWrapper(text, usage_info), usage_info

                qm.acquire()
                qm.actual_calls_count += 1
                daily_count = qm.increment_daily_requests_count()
                rpd_info = f", daily: {daily_count}/{qm.effective_rpd}" if qm.effective_rpd else f", daily: {daily_count}"
                print(f"📡 [GEMINI API CALL #{qm.actual_calls_count}{rpd_info}] Model: {model_name} (video+img request, interval >= {qm.min_interval:.1f}s)")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                wrapper, usage_info = _extract_text_and_usage(response)
                qm.set_cached(cache_key, wrapper.text, usage_info)
                return wrapper, usage_info

        except Exception as e:
            _handle_gemini_error(e, qm, retry_state, max_retries, op_name="video+img request")

    return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def request_gemini_video_img(
    prompt: str, video_path: str, image_path: str, log_id=None, max_tokens: int = 10000, max_retries: int = 2
):
    """Multimodal request to Gemini with video + reference image + text prompt."""
    wrapper, _ = request_gemini_video_img_token(
        prompt, video_path, image_path, log_id=log_id, max_tokens=max_tokens, max_retries=max_retries
    )
    return wrapper


# ============================================================================
# OpenAI / Claude Callers with Free Gemini Fallback Routing
# ============================================================================

def _call_openai_or_gemini_fallback(prompt, model_name: str, max_tokens: int = 8000, max_retries: int = 3):
    """If OPENAI_API_KEY is configured, call OpenAI; otherwise route to free Gemini."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key not in ("...", "YOUR_OPENAI_KEY"):
        try:
            import openai
            client = openai.OpenAI(
                api_key=openai_key,
                base_url=os.getenv("OPENAI_BASE_URL", None),
            )
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            usage_info = {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
            }
            return completion, usage_info
        except Exception as e:
            print(f"OpenAI call failed ({e}), falling back to Gemini...")

    # Route to free Gemini
    wrapper, usage_info = request_gemini_token(prompt, max_tokens=max_tokens, max_retries=max_retries)
    return wrapper, usage_info


def request_claude(prompt, log_id=None, max_tokens=16384, max_retries=3):
    wrapper, _ = _call_openai_or_gemini_fallback(prompt, "claude", max_tokens=max_tokens, max_retries=max_retries)
    return str(wrapper).strip()


def request_claude_token(prompt, log_id=None, max_tokens=10000, max_retries=3):
    return _call_openai_or_gemini_fallback(prompt, "claude", max_tokens=max_tokens, max_retries=max_retries)


def request_gpt4o(prompt, log_id=None, max_tokens=8000, max_retries=3):
    wrapper, _ = _call_openai_or_gemini_fallback(prompt, "gpt-4o", max_tokens=max_tokens, max_retries=max_retries)
    return wrapper


def request_gpt4o_token(prompt, log_id=None, max_tokens=8000, max_retries=3):
    return _call_openai_or_gemini_fallback(prompt, "gpt-4o", max_tokens=max_tokens, max_retries=max_retries)


def request_o4mini(prompt, log_id=None, max_tokens=8000, max_retries=3, thinking=False):
    wrapper, _ = _call_openai_or_gemini_fallback(prompt, "o4-mini", max_tokens=max_tokens, max_retries=max_retries)
    return wrapper


def request_o4mini_token(prompt, log_id=None, max_tokens=8000, max_retries=3, thinking=False):
    return _call_openai_or_gemini_fallback(prompt, "o4-mini", max_tokens=max_tokens, max_retries=max_retries)


def request_gpt5(prompt, log_id=None, max_tokens=1000, max_retries=3):
    wrapper, _ = _call_openai_or_gemini_fallback(prompt, "gpt-5", max_tokens=max_tokens, max_retries=max_retries)
    return wrapper


def request_gpt5_token(prompt, log_id=None, max_tokens=1000, max_retries=3):
    return _call_openai_or_gemini_fallback(prompt, "gpt-5", max_tokens=max_tokens, max_retries=max_retries)


def request_gpt41(prompt, log_id=None, max_tokens=1000, max_retries=3):
    wrapper, _ = _call_openai_or_gemini_fallback(prompt, "gpt-4.1", max_tokens=max_tokens, max_retries=max_retries)
    return wrapper


def request_gpt41_token(prompt, log_id=None, max_tokens=1000, max_retries=3):
    return _call_openai_or_gemini_fallback(prompt, "gpt-4.1", max_tokens=max_tokens, max_retries=max_retries)


def request_gpt41_img(prompt, image_path=None, log_id=None, max_tokens=1000, max_retries=2):
    """If image provided, route to Gemini vision, otherwise text."""
    if image_path and os.path.exists(image_path):
        qm = QuotaManager()
        extra_stat = _get_stable_file_stat(image_path)
        client = _get_gemini_client()
        config = types.GenerateContentConfig(max_output_tokens=max_tokens)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        model_name = _get_gemini_model()
        cache_key = qm.get_cache_key(prompt, model_name, extra=extra_stat)
        cached = qm.get_cached(cache_key)
        if cached:
            text, usage_info = cached
            return GeminiResponseWrapper(text, usage_info)

        retry_state = GeminiRetryState(max_rpm_retries=1, max_503_retries=4, max_other_retries=1)
        while retry_state.can_continue():
            try:
                with qm.api_execution_lock:
                    # Double-checked locking: check cache again inside lock before consuming quota slot
                    cached = qm.get_cached(cache_key)
                    if cached:
                        text, usage_info = cached
                        return GeminiResponseWrapper(text, usage_info)

                    qm.acquire()
                    qm.actual_calls_count += 1
                    daily_count = qm.increment_daily_requests_count()
                    rpd_info = f", daily: {daily_count}/{qm.effective_rpd}" if qm.effective_rpd else f", daily: {daily_count}"
                    print(f"📡 [GEMINI API CALL #{qm.actual_calls_count}{rpd_info}] Model: {model_name} (image request, interval >= {qm.min_interval:.1f}s)")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image_part],
                        config=config,
                    )
                    wrapper, usage_info = _extract_text_and_usage(response)
                    qm.set_cached(cache_key, wrapper.text, usage_info)
                    return wrapper
            except Exception as e:
                _handle_gemini_error(e, qm, retry_state, max_retries, op_name="image request")

    return request_gemini(prompt, max_tokens=max_tokens, max_retries=max_retries)


# ============================================================================
# Model-Agnostic LLM Abstraction Integration
# ============================================================================

def generate_response(prompt, system_prompt=None, response_format=None, stage=None, provider=None, model=None, **kwargs):
    """Centralized unified interface handling all LLM calls across all CineLearn components."""
    from llm import generate_response as _gen
    return _gen(prompt, system_prompt=system_prompt, response_format=response_format, stage=stage, provider=provider, model=model, **kwargs)


def generate_response_token(prompt, stage=None, max_tokens=None, **kwargs):
    """Unified interface returning (response, usage_dict)."""
    from llm import generate_response_token as _gen_tok
    return _gen_tok(prompt, stage=stage, max_tokens=max_tokens, **kwargs)


def get_llm(stage=None, provider=None, model=None):
    """Get the configured BaseLLMProvider for a given stage or override."""
    from llm import get_llm as _get_llm
    return _get_llm(stage=stage, provider=provider, model=model)


def make_stage_caller(stage):
    """Returns a callable bounded to a specific stage that returns (response, usage_dict)."""
    from llm import make_stage_caller as _msc
    return _msc(stage=stage)


def request_ollama_token(prompt: str, model_name: Optional[str] = None, max_tokens: int = 8000, **kwargs):
    """Send text prompt to local Ollama and return (response, usage_dict)."""
    from llm import generate_response_token as _gen_tok
    return _gen_tok(prompt, provider="ollama", model=model_name, max_tokens=max_tokens, **kwargs)


def request_ollama(prompt: str, model_name: Optional[str] = None, max_tokens: int = 8000, **kwargs):
    """Send text prompt to local Ollama and return response."""
    resp, _ = request_ollama_token(prompt, model_name=model_name, max_tokens=max_tokens, **kwargs)
    return resp


if __name__ == "__main__":
    print("Testing gpt_request configuration...")
    try:
        model = _get_gemini_model()
        print(f"Configured Gemini model: {model}")
        client = _get_gemini_client()
        print("Gemini client initialized successfully!")
    except Exception as e:
        print(f"Notice: {e}")
