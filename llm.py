"""A thin wrapper around the Gemini client that respects free-tier limits.

Free tier = three ceilings: requests/minute, tokens/minute, requests/day (resets midnight Pacific).
  * spacing:  wait so we never exceed `rpm` requests per minute
  * retries:  on HTTP 429, back off and try again (per-minute limits recover quickly)
  * counter:  track today's request count so you can see your daily budget draining
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from config import CACHE_DIR, SETTINGS

log = logging.getLogger(__name__)
USAGE_FILE = CACHE_DIR / "gemini_usage.json"


class QuotaExhausted(RuntimeError):
    pass


def _is_rate_limit(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    text = str(e)
    return code == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text


class GeminiCaller:
    def __init__(self, rpm: int = SETTINGS.rpm_limit, max_retries: int = 3):
        from google import genai  # pip install "google-genai>=2.3.0"

        self.client = genai.Client(api_key=SETTINGS.gemini_api_key or None)
        self.min_gap = 60.0 / max(rpm, 1)
        self.max_retries = max_retries
        self._last = 0.0

    # ------------------------------------------------------------------ usage counter
    @staticmethod
    def _today() -> str:
        return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()

    def requests_today(self) -> int:
        if not USAGE_FILE.exists():
            return 0
        data = json.loads(USAGE_FILE.read_text())
        return data.get(self._today(), 0)

    def _count(self):
        data = json.loads(USAGE_FILE.read_text()) if USAGE_FILE.exists() else {}
        data = {self._today(): data.get(self._today(), 0) + 1}  # keep only today
        USAGE_FILE.write_text(json.dumps(data))

    # ------------------------------------------------------------------ the call
    def create(self, **kwargs):
        """Same arguments as client.interactions.create, plus spacing and 429 retries.
        Pass _max_retries=N to override the instance default for this one call (e.g. web_search
        uses 0, since a 429 there is usually a separate, non-recovering grounding quota)."""
        max_retries = kwargs.pop("_max_retries", self.max_retries)
        for attempt in range(max_retries + 1):
            wait = self.min_gap - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self._count()
            try:
                return self.client.interactions.create(**kwargs)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                is_daily = "per day" in str(e).lower() or "PerDay" in str(e)
                if is_daily:
                    raise QuotaExhausted(
                        "Gemini daily free-tier limit reached. Resets at midnight Pacific; "
                        f"check aistudio.google.com/rate-limit. ({e})") from e
                if attempt == max_retries:
                    # We've retried and it's STILL 429, but nothing said "per day" — this is
                    # most often Search grounding, which has its own quota (often 0 on a
                    # no-billing project) separate from the RPM/RPD shown for plain requests.
                    # Don't claim it's a daily model quota; we don't actually know that.
                    raise QuotaExhausted(
                        "Gemini keeps returning 429 after retries. If this call used web_search, "
                        "Search grounding often has its own (sometimes zero) free quota, separate "
                        "from the RPM shown on aistudio.google.com/rate-limit for plain model "
                        f"calls. Otherwise, wait a bit and try again. ({e})") from e
                backoff = 20 * (attempt + 1)
                log.warning("429 rate limit, retrying in %ss", backoff)
                time.sleep(backoff)
