"""Central settings, read from environment / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "data_cache"
REPORT_DIR = ROOT / "reports"
for d in (MODEL_DIR, CACHE_DIR, REPORT_DIR):
    d.mkdir(exist_ok=True)


@dataclass
class Settings:
    league_id: str = os.getenv("SLEEPER_LEAGUE_ID", "")
    username: str = os.getenv("SLEEPER_USERNAME", "")
    owner_name: str = os.getenv("OWNER_NAME", "Charlie")
    scoring: str = os.getenv("SCORING", "ppr")
    first_season: int = 2020
    # --- Gemini (free tier) ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # Flash-Lite has a much bigger free daily allowance than full Flash, and agents burn
    # one request per loop step. Use full Flash for the once-a-week report if you like.
    chat_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    report_model: str = os.getenv("GEMINI_REPORT_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    rpm_limit: int = int(os.getenv("GEMINI_RPM", "10"))  # set to your AI Studio limit
    thinking_level: str = os.getenv("GEMINI_THINKING_LEVEL", "")  # optional, e.g. "low"


SETTINGS = Settings()
