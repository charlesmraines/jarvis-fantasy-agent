"""Kicker and defense projections from Sleeper's own numbers.

We don't model K/DEF (too little signal, and their fantasy points are mostly opponent-driven),
but leaving them out of a matchup total makes projected scores look ~15 points too low. So we
read Sleeper's public projections endpoint for those two positions only.

Caveat: this endpoint is public and read-only but undocumented, so the response shape could
change. Everything here is defensive: on any failure we fall back to flat league-average values
and say so, rather than breaking the report.
"""
from __future__ import annotations

import json
import logging
import time

import requests

from config import CACHE_DIR

log = logging.getLogger(__name__)
URL = "https://api.sleeper.app/projections/nfl/{season}/{week}"
CACHE_TTL = 6 * 3600

# Used only if the endpoint is unavailable. Rough league averages.
FALLBACK = {"K": 8.0, "DEF": 7.0}
# Week-to-week spread for these positions, for the win-probability math.
RMSE = {"K": 4.0, "DEF": 6.0}
POINTS_KEY = {"ppr": "pts_ppr", "half": "pts_half_ppr", "std": "pts_std"}


def _parse(payload, scoring: str) -> dict[str, float]:
    """Sleeper returns a list of {player_id, stats: {pts_ppr: ...}}; tolerate a dict too."""
    items = payload.values() if isinstance(payload, dict) else payload
    key = POINTS_KEY.get(scoring, "pts_ppr")
    out = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        pid = item.get("player_id") or (item.get("player") or {}).get("player_id")
        stats = item.get("stats") or {}
        pts = stats.get(key)
        if pts is None:  # K/DEF scoring is the same in every format, so any key will do
            pts = stats.get("pts_ppr", stats.get("pts_std", stats.get("pts_half_ppr")))
        if pid is not None and pts is not None:
            out[str(pid)] = float(pts)
    return out


def fetch(season: int, week: int, scoring: str = "ppr") -> dict[str, float]:
    """sleeper_id -> projected points for K and DEF only. Empty dict if unavailable."""
    cache = CACHE_DIR / f"sleeper_proj_{season}_{week}_{scoring}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL:
        return json.loads(cache.read_text())
    try:
        r = requests.get(URL.format(season=season, week=week),
                         params={"season_type": "regular", "position[]": ["K", "DEF"],
                                 "order_by": "ppr"}, timeout=15)
        r.raise_for_status()
        data = _parse(r.json(), scoring)
        if not data:
            raise ValueError("no K/DEF rows in response")
        cache.write_text(json.dumps(data))
        log.info("Sleeper K/DEF projections: %d players", len(data))
        return data
    except Exception as e:
        log.warning("Sleeper K/DEF projections unavailable (%s); using flat averages", e)
        return {}
