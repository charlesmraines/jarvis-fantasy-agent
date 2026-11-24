"""Read-only Sleeper league access (sleeper-api-wrapper + a couple of raw REST calls)."""
from __future__ import annotations

import difflib
import json
import time
from functools import cached_property

import requests
from sleeper_wrapper import League, Players

from config import CACHE_DIR

API = "https://api.sleeper.app/v1"
FANTASY_POS = {"QB", "RB", "WR", "TE", "K", "DEF"}


class SleeperClient:
    def __init__(self, league_id: str, username: str):
        self.league_id = league_id
        self.league = League(league_id)
        self.username = username

    # ---------------------------------------------------------------- basics
    def nfl_state(self) -> dict:
        return requests.get(f"{API}/state/nfl", timeout=10).json()

    @cached_property
    def my_user_id(self) -> str:
        return requests.get(f"{API}/user/{self.username}", timeout=10).json()["user_id"]

    @cached_property
    def info(self) -> dict:
        return self.league.get_league()

    @property
    def roster_positions(self) -> list[str]:
        return self.info["roster_positions"]

    def players(self) -> dict:
        """~5 MB dump; Sleeper asks you to fetch it at most once a day, so cache it."""
        path = CACHE_DIR / "sleeper_players.json"
        if not path.exists() or time.time() - path.stat().st_mtime > 86400:
            path.write_text(json.dumps(Players().get_all_players()))
        if not hasattr(self, "_players"):
            self._players = json.loads(path.read_text())
        return self._players

    # ---------------------------------------------------------------- league data
    def users(self) -> dict[str, dict]:
        return {u["user_id"]: u for u in self.league.get_users()}

    def rosters(self) -> list[dict]:
        return self.league.get_rosters()

    def matchups(self, week: int) -> list[dict]:
        return self.league.get_matchups(week) or []

    def team_name(self, roster: dict) -> str:
        u = self.users().get(roster.get("owner_id"), {})
        return (u.get("metadata") or {}).get("team_name") or u.get("display_name", "Unknown")

    def my_roster(self) -> dict:
        return next(r for r in self.rosters() if r["owner_id"] == self.my_user_id)

    def find_roster(self, query: str) -> dict | None:
        """Match on team name or owner display name (fuzzy)."""
        users = self.users()
        options = {}
        for r in self.rosters():
            u = users.get(r.get("owner_id"), {})
            options[self.team_name(r).lower()] = r
            options[u.get("display_name", "").lower()] = r
        hit = difflib.get_close_matches(query.lower(), list(options), n=1, cutoff=0.4)
        return options[hit[0]] if hit else None

    def rostered_ids(self) -> set[str]:
        return {p for r in self.rosters() for p in (r.get("players") or [])}

    def owner_of(self, player_id: str) -> str | None:
        for r in self.rosters():
            if player_id in (r.get("players") or []):
                return self.team_name(r)
        return None

    # ---------------------------------------------------------------- players
    def player_card(self, pid: str) -> dict:
        p = self.players().get(pid, {})
        return {
            "sleeper_id": pid,
            "name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid,
            "position": p.get("position"),
            "team": p.get("team"),
            "injury_status": p.get("injury_status"),  # Questionable / Doubtful / Out / IR ...
            "injury_body_part": p.get("injury_body_part"),
            "practice_participation": p.get("practice_participation"),
        }

    def search_player(self, name: str) -> str | None:
        if not hasattr(self, "_name_index"):
            self._name_index = {}
            for pid, p in self.players().items():
                if p.get("position") in FANTASY_POS and p.get("active") and p.get("full_name"):
                    self._name_index.setdefault(p["full_name"].lower(), pid)
        hit = difflib.get_close_matches(name.lower(), list(self._name_index), n=1, cutoff=0.6)
        return self._name_index[hit[0]] if hit else None
