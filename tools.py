"""Tools Jarvis can call.

Every tool is a *client-side function*: the model asks for it by name with JSON arguments,
OUR code runs it, and we send the JSON result back. Definitions below are provider-neutral
(name / description / JSON-Schema parameters); `gemini_tools()` adapts them to Gemini's format.
Swapping LLM providers only means writing a new adapter, not rewriting the tools.
"""
from __future__ import annotations

from projections import Projector
from sleeper_client import SleeperClient

# The description is the model's ONLY documentation for a tool. Write it like an API doc
# for a smart colleague: what it returns and when to use it.
TOOL_SCHEMAS = [
    {
        "name": "get_league_overview",
        "description": "Standings for every team in the league: team name, owner, record, "
                       "points for/against, and which team belongs to the user.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_my_team",
        "description": "The user's full roster with model projections, injury status, who is "
                       "currently starting, and the model's optimal lineup for this week. "
                       "Call this first for any lineup / start-sit question.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_team",
        "description": "Another league team's roster with projections, looked up by team name "
                       "or owner name (fuzzy match).",
        "parameters": {"type": "object",
                         "properties": {"team": {"type": "string"}}, "required": ["team"]},
    },
    {
        "name": "get_matchup_projection",
        "description": "This week's projected score and win probability for a matchup, from "
                       "each side's CURRENT starters. Totals include K/DEF (from Sleeper's own "
                       "projections, not our model). Defaults to the user's matchup.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string", "description": "Optional: any team in the league"}}},
    },
    {
        "name": "get_optimal_lineup",
        "description": "THE lineup recommendation: the best legal starting lineup for this week "
                       "by projection, every slot filled, each with the top model factors behind "
                       "the number; plus the exact bench/start swaps vs what is currently set and "
                       "how many points are being left on the bench. Use this for the weekly "
                       "report and any 'who should I start' question.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_player_projections",
        "description": "XGBoost projections for NFL players by name: projected PPR points, "
                       "floor/ceiling, opponent, Vegas implied team total, injury status, the "
                       "top model factors behind the number (feature, value, impact in points), "
                       "and which league team rosters each player.",
        "parameters": {"type": "object",
                         "properties": {"names": {"type": "array", "items": {"type": "string"},
                                                  "maxItems": 12}},
                         "required": ["names"]},
    },
    {
        "name": "get_waiver_targets",
        "description": "Highest-projected QB/RB/WR/TE NOT rostered by anyone in the league. "
                       "For K/DEF streamers use web_search instead.",
        "parameters": {"type": "object", "properties": {
            "position": {"type": "string", "enum": ["QB", "RB", "WR", "TE", "ANY"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 15}}},
    },
    {
        "name": "web_search",
        "description": "Search the web for current info the model can't know: injury and practice "
                       "reports, weather forecasts for outdoor games, depth-chart news, and weekly "
                       "kicker/defense rankings. Use a specific query, e.g. "
                       "'Caleb Williams injury status week 3 2026'. Each call is costly: batch topics.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
]


def gemini_tools() -> list[dict]:
    """Interactions API function declarations: {"type": "function", name, description, parameters}."""
    return [{"type": "function", **t} for t in TOOL_SCHEMAS]


class ToolBox:
    def __init__(self, sleeper: SleeperClient, projector: Projector, searcher=None):
        self.s, self.p = sleeper, projector
        self.searcher = searcher
        self.week = projector.week

    def _roster_view(self, roster: dict) -> dict:
        starters = [x for x in (roster.get("starters") or []) if x and x != "0"]
        players = []
        for pid in (roster.get("players") or []):
            r = self.p.project(pid)
            r["why"] = [f["reads_as"] for f in (r.pop("top_model_factors", None) or [])[:3]]
            players.append({**r, "starting": pid in starters})
        players.sort(key=lambda x: (not x["starting"], -(x["proj"] or 0)))
        return {"team_name": self.s.team_name(roster), "record": _record(roster),
                "players": players}

    # ------------------------------------------------------------------ handlers
    def get_league_overview(self):
        users = self.s.users()
        rows = [{"team_name": self.s.team_name(r),
                 "owner": users.get(r.get("owner_id"), {}).get("display_name"),
                 "is_user": r.get("owner_id") == self.s.my_user_id,
                 **_record(r)} for r in self.s.rosters()]
        return sorted(rows, key=lambda x: (-x["wins"], -x["points_for"]))

    def get_my_team(self):
        roster = self.s.my_roster()
        view = self._roster_view(roster)
        view["week"] = self.week
        view.update(self.p.lineup_advice(roster["players"], roster.get("starters") or []))
        return view

    def get_optimal_lineup(self):
        roster = self.s.my_roster()
        return {"week": self.week, **self.p.lineup_advice(roster["players"],
                                                          roster.get("starters") or [])}

    def get_team(self, team: str):
        roster = self.s.find_roster(team)
        return self._roster_view(roster) if roster else {"error": f"No team matching '{team}'"}

    def get_matchup_projection(self, team: str | None = None):
        roster = self.s.find_roster(team) if team else self.s.my_roster()
        if not roster:
            return {"error": f"No team matching '{team}'"}
        ms = self.s.matchups(self.week)
        mine = next((m for m in ms if m["roster_id"] == roster["roster_id"]), None)
        opp = mine and next((m for m in ms if m["matchup_id"] == mine["matchup_id"]
                             and m["roster_id"] != mine["roster_id"]), None)
        if not opp:
            return {"error": "No matchup found this week."}
        by_id = {r["roster_id"]: r for r in self.s.rosters()}

        def side(m):
            mu, var, rows = self.p.score_lineup(m["starters"])
            return mu, var, {
                "team": self.s.team_name(by_id[m["roster_id"]]),
                "projected_total": round(mu, 1),
                "live_points": m.get("points"),
                "starters": sorted(
                    [{"name": r["name"], "pos": r["position"], "proj": r["proj"],
                      "floor": r.get("floor"), "ceiling": r.get("ceiling"),
                      "opponent": r.get("opponent"),
                      "vegas_implied_team_total": r.get("vegas_implied_team_total"),
                      "injury": r["injury_status"], "source": r.get("projection_source"),
                      "why": [f["reads_as"] for f in (r.get("top_model_factors") or [])[:3]],
                      "note": r.get("note")} for r in rows],
                    key=lambda x: -(x["proj"] or 0)),
            }

        a_mu, a_var, a = side(mine)
        b_mu, b_var, b = side(opp)
        result = {"week": self.week, "side": a, "opponent": b,
                  "win_probability_as_currently_set": round(self.p.win_prob(a_mu, a_var, b_mu, b_var), 3)}
        if mine["roster_id"] == self.s.my_roster()["roster_id"]:
            advice = self.p.lineup_advice(by_id[mine["roster_id"]]["players"], mine["starters"])
            opt_mu = advice["optimal_lineup_projected"]
            result["if_you_start_the_optimal_lineup"] = {
                "projected_total": opt_mu,
                "points_left_on_bench": advice["points_left_on_bench"],
                "changes": advice["changes"],
                "win_probability": round(self.p.win_prob(opt_mu, a_var, b_mu, b_var), 3),
            }
        return result

    def get_player_projections(self, names: list[str]):
        out = []
        for n in names[:12]:
            pid = self.s.search_player(n)
            if not pid:
                out.append({"query": n, "error": "player not found"})
                continue
            out.append({**self.p.project(pid), "rostered_by": self.s.owner_of(pid) or "free agent"})
        return out

    def get_waiver_targets(self, position: str = "ANY", limit: int = 8):
        table = self.p.all_projections()
        cands = table[~table["sleeper_id"].isin(self.s.rostered_ids()) & table["opp"].notna()]
        if position != "ANY":
            cands = cands[cands["position"] == position]
        top = cands.sort_values("proj", ascending=False).head(limit)
        return [self.p.project(pid) for pid in top["sleeper_id"]]

    def web_search(self, query: str):
        if self.searcher is None:
            return {"error": "web search not configured"}
        return self.searcher.search(query)

    # ------------------------------------------------------------------ dispatch
    def run(self, name: str, args: dict):
        fn = getattr(self, name, None)
        if name.startswith("_") or name == "run" or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:  # model sent bad/missing arguments
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # hand errors back to the model so it can recover
            if type(e).__name__ == "QuotaExhausted":
                raise  # out of free requests: stop the whole loop, don't keep trying
            return {"error": f"{type(e).__name__}: {e}"}


def _record(r: dict) -> dict:
    s = r.get("settings") or {}
    return {"wins": s.get("wins", 0), "losses": s.get("losses", 0), "ties": s.get("ties", 0),
            "points_for": s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100,
            "points_against": s.get("fpts_against", 0) + s.get("fpts_against_decimal", 0) / 100}
