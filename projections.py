"""Live weekly projections: Sleeper id -> gsis id -> features -> XGBoost -> explained projection."""
from __future__ import annotations

import logging
import math

import pandas as pd
import xgboost as xgb

import model_registry
import sleeper_projections
from config import SETTINGS
from feature_config import MODEL_CONFIGS
from features import build_features, load_raw, normalize_team

log = logging.getLogger(__name__)
NOT_MODELED = {"K", "DEF"}
# Plain-English names so the agent can explain a projection instead of reciting column names.
FEATURE_GLOSSARY = {
    "fpts": "fantasy points", "targets": "targets", "target_share": "share of team targets",
    "wopr": "opportunity rating (WOPR)", "offense_pct": "snap share",
    "offense_snaps": "snaps", "receptions": "catches", "receiving_yards": "receiving yards",
    "receiving_air_yards": "air yards", "air_yards_per_target": "average depth of target",
    "rec_fantasy_points_exp": "expected receiving points",
    "rec_fp_over_exp": "points scored above expectation",
    "rz_targets": "red-zone targets", "gl_targets": "targets inside the 10",
    "avg_separation": "separation from coverage", "avg_cushion": "cushion given by corners",
    "catch_percentage": "catch rate", "avg_yac_above_expectation": "YAC above expectation",
    "percent_share_of_intended_air_yards": "share of team air yards",
    "carries": "carries", "carry_share": "share of team carries",
    "weighted_opps": "weighted opportunities (carries + 2x targets)",
    "rushing_yards": "rushing yards", "yards_per_carry": "yards per carry",
    "rushing_epa": "rushing EPA", "gl_carries": "goal-line carries (inside the 5)",
    "rz_carries": "red-zone carries", "rush_fantasy_points_exp": "expected rushing points",
    "efficiency": "rushing efficiency", "rush_yards_over_expected_per_att": "yards over expected per carry",
    "percent_attempts_gte_eight_defenders": "rate of stacked boxes faced",
    "total_fp_over_exp": "points scored above expectation",
    "attempts": "pass attempts", "dropbacks": "dropbacks", "passing_yards": "passing yards",
    "passing_air_yards": "air yards thrown", "epa_per_dropback": "EPA per dropback",
    "anya": "adjusted net yards per attempt", "passing_cpoe": "completion % over expected",
    "passer_rating": "passer rating", "avg_time_to_throw": "time to throw",
    "avg_intended_air_yards": "intended air yards", "aggressiveness": "throws into tight coverage",
    "scrambles": "scrambles", "designed_runs": "designed runs",
    "qb_rush_pts": "rushing fantasy points", "rush_share_of_pts": "share of points from rushing",
    "rz_pass_att": "red-zone pass attempts", "pass_fantasy_points_exp": "expected passing points",
    "implied_total": "Vegas implied team total", "spread": "point spread", "total": "game total",
    "is_home": "home game", "is_dome": "indoor game", "is_te": "tight end",
    "def_pts_allowed_roll6": "points this defense allows to the position",
    "def_pts_allowed": "points this defense allows to the position",
    "games_this_season": "games played this season",
}
SUFFIXES = {"_roll3": "3-game avg", "_roll6": "6-game avg", "_roll10": "10-game avg",
            "_std3": "week-to-week volatility", "_trend3": "recent trend", "_trend7": "longer trend"}


def humanize(feature: str) -> str:
    for suffix, label in SUFFIXES.items():
        if feature.endswith(suffix):
            base = feature[: -len(suffix)]
            return f"{FEATURE_GLOSSARY.get(base, base)} ({label})"
    return FEATURE_GLOSSARY.get(feature, feature)
FLEX_ELIGIBLE = {
    "FLEX": {"RB", "WR", "TE"}, "WRRB_FLEX": {"RB", "WR"}, "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}
OUT_STATUSES = {"Out", "IR", "PUP", "Sus", "NA", "COV"}


def _clean_id(x) -> str | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s or None


class Projector:
    def __init__(self, sleeper, season: int, week: int):
        self.sleeper, self.season, self.week = sleeper, season, week
        self.models = {}  # slot -> (version, model, meta)
        for slot, version in model_registry.active().items():
            model, meta = model_registry.load(version)
            self.models[slot] = (version, model, meta)
            log.info("slot %s -> %s (test RMSE %.2f)", slot, version, meta["test"]["rmse"])
        self.table: pd.DataFrame | None = None
        self.kdef: dict[str, float] = {}      # sleeper_id -> projection (K/DEF, from Sleeper)
        self.teams_playing: set[str] = set()  # teams with a real game this week

    # ------------------------------------------------------------------ id mapping
    def _id_map(self) -> dict[str, str]:
        """sleeper_id -> gsis_id. DynastyProcess crosswalk first, Sleeper's own field as fallback."""
        import nflreadpy as nfl
        ids = nfl.load_ff_playerids().to_pandas()
        mapping = {}
        for s, g in zip(ids["sleeper_id"], ids["gsis_id"]):
            s, g = _clean_id(s), _clean_id(g)
            if s and g:
                mapping[s] = g
        for pid, p in self.sleeper.players().items():
            g = _clean_id(p.get("gsis_id"))
            if g and pid not in mapping:
                mapping[pid] = g
        return mapping

    # ------------------------------------------------------------------ scoring
    def refresh(self):
        gsis_to_sleeper = {g: s for s, g in self._id_map().items()}
        sp = self.sleeper.players()
        overrides = {g: sp[s]["team"] for g, s in gsis_to_sleeper.items()
                     if s in sp and sp[s].get("team")}

        raw = load_raw([self.season - 1, self.season], include_pbp=True)

        # Who actually plays this week? Lets us tell a true bye from a broken team-code join.
        sched = raw["schedules"]
        this_week = sched[(sched["season"] == self.season) & (sched["week"] == self.week)]
        self.teams_playing = set(this_week["home_team"]) | set(this_week["away_team"])
        self.kdef = sleeper_projections.fetch(self.season, self.week, SETTINGS.scoring)

        frames = []
        for slot, (version, model, meta) in self.models.items():
            feats = meta["features"]
            df, built = build_features(raw, MODEL_CONFIGS[version], SETTINGS.scoring,
                                       upcoming=(self.season, self.week), team_overrides=overrides)
            if missing := set(feats) - set(built):
                log.warning("%s: %d trained features missing at inference: %s",
                            version, len(missing), sorted(missing)[:10])
            live = df[df["target"].isna() & (df["ctx_season"] == self.season)
                      & (df["ctx_week"] == self.week)].copy()
            if live.empty:
                continue
            X = live.reindex(columns=feats)
            live["proj"] = model.predict(X).clip(min=0)
            # SHAP values for trees, computed natively by XGBoost: one column per feature + bias
            contribs = model.get_booster().predict(xgb.DMatrix(X, feature_names=feats),
                                                   pred_contribs=True)
            live["why"] = [self._top_factors(c, X.iloc[i], feats) for i, c in enumerate(contribs)]
            live["rmse"] = meta["test"]["rmse"]
            live["model_version"] = version
            live["sleeper_id"] = live["player_id"].map(gsis_to_sleeper)
            frames.append(live)
        self.table = (pd.concat(frames, ignore_index=True)
                      .dropna(subset=["sleeper_id"])
                      .drop_duplicates("sleeper_id")
                      .set_index("sleeper_id", drop=False))
        log.info("Projected %d players for %s week %s", len(self.table), self.season, self.week)

    @staticmethod
    def _top_factors(contrib_row, x_row, feats, k=4):
        """TreeSHAP contributions, as sentences the agent can quote directly."""
        pairs = sorted(zip(feats, contrib_row[:-1]), key=lambda t: -abs(t[1]))[:k]
        out = []
        for f, c in pairs:
            val = None if pd.isna(x_row[f]) else round(float(x_row[f]), 2)
            out.append({
                "feature": f,
                "meaning": humanize(f),
                "value": val,
                "impact_pts": round(float(c), 2),
                "reads_as": f"{humanize(f)} = {val} -> {'+' if c >= 0 else ''}{round(float(c), 1)} pts",
            })
        return out

    # ------------------------------------------------------------------ lookups
    def project(self, sleeper_id: str) -> dict:
        if self.table is None:
            self.refresh()
        card = self.sleeper.player_card(sleeper_id)
        if card["position"] in NOT_MODELED:
            pos = card["position"]
            proj = self.kdef.get(str(sleeper_id))
            source = "Sleeper projection" if proj is not None else "flat league average"
            proj = sleeper_projections.FALLBACK[pos] if proj is None else proj
            rmse = sleeper_projections.RMSE[pos]
            return {**card, "proj": round(float(proj), 1), "rmse": rmse,
                    "floor": round(max(proj - rmse, 0), 1), "ceiling": round(proj + rmse, 1),
                    "projection_source": source,
                    "note": f"not from our XGBoost model ({source}); sanity-check streaming "
                            f"decisions with web_search"}
        if sleeper_id not in self.table.index:
            return {**card, "proj": 0.0, "rmse": 3.0,
                    "note": "no projection (bye week, no recent NFL games, or unmapped id)"}
        r = self.table.loc[sleeper_id]
        if pd.isna(r["opp"]):
            team = normalize_team(card.get("team"))
            if team and team in self.teams_playing:
                note = (f"no game context found for {card['team']} in week {self.week} even though "
                        f"they play: likely a data/id problem, NOT a bye. Say so instead of "
                        f"claiming a bye.")
            else:
                note = f"BYE week for {card.get('team')} in week {self.week}"
            return {**card, "proj": 0.0, "rmse": 0.0, "note": note}
        proj, rmse = float(r["proj"]), float(r["rmse"])
        return {
            **card,
            "proj": round(proj, 1),
            "floor": round(max(proj - rmse, 0), 1),
            "ceiling": round(proj + rmse, 1),
            "rmse": round(rmse, 2),
            "opponent": r["opp"],
            "home": bool(r["is_home"]),
            "dome": bool(r["is_dome"]),
            "vegas_implied_team_total": None if pd.isna(r["implied_total"]) else float(r["implied_total"]),
            "last_game": f"{int(r['season'])} wk{int(r['week'])}: {r['fpts']:.1f} pts",
            "top_model_factors": r["why"],
            "note": ("player is listed OUT: treat as 0" if card["injury_status"] in OUT_STATUSES
                     else None),
        }

    def all_projections(self) -> pd.DataFrame:
        if self.table is None:
            self.refresh()
        return self.table

    # ------------------------------------------------------------------ lineup & matchup
    @staticmethod
    def effective(p: dict) -> float:
        return 0.0 if p["injury_status"] in OUT_STATUSES or p.get("proj") is None else p["proj"]

    def optimal_lineup(self, player_ids: list[str], current_starters: list[str] | None = None) -> list[dict]:
        """Greedy: fill single-position slots first, then flex, highest projection wins.
        K/DEF are included using Sleeper's projections."""
        pool = sorted((self.project(pid) for pid in player_ids), key=lambda p: -self.effective(p))
        slots = [s for s in self.sleeper.roster_positions if s not in ("BN", "IR", "TAXI")]
        used, lineup = set(), [None] * len(slots)
        for i in sorted(range(len(slots)), key=lambda i: slots[i] in FLEX_ELIGIBLE):
            ok = FLEX_ELIGIBLE.get(slots[i], {slots[i]})
            pick = next((p for p in pool if p["sleeper_id"] not in used and p["position"] in ok), None)
            if pick:
                used.add(pick["sleeper_id"])
                lineup[i] = {"slot": slots[i], "name": pick["name"], "sleeper_id": pick["sleeper_id"],
                             "proj": self.effective(pick), "injury_status": pick["injury_status"]}
        return [x for x in lineup if x]

    def lineup_advice(self, player_ids: list[str], current_starters: list[str]) -> dict:
        """Optimal lineup, what's currently set, and the exact swaps between them."""
        optimal = self.optimal_lineup(player_ids)
        current = [p for p in (current_starters or []) if p and p != "0"]
        opt_ids = {x["sleeper_id"] for x in optimal}
        cur_ids = set(current)

        bench_them = [self.project(p) for p in current if p not in opt_ids]
        start_them = [self.project(x["sleeper_id"]) for x in optimal if x["sleeper_id"] not in cur_ids]
        cur_total = sum(self.effective(self.project(p)) for p in current)
        opt_total = sum(x["proj"] or 0 for x in optimal)
        return {
            "optimal_lineup": [
                {**x, "why": (self.project(x["sleeper_id"]).get("top_model_factors") or [])[:3]}
                for x in optimal
            ],
            "changes": [
                {"bench": b["name"], "bench_proj": b["proj"], "bench_injury": b["injury_status"],
                 "start": s["name"], "start_proj": s["proj"], "start_injury": s["injury_status"],
                 "gain_pts": round((s["proj"] or 0) - (b["proj"] or 0), 1)}
                for b, s in zip(bench_them, start_them)
            ],
            "current_lineup_projected": round(cur_total, 1),
            "optimal_lineup_projected": round(opt_total, 1),
            "points_left_on_bench": round(opt_total - cur_total, 1),
        }

    def score_lineup(self, starter_ids: list[str]) -> tuple[float, float, list[dict]]:
        """Full projected total (QB/RB/WR/TE from our model, K/DEF from Sleeper) and its variance."""
        rows = [self.project(p) for p in starter_ids if p and p != "0"]
        scored = [r for r in rows if r.get("proj") is not None]
        mean = sum(self.effective(r) for r in scored)
        var = sum(r.get("rmse", 4.0) ** 2 for r in scored)
        return mean, var, rows

    @staticmethod
    def win_prob(my_mean, my_var, opp_mean, opp_var) -> float:
        # Difference of two independent normals is normal: P(diff > 0)
        sd = math.sqrt(my_var + opp_var) or 1.0
        return 0.5 * (1 + math.erf((my_mean - opp_mean) / (sd * math.sqrt(2))))
