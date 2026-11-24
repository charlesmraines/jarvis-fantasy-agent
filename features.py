"""Feature engineering shared by training AND live inference, driven by feature_config.py.

Row = one player-game. Target = the player's fantasy points in his NEXT game that season.
Every feature uses only information known after the current game, so there is no leakage.
"""
from __future__ import annotations

import logging
from functools import reduce

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
KEYS = ["player_id", "season", "week"]
MAX_REG_WEEK = 18
# Sleeper and nflverse disagree on some team abbreviations. Sleeper says LAR, nflverse says LA;
# without this, every Rams player joins to no schedule row and looks like he's on a bye.
TEAM_ALIASES = {
    "LAR": "LA", "STL": "LA", "SL": "LA",
    "JAC": "JAX", "WSH": "WAS", "WFT": "WAS",
    "OAK": "LV", "SD": "LAC", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}


def normalize_team(team):
    """Map any known alias onto the nflverse abbreviation."""
    if not isinstance(team, str):
        return team
    return TEAM_ALIASES.get(team.upper(), team.upper())
PBP_COLUMNS = ["season", "week", "season_type", "yardline_100", "rush_attempt", "pass_attempt",
               "qb_scramble", "rusher_player_id", "receiver_player_id", "passer_player_id"]

# Identifier / context columns that must never be fed to the model
NON_FEATURES = set(KEYS) | {
    "player_display_name", "position", "team", "opponent_team", "opp", "target",
    "next_week", "ctx_season", "ctx_week", "ord", "roof",
    "rush_attempt_team",  # only used as a denominator for carry_share
}


# ============================================================================ loading
def _pd(x) -> pd.DataFrame:
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def _num_keys(df: pd.DataFrame) -> pd.DataFrame:
    for k in ("season", "week"):
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def load_raw(seasons: list[int], include_pbp: bool = True) -> dict:
    """Download everything once; build_features() is then called once per model config."""
    import nflreadpy as nfl

    log.info("Loading nflverse data for %s ...", seasons)
    raw = {
        "stats": _pd(nfl.load_player_stats(seasons)),
        "snaps": _pd(nfl.load_snap_counts(seasons)),
        "opp": _pd(nfl.load_ff_opportunity(seasons)),
        "players": _pd(nfl.load_players()),
        "schedules": _pd(nfl.load_schedules(seasons)),
        "ngs": {t: _pd(nfl.load_nextgen_stats(seasons, stat_type=t))
                for t in ("receiving", "rushing", "passing")},
        "pbp": _load_pbp_features(seasons) if include_pbp else None,
    }
    for k in ("stats", "snaps", "opp", "schedules"):
        _num_keys(raw[k])
    for df in raw["ngs"].values():
        _num_keys(df)
    return raw


def _load_pbp_features(seasons) -> pd.DataFrame:
    """Play-by-play is huge (~370 columns): load one season at a time, keep only what we need."""
    import nflreadpy as nfl

    parts = []
    for s in seasons:
        pbp = nfl.load_pbp([s])
        cols = [c for c in PBP_COLUMNS if c in pbp.columns]
        parts.append(aggregate_pbp(pbp.select(cols).to_pandas()))
    return pd.concat(parts, ignore_index=True)


def aggregate_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per player-game red-zone / goal-line / scramble counts."""
    pbp = _num_keys(pbp[pbp["season_type"] == "REG"].copy())
    for c in ("rush_attempt", "pass_attempt", "qb_scramble"):
        pbp[c] = pbp[c].fillna(0)
    yl = pbp["yardline_100"]
    scramble = pbp["qb_scramble"] == 1
    rush = (pbp["rush_attempt"] == 1) & ~scramble
    targeted = (pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()
    passed = (pbp["pass_attempt"] == 1) & ~scramble & pbp["passer_player_id"].notna()

    def count(mask, ids, name):
        d = pbp.loc[mask, ["season", "week"]].assign(player_id=ids[mask])
        return d.dropna(subset=["player_id"]).groupby(KEYS).size().rename(name)

    rusher, receiver, passer = (pbp["rusher_player_id"], pbp["receiver_player_id"],
                                pbp["passer_player_id"])
    series = [
        count(scramble, rusher.fillna(passer), "scrambles"),
        count(rush, rusher, "designed_runs"),
        count(rush & (yl <= 5), rusher, "gl_carries"),
        count(rush & (yl <= 20), rusher, "rz_carries"),
        count(targeted & (yl <= 20), receiver, "rz_targets"),
        count(targeted & (yl <= 10), receiver, "gl_targets"),
        count(passed & (yl <= 20), passer, "rz_pass_att"),
    ]
    return pd.concat(series, axis=1).fillna(0).reset_index()


# ============================================================================ helpers
def _avail(df: pd.DataFrame, cols: list[str], label: str) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.warning("%s: columns not found, skipping: %s", label, missing)
    return [c for c in cols if c in df.columns]


def fantasy_points(stats: pd.DataFrame, scoring: str) -> pd.Series:
    ppr, std = stats["fantasy_points_ppr"], stats["fantasy_points"]
    return {"ppr": ppr, "half": (ppr + std) / 2, "std": std}[scoring]


def game_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with Vegas lines. nflverse spread_line > 0 means HOME is favored."""
    s = schedules[schedules["game_type"] == "REG"]
    common = {"ctx_season": s["season"].astype(float), "ctx_week": s["week"].astype(float),
              "total": s["total_line"], "roof": s["roof"]}
    home = pd.DataFrame({**common, "team": s["home_team"], "opp": s["away_team"],
                         "is_home": 1, "spread": s["spread_line"]})
    away = pd.DataFrame({**common, "team": s["away_team"], "opp": s["home_team"],
                         "is_home": 0, "spread": -s["spread_line"]})
    ctx = pd.concat([home, away], ignore_index=True)
    ctx["implied_total"] = (ctx["total"] + ctx["spread"]) / 2
    ctx["is_dome"] = ctx["roof"].isin(["dome", "closed"]).astype(int)
    return ctx


def defense_allowed(stats: pd.DataFrame, positions: list[str]) -> pd.DataFrame:
    """Rolling fantasy points a defense allowed to this position group over its last 6 games."""
    d = (stats[stats["position"].isin(positions)]
         .groupby(["season", "week", "opponent_team"])["fpts"].sum().reset_index()
         .rename(columns={"opponent_team": "opp", "fpts": "pts_allowed"})
         .sort_values(["season", "week"]))
    d["def_pts_allowed_roll6"] = (d.groupby("opp")["pts_allowed"]
                                  .transform(lambda x: x.rolling(6, min_periods=1).mean()))
    d["ord"] = (d["season"] * 100 + d["week"]).astype("int64")
    return d[["opp", "ord", "def_pts_allowed_roll6"]].sort_values("ord")


# ============================================================================ builder
def build_features(raw: dict, cfg: dict, scoring: str = "ppr",
                   upcoming: tuple[int, int] | None = None,
                   team_overrides: dict[str, str] | None = None):
    """Return (df, feature_list) for one model config.

    upcoming=(season, week): attach next-game context to each player's latest row so it can
    be scored live. team_overrides: gsis_id -> current team (from Sleeper) for players who moved.
    """
    name = cfg["model_key"]
    pos = cfg["positions"]

    # --- 1. base rows: regular-season games actually played
    stats = raw["stats"]
    stats = stats[stats["season_type"] == "REG"].copy()
    stats["fpts"] = fantasy_points(stats, scoring)
    stat_cols = _avail(stats, cfg["stats_cols"], f"{name}/stats")
    base = stats[stats["position"].isin(pos)][
        KEYS + ["player_display_name", "position", "team", "opponent_team", "fpts"] + stat_cols
    ].drop_duplicates(KEYS)

    # --- 2. snaps (snap counts use PFR ids -> translate to gsis ids)
    snaps = raw["snaps"].rename(columns={"pfr_player_id": "pfr_id"})
    snaps = snaps[snaps["game_type"] == "REG"]
    ids = raw["players"][["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    snaps = (snaps.merge(ids, on="pfr_id", how="inner")
             .rename(columns={"gsis_id": "player_id"})[KEYS + ["offense_snaps", "offense_pct"]]
             .drop_duplicates(KEYS))

    # --- 3. expected fantasy points (ff_opportunity has no season_type column)
    opp = raw["opp"]
    opp = opp[opp["week"] <= MAX_REG_WEEK]
    opp_cols = _avail(opp, cfg["opp_cols"], f"{name}/ff_opportunity")
    opp = opp[KEYS + opp_cols].drop_duplicates(KEYS)

    # --- 4. Next Gen Stats (week 0 = full-season aggregate -> future leakage, drop it)
    ngs = raw["ngs"][cfg["ngs_type"]].rename(columns={"player_gsis_id": "player_id"})
    ngs = ngs[(ngs["season_type"] == "REG") & (ngs["week"] > 0)]
    ngs = ngs[KEYS + _avail(ngs, cfg["ngs_cols"], f"{name}/ngs")].drop_duplicates(KEYS)

    parts = [base, snaps, opp, ngs]
    pbp_cols = []
    if cfg.get("pbp_cols") and raw.get("pbp") is not None:
        pbp_cols = _avail(raw["pbp"], cfg["pbp_cols"], f"{name}/pbp")
        parts.append(raw["pbp"][KEYS + pbp_cols])
    df = reduce(lambda l, r: l.merge(r, on=KEYS, how="left"), parts)

    # Counting stats: missing = zero. NGS stays NaN (XGBoost learns what "missing" means).
    zero_cols = stat_cols + opp_cols + pbp_cols + ["offense_snaps", "offense_pct"]
    df[zero_cols] = df[zero_cols].fillna(0)

    # --- 5. derived per-game features from the config
    for col, fn in cfg.get("derived", {}).items():
        try:
            df[col] = fn(df)
        except KeyError as e:
            log.warning("%s: derived feature %s skipped (missing column %s)", name, col, e)

    df = df.sort_values(KEYS).reset_index(drop=True)
    if team_overrides:
        latest = df.groupby("player_id").tail(1).index
        override = df.loc[latest, "player_id"].map(team_overrides).map(normalize_team)
        df.loc[latest, "team"] = override.fillna(df.loc[latest, "team"])

    # --- 6. target: next game in the same season
    gs = df.groupby(["player_id", "season"])
    df["target"] = gs["fpts"].shift(-1)
    df["next_week"] = gs["week"].shift(-1)
    df["games_this_season"] = gs.cumcount() + 1

    # --- 7. rolling form ACROSS seasons (window includes the current game)
    g = df.groupby("player_id")
    new = {}  # build all columns first, then concat once (adding 100+ columns one by one is slow)
    for col in [c for c in cfg["roll_cols"] if c in df.columns]:
        s = g[col]
        for w in cfg.get("windows", [3, 6, 10]):
            new[f"{col}_roll{w}"] = s.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        new[f"{col}_std3"] = s.rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
        new[f"{col}_trend3"] = (df[col] - s.shift(2)) / 2
        new[f"{col}_trend7"] = (df[col] - s.shift(6)) / 6
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # --- 8. next-game context (Vegas lines + opponent defense)
    df["ctx_season"] = np.where(df["next_week"].notna(), df["season"], np.nan)
    df["ctx_week"] = df["next_week"]
    if upcoming:
        is_last = df.groupby("player_id").cumcount(ascending=False) == 0
        live = is_last & df["target"].isna()
        df.loc[live, "ctx_season"] = upcoming[0]
        df.loc[live, "ctx_week"] = upcoming[1]

    df = df.merge(game_context(raw["schedules"]), on=["ctx_season", "ctx_week", "team"], how="left")

    has = df["opp"].notna()
    left = df[has].copy()
    left["ord"] = (left["ctx_season"] * 100 + left["ctx_week"]).astype("int64")
    # merge_asof = "latest defense value strictly BEFORE the game we're predicting"
    left = pd.merge_asof(left.sort_values("ord"), defense_allowed(stats, pos),
                         on="ord", by="opp", allow_exact_matches=False, direction="backward")
    df = pd.concat([left, df[~has]], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- 9. feature list
    auto = [c for c in df.columns
            if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    features = list(cfg["features"]) if cfg.get("features") else auto
    missing = [f for f in features if f not in df.columns]
    if missing:
        log.warning("%s: requested features not built: %s", name, missing)
        features = [f for f in features if f in df.columns]
    return df, features
