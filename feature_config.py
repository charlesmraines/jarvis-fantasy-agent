"""Model recipes. A config IS a model version.

To experiment: copy an entry (e.g. "RB_v1" -> "RB_v2"), change it, then
    python cli.py train --configs RB_v2
    python cli.py models            # compare test metrics across versions
    python cli.py promote RB_v2     # hot-swap it into the live agent
Treat entries as append-only once trained, since saved models depend on them.

Keys per config
  model_key     which slot it fills: "WR" (WR+TE), "RB", "QB"
  positions     nflverse positions included in the training rows
  stats_cols    from load_player_stats (same-game actuals)
  opp_cols      from load_ff_opportunity (expected & actual fantasy components)
  ngs_type/cols Next Gen Stats table and columns
  pbp_cols      from our play-by-play aggregation (see features.PBP_FEATURES)
  derived       {new_col: function(df) -> Series}, computed per game BEFORE rolling
  roll_cols     columns to turn into rolling mean / std / trend features
  windows       rolling window sizes
  features      None = use every numeric engineered column; or an explicit list to prune
  xgb_params    overrides for train.DEFAULT_PARAMS
"""
import numpy as np
import pandas as pd


def safe_div(a, b):
    """Division that returns NaN instead of inf when b == 0."""
    return (a / b.replace(0, np.nan)).astype(float)


# ------------------------------------------------------------------------------ WR / TE
WR_v1 = {
    "model_key": "WR",
    "positions": ["WR", "TE"],
    "stats_cols": ["targets", "receptions", "receiving_yards", "receiving_tds",
                   "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
                   "receiving_first_downs", "target_share", "air_yards_share", "wopr", "racr"],
    "opp_cols": ["rec_fantasy_points", "rec_fantasy_points_exp", "receptions_exp",
                 "rec_yards_gained_exp", "rec_touchdown_exp", "rec_air_yards",
                 "pass_attempt_team", "total_fantasy_points_exp_team"],
    "ngs_type": "receiving",
    "ngs_cols": ["avg_cushion", "avg_separation", "avg_intended_air_yards",
                 "percent_share_of_intended_air_yards", "catch_percentage",
                 "avg_yac_above_expectation"],
    "pbp_cols": ["rz_targets", "gl_targets"],
    "derived": {
        "is_te": lambda d: (d["position"] == "TE").astype(int),
        # Scoring above expectation tends to regress -> useful signal
        "rec_fp_over_exp": lambda d: d["rec_fantasy_points"] - d["rec_fantasy_points_exp"],
        "air_yards_per_target": lambda d: safe_div(d["receiving_air_yards"], d["targets"]),
        "team_pass_volume": lambda d: d["pass_attempt_team"],
    },
    "roll_cols": ["fpts", "targets", "target_share", "wopr", "offense_pct", "receiving_yards",
                  "receiving_air_yards", "rec_fantasy_points_exp", "rec_fp_over_exp",
                  "receiving_epa", "rz_targets", "team_pass_volume"],
    "windows": [3, 6, 10],
    "features": None,
    "xgb_params": {},
}

# ------------------------------------------------------------------------------ RB
# PPR backs: receiving-work features borrowed from the WR model + rushing volume/efficiency
# + goal-line work (the biggest TD driver).
RB_v1 = {
    "model_key": "RB",
    "positions": ["RB"],
    "stats_cols": [
        # receiving (top WR-model signals)
        "targets", "receptions", "receiving_yards", "receiving_yards_after_catch",
        "receiving_tds", "target_share", "wopr",
        # rushing
        "carries", "rushing_yards", "rushing_tds", "rushing_epa", "rushing_first_downs",
        "rushing_fumbles_lost",
    ],
    "opp_cols": ["rush_attempt", "rush_attempt_team", "rush_fantasy_points",
                 "rush_fantasy_points_exp", "rush_yards_gained_exp", "rush_touchdown_exp",
                 "rec_fantasy_points_exp", "receptions_exp", "total_fantasy_points",
                 "total_fantasy_points_exp", "total_touchdown_exp",
                 "total_fantasy_points_exp_team"],
    "ngs_type": "rushing",
    "ngs_cols": ["efficiency", "percent_attempts_gte_eight_defenders", "avg_time_to_los",
                 "rush_yards_over_expected_per_att", "rush_pct_over_expected"],
    "pbp_cols": ["gl_carries", "rz_carries", "rz_targets"],
    "derived": {
        "carry_share": lambda d: safe_div(d["rush_attempt"], d["rush_attempt_team"]),
        # A target is worth ~2 carries in PPR
        "weighted_opps": lambda d: d["carries"] + 2 * d["targets"],
        "yards_per_carry": lambda d: safe_div(d["rushing_yards"], d["carries"]),
        "total_fp_over_exp": lambda d: d["total_fantasy_points"] - d["total_fantasy_points_exp"],
        "rec_pts_share": lambda d: safe_div(d["receptions"] + 0.1 * d["receiving_yards"], d["fpts"]),
    },
    "roll_cols": ["fpts", "carries", "carry_share", "targets", "target_share", "weighted_opps",
                  "offense_pct", "rushing_yards", "rushing_epa", "gl_carries", "rz_carries",
                  "rush_fantasy_points_exp", "rec_fantasy_points_exp", "total_fp_over_exp"],
    "windows": [3, 6, 10],
    "features": None,
    "xgb_params": {},
}

# ------------------------------------------------------------------------------ QB
# Efficiency (EPA/dropback, ANY/A, CPOE, passer rating) + volume + rushing upside.
QB_v1 = {
    "model_key": "QB",
    "positions": ["QB"],
    "stats_cols": ["attempts", "completions", "passing_yards", "passing_tds",
                   "passing_interceptions", "sacks_suffered", "sack_yards_lost",
                   "passing_air_yards", "passing_first_downs", "passing_epa", "passing_cpoe",
                   "pacr", "carries", "rushing_yards", "rushing_tds", "rushing_epa",
                   "rushing_first_downs"],
    "opp_cols": ["pass_fantasy_points", "pass_fantasy_points_exp", "pass_touchdown_exp",
                 "pass_yards_gained_exp", "rush_fantasy_points_exp",
                 "total_fantasy_points_exp", "total_fantasy_points_exp_team"],
    "ngs_type": "passing",
    "ngs_cols": ["passer_rating", "avg_time_to_throw", "avg_intended_air_yards",
                 "aggressiveness", "completion_percentage_above_expectation",
                 "avg_air_yards_to_sticks"],
    "pbp_cols": ["scrambles", "designed_runs", "gl_carries", "rz_pass_att"],
    "derived": {
        "dropbacks": lambda d: d["attempts"] + d["sacks_suffered"],
        "epa_per_dropback": lambda d: safe_div(d["passing_epa"], d["attempts"] + d["sacks_suffered"]),
        # Adjusted Net Yards per Attempt: a classic, very predictive QB efficiency stat
        "anya": lambda d: safe_div(
            d["passing_yards"] + 20 * d["passing_tds"] - 45 * d["passing_interceptions"]
            - d["sack_yards_lost"], d["attempts"] + d["sacks_suffered"]),
        "qb_rush_pts": lambda d: 0.1 * d["rushing_yards"] + 6 * d["rushing_tds"],
        "rush_share_of_pts": lambda d: safe_div(0.1 * d["rushing_yards"] + 6 * d["rushing_tds"], d["fpts"]),
        "pass_fp_over_exp": lambda d: d["pass_fantasy_points"] - d["pass_fantasy_points_exp"],
    },
    "roll_cols": ["fpts", "dropbacks", "epa_per_dropback", "anya", "passing_cpoe",
                  "passing_air_yards", "passing_yards", "carries", "rushing_yards",
                  "scrambles", "designed_runs", "qb_rush_pts", "rz_pass_att",
                  "pass_fantasy_points_exp", "pass_fp_over_exp"],
    "windows": [3, 6, 10],
    "features": None,
    "xgb_params": {"min_child_weight": 8},  # fewer QB rows -> regularize a bit more
}

MODEL_CONFIGS = {"WR_v1": WR_v1, "RB_v1": RB_v1, "QB_v1": QB_v1}
DEFAULT_ACTIVE = {"WR": "WR_v1", "RB": "RB_v1", "QB": "QB_v1"}
