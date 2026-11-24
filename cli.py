"""Jarvis command line.

  python cli.py train                          # train all *_v1 configs and make them active
  python cli.py train --configs RB_v2          # train an experiment (not live yet)
  python cli.py models                         # compare every trained version
  python cli.py promote RB_v2                  # hot-swap it into the agent
  python cli.py project "Caleb Williams" "Bryce Young"
  python cli.py chat [--trace]                 # talk to Jarvis; --trace shows the agent loop
  python cli.py report                         # weekly markdown report -> reports/
  python cli.py league                         # draft replies for the Sleeper league chat
  python cli.py usage                          # Gemini requests used today (free-tier budget)
  python cli.py doctor                         # data health check: ids, teams, byes, K/DEF
  Add --week N to override the current NFL week.
"""
import argparse
import json
import logging

from config import SETTINGS
from sleeper_client import SleeperClient


def current_season_week(sleeper) -> tuple[int, int]:
    state = sleeper.nfl_state()
    return int(state["season"]), int(state["week"])


def build(week_override=None, trace=False):
    from agent import Jarvis
    from llm import GeminiCaller
    from projections import Projector
    from tools import ToolBox
    from web_search import WebSearcher

    sleeper = SleeperClient(SETTINGS.league_id, SETTINGS.username)
    season, week = current_season_week(sleeper)
    week = week_override or week
    print(f"Loading data and projecting {season} week {week} ...")
    projector = Projector(sleeper, season, week)
    projector.refresh()

    caller = GeminiCaller()                                   # one shared, rate-limited client
    searcher = WebSearcher(caller, SETTINGS.chat_model)
    toolbox = ToolBox(sleeper, projector, searcher)
    jarvis = Jarvis(toolbox, season, week, caller, trace=trace)
    print(f"Jarvis on {SETTINGS.chat_model} | Gemini requests used today: {caller.requests_today()}")
    return toolbox, jarvis


def doctor(toolbox):
    """Check the joins that silently produce wrong answers: id mapping, team codes, byes, K/DEF."""
    p, s = toolbox.p, toolbox.s
    print(f"\nWeek {p.week}: {len(p.teams_playing)} teams have a game "
          f"({32 - len(p.teams_playing)} on bye: "
          f"{', '.join(sorted({'ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC','LA','LAC','LV','MIA','MIN','NE','NO','NYG','NYJ','PHI','PIT','SEA','SF','TB','TEN','WAS'} - p.teams_playing)) or 'none'})")
    print(f"Model projections built for {len(p.all_projections())} players")
    print(f"Sleeper K/DEF projections: {len(p.kdef) or 'UNAVAILABLE (using flat averages)'}")

    problems = []
    for roster in s.rosters():
        for pid in (roster.get("players") or []):
            r = p.project(pid)
            note = (r.get("note") or "")
            if "NOT a bye" in note or "unmapped" in note or "no recent" in note:
                problems.append((s.team_name(roster), r["name"], r.get("team"), note))
    print(f"\n{len(problems)} rostered players with questionable projections:")
    for team, name, nfl_team, note in problems[:25]:
        print(f"  {name:22s} {str(nfl_team):4s} ({team[:18]:18s}) {note[:70]}")
    if not problems:
        print("  none - ids, team codes and schedule joins all look healthy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["train", "models", "promote", "project", "chat", "report", "league", "usage", "doctor"])
    ap.add_argument("args", nargs="*", help="player names (project) or version (promote)")
    ap.add_argument("--configs", nargs="+", help="config versions to train")
    ap.add_argument("--week", type=int)
    ap.add_argument("--trace", action="store_true", help="print each step of the agent loop")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd == "train" else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if a.cmd == "train":
        from feature_config import DEFAULT_ACTIVE
        from train import main as train_main
        season, _ = current_season_week(SleeperClient(SETTINGS.league_id, SETTINGS.username))
        versions = a.configs or list(DEFAULT_ACTIVE.values())
        train_main(season, versions, promote=not a.configs)  # v1 defaults go live immediately
        return
    if a.cmd == "models":
        import model_registry
        for r in model_registry.list_versions():
            star = "*" if r["active"] else " "
            print(f"{star} {r['version']:10s} slot={r['slot']:3s} RMSE={r['test_rmse']:.2f} "
                  f"R2={r['test_r2']:.3f} (baseline RMSE {r['baseline_rmse']:.2f}) "
                  f"features={r['n_features']} trained={r['trained']}")
        return
    if a.cmd == "usage":
        from llm import USAGE_FILE
        print(USAGE_FILE.read_text() if USAGE_FILE.exists() else "no Gemini requests yet today")
        return
    if a.cmd == "promote":
        import model_registry
        print("active models:", model_registry.promote(a.args[0]))
        return

    toolbox, jarvis = build(a.week, a.trace)
    if a.cmd == "doctor":
        doctor(toolbox)
    elif a.cmd == "project":
        print(json.dumps(toolbox.get_player_projections(a.args), indent=2, default=str))
    elif a.cmd == "report":
        print(jarvis.weekly_report())
    elif a.cmd == "league":
        from league_chat import repl
        repl(jarvis)
    elif a.cmd == "chat":
        history: list[dict] = []
        print("Jarvis online. Ctrl+C to exit.\n")
        while True:
            q = input("you > ").strip()
            if q:
                print(f"\njarvis > {jarvis.ask(q, history)}\n")


if __name__ == "__main__":
    main()
