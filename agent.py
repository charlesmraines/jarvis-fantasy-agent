"""Jarvis on Gemini: a tool-using agent built on a hand-rolled loop (Interactions API).

Same design as before, different wire format:
    history = [user_input]
    loop:
        interaction = gemini(history, tools)          # stateless: we send the whole history
        history += interaction.steps                   # thoughts, function_calls, model_output
        if no function_call steps: return interaction.output_text
        for each call: run it locally, history += function_result(call_id=call.id)
"""
from __future__ import annotations

import json
import logging

from config import REPORT_DIR, SETTINGS
from llm import GeminiCaller, QuotaExhausted
from tools import ToolBox, gemini_tools

log = logging.getLogger(__name__)
MAX_STEPS = 10        # safety valve: each step is one request from your free daily budget
MAX_TURNS_KEPT = 6    # chat memory: keep the last N user questions (and everything after them)

SYSTEM_PROMPT = """You are Jarvis, the personal fantasy football GM for {owner} (team: "{team}") \
in a Sleeper league. It is NFL {season}, week {week}. Scoring: full PPR.

## Personality
- Toward {owner} and their team: loyal, supportive, confident. You're in their corner.
- Toward every other team in the league: smug, condescending, theatrically unimpressed. Roast \
their rosters, lineup choices, records and projections. Keep it to fantasy-football trash talk. \
Never insult anyone's real-life appearance, identity, family, or job.
- Witty and concise: a butler-grade AI who knows he's the smartest one in the league.

## How to decide
1. Pull data with tools before making claims. Never invent projections, stats, or injuries.
2. The XGBoost projection is the baseline for QB/RB/WR/TE. Every projection ships with \
`top_model_factors` / `why`, already written in plain English (e.g. "goal-line carries (3-game \
avg) = 2.3 -> +2.1 pts"). ALWAYS justify a projection with these instead of asserting it, and \
never print raw feature names like `gl_carries_roll3` at the user.
3. Adjust for what the model can't see: injury designations and practice reports, weather for \
outdoor games (wind > 15 mph or heavy rain hurts passing and kicking), late-week news, role \
changes. Use web_search for these when a decision depends on them. Searches are limited: \
combine players into one query when you can, and use at most 3 searches per answer.
4. Kicker and defense projections come from Sleeper, not our model (`projection_source` says \
which). They're included in matchup totals. Before recommending a K/DEF change, sanity-check \
with web_search: opponent offense quality, Vegas total, weather, and weekly rankings.
5. Questionable + missed practice = risky. Out/IR = zero.
6. Projections have error (floor/ceiling ≈ ±1 RMSE). If two options are within ~1.5 points, \
say it's close and decide with the tiebreakers above.
7. If a tool returns an error, work with what you have and say what you couldn't check. If a \
player's `note` says a bye could not be confirmed or that context is missing, report it as a data \
problem you could not verify. NEVER state that a team is on a bye unless the note says so.
8. Give a clear recommendation, your confidence, and what would change your mind.

## Output format
{format_rules}
"""

CHAT_FORMAT = "Conversational. Short paragraphs; bullets only for lists of players."

REPORT_PROMPT = """Produce the weekly strategy report for week {week}.

Required tool calls, in this order: get_optimal_lineup, get_matchup_projection, get_team (my \
opponent, to scout them), get_waiver_targets. Then web_search only where a decision actually \
hinges on injury news or weather (max 3 searches).

The point of this report is to TELL ME WHAT LINEUP TO SET, not to describe the lineup I happen \
to have. Recommend the optimal lineup, then justify it.

Every projection you cite must come with a reason drawn from the `why` / `top_model_factors` \
fields (they are already in plain English, e.g. "red-zone targets (3-game avg) = 2.7 -> +1.8 pts"). \
Turn those into natural sentences. Never print raw feature names like `rz_targets_roll3`.

Use exactly these markdown sections:

# Jarvis Weekly Report: Week {week}
## TL;DR
3 bullets: projected score, win probability, and the single most important lineup change.
## Start These (the optimal lineup)
A table of EVERY starting slot: slot | player | proj | floor-ceiling | opponent | status.
Below the table, list each change vs. my current lineup as "BENCH X (n.n) -> START Y (n.n), +n.n pts",
and state the total points currently being left on the bench.
## Why These Players
One short paragraph per starter who is either a change, a top-3 projected scorer, or a close \
call. Each must explain the projection using its model factors (usage, red-zone work, Vegas \
implied team total, matchup) in plain English.
## Matchup Outlook
My projected total (with the optimal lineup) vs. my opponent's projected total, win probability \
both as-currently-set and if I make the changes, and the 2-3 swing players on each side.
## Scouting the Opponent
Their projected starters, their 3 biggest scorers WITH the model reasons behind those numbers, \
and where their lineup is weak. Roast them, but use the data.
## Kicker & Defense
Note these come from Sleeper's projections, not our model, and whether a streamer would be better.
## Injury & Weather Watch
## Waiver Wire Targets
Top 3 with reasoning and who to drop.
## Bold Prediction
"""


def user_input(text: str) -> dict:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def function_result(call, output) -> dict:
    return {"type": "function_result", "name": call.name, "call_id": call.id,
            "result": [{"type": "text", "text": json.dumps(output, default=str)}]}


def _args(call) -> dict:
    a = call.arguments or {}
    return json.loads(a) if isinstance(a, str) else dict(a)


def trim_history(history: list[dict], max_turns: int = MAX_TURNS_KEPT) -> list[dict]:
    """Drop old turns, cutting only at user_input boundaries so every function_call
    stays paired with its function_result."""
    starts = [i for i, s in enumerate(history) if s.get("type") == "user_input"]
    if len(starts) <= max_turns:
        return history
    return history[starts[-max_turns]:]


class Jarvis:
    def __init__(self, toolbox: ToolBox, season: int, week: int,
                 caller: GeminiCaller, trace: bool = False):
        self.caller = caller
        self.tools = toolbox
        self.season, self.week = season, week
        self.trace = trace
        self.team_name = toolbox.s.team_name(toolbox.s.my_roster())

    def _system(self, format_rules: str) -> str:
        return SYSTEM_PROMPT.format(owner=SETTINGS.owner_name, team=self.team_name,
                                    season=self.season, week=self.week, format_rules=format_rules)

    def _log(self, *parts):
        if self.trace:
            print("  [trace]", *parts)

    def run(self, history: list[dict], format_rules: str = CHAT_FORMAT,
            model: str | None = None) -> str:
        """The agent loop. Mutates `history` in place so callers keep the conversation."""
        model = model or SETTINGS.chat_model
        params = {"model": model, "system_instruction": self._system(format_rules),
                  "tools": gemini_tools(), "store": False}  # store=False: we own the history
        if SETTINGS.thinking_level:
            params["generation_config"] = {"thinking_level": SETTINGS.thinking_level}

        try:
            for step in range(1, MAX_STEPS + 1):
                interaction = self.caller.create(input=history, **params)

                # Append the model's steps VERBATIM. Thought/function_call steps carry
                # signatures that Gemini needs to see again on the next request.
                history.extend(s.model_dump(mode="json", exclude_none=True)
                               for s in interaction.steps)

                calls = [s for s in interaction.steps if s.type == "function_call"]
                self._log(f"step {step}: {len(interaction.steps)} steps, {len(calls)} tool calls, "
                          f"usage={getattr(interaction, 'usage', None)} "
                          f"| requests today: {self.caller.requests_today()}")
                if not calls:
                    return (interaction.output_text or "").strip()

                for call in calls:  # the model may ask for several tools in parallel
                    args = _args(call)
                    out = self.tools.run(call.name, args)
                    self._log(f"   tool {call.name}({args}) -> {len(json.dumps(out, default=str))} chars")
                    history.append(function_result(call, out))
            return "I've overthought this one. Ask me again, more simply."
        except QuotaExhausted as e:
            return f"(Jarvis is out of free Gemini requests for now.) {e}"

    # ------------------------------------------------------------------ entry points
    def ask(self, question: str, history: list[dict] | None = None,
            format_rules: str = CHAT_FORMAT, model: str | None = None) -> str:
        history = history if history is not None else []
        history[:] = trim_history(history)  # in place, so the caller's list is trimmed too
        history.append(user_input(question))
        return self.run(history, format_rules, model)

    def weekly_report(self) -> str:
        text = self.ask(REPORT_PROMPT.format(week=self.week), model=SETTINGS.report_model)
        path = REPORT_DIR / f"{self.season}_week{self.week:02d}.md"
        path.write_text(text)
        print(f"(saved to {path})")
        return text
