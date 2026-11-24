# Jarvis: an ML-powered Sleeper fantasy GM

XGBoost projections (WR/TE, RB, QB) trained on 6+ seasons of nflverse data, wrapped in a
tool-using agent on Google Gemini's **free tier** that reads your Sleeper league, searches the web
for injuries, weather and K/DEF rankings, writes weekly reports, chats, and drafts league-chat trash talk.

See **WALKTHROUGH.md** for how every piece works and why.

```
feature_config.py   model recipes (a config = a model version). Edit this to experiment.
features.py         config-driven feature engineering (same code for training and live use)
train.py            time-based train/val/test, early stopping, refit, baseline comparison
model_registry.py   versioned models + active.json (hot-swapping)
projections.py      Sleeper id -> gsis id -> features -> projection + SHAP explanation,
                    lineup optimizer, win probability
sleeper_client.py   league, rosters, matchups, injuries (read-only public API)
tools.py            the agent's tools (provider-neutral definitions + Gemini adapter)
agent.py            persona + Gemini agent loop + weekly report
llm.py              rate-limited Gemini client: spacing, 429 retries, daily request counter
web_search.py       search tool: Gemini grounding sub-call, DuckDuckGo fallback
league_chat.py      copy-paste league-chat mode (with the $100/month paywall)
cli.py              entry point
```

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env               # free key from aistudio.google.com, league id, username
python cli.py train                # trains WR_v1, RB_v1, QB_v1 and makes them active
python cli.py project "Caleb Williams" "Bryce Young"
python cli.py chat --trace         # --trace prints every step of the agent loop
python cli.py report               # reports/2026_week03.md
python cli.py league               # paste a chat message, get a reply to paste back
python cli.py usage                # Gemini requests used today
python cli.py doctor               # data health check (id mapping, team codes, byes, K/DEF)
```
Google Colab: upload the folder, `%cd jarvis`, then `!python cli.py train` etc. Put your keys
in Colab Secrets and export them with `os.environ[...]` before running.

Run projections Tuesday–Wednesday: nflverse updates a day or so after games, and players who
already played this week (TNF) won't have a row for the upcoming game.

## Staying free
Each question costs about 3–5 Gemini requests (one per agent step, plus searches). Check your real
limits at aistudio.google.com/rate-limit and set `GEMINI_RPM` to match. Flash-Lite
(`gemini-3.5-flash-lite`) has by far the largest free daily allowance; full Flash is smarter, but
you get far fewer free requests, so consider it only for `GEMINI_REPORT_MODEL`. Never enable
billing on the project if you want a hard guarantee of $0.

## Improving a model later
```bash
# 1. copy RB_v1 -> RB_v2 in feature_config.py (add to MODEL_CONFIGS), change it
python cli.py train --configs RB_v2   # trains + evaluates, NOT live yet
python cli.py models                  # compare RMSE vs. v1 and vs. the naive baseline
python cli.py promote RB_v2           # the agent uses it on its next start
```

## League chat
Automating Sleeper's chat violates their ToS, so `league` mode keeps you as the messenger.
Paste who said it and what they said. If `@jarvis` isn't in the message, Jarvis stays quiet.
Asking for a "report"/"rundown" in league mode gets a **chat-sized** (~180 word, plain text)
version you can actually paste into Sleeper — not the markdown file. For the full 8-section
report with tables, run `python cli.py report` and read it yourself in `reports/`.
If someone other than you asks, they get a canned paywall reply (no API cost). If you ask, Jarvis
answers in short, chat-sized trash talk. With `pyperclip` installed, the reply is copied
automatically.
