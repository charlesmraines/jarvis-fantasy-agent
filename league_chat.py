"""League-chat mode, the ToS-safe version: YOU are the transport.

Paste a message from the Sleeper chat, say who wrote it, and Jarvis drafts a reply you paste back.
Everything else (the @jarvis check, the paywall, the persona) works like a real bot would.
"""
from __future__ import annotations

import random
import re

from config import SETTINGS

MENTION = re.compile(r"@jarvis\b", re.I)
PREFIX = "🤖 Jarvis: "

LEAGUE_CHAT_FORMAT = (
    "You are replying in the league's group chat, where everyone can read it. Plain text, "
    "no markdown, under ~100 words, maximum swagger. The message was written by {author}."
)

# A "report" asked for in the group chat is NOT the markdown file from `cli.py report` — tables
# and headers don't render in Sleeper. This is a postable, chat-sized version of the same idea.
REPORT_REQUEST = re.compile(r"\b(report|rundown|preview|breakdown|state of the league)\b", re.I)

CHAT_REPORT_FORMAT = (
    "You are posting a weekly rundown in the league's group chat. Plain text only, no markdown, "
    "no tables, no headers. Under ~300 words. Use short lines, at most one emoji per line. "
    "Maximum swagger: the whole league reads this."
)

CHAT_REPORT_REQUEST = (
    "Post your week {week} rundown for the league chat. Call get_optimal_lineup and "
    "get_matchup_projection first. Cover, briefly: my projected total vs my opponent's projected "
    "total and the win probability, model-based reasoning for at least 3 of my players, "
    "every lineup change I should make (as 'bench X, start Y') "
    "with a few words of model-based reasoning each, my opponent's single biggest threat and why, "
    "and my biggest injury or weather risk. Check injuries/weather with web_search only if a "
    "start-sit call actually hinges on it. End with a one-line bold prediction."
)

PAYWALL_REPLIES = [
    "Access denied, {author}. Jarvis Premium is $100/month, payable to {owner}. "
    "Judging by your lineup, it's the best investment you could make.",
    "I only work for {owner}. For $100/month he *might* rent me out. "
    "{author}, your roster says you can't afford NOT to.",
    "Cute question, {author}. Subscriptions start at $100/month (see {owner}). "
    "Until then, keep guessing like you have been all season.",
]


def _copy(text: str) -> bool:
    try:
        import pyperclip  # optional: pip install pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def reply(jarvis, author: str, message: str, history: list[dict]) -> str | None:
    """Return the text to post, or None if Jarvis wasn't addressed."""
    if not MENTION.search(message):
        return None
    if author.strip().lower() != SETTINGS.owner_name.lower():
        # Canned: no API call, so friends can't run up your bill.
        return PREFIX + random.choice(PAYWALL_REPLIES).format(author=author, owner=SETTINGS.owner_name)
    question = re.sub(r"\s+", " ", MENTION.sub("", message)).strip()
    if REPORT_REQUEST.search(question):
        # Swap in the chat-sized report prompt + looser word budget.
        return PREFIX + jarvis.ask(CHAT_REPORT_REQUEST.format(week=jarvis.week), history,
                                   format_rules=CHAT_REPORT_FORMAT)
    answer = jarvis.ask(question, history, format_rules=LEAGUE_CHAT_FORMAT.format(author=author))
    return PREFIX + answer


def repl(jarvis):
    print("League-chat mode. Paste a message from the Sleeper chat; Ctrl+C to quit.\n")
    history: list[dict] = []
    while True:
        author = input("who wrote it? > ").strip() or SETTINGS.owner_name
        message = input("message      > ").strip()
        out = reply(jarvis, author, message, history)
        if out is None:
            print("(no @jarvis in that message, so Jarvis stays quiet)\n")
            continue
        copied = " (copied to clipboard)" if _copy(out) else ""
        print(f"\n{out}\n{copied}\n")
