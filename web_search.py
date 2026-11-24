"""web_search as an ordinary client-side tool.

Why not just add Gemini's built-in google_search next to our functions? Mixing built-in tools with
custom functions in one request is still a preview feature (Gemini 3 only), and Search grounding
may not be free on your tier. Wrapping search in its own function isolates it:
  1. try a small *sub-agent* call: Gemini + google_search, returning a short cited summary
  2. if that's unavailable, fall back to free DuckDuckGo results (pip install ddgs)
  3. if both fail, return an error, and the main agent carries on without live news
"""
from __future__ import annotations

import logging

from llm import GeminiCaller, QuotaExhausted, _is_rate_limit

log = logging.getLogger(__name__)

SEARCH_PROMPT = (
    "Search the web and answer concisely (max 120 words) for a fantasy football decision. "
    "Prefer reputable sources (ESPN, NFL.com, team sites, Rotowire, weather.gov) and include "
    "the source site names and dates. Question: {q}"
)


class WebSearcher:
    def __init__(self, caller: GeminiCaller, model: str):
        self.caller, self.model = caller, model
        self.grounding_ok = True  # flips off after the first failure to save requests

    def search(self, query: str, max_results: int = 5) -> dict:
        if self.grounding_ok:
            try:
                it = self.caller.create(model=self.model, input=SEARCH_PROMPT.format(q=query),
                                        tools=[{"type": "google_search"}], store=False,
                                        _max_retries=0)  # fail fast: grounding quota rarely recovers in 20s
                if it.output_text:
                    return {"query": query, "source": "google_search", "summary": it.output_text}
            except (QuotaExhausted, Exception) as e:
                # Search grounding commonly has its own (often zero, on no-billing projects) quota,
                # separate from the RPM/RPD shown for plain model calls — a 429 here doesn't mean
                # the whole agent is out of requests. Switch to DuckDuckGo and don't try grounding again.
                if isinstance(e, QuotaExhausted) or _is_rate_limit(e):
                    log.warning("Search grounding quota unavailable; using DuckDuckGo from now on")
                else:
                    log.warning("Search grounding failed (%s); using DuckDuckGo from now on", e)
                self.grounding_ok = False
        try:
            from ddgs import DDGS
            hits = DDGS().text(query, max_results=max_results) or []
            return {"query": query, "source": "duckduckgo",
                    "results": [{"title": h.get("title"), "url": h.get("href"),
                                 "snippet": h.get("body")} for h in hits]}
        except ImportError:
            return {"error": "web search unavailable (grounding off and `ddgs` not installed)"}
        except Exception as e:
            return {"error": f"web search failed: {e}"}
