"""
tools.py
--------
The actual functions the agent is allowed to call. Each tool does one small,
verifiable thing and returns structured data -- never prose. The agent's job
is to decide *when* to call these and how to weigh what comes back.

web_search() tries a live DuckDuckGo search first (via the `ddgs` package,
no API key required) and falls back to a small offline mock dataset if
there's no network access or the package isn't installed. This means the
demo always runs end-to-end, online or offline.
"""

from __future__ import annotations
import re
import socket


# ---------------------------------------------------------------------------
# Offline fallback data, keyed by company name (lowercase, substring match).
# Only used if a live web search isn't available. Keeps the fictional demo
# leads self-consistent when running with no internet / no API key.
# ---------------------------------------------------------------------------
_MOCK_SEARCH_DB = {
    "northwind robotics": [
        {
            "title": "Northwind Robotics | About",
            "snippet": "Northwind Robotics builds warehouse automation hardware. "
                       "Series B, ~85 employees. HQ listed as Austin, TX.",
            "url": "https://northwindrobotics.com/about",
        },
        {
            "title": "Aria Chen - Northwind Robotics",
            "snippet": "Aria Chen, Head of Partnerships at Northwind Robotics, "
                       "previously worked in BD at a logistics startup.",
            "url": "https://northwindrobotics.com/team",
        },
    ],
    "bluepeak analytics": [
        {
            "title": "BluePeak Analytics - Data platform for retailers",
            "snippet": "BluePeak Analytics offers a retail forecasting SaaS platform. "
                       "Small team (~20), based in Lisbon.",
            "url": "https://bluepeakanalytics.io",
        },
        {
            "title": "Marco Silva | BluePeak Analytics",
            "snippet": "Marco Silva is listed as VP of Sales at BluePeak Analytics.",
            "url": "https://bluepeakanalytics.io/team/marco-silva",
        },
    ],
    "fernbridge health": [
        {
            "title": "Fernbridge Health",
            "snippet": "Fernbridge Health is a digital clinic network operating in the UK. "
                       "No public employee count found.",
            "url": "https://fernbridgehealth.co.uk",
        },
    ],
    "solace freight": [
        {
            "title": "Solace Freight | Trucking & Logistics",
            "snippet": "Solace Freight is a freight brokerage founded 2019, "
                       "headquartered in Chicago.",
            "url": "https://solacefreight.com",
        },
        {
            "title": "Tom Becker - Solace Freight",
            "snippet": "Tom Becker, Operations Manager, Solace Freight.",
            "url": "https://solacefreight.com/about/leadership",
        },
    ],
    # Deliberately sparse / low-signal entry to show a low-confidence result.
    "loom": [
        {
            "title": "Loom & Co - handmade textiles",
            "snippet": "A small studio shop. Little public information available online.",
            "url": "https://example.com/loomandco",
        },
    ],
}


def _mock_search(query: str) -> list[dict]:
    q = query.lower()
    for key, results in _MOCK_SEARCH_DB.items():
        if key in q:
            return results
    return []


def web_search(query: str) -> list[dict]:
    """
    Search the web for public info about a person or company.
    Returns a list of {title, snippet, url} dicts, or [] if nothing found.
    Tries a live search first, falls back to offline mock data.
    """
    try:
        from ddgs import DDGS  # pip install ddgs

        with DDGS() as ddgs:
            live = [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                for r in ddgs.text(query, max_results=5)
            ]
        if live:
            return live
    except Exception:
        pass  # no network / package missing / rate-limited -> fall back

    return _mock_search(query)


def guess_domain(company_name: str) -> str:
    """
    Cheap heuristic for a company's likely primary domain: strip legal
    suffixes and punctuation, lowercase, join, add .com. This is a *guess*
    that check_domain() below is meant to verify -- it is not itself
    evidence of anything.
    """
    name = re.sub(r"\b(inc|llc|ltd|corp|co|gmbh|studio)\b", "", company_name, flags=re.I)
    name = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
    return f"{name}.com"


def check_domain(domain: str) -> dict:
    """
    Verify a domain actually resolves (real DNS lookup). This is the one
    tool result that's genuinely binary/trustworthy -- it either resolves
    or it doesn't -- which is why it's weighted heavily in the confidence
    score downstream.
    """
    try:
        socket.gethostbyname(domain)
        return {"domain": domain, "resolves": True}
    except Exception:
        return {"domain": domain, "resolves": False}


# ---------------------------------------------------------------------------
# Tool schemas in Anthropic's tool-use format, shared by the real agent.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": (
            "Search the public web for information about a person or company. "
            "Use this to verify a person's role/company and find a short "
            "public summary. Returns a list of title/snippet/url results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'Aria Chen Northwind Robotics'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_domain",
        "description": (
            "Verify whether a candidate company domain actually resolves on "
            "the public internet. Use this after guessing a domain from the "
            "company name, to confirm it's real before reporting it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Candidate domain, e.g. 'acme.com'"},
            },
            "required": ["domain"],
        },
    },
]


def run_tool(name: str, tool_input: dict) -> dict | list:
    """Dispatch a tool call by name. Used by both the real and mock agents."""
    if name == "web_search":
        return web_search(tool_input["query"])
    if name == "check_domain":
        return check_domain(tool_input["domain"])
    raise ValueError(f"Unknown tool: {name}")
