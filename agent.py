"""
agent.py
--------
Two implementations of the same interface: enrich(lead) -> dict.

ClaudeAgent   - the real thing: a Claude tool-calling loop. The model decides
                which tools to call, sees the results, and only then produces
                a final structured verdict. This is the pattern worth
                reviewing.

MockAgent     - a small rule-based stand-in with the identical output shape,
                used automatically when there's no ANTHROPIC_API_KEY set (or
                via --provider mock). It calls the *same* tools.py functions,
                so the "evidence gathering" step is real either way -- only
                the reasoning step is swapped for a heuristic.

Both return a dict matching VERDICT_SCHEMA_DESCRIPTION below.
"""

from __future__ import annotations
import json
from tools import TOOL_SCHEMAS, run_tool, guess_domain

MODEL = "claude-haiku-4-5-20251001"

VERDICT_SCHEMA_DESCRIPTION = """
Return ONLY a single JSON object (no markdown, no prose) with exactly these
fields:
{
  "name": string,
  "company": string,
  "likely_domain": string,
  "domain_verified": boolean,
  "likely_role": string,
  "linkedin_style_summary": string,   // 1-2 sentences, factual, no fluff
  "confidence_score": number,         // 0.0-1.0
  "reasoning": string,                // 1-2 sentences: what evidence was checked and why the score landed where it did
  "sources_checked": [string]         // tool calls / URLs actually used
}
""".strip()

SYSTEM_PROMPT = f"""You are a lead enrichment agent used ahead of outbound sales outreach.

For each lead (name + company) you are given, you must:
1. Use the check_domain tool to verify a likely company domain (guess it from
   the company name if you're not given one, e.g. "Acme Robotics" -> "acmerobotics.com").
2. Use the web_search tool at least once to try to confirm the person's role
   and find a short factual public summary of them or their company.
3. Only THEN produce a final verdict. Do not guess facts you have not checked
   with a tool. If a tool returns nothing useful, say so in "reasoning" and
   lower the confidence_score accordingly -- do not invent details to fill gaps.

IMPORTANT: your final response (the one that is not a tool call) must contain
ONLY the JSON object below. No preamble like "Here is the verdict", no
explanation before or after it, no markdown code fences. The very first
character of your final response must be an opening curly brace.

Confidence guidance:
- 0.8-1.0: domain verified AND search returned a specific, relevant hit for this person/company.
- 0.4-0.79: only one of the two checks succeeded, or search results were generic/ambiguous.
- 0.0-0.39: neither check produced solid evidence.

{VERDICT_SCHEMA_DESCRIPTION}
"""


class ClaudeAgent:
    """Real tool-calling agent, backed by the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None):
        import anthropic  # imported lazily so mock mode has no hard dependency

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def enrich(self, lead: dict) -> dict:
        user_msg = (
            f"Enrich this lead:\nName: {lead['name']}\nCompany: {lead['company']}"
        )
        messages = [{"role": "user", "content": user_msg}]

        # Tool-calling loop: keep going until Claude stops asking for tools.
        for _ in range(6):  # hard cap so a misbehaving loop can't run forever
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                final_text = "".join(b.text for b in response.content if b.type == "text")
                return _parse_verdict(final_text, lead)

            # Model asked for one or more tools. Run them, feed results back.
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        # Safety net: loop cap hit without a final answer.
        return _fallback_verdict(lead, reasoning="Agent exceeded tool-call budget without producing a final verdict.")


class MockAgent:
    """
    Deterministic, offline-friendly stand-in for ClaudeAgent. Calls the same
    tools, then applies a simple rule-based scorer instead of an LLM. Used
    automatically when no ANTHROPIC_API_KEY is set, so the demo always runs.
    """

    def enrich(self, lead: dict) -> dict:
        name, company = lead["name"], lead["company"]
        sources_checked = []

        domain = guess_domain(company)
        domain_result = run_tool("check_domain", {"domain": domain})
        sources_checked.append(f"check_domain({domain})")

        search_results = run_tool("web_search", {"query": f"{name} {company}"})
        sources_checked.append(f"web_search('{name} {company}')")

        domain_verified = bool(domain_result.get("resolves"))
        has_search_hit = len(search_results) > 0
        person_specific_hit = any(name.split()[0].lower() in r.get("snippet", "").lower() for r in search_results)

        # Try to lift a role mention out of search snippets.
        likely_role = "Unknown"
        for r in search_results:
            snippet = r.get("snippet", "")
            if name.split()[0] in snippet:
                for marker in [",", " at ", " - "]:
                    if marker in snippet:
                        # crude extraction, good enough for a heuristic demo
                        after = snippet.split(name.split()[0], 1)[-1]
                        likely_role = after.strip(" ,-").split(".")[0][:60] or "Unknown"
                        break

        summary_source = search_results[0]["snippet"] if search_results else None
        summary = summary_source or f"No public summary found for {name} at {company}."

        if domain_verified and person_specific_hit:
            score = 0.85
            reason = "Domain resolved and search returned a hit naming this person directly."
        elif domain_verified and has_search_hit:
            score = 0.6
            reason = "Domain resolved; search returned company info but no result naming this person specifically."
        elif domain_verified or has_search_hit:
            score = 0.4
            reason = "Only one of domain verification or web search produced evidence."
        else:
            score = 0.15
            reason = "Neither domain verification nor web search produced usable evidence."

        return {
            "name": name,
            "company": company,
            "likely_domain": domain,
            "domain_verified": domain_verified,
            "likely_role": likely_role,
            "linkedin_style_summary": summary,
            "confidence_score": score,
            "reasoning": reason,
            "sources_checked": sources_checked,
        }


def _parse_verdict(text: str, lead: dict) -> dict:
    """
    Extract the verdict JSON from the model's final text response. The
    system prompt asks for JSON only, but models sometimes add a stray
    sentence before it ("Here is the verdict:") or wrap it in a ```json
    fence anyway -- so this pulls out the {...} block rather than assuming
    the whole string is clean JSON.
    """
    import re

    # Prefer a fenced ```json ... ``` block if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else None

    # Otherwise, take the substring from the first '{' to the last '}'.
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]

    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return _fallback_verdict(lead, reasoning=f"Could not parse model output as JSON: {text[:200]!r}")


def _fallback_verdict(lead: dict, reasoning: str) -> dict:
    return {
        "name": lead["name"],
        "company": lead["company"],
        "likely_domain": guess_domain(lead["company"]),
        "domain_verified": False,
        "likely_role": "Unknown",
        "linkedin_style_summary": "N/A",
        "confidence_score": 0.0,
        "reasoning": reasoning,
        "sources_checked": [],
    }