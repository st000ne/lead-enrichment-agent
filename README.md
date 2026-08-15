# Lead Enrichment Agent

Given a list of leads (name + company), an agent uses tool-calling to gather
public evidence about each one — a verified company domain, a role, a short
summary — then outputs a structured verdict with a confidence score and a
one-line reasoning note.

## Quickstart

```bash
pip install -r requirements.txt

# Runs immediately, no API key needed (offline mock agent):
python main.py

# With the real Claude tool-calling agent:
export ANTHROPIC_API_KEY=sk-...
python main.py --provider claude

# Your own leads, results saved to a file:
python main.py --input my_leads.csv --output results.json
```

## What it does

For each lead, the agent:
1. Calls `check_domain` to verify a likely company domain actually resolves.
2. Calls `web_search` to confirm the person's role and find a short public summary.
3. Only then produces a final verdict — it's instructed not to invent facts it hasn't checked with a tool.

Output per lead:

```json
{
  "name": "Marco Silva",
  "company": "BluePeak Analytics",
  "likely_domain": "bluepeakanalytics.com",
  "domain_verified": true,
  "likely_role": "VP of Sales",
  "linkedin_style_summary": "BluePeak Analytics offers a retail forecasting SaaS platform...",
  "confidence_score": 0.85,
  "reasoning": "Domain resolved and search returned a hit naming this person directly.",
  "sources_checked": ["check_domain(bluepeakanalytics.com)", "web_search('Marco Silva BluePeak Analytics')"]
}
```

## Design choices

**Tool-calling over one big prompt.** A single prompt asking an LLM to
"enrich this lead" will happily produce a plausible-looking domain, role,
and summary with no grounding — you can't tell fabricated output from
correct output. Splitting the task into tools (`web_search`, `check_domain`)
forces every claim in the final verdict to trace back to a concrete,
inspectable result, and lets the knowledge layer be swapped or upgraded
independently of the reasoning layer.

**Confidence reflects what was verified, not what the model feels.**
Domain resolves *and* search names the person → high confidence. Only one
signal fires → medium. Neither fires → low, stated explicitly rather than
papered over.

**Offline mock mode.** With no `ANTHROPIC_API_KEY`, `main.py` falls back to
`MockAgent` — same tools, a heuristic instead of an LLM for the reasoning
step — so the repo runs for anyone with no setup, while the real
`ClaudeAgent` tool-calling loop is still there to read and run.

## Files

- `main.py` — CLI: loads leads, picks a provider, prints + optionally saves results.
- `agent.py` — `ClaudeAgent` (real tool-calling loop) and `MockAgent` (offline fallback).
- `tools.py` — tool implementations (`web_search`, `check_domain`) and their schemas.
- `leads_sample.csv` — 5 fictional sample leads.
