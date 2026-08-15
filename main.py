#!/usr/bin/env python3
"""
main.py
-------
CLI entrypoint for the mini lead-enrichment agent.

Usage:
    python main.py                          # run on the bundled sample leads
    python main.py --input my_leads.csv      # run on your own leads (name,company columns)
    python main.py --output results.json     # also write results to a file
    python main.py --provider mock           # force offline mock mode
    python main.py --provider claude         # force the real Claude agent

By default, the provider is chosen automatically: Claude if ANTHROPIC_API_KEY
is set in the environment, otherwise the offline mock agent.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()  # reads ANTHROPIC_API_KEY (and anything else) from a local .env file, if present

from agent import ClaudeAgent, MockAgent


def load_leads(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        leads = [{"name": row["name"].strip(), "company": row["company"].strip()} for row in reader]
    if not leads:
        raise ValueError(f"No leads found in {path}")
    return leads


def pick_provider(choice: str):
    if choice == "mock":
        return MockAgent(), "mock"
    if choice == "claude":
        return ClaudeAgent(), "claude"
    # auto
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeAgent(), "claude"
    return MockAgent(), "mock"


def print_verdict(v: dict) -> None:
    bar_len = round(v.get("confidence_score", 0) * 20)
    bar = "#" * bar_len + "-" * (20 - bar_len)
    print(f"\n{v['name']}  -  {v['company']}")
    print(f"  domain:      {v['likely_domain']}  (verified: {v['domain_verified']})")
    print(f"  role:        {v['likely_role']}")
    print(f"  summary:     {v['linkedin_style_summary']}")
    print(f"  confidence:  [{bar}] {v.get('confidence_score', 0):.2f}")
    print(f"  reasoning:   {v['reasoning']}")
    print(f"  checked:     {', '.join(v.get('sources_checked', [])) or 'n/a'}")


def main():
    parser = argparse.ArgumentParser(description="Mini lead-enrichment agent")
    parser.add_argument("--input", default="leads_sample.csv", help="CSV with name,company columns")
    parser.add_argument("--output", default=None, help="Optional path to write results as JSON")
    parser.add_argument("--provider", choices=["auto", "claude", "mock"], default="auto")
    args = parser.parse_args()

    try:
        leads = load_leads(args.input)
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    agent, provider_name = pick_provider(args.provider)
    print(f"Running with provider: {provider_name}  ({len(leads)} leads)")
    if provider_name == "mock":
        print("(No ANTHROPIC_API_KEY found / --provider mock used - running the offline "
              "rule-based agent. Same tools, simulated reasoning. Set ANTHROPIC_API_KEY "
              "and re-run to see the real Claude tool-calling agent.)")

    results = []
    for lead in leads:
        try:
            verdict = agent.enrich(lead)
        except Exception as e:
            verdict = {
                "name": lead["name"], "company": lead["company"],
                "likely_domain": "", "domain_verified": False,
                "likely_role": "Unknown", "linkedin_style_summary": "N/A",
                "confidence_score": 0.0, "reasoning": f"Agent error: {e}",
                "sources_checked": [],
            }
        results.append(verdict)
        print_verdict(verdict)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
