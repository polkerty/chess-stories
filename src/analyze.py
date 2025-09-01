#!/usr/bin/env python3
"""
analyze.py — Lichess game analyzer (parallel OpenAI calls)

3-phase workflow:
  1) Fetch PGN by Lichess URL/id, parse positions + move clocks.
  2) Analyze each position (before each move) in PARALLEL via OpenAI.
     - Adds a per-position "winner_pred" (White|Black|Draw).
  3) Final summary model sees every move + (parsed) clock info and all micro-analyses.
     - Also renders a prediction trajectory chart and prints summary stats.

Defaults:
  - Phase 2 model (parallel positions): gpt5-mini
  - Phase 3 model (final summary):     gpt5-thinking

Setup
  pip install -U openai python-chess requests rich python-dotenv matplotlib

.env (example)
  OPENAI_API_KEY=sk-...
  # Optional (for private games)
  # LICHESS_TOKEN=lip_...

Run
  python analyze.py https://lichess.org/uLHo0b8iWH4o --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from prompts import load_prompt, DEFAULT_POSITION_PROMPT, DEFAULT_FINAL_PROMPT
from lichess_io import (
    extract_lichess_game_id,
    fetch_pgn_for_game,
    parse_positions_and_clocks,
    PositionInfo,
    MoveClock,
)
from openai_analyze import (
    analyze_positions_parallel,
    final_summary,
    PositionResult,
)
from prediction import (
    compute_prediction_stats,
    plot_prediction_series,
)
from reporting import show_summary_tables


console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze a Lichess game with the OpenAI API (parallel per-position).")
    p.add_argument("url_or_id", help="Lichess game URL (e.g., https://lichess.org/abcdefgh) or raw game id.")
    p.add_argument("--concurrency", type=int, default=10, help="Max parallel OpenAI calls (default: 10).")
    p.add_argument("--limit", type=int, default=0, help="Limit to first N positions (0 = all).")
    p.add_argument("--include-evals", action="store_true", help="Request eval numbers in PGN export (not used in prompts).")

    # Different models per phase (requested defaults)
    p.add_argument("--model", default="gpt5-mini",
                   help="Model for per-position analysis (phase 2). Default: gpt5-mini.")
    p.add_argument("--final-model", default="gpt5-thinking",
                   help="Model for final summary (phase 3). Default: gpt5-thinking.")

    p.add_argument("--position-prompt-file", help="Path to custom template for position analysis.")
    p.add_argument("--final-prompt-file", help="Path to custom template for the final summary.")
    p.add_argument("--position-max-output-tokens", type=int, default=300)
    p.add_argument("--final-max-output-tokens", type=int, default=800)

    p.add_argument("--lichess-token", help="Optional Lichess API token (env LICHESS_TOKEN also supported).")
    p.add_argument("--plot", default="prediction_trajectory.png",
                   help="Path to save prediction plot (PNG). Set empty string to skip plotting.")
    return p


async def main_async(args: argparse.Namespace) -> int:
    console.print(Panel("Phase 1: Fetching PGN & parsing positions", style="cyan"))

    # Get Lichess token (optional)
    lichess_token = args.lichess_token or os.environ.get("LICHESS_TOKEN")

    # Extract canonical id & fetch
    m = re.search(r"([A-Za-z0-9]{8,12})", args.url_or_id)
    raw_id = m.group(1) if m else args.url_or_id
    canonical_id = extract_lichess_game_id(raw_id)

    pgn = fetch_pgn_for_game(canonical_id, include_evals=args.include_evals, token=lichess_token)
    positions, headers, move_clocks = parse_positions_and_clocks(pgn)

    if args.limit and args.limit > 0:
        positions = positions[: args.limit]
        move_clocks = [mc for mc in move_clocks if mc.ply <= args.limit]

    console.print(f"[bold]Game:[/] {headers.get('White','?')} vs {headers.get('Black','?')}  "
                  f"[dim]({headers.get('Event','')} {headers.get('Date','')})[/]")
    console.print(f"[bold]Total positions to analyze:[/] {len(positions)}")

    # Prompts
    position_prompt = load_prompt(args.position_prompt_file, DEFAULT_POSITION_PROMPT)
    final_prompt = load_prompt(args.final_prompt_file, DEFAULT_FINAL_PROMPT)

    # Phase 2 — parallel per-position analysis
    results, stats = await analyze_positions_parallel(
        positions,
        model=args.model,
        position_prompt=position_prompt,
        concurrency=args.concurrency,
    )

    # Winner prediction stats + (optional) plot
    pred_stats = compute_prediction_stats(results)
    plot_path = None
    if args.plot is not None and len(args.plot) > 0:
        plot_path = plot_prediction_series(pred_stats, args.plot)

    # Phase 3 — final summary (includes moves+clocks JSON)
    console.print(Panel("Phase 3: Final summary synthesis", style="cyan"))
    final_text, final_in, final_out = await final_summary(
        model=args.final_model,
        headers=headers,
        results=results,
        move_clocks=move_clocks,
        final_prompt_template=final_prompt,
    )

    show_summary_tables(stats, results, final_text, final_in, final_out, pred_stats, plot_path)
    return 0


def main() -> int:
    load_dotenv()  # load .env first

    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY is not set (load via .env or export it).[/red]")
        return 2

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        return 130
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
