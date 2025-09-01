#!/usr/bin/env python3
"""
analyze.py — Lichess game analyzer (parallel OpenAI calls)

Improvements in this version:
- Handles 12-char Lichess "full ids" by canonicalizing to 8-char ids for export.
- Falls back to {base}.pgn download if the export endpoint 404s.
- Loads OPENAI_API_KEY (and optional LICHESS_TOKEN) via python-dotenv.

Usage:
  pip install -U openai python-chess requests rich python-dotenv
  echo "OPENAI_API_KEY=sk-..." > .env
  python analyze.py https://lichess.org/uLHo0b8iWH4o --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import io
import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn,
    MofNCompleteColumn, SpinnerColumn, TextColumn
)
from rich.table import Table
from rich.panel import Panel

import chess
import chess.pgn

from openai import AsyncOpenAI
import openai  # for exception classes

console = Console()

# ----------------------------- Prompts -----------------------------

DEFAULT_POSITION_PROMPT = """\
You are a strong chess analyst. Analyze the position **before** the listed move is played.
Return a compact JSON object (no backticks) with these fields and nothing else:
{{
  "eval": "winning|better|equal|worse|lost",
  "plan": "1-2 short sentences (≤30 words)",
  "motif": "a single tactical/positional motif",
  "comment": "≤25 words single-sentence note about the chosen move"
}}
Be concise and do not exceed the word limits.

Context:
- FEN: {fen}
- Side to move: {side_to_move}
- Ply (from 1): {ply}
- Move about to be played: {move_san} ({move_uci})
"""

DEFAULT_FINAL_PROMPT = """\
You are a chess coach. You will receive a list of per-move JSON analyses for a game.
Produce (in plain text):
1) A 5-bullet high-level narrative of the game (openings, middlegame plan shifts, endgame).
2) 3 critical turning points with move numbers and a one-liner for each.
3) The main recurring themes (2–4).
4) One training recommendation for the weaker side.
Keep it under ~300 words. Avoid redundancy. Use clear, punchy language.

Game headers:
{headers}

Players: {white} vs {black}

Per-move analyses (JSON per ply, trimmed):
{micro_json}
"""

# ----------------------------- Data classes -----------------------------

@dataclasses.dataclass
class PositionInfo:
    ply: int
    side_to_move: str
    fen: str
    move_san: str
    move_uci: str
    move_number: int

@dataclasses.dataclass
class PositionResult:
    ply: int
    move_san: str
    move_uci: str
    side_to_move: str
    usage_input_tokens: int
    usage_output_tokens: int
    analysis_raw: str
    analysis_json: Optional[Dict[str, Any]]
    error: Optional[str] = None

# ----------------------------- Helpers -----------------------------

def load_prompt(path: Optional[str], default_text: str) -> str:
    if not path:
        return default_text
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_position_prompt(template: str, p: PositionInfo) -> str:
    return template.format(
        fen=p.fen,
        side_to_move=p.side_to_move,
        ply=p.ply,
        move_san=p.move_san,
        move_uci=p.move_uci,
        move_number=p.move_number,
    )

def extract_lichess_game_id(url_or_id: str) -> str:
    """
    Accepts a Lichess URL or raw id.
    - Lichess "full ids" can be 12 chars (8-char game id + 4-char POV suffix).
      Export endpoints use the canonical 8-char id, so we trim if length == 12.
    """
    # Raw id?
    m_raw = re.fullmatch(r"[A-Za-z0-9]{8,12}", url_or_id)
    if m_raw:
        gid = m_raw.group(0)
    else:
        m = re.search(r"lichess\.org/([A-Za-z0-9]{8,12})", url_or_id)
        if not m:
            raise ValueError(
                f"Could not extract a Lichess game id from '{url_or_id}'. "
                "Provide a URL like https://lichess.org/abcdefgh or a raw id."
            )
        gid = m.group(1)

    # Canonicalize 12->8
    if len(gid) == 12:
        canonical = gid[:8]
        console.print(f"[dim]Canonicalizing 12-char id '{gid}' → '{canonical}' for export[/dim]")
        return canonical
    return gid

def fetch_pgn_for_game(game_id_or_full: str, *, include_evals: bool = False, token: Optional[str] = None) -> str:
    """
    Fetch PGN via official export endpoint, auto-canonicalizing IDs.
    Fallback to ".pgn" direct URL if needed.
    """
    # First pass: canonicalize to 8-char id if 12 given.
    canonical_id = game_id_or_full[:8] if len(game_id_or_full) == 12 else game_id_or_full

    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {
        "moves": "true",
        "clocks": "false",
        "evals": "true" if include_evals else "false",
        "opening": "true",
    }

    # Try official export
    urls = [
        f"https://lichess.org/game/export/{canonical_id}.pgn",
        # Fallback: direct .pgn on game page (legacy but often works)
        f"https://lichess.org/{canonical_id}.pgn",
    ]

    errors: List[str] = []
    for u in urls:
        try:
            resp = requests.get(u, params=params if "export" in u else None, headers=headers, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                console.print(f"[dim]PGN fetched via {u}[/dim]")
                return resp.text
            errors.append(f"{u} -> {resp.status_code}")
        except Exception as e:
            errors.append(f"{u} -> {type(e).__name__}: {e}")

    raise RuntimeError("Failed to fetch PGN. Tried:\n  " + "\n  ".join(errors))

def parse_positions_from_pgn(pgn_text: str) -> Tuple[List[PositionInfo], Dict[str, str]]:
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("Unable to parse PGN from Lichess export.")

    headers = dict(game.headers)
    board = game.board()

    positions: List[PositionInfo] = []
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        side = "White" if board.turn == chess.WHITE else "Black"
        fen = board.fen()
        move_san = board.san(move)
        move_uci = move.uci()
        move_number = board.fullmove_number
        positions.append(
            PositionInfo(
                ply=ply,
                side_to_move=side,
                fen=fen,
                move_san=move_san,
                move_uci=move_uci,
                move_number=move_number,
            )
        )
        board.push(move)

    return positions, headers

# ----------------------------- OpenAI calls -----------------------------

async def call_openai_responses(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int = 300,
    temperature: float = 0.2,
) -> Tuple[str, int, int]:
    """
    Async call to Responses API; returns (text, input_tokens, output_tokens).
    SDK exposes response.output_text & response.usage.  See official README.
    """
    resp = await client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    text = getattr(resp, "output_text", "")
    usage = getattr(resp, "usage", None)
    in_toks = getattr(usage, "input_tokens", 0) if usage else 0
    out_toks = getattr(usage, "output_tokens", 0) if usage else 0
    return text, in_toks, out_toks

async def analyze_one_position(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    p: PositionInfo,
    template: str,
    max_output_tokens: int,
    temperature: float,
    attempt_limit: int = 4,
    base_backoff: float = 0.8,
) -> PositionResult:
    prompt = render_position_prompt(template, p)
    attempt = 0
    last_err = None
    async with sem:
        while attempt < attempt_limit:
            attempt += 1
            try:
                text, in_toks, out_toks = await call_openai_responses(
                    client, model, prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature
                )
                # best-effort JSON parse
                as_json = None
                try:
                    cleaned = text.strip()
                    if cleaned.startswith("```"):
                        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
                        cleaned = re.sub(r"```$", "", cleaned).strip()
                    as_json = json.loads(cleaned)
                except Exception:
                    as_json = None

                return PositionResult(
                    ply=p.ply,
                    move_san=p.move_san,
                    move_uci=p.move_uci,
                    side_to_move=p.side_to_move,
                    usage_input_tokens=in_toks,
                    usage_output_tokens=out_toks,
                    analysis_raw=text.strip(),
                    analysis_json=as_json,
                )
            except (openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)))
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                break

    return PositionResult(
        ply=p.ply,
        move_san=p.move_san,
        move_uci=p.move_uci,
        side_to_move=p.side_to_move,
        usage_input_tokens=0,
        usage_output_tokens=0,
        analysis_raw="",
        analysis_json=None,
        error=last_err or "Unknown error",
    )

async def analyze_positions_parallel(
    positions: List[PositionInfo],
    model: str,
    position_prompt: str,
    *,
    concurrency: int = 10,
    max_output_tokens: int = 300,
    temperature: float = 0.2,
) -> Tuple[List[PositionResult], Dict[str, int]]:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    results: List[PositionResult] = []
    total_in = 0
    total_out = 0
    ok = 0
    failed = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Phase 2: analyzing positions[/]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
        console=console,
    )
    task_id = progress.add_task("positions", total=len(positions))

    start = time.time()
    with progress:
        coros = [
            analyze_one_position(
                sem, client, model, p, position_prompt, max_output_tokens, temperature
            ) for p in positions
        ]
        for coro in asyncio.as_completed(coros):
            res = await coro
            results.append(res)
            if res.error:
                failed += 1
            else:
                ok += 1
                total_in += res.usage_input_tokens
                total_out += res.usage_output_tokens
            progress.update(task_id, advance=1)

    elapsed = time.time() - start
    stats = {
        "ok": ok,
        "failed": failed,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "elapsed_sec": int(elapsed),
    }
    return results, stats

def trim_per_move_payload_for_final(results: List[PositionResult], max_chars_per: int = 240) -> List[Dict[str, Any]]:
    payload = []
    for r in sorted(results, key=lambda x: x.ply):
        j = r.analysis_json if r.analysis_json else {"comment": r.analysis_raw[:max_chars_per]}
        payload.append({
            "ply": r.ply,
            "move": r.move_san,
            "side_to_move": r.side_to_move,
            "summary": j,
        })
    return payload

async def final_summary(
    model: str,
    headers: Dict[str, str],
    results: List[PositionResult],
    final_prompt_template: str,
    *,
    max_output_tokens: int = 800,
    temperature: float = 0.2,
) -> Tuple[str, int, int]:
    client = AsyncOpenAI()

    white = headers.get("White", "?")
    black = headers.get("Black", "?")
    micro_json = json.dumps(trim_per_move_payload_for_final(results), ensure_ascii=False)
    headers_kv = ", ".join(
        f"{k}={v}" for k, v in headers.items() if k in {"Event","Site","Date","Round","Result","Opening"}
    )

    prompt = final_prompt_template.format(
        headers=headers_kv,
        white=white,
        black=black,
        micro_json=micro_json,
    )
    text, in_toks, out_toks = await call_openai_responses(
        client, model, prompt, max_output_tokens=max_output_tokens, temperature=temperature
    )
    return text.strip(), in_toks, out_toks

def show_summary_tables(
    phase2_stats: Dict[str, int],
    phase2_results: List[PositionResult],
    final_text: str,
    final_in: int,
    final_out: int,
) -> None:
    t = Table(title="Phase 2 — Position Analyses (parallel)")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Positions analyzed (OK)", str(phase2_stats["ok"]))
    t.add_row("Positions failed", str(phase2_stats["failed"]))
    t.add_row("Input tokens (sum)", str(phase2_stats["total_input_tokens"]))
    t.add_row("Output tokens (sum)", str(phase2_stats["total_output_tokens"]))
    t.add_row("Elapsed (sec)", str(phase2_stats["elapsed_sec"]))
    console.print(t)

    failures = [r for r in phase2_results if r.error]
    if failures:
        ft = Table(title="Failures (first 5)", show_lines=True)
        ft.add_column("Ply")
        ft.add_column("Move")
        ft.add_column("Error")
        for r in failures[:5]:
            ft.add_row(str(r.ply), r.move_san, r.error or "")
        console.print(ft)

    ftok = Table(title="Phase 3 — Final Summary Tokens")
    ftok.add_column("Metric")
    ftok.add_column("Value", justify="right")
    ftok.add_row("Final input tokens", str(final_in))
    ftok.add_row("Final output tokens", str(final_out))
    console.print(ftok)

    console.print(Panel.fit(final_text, title="Final Game Summary", border_style="green"))

# ----------------------------- CLI -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze a Lichess game with the OpenAI API (parallel per-position).")
    p.add_argument("url_or_id", help="Lichess game URL (e.g., https://lichess.org/abcdefgh) or raw game id.")
    p.add_argument("--concurrency", type=int, default=10, help="Max parallel OpenAI calls (default: 10).")
    p.add_argument("--limit", type=int, default=0, help="Limit to first N positions (0 = all).")
    p.add_argument("--include-evals", action="store_true", help="Request eval numbers in PGN export (not used in prompts).")

    p.add_argument("--model", default="gpt-4o-mini", help="Model for per-position analysis.")
    p.add_argument("--final-model", default="gpt-4o-mini", help="Model for final summary.")

    p.add_argument("--position-prompt-file", help="Path to custom template for position analysis.")
    p.add_argument("--final-prompt-file", help="Path to custom template for the final summary.")
    p.add_argument("--position-max-output-tokens", type=int, default=300)
    p.add_argument("--final-max-output-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.2)

    p.add_argument("--lichess-token", help="Optional Lichess API token (env LICHESS_TOKEN also supported).")
    return p

async def main_async(args: argparse.Namespace) -> int:
    console.print(Panel("Phase 1: Fetching PGN & parsing positions", style="cyan"))

    # Get Lichess token (optional)
    lichess_token = args.lichess_token or os.environ.get("LICHESS_TOKEN")

    # Extract canonical id & fetch
    game_id_input = re.search(r"([A-Za-z0-9]{8,12})", args.url_or_id)
    raw_id = game_id_input.group(1) if game_id_input else args.url_or_id
    canonical_id = extract_lichess_game_id(raw_id)

    pgn = fetch_pgn_for_game(canonical_id, include_evals=args.include_evals, token=lichess_token)
    positions, headers = parse_positions_from_pgn(pgn)

    if args.limit and args.limit > 0:
        positions = positions[: args.limit]

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
        max_output_tokens=args.position_max_output_tokens,
        temperature=args.temperature,
    )

    # Phase 3 — final summary
    console.print(Panel("Phase 3: Final summary synthesis", style="cyan"))
    final_text, final_in, final_out = await final_summary(
        model=args.final_model,
        headers=headers,
        results=results,
        final_prompt_template=final_prompt,
        max_output_tokens=args.final_max_output_tokens,
        temperature=args.temperature,
    )

    show_summary_tables(stats, results, final_text, final_in, final_out)
    return 0

def main() -> int:
    load_dotenv()  # NEW: load .env first thing

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
