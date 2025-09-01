from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
import openai  # for exception classes
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn,
    MofNCompleteColumn, SpinnerColumn, TextColumn
)

from lichess_io import PositionInfo, MoveClock

console = Console()


@dataclass
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


# ----------------------------- OpenAI calls -----------------------------

async def call_openai_responses(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int = 300,
) -> Tuple[str, int, int]:
    """
    Async call to Responses API; returns (text, input_tokens, output_tokens).
    """
    resp = await client.responses.create(
        model=model,
        input=prompt,
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
    attempt_limit: int = 4,
    base_backoff: float = 0.8,
) -> PositionResult:
    prompt = template.format(
        fen=p.fen,
        side_to_move=p.side_to_move,
        ply=p.ply,
        move_san=p.move_san,
        move_uci=p.move_uci,
        move_number=p.move_number,
        white_clock=p.white_clock if p.white_clock else "unknown",
        black_clock=p.black_clock if p.black_clock else "unknown",
    )
    attempt = 0
    last_err = None
    async with sem:
        while attempt < attempt_limit:
            attempt += 1
            try:
                text, in_toks, out_toks = await call_openai_responses(
                    client, model, prompt,
                    max_output_tokens=max_output_tokens,
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

    with progress:
        coros = [
            analyze_one_position(
                sem, client, model, p, position_prompt, max_output_tokens
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

    stats = {
        "ok": ok,
        "failed": failed,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
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


def moves_with_clocks_jsonable(move_clocks: List[MoveClock]) -> List[Dict[str, Any]]:
    out = []
    for m in move_clocks:
        out.append({
            "ply": m.ply,
            "move": m.move_san,
            "mover": m.mover,
            "white_clock_after": m.white_clock_after,
            "black_clock_after": m.black_clock_after,
        })
    return out


async def final_summary(
    model: str,
    headers: Dict[str, str],
    results: List[PositionResult],
    move_clocks: List[MoveClock],
    final_prompt_template: str,
    *,
    max_output_tokens: int = 800,
) -> Tuple[str, int, int]:
    client = AsyncOpenAI()

    white = headers.get("White", "?")
    black = headers.get("Black", "?")

    micro_json = json.dumps(trim_per_move_payload_for_final(results), ensure_ascii=False)
    moves_with_clocks = json.dumps(moves_with_clocks_jsonable(move_clocks), ensure_ascii=False)

    headers_kv = ", ".join(
        f"{k}={v}" for k, v in headers.items()
        if k in {"Event","Site","Date","Round","Result","Opening"}
    )

    prompt = final_prompt_template.format(
        headers=headers_kv,
        white=white,
        black=black,
        micro_json=micro_json,
        moves_with_clocks_json=moves_with_clocks,
    )
    text, in_toks, out_toks = await call_openai_responses(
        client, model, prompt, max_output_tokens=max_output_tokens
    )
    return text.strip(), in_toks, out_toks
