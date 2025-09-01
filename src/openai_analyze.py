from __future__ import annotations

import asyncio
import json
import os
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

DEBUG = os.environ.get("CHESS_ANALYZE_DEBUG", "0").lower() not in ("", "0", "false", "no")


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


# ----------------------------- Helpers -----------------------------

_CODE_FENCE_START_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_CODE_FENCE_END_RE = re.compile(r"\s*```$")

def _ensure_debug_dir() -> str:
    d = "debug"
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = _CODE_FENCE_START_RE.sub("", t)
    t = _CODE_FENCE_END_RE.sub("", t)
    return t.strip()

def _find_first_json_object(text: str) -> Optional[str]:
    """
    Return the first top-level {...} object as a string, handling quotes and escapes.
    If none found, returns None.
    """
    if not text:
        return None
    s = _strip_code_fences(text)

    brace_level = 0
    in_string = False
    escape = False
    start_idx = None

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if brace_level == 0:
                start_idx = i
            brace_level += 1
        elif ch == "}":
            if brace_level > 0:
                brace_level -= 1
                if brace_level == 0 and start_idx is not None:
                    return s[start_idx:i+1]
    return None

def _extract_first_json_object(text: str) -> Optional[dict]:
    obj = _find_first_json_object(text)
    if obj is None:
        return None
    try:
        return json.loads(obj)
    except Exception:
        return None

def _infer_winner_pred(analysis_json: dict, side_to_move: str) -> Optional[str]:
    """
    If 'winner_pred' missing, infer from 'eval' and side_to_move.
    winning/better => side_to_move
    lost/worse     => opposite
    equal          => Draw
    """
    if not isinstance(analysis_json, dict):
        return None
    if analysis_json.get("winner_pred"):
        return analysis_json["winner_pred"]
    ev = (analysis_json.get("eval") or "").strip().lower()
    if not ev:
        return None
    if ev in {"winning", "better"}:
        return side_to_move
    if ev in {"lost", "worse"}:
        return "White" if side_to_move == "Black" else "Black"
    if ev == "equal":
        return "Draw"
    return None


# ----------------------------- OpenAI calls -----------------------------

async def _call_responses_and_text(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
) -> Tuple[str, int, int]:
    """
    Responses API call. If output_text empty, try to reconstruct from structured output.
    """
    resp = await client.responses.create(
        model=model,
        input=prompt,
    )

    # Primary convenience accessor
    text = getattr(resp, "output_text", None)
    if not text:
        # Fallback: stitch together any structured text parts
        try:
            parts: List[str] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    # some SDK versions have c.text or c.text.value; handle both
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t:
                        parts.append(t)
                    elif hasattr(t, "value") and isinstance(t.value, str):
                        parts.append(t.value)
            text = "\n".join(parts).strip() if parts else ""
        except Exception:
            text = ""

    usage = getattr(resp, "usage", None)
    in_toks = getattr(usage, "input_tokens", 0) if usage else 0
    out_toks = getattr(usage, "output_tokens", 0) if usage else 0
    return text or "", in_toks, out_toks


async def _call_chat_fallback(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
) -> Tuple[str, int, int]:
    """
    Hard fallback: use Chat Completions if Responses returned nothing.
    """
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    usage = getattr(resp, "usage", None)
    in_toks = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_toks = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, in_toks, out_toks


async def call_openai_text(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
) -> Tuple[str, int, int, bool]:
    """
    Returns (text, input_tokens, output_tokens, used_fallback).
    """
    text, in_toks, out_toks = await _call_responses_and_text(
        client, model, prompt
    )
    used_fallback = False

    if not text:
        # Try chat fallback once
        text, in2, out2 = await _call_chat_fallback(
            client, model, prompt
        )
        used_fallback = True
        # Prefer non-zero usage figures if chat returned them
        in_toks = in2 or in_toks
        out_toks = out2 or out_toks

    return text, in_toks, out_toks, used_fallback


# ----------------------------- Position analysis -----------------------------

async def analyze_one_position(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    p: PositionInfo,
    template: str,
    attempt_limit: int = 4,
    base_backoff: float = 0.8,
    progress: Optional[Progress] = None,
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
                text, in_toks, out_toks, used_fallback = await call_openai_text(
                    client, model, prompt
                )

                if DEBUG and used_fallback and progress:
                    progress.console.print(
                        f"[yellow]DEBUG:[/] ply {p.ply} used chat.fallback (responses output was empty)"
                    )

                # Always dump a few samples for inspection (ply 1–3)
                if DEBUG and p.ply <= 3:
                    d = _ensure_debug_dir()
                    path = os.path.join(d, f"sample_ply_{p.ply}.txt")
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("=== PROMPT ===\n")
                            f.write(prompt)
                            f.write("\n\n=== RAW OUTPUT ===\n")
                            f.write(text or "")
                    except Exception:
                        pass

                # Robust JSON parse and winner_pred inference
                as_json = _extract_first_json_object(text)
                if as_json is None:
                    if DEBUG and progress:
                        preview = (text or "").replace("\n", " ")[:160]
                        progress.console.print(
                            f"[yellow]DEBUG:[/] JSON parse failed for ply {p.ply} ({p.move_san}). Preview: {preview}"
                        )
                    # Also dump per-ply file so we can inspect
                    if DEBUG:
                        d = _ensure_debug_dir()
                        path = os.path.join(d, f"position_ply_{p.ply}.txt")
                        try:
                            with open(path, "w", encoding="utf-8") as f:
                                f.write("=== PROMPT ===\n")
                                f.write(prompt)
                                f.write("\n\n=== RAW OUTPUT ===\n")
                                f.write(text or "")
                        except Exception:
                            pass
                    as_json = {}

                wp = _infer_winner_pred(as_json, p.side_to_move)
                if wp:
                    as_json["winner_pred"] = wp

                norm_json = as_json if as_json else None

                return PositionResult(
                    ply=p.ply,
                    move_san=p.move_san,
                    move_uci=p.move_uci,
                    side_to_move=p.side_to_move,
                    usage_input_tokens=in_toks,
                    usage_output_tokens=out_toks,
                    analysis_raw=(text or "").strip(),
                    analysis_json=norm_json,
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
) -> Tuple[List[PositionResult], Dict[str, int]]:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    results: List[PositionResult] = []
    total_in = 0
    total_out = 0
    ok = 0
    failed = 0
    parsed = 0  # how many yielded a non-empty JSON

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
                sem, client, model, p, position_prompt,
                progress=progress
            ) for p in positions
        ]
        for coro in asyncio.as_completed(coros):
            res = await coro
            results.append(res)
            if res.error:
                failed += 1
            else:
                ok += 1
                if res.analysis_json:
                    parsed += 1
                total_in += res.usage_input_tokens
                total_out += res.usage_output_tokens
            progress.update(task_id, advance=1)

    if DEBUG:
        progress.console.print(
            f"[yellow]DEBUG:[/] Phase 2 summary: ok={ok}, failed={failed}, parsed_json={parsed}/{len(positions)}"
        )
        # dump a small index of first 5 results
        d = _ensure_debug_dir()
        idx_path = os.path.join(d, "phase2_first5.txt")
        try:
            with open(idx_path, "w", encoding="utf-8") as f:
                for r in sorted(results, key=lambda x: x.ply)[:5]:
                    f.write(f"ply={r.ply} move={r.move_san} json={bool(r.analysis_json)}\n")
                    f.write(f"raw[:160]={ (r.analysis_raw or '').replace(chr(10),' ')[:160] }\n\n")
        except Exception:
            pass

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
        j = r.analysis_json if r.analysis_json else {"comment": (r.analysis_raw or "")[:max_chars_per]}
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

    text, in_toks, out_toks, used_fallback = await call_openai_text(
        client, model, prompt
    )

    if DEBUG:
        d = _ensure_debug_dir()
        path = os.path.join(d, "final_summary.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n=== RAW OUTPUT ===\n")
                f.write(text or "")
        except Exception:
            pass
        preview = (text or "").replace("\n", " ")[:200]
        console.print(f"[yellow]DEBUG:[/] Final summary preview: {preview}")
        if used_fallback:
            console.print("[yellow]DEBUG:[/] final summary used chat.fallback")

    return (text or "").strip(), in_toks, out_toks
