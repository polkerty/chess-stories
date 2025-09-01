from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn
import requests
from rich.console import Console

console = Console()

# Extract [%clk hh:mm:ss] from move comments
import re as _re
_CLK_RE = _re.compile(r"\[%clk\s+([0-9:.]+)\]")

def _extract_clock_from_comment(s: str) -> Optional[str]:
    if not s:
        return None
    m = _CLK_RE.search(s)
    return m.group(1) if m else None


@dataclass
class PositionInfo:
    ply: int
    side_to_move: str
    fen: str
    move_san: str
    move_uci: str
    move_number: int
    white_clock: Optional[str] = None  # BEFORE the move
    black_clock: Optional[str] = None  # BEFORE the move


@dataclass
class MoveClock:
    """Clocks attached to the game AFTER a given move is played (from PGN comments)."""
    ply: int
    move_san: str
    mover: str  # "White" or "Black"
    white_clock_after: Optional[str]
    black_clock_after: Optional[str]


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
    canonical_id = game_id_or_full[:8] if len(game_id_or_full) == 12 else game_id_or_full

    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {
        "moves": "true",
        "clocks": "true",  # request clocks
        "evals": "true" if include_evals else "false",
        "opening": "true",
    }

    urls = [
        f"https://lichess.org/game/export/{canonical_id}.pgn",
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


def parse_positions_and_clocks(pgn_text: str) -> Tuple[List[PositionInfo], Dict[str, str], List[MoveClock]]:
    """
    Parse PGN:
      - Returns PositionInfo list (one per ply BEFORE the move is played).
      - Returns headers dict.
      - Returns MoveClock list (clock after each move, from comments).
    Also attempts to set "last known clocks" on each PositionInfo to provide time context.
    """
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("Unable to parse PGN from Lichess export.")

    headers = dict(game.headers)
    board = game.board()

    positions: List[PositionInfo] = []
    move_clocks: List[MoveClock] = []

    # Track rolling last-known clocks (after the last move made by each side)
    last_white_clock: Optional[str] = None
    last_black_clock: Optional[str] = None

    ply = 0
    node = game
    for next_node in game.mainline():
        move = next_node.move
        ply += 1
        side_to_move_str = "White" if board.turn == chess.WHITE else "Black"
        fen = board.fen()
        move_san = board.san(move)
        move_uci = move.uci()
        move_number = board.fullmove_number

        # Create PositionInfo BEFORE the move is pushed.
        positions.append(
            PositionInfo(
                ply=ply,
                side_to_move=side_to_move_str,
                fen=fen,
                move_san=move_san,
                move_uci=move_uci,
                move_number=move_number,
                white_clock=last_white_clock,
                black_clock=last_black_clock,
            )
        )

        # Play the move; the clock tag typically lives in the *resulting* node's comment.
        board.push(move)

        # Clock after the move the side just played:
        clk = _extract_clock_from_comment(next_node.comment)
        mover = side_to_move_str  # the side who just moved
        if mover == "White":
            last_white_clock = clk or last_white_clock
        else:
            last_black_clock = clk or last_black_clock

        move_clocks.append(
            MoveClock(
                ply=ply,
                move_san=move_san,
                mover=mover,
                white_clock_after=last_white_clock,
                black_clock_after=last_black_clock,
            )
        )

        node = next_node

    return positions, headers, move_clocks
