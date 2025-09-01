from __future__ import annotations

from typing import Optional

# NOTE: Because we use .format(...) later, any literal curly braces in this template
# must be doubled ({{ and }}). Placeholders like {fen} remain single-braced.
DEFAULT_POSITION_PROMPT = """\
You are a strong chess analyst. Analyze the position **before** the listed move is played.
Return a compact JSON object (no backticks) with these fields and nothing else:
{{
  "eval": "winning|better|equal|worse|lost",
  "plan": "1-2 short sentences (≤30 words)",
  "motif": "a single tactical/positional motif",
  "comment": "≤25 words single-sentence note about the chosen move",
  "winner_pred": "White|Black|Draw"
}}
Be concise and do not exceed the word limits. Keep in mind that it's possible for people to time out, 
so clocks should be a factor, along with the position itself, in your prediction.

Context:
- FEN: {fen}
- Side to move: {side_to_move}
- Ply (from 1): {ply}
- Move about to be played: {move_san} ({move_uci})
- Last known clocks: White={white_clock}, Black={black_clock}
"""

DEFAULT_FINAL_PROMPT = """\
You are a chess coach. You will receive:
  • Game headers
  • A per-move list including SAN and clock after the move (when available)
  • Per-ply micro-analyses (JSON) including a winner prediction at each ply

Please produce (plain text):
1) A 5-bullet high-level narrative of the game (opening, plans, momentum shifts, endgame).
2) 3 critical turning points with move numbers and a one-liner for each (be specific).
3) The main recurring themes (2–4).
4) One training recommendation for the weaker side.
Keep it under ~300 words. Avoid redundancy.

Game headers:
{headers}

Players: {white} vs {black}

Per-move (ply-indexed) list with move and clocks (JSON):
{moves_with_clocks_json}

Per-ply micro-analyses (trimmed JSON):
{micro_json}
"""

def load_prompt(path: Optional[str], default_text: str) -> str:
    if not path:
        return default_text
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
