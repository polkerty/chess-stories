from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from lichess_io import (
    extract_lichess_game_id,
    fetch_pgn_for_game,
    parse_positions_and_clocks,
    PositionInfo,
)
from openai_analyze import analyze_one_position, PositionResult
from prompts import DEFAULT_POSITION_PROMPT

load_dotenv()

app = FastAPI(title="Chess Stories Web")
app.mount("/static", StaticFiles(directory="static"), name="static")

class GameRequest(BaseModel):
    url: str

class AnalysisResponse(BaseModel):
    ply: int
    move_san: str
    side_to_move: str
    analysis_raw: str
    analysis_json: dict | None

# In-memory store for loaded games
_games: Dict[str, Dict[str, any]] = {}

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/api/game")
async def load_game(req: GameRequest) -> Dict[str, any]:
    game_id = extract_lichess_game_id(req.url)
    pgn = fetch_pgn_for_game(game_id)
    positions, headers, move_clocks = parse_positions_and_clocks(pgn)
    pos_meta = [
        {
            "ply": p.ply,
            "fen": p.fen,
            "move": p.move_san,
            "move_number": p.move_number,
            "side_to_move": p.side_to_move,
        }
        for p in positions
    ]
    _games[game_id] = {
        "positions": positions,
        "headers": headers,
        "analyses": {},
    }
    return {"game_id": game_id, "positions": pos_meta, "headers": headers}

@app.get("/api/game/{game_id}/analysis/{ply}")
async def analyze_position(game_id: str, ply: int) -> AnalysisResponse:
    game = _games.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Unknown game id")
    if ply in game["analyses"]:
        res: PositionResult = game["analyses"][ply]
    else:
        pos_list: List[PositionInfo] = game["positions"]
        if ply < 1 or ply > len(pos_list):
            raise HTTPException(status_code=404, detail="Invalid ply")
        position = pos_list[ply - 1]
        client_sem = asyncio.Semaphore(1)
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        res = await analyze_one_position(
            client_sem,
            client,
            model="gpt5-mini",
            p=position,
            template=DEFAULT_POSITION_PROMPT,
        )
        game["analyses"][ply] = res
    data = asdict(res)
    return AnalysisResponse(
        ply=data["ply"],
        move_san=data["move_san"],
        side_to_move=data["side_to_move"],
        analysis_raw=data["analysis_raw"],
        analysis_json=data.get("analysis_json"),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.webapp:app", host="0.0.0.0", port=8000, reload=False)
