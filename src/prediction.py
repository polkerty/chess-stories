from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

from openai_analyze import PositionResult


def winner_pred_to_scalar(v: Optional[str]) -> Optional[float]:
    if not v:
        return None
    vv = v.strip().lower()
    if vv == "white":
        return 1.0
    if vv == "draw":
        return 0.5
    if vv == "black":
        return 0.0
    return None


def compute_prediction_stats(results: List[PositionResult]) -> Dict[str, Any]:
    preds: List[Tuple[int, Optional[float], Optional[str]]] = []
    for r in sorted(results, key=lambda x: x.ply):
        wp = None
        label = None
        if r.analysis_json and isinstance(r.analysis_json, dict):
            label = r.analysis_json.get("winner_pred")
            wp = winner_pred_to_scalar(label)
        preds.append((r.ply, wp, label))

    # Summary counts
    counts = {"White": 0, "Black": 0, "Draw": 0, "Unknown": 0}
    series = []
    for _, s, label in preds:
        if label in counts:
            counts[label] += 1
        else:
            counts["Unknown"] += 1
        series.append(s)

    # Lead changes (based on scalar with draw treated as 0.5)
    lead_changes = 0
    last_leader: Optional[str] = None
    for s in series:
        leader = None
        if s is None:
            leader = None
        elif s > 0.5:
            leader = "White"
        elif s < 0.5:
            leader = "Black"
        else:
            leader = "Draw"
        if last_leader is None:
            last_leader = leader
        elif leader is not None and leader != last_leader:
            lead_changes += 1
            last_leader = leader

    # Longest streak of same leader (ignoring None)
    longest_streak = 0
    current_streak = 0
    current_leader = None
    for s in series:
        leader = None
        if s is None:
            leader = None
        elif s > 0.5:
            leader = "White"
        elif s < 0.5:
            leader = "Black"
        else:
            leader = "Draw"
        if leader is not None and leader == current_leader:
            current_streak += 1
        elif leader is not None:
            current_leader = leader
            current_streak = 1
        else:
            current_leader = None
            current_streak = 0
        longest_streak = max(longest_streak, current_streak)

    total = len(series) if series else 0
    pct = {k: (v / total if total else 0.0) for k, v in counts.items()}

    return {
        "counts": counts,
        "percentages": pct,
        "lead_changes": lead_changes,
        "longest_streak": longest_streak,
        "series": preds,  # list of (ply, scalar, label)
    }


def plot_prediction_series(stats: Dict[str, Any], out_path: str) -> Optional[str]:
    if not _HAVE_MPL:
        return None
    series = stats.get("series", [])
    xs = [ply for (ply, _, _) in series]
    ys = [s if s is not None else float("nan") for (_, s, _) in series]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.5, 1.0], ["Black", "Draw", "White"])
    plt.xlabel("Ply")
    plt.ylabel("Winner prediction")
    plt.title("Winner prediction over time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    plt.close()
    return out_path
