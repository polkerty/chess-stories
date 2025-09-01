from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from openai_analyze import PositionResult


def show_summary_tables(
    phase2_stats: Dict[str, int],
    phase2_results: List[PositionResult],
    final_text: str,
    final_in: int,
    final_out: int,
    pred_stats: Dict[str, Any],
    plot_path: Optional[str],
) -> None:
    console = Console()

    # Phase 2 stats
    t = Table(title="Phase 2 — Position Analyses (parallel)")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Positions analyzed (OK)", str(phase2_stats["ok"]))
    t.add_row("Positions failed", str(phase2_stats["failed"]))
    t.add_row("Input tokens (sum)", str(phase2_stats["total_input_tokens"]))
    t.add_row("Output tokens (sum)", str(phase2_stats["total_output_tokens"]))
    console.print(t)

    # Prediction stats
    ps = Table(title="Winner Prediction — Summary")
    ps.add_column("Metric")
    ps.add_column("Value", justify="right")
    counts = pred_stats["counts"]
    pct = pred_stats["percentages"]
    ps.add_row("Pred: White", f"{counts['White']} ({pct['White']*100:.1f}%)")
    ps.add_row("Pred: Draw",  f"{counts['Draw']} ({pct['Draw']*100:.1f}%)")
    ps.add_row("Pred: Black", f"{counts['Black']} ({pct['Black']*100:.1f}%)")
    ps.add_row("Unknown",     f"{counts['Unknown']} ({pct['Unknown']*100:.1f}%)")
    ps.add_row("Lead changes", str(pred_stats["lead_changes"]))
    ps.add_row("Longest same-leader streak", str(pred_stats["longest_streak"]))
    if plot_path:
        ps.add_row("Plot saved", plot_path)
    console.print(ps)

    # Sample failures (if any)
    failures = [r for r in phase2_results if r.error]
    if failures:
        ft = Table(title="Failures (first 5)", show_lines=True)
        ft.add_column("Ply")
        ft.add_column("Move")
        ft.add_column("Error")
        for r in failures[:5]:
            ft.add_row(str(r.ply), r.move_san, r.error or "")
        console.print(ft)

    # Final tokens
    ftok = Table(title="Phase 3 — Final Summary Tokens")
    ftok.add_column("Metric")
    ftok.add_column("Value", justify="right")
    ftok.add_row("Final input tokens", str(final_in))
    ftok.add_row("Final output tokens", str(final_out))
    console.print(ftok)

    console.print(Panel.fit(final_text, title="Final Game Summary", border_style="green"))
