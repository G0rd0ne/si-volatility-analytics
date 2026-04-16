"""
05_3_dashboard_viz.py
Cell ID: zUZl2oMsMNe6
Exported: 2026-04-16T10:12:23.218688
"""

"""
Si Dual-Contract Volatility State v2.6.0 - Cell 5.3/5.5
Analytics Dashboard Visualization

6-panel dashboard: Vol Term Structure, Gap Quantiles, Volume, Samuelson, Momentum, Stress
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════
def plot_analytics_dashboard(output: dict, vol_data: dict, roc: dict, cfg: PipelineConfig, schema_version: str) -> None:
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax1, ax2, ax3, ax4, ax5, ax6 = axes
    wins = [5, 20, 60, 250]

    # 1. Vol Term Structure
    f1v = [vscalar(vol_data.get("F1", {}), f"YZ_{w}d") or 0 for w in wins]
    f2v = [vscalar(vol_data.get("F2", {}), f"YZ_{w}d") or 0 for w in wins]
    ax1.bar([w - 0.2 for w in wins], f1v, 1.5, color="#1f77b4", label="F1")
    ax1.bar([w + 0.2 for w in wins], f2v, 1.5, color="#ff7f0e", label="F2")
    ax1.set_xticks(wins); ax1.set_xticklabels([f"{w}d" for w in wins])
    ax1.set_ylabel("Ann. Volatility"); ax1.set_title("Vol Term Structure (F1 vs F2)"); ax1.legend()
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # 2. Gap Quantiles F1 20d
    f1_gap_20 = output.get("gap_analysis", {}).get("F1_current", {}).get("20d", {})
    gq = f1_gap_20.get("gap_quantiles", {}) if isinstance(f1_gap_20, dict) else {}
    if gq and all(v is not None for v in gq.values()):
        qs = [int(k[1:]) for k in gq.keys()]; vals = list(gq.values())
        ax2.plot(qs, vals, marker="o", color="#d62728", lw=2)
        ax2.fill_between(qs, vals, alpha=0.2, color="#d62728")
        ax2.set_xticks([10,25,50,75,90,95]); ax2.set_title("F1 20d Gap Amplitude Quantiles"); ax2.set_ylabel("Log-Amplitude")
    else: ax2.text(0.5, 0.5, "Insufficient gap data", ha="center", va="center", transform=ax2.transAxes)

    # 3-6: остальные панели (placeholder — заполняются по мере развития)
    ax3.text(0.5, 0.5, "Volume Ratio", ha="center", va="center", transform=ax3.transAxes)
    ax4.text(0.5, 0.5, "Samuelson Daily", ha="center", va="center", transform=ax4.transAxes)
    ax5.text(0.5, 0.5, "Momentum ROC", ha="center", va="center", transform=ax5.transAxes)
    ax6.text(0.5, 0.5, "Stress Components", ha="center", va="center", transform=ax6.transAxes)

    fig.suptitle(f"Si Volatility Analytics Dashboard v{schema_version}", fontsize=14, fontweight="bold")
    dash_path = cfg.chart_dir / f"analytics_dashboard_{schema_version}.png"
    fig.savefig(dash_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved: %s", dash_path)


if __name__ == "__main__":
    print("Cell 5.3/5.5: Dashboard Module загружен")
