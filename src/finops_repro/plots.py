from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .counterfactual import fit_quadratic_demand, quadratic_planning_projection
from .interrupted_time_series import fitted_primary_series


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure1(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    blocks = [
        (10.3, "Data sources", "Azure billing; service telemetry; vehicle and message indices; service categories"),
        (8.75, "Preprocessing", "Monthly aggregation; one shared boundary; indexing; unit-cost features"),
        (7.2, "FinOps decision rules", "Inform evidence; Optimize ranking; Operate feedback and guardrails"),
        (5.65, "Forecast benchmark", "Common folds; ARIMA, TCN, Transformer, LSTM and probabilistic LSTM"),
        (4.1, "Counterfactual + ITS", "Demand control; functional forms; segmented log unit cost; HAC uncertainty"),
        (2.55, "Uncertainty", "50/80/95% intervals; horizon reliability; low/reference/high demand"),
        (1.0, "Outputs", "Observed KPIs; bounded attribution; budgets; anomaly thresholds; governance actions"),
    ]
    facecolors = ["#f6e7a6", "#f6e7a6", "#cfe2f3", "#d9ead3", "#d9d2e9", "#fce5cd", "#eeeeee"]
    for (y, title, text), color in zip(blocks, facecolors):
        box = FancyBboxPatch(
            (1.0, y),
            8.0,
            1.05,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="0.35",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(5.0, y + 0.72, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(5.0, y + 0.33, text, ha="center", va="center", fontsize=8.6, wrap=True)
    for y in [10.3, 8.75, 7.2, 5.65, 4.1, 2.55]:
        ax.annotate("", xy=(5.0, y - 0.30), xytext=(5.0, y - 0.02), arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.set_title("FinOps evidence architecture for EV telematics", fontsize=15, fontweight="bold", pad=12)
    _save(fig, path)


def plot_figure2(calibration: pd.DataFrame, path: Path) -> None:
    cost = calibration["cloud_cost_index"].to_numpy(float)
    vehicles = calibration["vehicles_index"].to_numpy(float)
    cost_future = quadratic_planning_projection(cost, 40)
    vehicle_future = quadratic_planning_projection(vehicles, 40)
    t = np.arange(12)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, observed, future, ylabel, title in [
        (axes[0], cost, cost_future, "Cloud cost index", "Cloud-cost time comparator"),
        (axes[1], vehicles, vehicle_future, "Vehicle index", "Vehicle-growth planning model"),
    ]:
        ax.scatter(t, observed, label="Observed calibration", zorder=3)
        ax.plot(future["month_index"], future["mean"], linestyle="--", label="Planning projection")
        ax.fill_between(future["month_index"], future["lower"], future["upper"], alpha=0.18, label="95% PI")
        ax.axvline(11, linestyle=":", linewidth=1.2, label="June 2021 boundary")
        ax.set_xlabel("Month index from July 2020")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 2. Pre-FinOps observations and planning projections", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_figure3(calibration: pd.DataFrame, path: Path) -> None:
    vehicles = calibration["vehicles_index"].to_numpy(float)
    cost = calibration["cloud_cost_index"].to_numpy(float)
    fit = fit_quadratic_demand(vehicles, cost)
    grid = np.linspace(vehicles.min() - 2, vehicles.max() + 5, 250)
    pred = fit.predict(grid, alpha=0.05)
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    ax.scatter(vehicles, cost, label="Observed pre-FinOps", zorder=3)
    ax.plot(grid, pred["mean"], label="Quadratic demand fit")
    ax.fill_between(grid, pred["obs_ci_lower"], pred["obs_ci_upper"], alpha=0.20, label="95% prediction interval")
    ax.set_xlabel("Active-vehicle index (June 2021 = 100)")
    ax.set_ylabel("Cloud-cost index (June 2021 = 100)")
    ax.set_title("Figure 3. Local pre-FinOps cost-demand relationship")
    ax.text(0.03, 0.94, r"$C(V)=39.64-0.334V+0.00942V^2$", transform=ax.transAxes, va="top")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_figure4(
    monthly: pd.DataFrame,
    calibration: pd.DataFrame,
    implementation: pd.DataFrame,
    long_scenarios: pd.DataFrame,
    path: Path,
) -> None:
    pre_x = np.arange(12)
    future_x = long_scenarios["month_index"].to_numpy(float)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 12.0), gridspec_kw={"height_ratios": [1, 1, 0.9]})

    ax = axes[0]
    ax.scatter(pre_x, calibration["cloud_cost_index"], label="Observed (pre-FinOps)", zorder=4)
    for mean, lo, hi, label in [
        ("arima", "arima_lower_95", "arima_upper_95", "ARIMA"),
        ("prophet_linear_growth", "prophet_lower_95", "prophet_upper_95", "Prophet (linear growth)"),
        ("lstm_archived_scenario", "lstm_lower_95", "lstm_upper_95", "LSTM scenario"),
    ]:
        ax.plot(future_x, long_scenarios[mean], label=label)
        ax.fill_between(future_x, long_scenarios[lo], long_scenarios[hi], alpha=0.12)
    ax.axvline(11, linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 410)
    ax.set_ylabel("Cloud-cost index")
    ax.set_title("(a) Descriptive long-horizon forecasting scenarios")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    vehicles = calibration["vehicles_index"].to_numpy(float)
    cost = calibration["cloud_cost_index"].to_numpy(float)
    fit = fit_quadratic_demand(vehicles, cost)
    post = monthly.loc[monthly["date"] >= pd.Timestamp("2021-06-01")].copy()
    cf = fit.predict(post["vehicles_index"].to_numpy(float), alpha=0.05)
    post_x = np.arange(11, 11 + len(post))
    ax.scatter(pre_x, cost, label="Pre-FinOps observed", zorder=4)
    ax.plot(post_x, cf["mean"], linestyle="--", label="No-FinOps quadratic counterfactual")
    ax.fill_between(post_x, cf["obs_ci_lower"], cf["obs_ci_upper"], alpha=0.18, label="95% PI")
    ax.scatter(post_x, post["cloud_cost_index"], s=18, label="Observed with FinOps", zorder=4)
    ax.fill_between(post_x, post["cloud_cost_index"].to_numpy(float), cf["mean"].to_numpy(float), alpha=0.22, label="Counterfactual gap")
    ax.axvline(11, linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 500)
    ax.set_ylabel("Cloud-cost index")
    ax.set_title("(b) Demand-controlled counterfactual versus observed cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[2]
    x = implementation["implementation_month"].to_numpy(float)
    efficiency = implementation["cost_per_billion_messages_index"].to_numpy(float)
    ax.plot(x, efficiency, marker="o", markersize=3, label="Cost per billion messages")
    ax.fill_between(x, 0, efficiency, alpha=0.25)
    ax.axhline(100, linestyle="--", linewidth=1.0, label="Boundary level = 100")
    ax.annotate(
        "22.3x efficiency improvement",
        xy=(39, efficiency[-1]),
        xytext=(27, 24),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=9,
    )
    ax.set_xlabel("Months from FinOps boundary")
    ax.set_ylabel("Cost per billion messages")
    ax.set_title("(c) Directly observed throughput-normalized efficiency")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, path)


def plot_figure5(
    monthly: pd.DataFrame,
    calibration: pd.DataFrame,
    implementation: pd.DataFrame,
    path: Path,
) -> None:
    its = fitted_primary_series(monthly)
    fit = fit_quadratic_demand(
        calibration["vehicles_index"].to_numpy(float),
        calibration["cloud_cost_index"].to_numpy(float),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 11.5))

    ax = axes[0]
    ax.scatter(its["date"], its["observed_cost_per_vehicle"], s=24, label="Observed cost per active vehicle")
    ax.plot(its["date"], its["segmented_fit"], linewidth=2.2, label="Segmented ITS fit")
    ax.axvline(pd.Timestamp("2021-06-01"), linestyle="--", linewidth=1.2, label="FinOps boundary")
    ax.annotate(
        "Immediate level change: -45.6%\nAdditional slope change: -3.6%/month",
        xy=(pd.Timestamp("2021-07-01"), float(its.loc[its["date"] == pd.Timestamp("2021-07-01"), "observed_cost_per_vehicle"].iloc[0])),
        xytext=(pd.Timestamp("2022-02-01"), 78),
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )
    ax.set_ylabel("Cost per active vehicle\n(index, June 2021 = 100)")
    ax.set_title("(a) Interrupted time-series analysis")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate(rotation=30)

    ax = axes[1]
    x = implementation["implementation_month"].to_numpy(float)
    vehicles = implementation["vehicles_index"].to_numpy(float)
    observed = implementation["cloud_cost_index"].to_numpy(float)
    p95 = fit.predict(vehicles, alpha=0.05)
    p80 = fit.predict(vehicles, alpha=0.20)
    p50 = fit.predict(vehicles, alpha=0.50)
    low = fit.predict(0.85 * vehicles, alpha=0.05)
    high = fit.predict(1.15 * vehicles, alpha=0.05)
    ax.fill_between(x, p95["obs_ci_lower"], p95["obs_ci_upper"], alpha=0.16, label="95% PI")
    ax.fill_between(x, p80["obs_ci_lower"], p80["obs_ci_upper"], alpha=0.20, label="80% PI")
    ax.fill_between(x, p50["obs_ci_lower"], p50["obs_ci_upper"], alpha=0.26, label="50% PI")
    ax.plot(x, low["mean"], linestyle="--", label="Low demand (-15%)")
    ax.plot(x, p95["mean"], linewidth=2.0, label="Reference demand")
    ax.plot(x, high["mean"], linestyle=":", linewidth=2.0, label="High demand (+15%)")
    ax.scatter(x, observed, s=20, label="Observed post-FinOps cost", zorder=4)
    ax.set_xlabel("Months from FinOps boundary")
    ax.set_ylabel("Cloud-cost index\n(June 2021 = 100)")
    ax.set_title("(b) Counterfactual uncertainty and demand scenarios")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, path)
