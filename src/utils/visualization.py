#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt


# Publication quality settings
def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


# Color scheme
COLORS = {
    'observed': '#2E86AB',      # Blue
    'polynomial': '#A23B72',    # Magenta
    'arima': '#F18F01',         # Orange  
    'prophet': '#C73E1D',       # Red
    'lstm': '#3B1F2B',          # Dark purple
    'actual': '#28A745',        # Green
    'counterfactual': '#DC3545' # Red
}


def plot_figure3_column(results, save_path=None):
    """
    Generate Figure 3 with 3 panels in VERTICAL COLUMN layout.
    
    Panel (a): Forecasting models comparison
    Panel (b): Counterfactual analysis  
    Panel (c): Efficiency evolution
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_complete_analysis()
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    setup_publication_style()
    
    pre_finops = results['pre_finops']
    post_finops = results['post_finops']
    forecasts = results['forecasts']
    counterfactual = results['counterfactual']
    
    # Create figure with 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    
    n_forecast = len(post_finops)
    future_months = np.arange(12, 12 + n_forecast)
    
    # =========================================================================
    # Panel (a): Forecasting Models Comparison
    # =========================================================================
    ax1 = axes[0]
    
    # Observed data (pre-FinOps)
    ax1.scatter(pre_finops['month_index'], pre_finops['cloud_cost_index'], 
                s=80, c=COLORS['observed'], label='Observed (Pre-FinOps)', 
                zorder=5, edgecolors='white', linewidth=1)
    
    # Plot each model forecast
    max_reasonable = 400
    model_colors = {
        'Polynomial': COLORS['polynomial'],
        'ARIMA': COLORS['arima'],
        'Prophet': COLORS['prophet'],
        'LSTM': COLORS['lstm']
    }
    
    for name, data in forecasts.items():
        if data['pred'] is not None:
            pred = np.array(data['pred'])
            if np.max(pred) < max_reasonable:
                ax1.plot(future_months, pred, '-', linewidth=2, 
                         color=model_colors[name], label=f'{name}', alpha=0.9)
                
                if data['lower'] is not None:
                    lower = np.clip(data['lower'], 0, max_reasonable)
                    upper = np.clip(data['upper'], 0, max_reasonable)
                    ax1.fill_between(future_months, lower, upper, 
                                     alpha=0.15, color=model_colors[name])
    
    ax1.axvline(x=11.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(10, 350, 'Forecast\nStart', ha='right', va='top', fontsize=9, color='gray')
    
    ax1.set_xlabel('Month Index (from Jul 2020)')
    ax1.set_ylabel('Cloud Cost (Indexed, Jun 2021 = 100)')
    ax1.set_title('(a) Forecasting Models Comparison', fontweight='bold', pad=10, loc='left')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(0, max_reasonable)
    ax1.set_xlim(-1, 55)
    
    # =========================================================================
    # Panel (b): Counterfactual Analysis (FinOps Impact)
    # =========================================================================
    ax2 = axes[1]
    
    # Pre-FinOps observed
    ax2.scatter(pre_finops['month_index'], pre_finops['cloud_cost_index'], 
                s=80, c=COLORS['observed'], label='Pre-FinOps Observed', 
                zorder=5, edgecolors='white', linewidth=1)
    
    # Counterfactual projection with 95% PI (orange shading)
    cf_months = counterfactual['months']
    cf_cost = counterfactual['cost_counterfactual']
    cf_lower = counterfactual['cost_cf_lower']
    cf_upper = counterfactual['cost_cf_upper']
    
    ax2.fill_between(cf_months, cf_lower, cf_upper, 
                     alpha=0.25, color='#FFA500', label='95% Prediction Interval')
    ax2.plot(cf_months, cf_cost, '--', color=COLORS['counterfactual'], 
             linewidth=2.5, label='Counterfactual (No FinOps)')
    
    # Actual post-FinOps cost
    actual_months = np.arange(12, 12 + len(post_finops))
    ax2.scatter(actual_months, post_finops['cloud_cost_index'], 
                s=50, c=COLORS['actual'], alpha=0.7, label='Actual (With FinOps)',
                edgecolors='white', linewidth=0.5, zorder=4)
    
    ax2.axvline(x=11.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Shade savings area
    actual_cost = post_finops['cloud_cost_index'].values
    ax2.fill_between(actual_months, actual_cost, cf_cost, 
                     alpha=0.3, color='green', label='FinOps Savings')
    
    ax2.set_xlabel('Month Index (from Jul 2020)')
    ax2.set_ylabel('Cloud Cost (Indexed)')
    ax2.set_title('(b) FinOps Impact: Counterfactual vs Actual', fontweight='bold', pad=10, loc='left')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax2.set_ylim(0, min(500, max(cf_upper) * 1.1))
    ax2.set_xlim(-1, 55)
    
    # =========================================================================
    # Panel (c): Cost Efficiency Evolution
    # =========================================================================
    ax3 = axes[2]
    
    months_post = post_finops['month_index'].values
    cost_per_billion = post_finops['cost_per_billion_msg'].values
    
    ax3.plot(months_post, cost_per_billion, '-', color=COLORS['actual'], 
             linewidth=2.5, marker='o', markersize=4, label='Cost per Billion Messages')
    
    ax3.axhline(y=cost_per_billion[0], color=COLORS['counterfactual'], 
                linestyle='--', alpha=0.7, linewidth=1.5, label='Initial Level (100)')
    
    ax3.fill_between(months_post, cost_per_billion, cost_per_billion[0], 
                     alpha=0.3, color='green')
    
    efficiency_factor = cost_per_billion[0] / cost_per_billion[-1]
    ax3.annotate(f'{efficiency_factor:.1f}× improvement', 
                 xy=(months_post[-1], cost_per_billion[-1]),
                 xytext=(months_post[-1]-12, cost_per_billion[-1]+20),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    
    ax3.set_xlabel('Month Index (Post-FinOps)')
    ax3.set_ylabel('Cost per Billion Messages (Indexed)')
    ax3.set_title('(c) Cloud Cost Efficiency Evolution', fontweight='bold', pad=10, loc='left')
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_ylim(0, 110)
    
    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    return fig
