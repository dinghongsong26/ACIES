"""IEEE-style grouped bar charts of latency and cost per task, with vs without lessons."""
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import AutoMinorLocator

# -----------------------------
# Tasks (DO NOT CHANGE)
# -----------------------------
tasks = [
    "Single-robot Warehouse",
    "Multi-robot Warehouse",
    "Patrol Robot",
    "Static_2R_3B",
    "Dynamic_2R_3B"
]

# -----------------------------
# Data
# -----------------------------
time_wo = [126.7, 391.1, 252.8, 220.7, 441.9]
time_w  = [78.3, 344.8, 153.6, 121.3, 263.4]

total_cost_wo = [0.322, 0.802, 0.612, 0.545, 1.199]
total_cost_w  = [0.213, 0.646, 0.397, 0.369, 0.732]

translator_cost_wo = [0.065, 0.146, 0.103, 0.164, 0.182]
translator_cost_w  = [0.062, 0.151, 0.102, 0.158, 0.181]

decomposer_cost_wo = [0.256, 0.655, 0.509, 0.38, 1.016]
decomposer_cost_w  = [0.151, 0.496, 0.294, 0.21, 0.55]

# -----------------------------
# Style (UNCHANGED)
# -----------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 13


def plot_ieee_bar_chart(data_wo, data_w, ylabel, filename):
    """Save a paired (w/o vs with lessons) bar chart over tasks to `task_<filename>`."""

    fig, ax = plt.subplots(figsize=(7.6, 2.8))

    x = np.arange(len(tasks))
    width = 0.32

    color_wo = 'white'
    color_w = '#4682B4'

    ax.bar(x - width/2, data_wo, width,
           label='w/o Lessons',
           color=color_wo,
           edgecolor='black',
           hatch='////',
           linewidth=0.8)

    ax.bar(x + width/2, data_w, width,
           label='with Lessons',
           color=color_w,
           edgecolor='black',
           linewidth=0.8)

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xticks(x)

    ax.set_xticklabels(tasks,
                       rotation=8,
                       ha='center',
                       va='top',
                       rotation_mode='anchor')

    # -----------------------------
    # IMPORTANT: extra headroom for legend
    # -----------------------------
    max_val = max(max(data_wo), max(data_w))
    ax.set_ylim(0, max_val * 1.35)

    # -----------------------------
    # LEGEND (inside box, horizontal)
    # -----------------------------
    ax.legend(
        loc='upper center',
        ncol=2,
        frameon=True,
        edgecolor='black',
        fancybox=False,
        framealpha=1
    )

    # -----------------------------
    # Grid (unchanged)
    # -----------------------------
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.grid(axis='y', which='minor', linestyle=':', alpha=0.25, linewidth=0.4)

    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), "task_" + filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')

    print(f"Saved: task_{filename}")
    plt.close()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    plot_ieee_bar_chart(time_wo, time_w,
                         'E2E Latency (s)',
                         'fig_latency.pdf')

    plot_ieee_bar_chart(total_cost_wo, total_cost_w,
                         'Total Cost ($)',
                         'fig_total_cost.pdf')

    plot_ieee_bar_chart(translator_cost_wo, translator_cost_w,
                         'Translator Cost ($)',
                         'fig_translator_cost.pdf')

    plot_ieee_bar_chart(decomposer_cost_wo, decomposer_cost_w,
                         'Decomposer Cost ($)',
                         'fig_decomposer_cost.pdf')