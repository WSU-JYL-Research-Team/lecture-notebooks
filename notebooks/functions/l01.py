import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection

# Color definitions
BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"

### Example 1: Angular Position of a Pointer ###

# Angle definitions
QUARTER_ANGLES = {0: "0", np.pi / 2: "π/2", np.pi: "π", 3 * np.pi / 2: "3π/2"}

# Plots the circle
def plot_circle_ex1(theta=5 * np.pi / 6):
    radius = 1
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle((0, 0), radius, facecolor=BLUE, alpha=0.12, edgecolor="none"))
    ax.add_patch(plt.Circle((0, 0), radius, facecolor="none", edgecolor=INK, linewidth=2))
    for phi, label in QUARTER_ANGLES.items():
            x, y = radius * np.cos(phi), radius * np.sin(phi)
            ax.plot([0, x], [0, y], linestyle=":", linewidth=1.2, color=MUTED, alpha=0.8)
            ax.text(x * 1.1, y * 1.1, label, ha="center", va="center", color=MUTED, fontsize=10, fontweight="bold")

    wedge = Wedge((0, 0), radius, 0, np.degrees(theta), facecolor=RED, alpha=0.4, edgecolor="none")
    ax.add_patch(wedge)

    for phi, label in [(0, "a"), (theta, "b")]:
        x, y = radius * np.cos(phi), radius * np.sin(phi)
        ax.plot([0, x], [0, y], linestyle="-", linewidth=2.5, color=RED)
        ax.text(x * 1.15, y * 1.15, label, ha="center", va="center", color=RED, fontsize=13, fontweight="bold")

    ax.plot(0, 0, marker="o", markersize=7, color=INK, zorder=3)

    margin = radius * 1.2
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Angular position of a pointer", fontsize=13, color=INK, fontweight="bold")
    plt.show()

# Generate random radius lines (simulated pointer spins) and estimate P(a < theta < b) empirically
def compute_empirical_solution(num_lines, a, b):
    radius = 1
    rng = np.random.default_rng()
    angles = rng.uniform(0, 2 * np.pi, size=num_lines)

    in_range = (angles >= a) & (angles <= b)
    empirical_prob = in_range.mean()

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    origins = np.zeros((num_lines, 2))
    tips = np.column_stack((x, y))
    segments = np.stack((origins, tips), axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.add_patch(plt.Circle((0, 0), radius, facecolor=BLUE, alpha=0.1, edgecolor="none"))
    ax.add_patch(plt.Circle((0, 0), radius, facecolor="none", edgecolor=INK, linewidth=2))
    ax.add_collection(LineCollection(segments[~in_range], colors=MUTED, linewidths=0.6, alpha=0.15, zorder=2))
    ax.add_collection(LineCollection(segments[in_range], colors=RED, linewidths=1.0, alpha=0.6, zorder=3))

    for phi, label in [(a, "a"), (b, "b")]:
        x, y = radius * np.cos(phi), radius * np.sin(phi)
        ax.plot([0, x], [0, y], linestyle="-", linewidth=2.5, color=RED)
        ax.text(x * 1.15, y * 1.15, label, ha="center", va="center", color=RED, fontsize=13, fontweight="bold")

    margin = radius * 1.2
    ax.text(-margin * 0.95, margin * 0.92, f"{empirical_prob:.1%}", fontsize=26, fontweight="bold", color=RED, va="top")

    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Randomly generating {num_lines:,} lines", fontsize=13, color=INK, fontweight="bold")
    plt.show()

    return empirical_prob

