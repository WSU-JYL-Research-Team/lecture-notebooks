import numpy as np
import matplotlib.pyplot as plt

# Color definitions
BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"

# Damage state definitions
DAMAGE_LABELS = {1: r"$\theta_1$", 2: r"$\theta_2$", 3: r"$\theta_3$"}

# Plots the CDF of the building damage state
def plot_damage_cdf(p1, p2, p3):
    if p1 + p2 + p3 > 1.0:
        raise ValueError("Probabilities must not sum to more than 1.0")

    thetas = np.array([1, 2, 3])
    probs = np.array([p1, p2, p3])
    cdf = np.cumsum(probs)

    fig, ax = plt.subplots(figsize=(6, 4))

    x_start, x_end = thetas[0] - 1, thetas[-1] + 1
    x = np.concatenate(([x_start], thetas, [x_end]))
    y = np.concatenate(([0], cdf, [cdf[-1]]))
    ax.step(x, y, where="post", color=BLUE, linewidth=2)

    for i, theta in enumerate(thetas):
        y_before = cdf[i - 1] if i > 0 else 0
        ax.plot(theta, y_before, marker="o", markerfacecolor="white", markeredgecolor=BLUE, markersize=7, zorder=3)
        ax.plot(theta, cdf[i], marker="o", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=7, zorder=3)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(thetas)
    ax.set_xticklabels([DAMAGE_LABELS[t] for t in thetas], fontsize=12)
    ax.set_xlabel(r"Damage state $\theta$", color=INK)
    ax.set_ylabel(r"$F_\Theta(\theta)$", color=INK)
    ax.set_title("CDF of the building damage state", fontsize=13, color=INK, fontweight="bold")
    ax.grid(alpha=0.2)
    plt.show()

# Plots the damage states proportioned on a horizontal bar from 0 to 1
def plot_damage_bar(p1, p2, p3):
    if p1 + p2 + p3 > 1.0:
        raise ValueError("Probabilities must not sum to more than 1.0")

    thetas = np.array([1, 2, 3])
    probs = np.array([p1, p2, p3])
    boundaries = np.concatenate(([0], np.cumsum(probs)))
    colors = [BLUE, RED, MUTED]

    fig, ax = plt.subplots(figsize=(8, 1.8))

    for i, theta in enumerate(thetas):
        left, width = boundaries[i], probs[i]
        ax.barh(0, width, left=left, height=0.6, color=colors[i % len(colors)], edgecolor="white", linewidth=1.5)
        ax.text(left + width / 2, 0, f"{DAMAGE_LABELS[theta]}\n{width:.1f}", ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    for boundary in boundaries:
        ax.axvline(boundary, color=INK, linewidth=0.8, ymin=0.15, ymax=0.85)
        ax.text(boundary, -0.55, f"{boundary:.1f}", ha="center", va="top", color=INK, fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.set_title("Damage states", fontsize=13, color=INK, fontweight="bold")
    plt.show()

# Draws one random sample from the CDF and visualizes where it lands on the bar
def sample_damage_state(p1, p2, p3):
    if p1 + p2 + p3 > 1.0:
        raise ValueError("Probabilities must not sum to more than 1.0")

    thetas = np.array([1, 2, 3])
    probs = np.array([p1, p2, p3])
    boundaries = np.concatenate(([0], np.cumsum(probs)))
    colors = [BLUE, RED, MUTED]

    rng = np.random.default_rng()
    u = rng.uniform(0, 1)
    sampled_theta = thetas[np.searchsorted(boundaries, u, side="right") - 1]

    fig, ax = plt.subplots(figsize=(8, 1.8))

    for i, theta in enumerate(thetas):
        left, width = boundaries[i], probs[i]
        ax.barh(0, width, left=left, height=0.6, color=colors[i % len(colors)], edgecolor="white", linewidth=1.5)
        ax.text(left + width / 2, 0, f"{DAMAGE_LABELS[theta]}\n{width:.1f}", ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    for boundary in boundaries:
        ax.axvline(boundary, color=INK, linewidth=0.8, ymin=0.15, ymax=0.85)
        ax.text(boundary, -0.55, f"{boundary:.1f}", ha="center", va="top", color=INK, fontsize=10)

    ax.plot(u, 0, marker="v", markersize=14, color=INK, zorder=4)
    ax.text(u, 0.55, f"u = {u:.2f}", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.75)
    ax.axis("off")
    ax.set_title(f"Sampled damage state: {DAMAGE_LABELS[sampled_theta]}", fontsize=13, color=INK, fontweight="bold")
    plt.show()

    return sampled_theta

# Draws n_points random samples from the CDF using inverse-CDF sampling
def sample_damage_states(n_points, p1, p2, p3):
    if p1 + p2 + p3 > 1.0:
        raise ValueError("Probabilities must not sum to more than 1.0")

    thetas = np.array([1, 2, 3])
    probs = np.array([p1, p2, p3])
    boundaries = np.concatenate(([0], np.cumsum(probs)))

    rng = np.random.default_rng()
    u = rng.uniform(0, 1, size=n_points)
    indices = np.searchsorted(boundaries, u, side="right") - 1
    return thetas[indices]

# Computes the expected cost, theoretical vs. sample-based, given each damage state's probability and cost
def expected_cost(n_points, p1, p2, p3, c1, c2, c3):
    if p1 + p2 + p3 > 1.0:
        raise ValueError("Probabilities must not sum to more than 1.0")

    theoretical_cost = p1 * c1 + p2 * c2 + p3 * c3

    costs = {1: c1, 2: c2, 3: c3}
    samples = sample_damage_states(n_points, p1, p2, p3)
    sample_cost = np.array([costs[s] for s in samples]).mean()

    print(f"Theoretical expected cost: ${theoretical_cost:,.2f}")
    print(f"Sample mean cost:          ${sample_cost:,.2f}")

    return theoretical_cost, sample_cost
