import numpy as np
import matplotlib.pyplot as plt
from blockinator import block


def plot_w_std(
        ax, xs, ys, stds, label, color, alpha=0.4,
        markersize=None, marker=".", linestyle="-"):
    if markersize is None:
        if len(xs) > 1000:
            markersize = 4
        elif len(xs) > 100:
            markersize = 8
        else:
            markersize = 12
    ax.fill_between(
        xs, ys - stds, ys + stds,
        color=color, alpha=alpha, edgecolor=color
    )
    ax.plot(
        xs, ys, color=color, label=label,
        markersize=markersize, linestyle=linestyle,
        marker=marker
    )


def plotMs(N, d, mc_type, folder):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    max_end = 0
    min_end = float("Infinity")

    Ms, blockEs, _ = np.loadtxt(f"data/{folder}/N{N}d{d}_{mc_type}_blockavg.dat").T
    for M, blockE, c in zip(Ms[::-1], blockEs[::-1], colors):
        idx, E, _ = np.loadtxt(f"data/{folder}/N{N}d{d}M{int(M)}_{mc_type}_grad.dat").T
        plt.plot(idx, E, label=f"M={M}", color=c)
        plt.axhline(blockE, linestyle="--", color=c, linewidth=1)

        max_end = max(max_end, blockE)
        min_end = min(min_end, blockE)

    true_val = 0.5 * N * d
    if folder == "interactions":
        true_val = 3
    plt.axhline(true_val, linestyle="--", color="k", label="analytic")

    max_end = max(max_end, true_val)
    min_end = min(min_end, true_val)

    padd = (max_end - min_end) / 10
    plt.ylim(min_end - padd, max_end + padd)
    plt.legend()

    plt.xlabel("Step []")
    plt.ylabel("$E_L$ []")
    plt.savefig(f"figures/{folder}_d{d}N{N}_{mc_type}.png", dpi=200)
    plt.clf()


def plot_lrs(N, d, M, mc_type, folder):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lrs, blockEs, blockEstds = np.loadtxt(f"data/{folder}/N{N}d{d}M{M}_{mc_type}_blockavg.dat").T
    
    analytical = 0.5 * N * d
    plot_w_std(plt, lrs, np.abs(blockEs - analytical), blockEstds, label="", color=colors[0])
    plt.axvline(0.0035)
    print(lrs[np.argmin(blockEs)])

    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel("Learning rate []")
    plt.ylabel("$E_{{ana}} - E_{{num}}$ []")
    plt.savefig(f"figures/{folder}_d{d}N{N}M{M}_{mc_type}.png", dpi=200)
    plt.clf()


if __name__ == "__main__":
    plotMs(1, 1, "met", "MComp")
    plotMs(1, 1, "methas", "MComp")

    plotMs(2, 3, "met", "MComp")
    plotMs(2, 3, "methas", "MComp")

    plotMs(2, 2, "met", "interactions")
    plotMs(2, 2, "methas", "interactions")

    plot_lrs(1, 1, 10, "met", "lrComp")
    plot_lrs(1, 1, 10, "methas", "lrComp")
