import numpy as np
import matplotlib.pyplot as plt
from blockinator import block

def plot_w_std(
        ax, xs, ys, stds, label, color, alpha=0.4,
        markersize=4, marker="", linestyle="-"):
    ax.fill_between(
        xs, ys - stds, ys + stds,
        color=color, alpha=alpha, edgecolor=color
    )
    ax.plot(
        xs, ys, color=color, label=label,
        markersize=markersize, linestyle=linestyle,
        marker=marker
    )


def plotMs(N, d, Ms, mc_type, folder):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    print(f"{folder} - d{d}N{N} - {mc_type}")
    max_end = 0
    min_end = float("Infinity")
    for M, c in zip(Ms[::-1], colors):
        print(f" M{M}")
        idx, E, Estd = np.loadtxt(f"data/{folder}/N{N}d{d}M{M}_{mc_type}_grad.dat").T
        # plot_w_std(plt, idx, E, Estd, f"M={M}", c)
        plt.plot(idx, E, label=f"M={M}")

        max_end = max(max_end, np.max(E[-int(len(E)/4):]))
        min_end = min(min_end, np.min(E[-int(len(E)/4):]))
        min_idx = len(E) - 10 + np.argmin(E[-10:])

        blockE, blockStd = block(np.loadtxt(f"data/{folder}/N{N}d{d}M{M}_{mc_type}_samples.dat").T.flatten())
        print(f"  Grad min: {E[min_idx]} ± {Estd[min_idx]}")
        print(f"  True min: {blockE} ± {blockStd}")

    true_val = 0.5 * N * d
    if folder == "interactions":
        true_val = 3
    plt.plot([0, idx[-1]], [true_val, true_val], "--k", label="analytic")

    max_end = max(max_end, true_val)
    min_end = min(min_end, true_val)

    padd = (max_end - min_end) / 10
    plt.ylim(min_end - padd, max_end + padd)
    plt.legend()

    plt.xlabel("Step []")
    plt.ylabel("$E_L$ []")
    plt.savefig(f"figures/{folder}_d{d}N{N}_{mc_type}.png", dpi=200)
    plt.clf()


if __name__ == "__main__":
    plotMs(1, 1, [1, 2, 5, 10], "met", "MComp")
    plotMs(1, 1, [1, 2, 5, 10], "methas", "MComp")

    plotMs(2, 3, [2, 5], "met", "MComp")
    plotMs(2, 3, [2, 5], "methas", "MComp")

    plotMs(2, 2, [1, 2, 5, 10], "met", "interactions")
    plotMs(2, 2, [1, 2, 5, 10], "methas", "interactions")
    
