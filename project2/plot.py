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


def plot_lrs(N, d, M, mc_type, folder):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lrs, blockEs, blockEstds = np.loadtxt(
        f"data/{folder}/N{N}d{d}M{M}_{mc_type}_blockavg.dat").T

    analytical = 0.5 * N * d
    plot_w_std(
        plt, lrs, np.abs(blockEs - analytical),
        blockEstds, label="", color=colors[0]
    )
    plt.axvline(0.30392)
    # print(lrs[np.argmin(blockEs)])

    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel("Learning rate []")
    plt.ylabel("$|E_{{ana}} - E_{{num}}|$ [a.u.]")
    plt.savefig(f"figures/{folder}_d{d}N{N}M{M}_{mc_type}.png", dpi=200, bbox_inches='tight')
    plt.clf()


def plot_Ms(N, d, mc_type, folder):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    max_end = 0
    min_end = float("Infinity")

    Ms, blockEs, _ = np.loadtxt(
        f"data/{folder}/N{N}d{d}_{mc_type}_blockavg.dat"
    ).T
    for M, blockE, c in zip(Ms[::-1], blockEs[::-1], colors):
        idx, E, _ = np.loadtxt(
            f"data/{folder}/N{N}d{d}M{int(M)}_{mc_type}_grad.dat"
        ).T
        plt.plot(idx, E, label=f"M={M}", color=c)
        plt.axhline(blockE, linestyle="--", color=c, linewidth=1)

        if M < 20:
            max_end = max(max_end, blockE)
            min_end = min(min_end, blockE)

    true_val = 0.5 * N * d
    if folder == "interactions":
        true_val = 3
    plt.axhline(true_val, linestyle="--", color="k", label="analytic")

    max_end = max(max_end, true_val)
    min_end = min(min_end, true_val)

    padd = (max_end - min_end)
    plt.ylim(min_end - padd, max_end + padd)
    plt.legend()

    plt.xlabel("Step []")
    plt.ylabel(r"$\left<E\right>$ [a.u.]")
    plt.savefig(f"figures/{folder}_d{d}N{N}_{mc_type}.png", dpi=200, bbox_inches='tight')
    plt.clf()


def plot_comp(N, d, folder):
    Ms, metE, metEstd = np.loadtxt(
        f"data/{folder}/N{N}d{d}_met_blockavg.dat"
    ).T
    _, methasE, methasEstd = np.loadtxt(
        f"data/{folder}/N{N}d{d}_methas_blockavg.dat"
    ).T

    true_val = 0.5 * N * d
    if folder == "interactions":
        true_val = 3

    met_diff = np.abs(metE - true_val)
    methas_diff = np.abs(methasE - true_val)

    plt.errorbar(
        Ms, met_diff, metEstd,
        fmt=".--", label="Metropolis"
    )
    plt.errorbar(
        Ms, methas_diff, methasEstd,
        fmt=".--", label="Metropolis-Hastings"
    )

    combimax_diff = np.array([max(a, b) for a, b in zip(met_diff, methas_diff)])
    combimin_diff = np.array([min(a, b) for a, b in zip(met_diff, methas_diff)])
    
    sort_idxs = np.argsort(combimax_diff)
    sortmax = combimax_diff[sort_idxs]
    sortmin = combimin_diff[sort_idxs]

    do_lim = False
    if sortmax[-1] > 10*sortmax[-2]:
        do_lim = True
        sortmax = sortmax[:-1]

    if sortmax[-1] > 10*sortmax[0]:
        plt.yscale("log")
    plt.xscale("log")
    plt.xticks(Ms, [int(M) for M in Ms])

    if do_lim:
        diff = sortmax[-1] - sortmin[0]
        plt.ylim(sortmin[0] - 0.1*diff, sortmax[-1] + 0.1*diff)

    plt.legend()
    plt.xlabel("M []")
    plt.ylabel("$E_{{num}} - E_{{ana}}$ [a.u.]")
    plt.savefig(f"figures/{folder}_d{d}N{N}_comp.png", dpi=200, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    
    plot_lrs(1, 1, 10, "met", "lrComp")
    plot_lrs(1, 1, 10, "methas", "lrComp")

    plot_Ms(1, 1, "met", "MComp")
    plot_Ms(1, 1, "methas", "MComp")
    plot_comp(1, 1, "MComp")

    plot_Ms(2, 3, "met", "MComp")
    plot_Ms(2, 3, "methas", "MComp")
    plot_comp(2, 3, "MComp")

    plot_Ms(2, 2, "met", "interactions")
    plot_Ms(2, 2, "methas", "interactions")
    plot_comp(2, 2, "interactions")
