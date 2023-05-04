import numpy as np
import matplotlib.pyplot as plt


def plot_lrs(N, d, M, folder):
    met_lrs, met_blockEs, met_blockEstds = np.loadtxt(
        f"data/{folder}/N{N}d{d}M{M}_met_blockavg.dat").T
    methas_lrs, methas_blockEs, methas_blockEstds = np.loadtxt(
        f"data/{folder}/N{N}d{d}M{M}_methas_blockavg.dat").T

    optimal_lr = 0.30392
    plt.axvline(optimal_lr, color="k", linestyle="--")
    plt.text(optimal_lr*1.05, 10, f"{optimal_lr}")

    analytical = 0.5 * N * d
    plt.errorbar(
        met_lrs, np.abs(met_blockEs - analytical),
        met_blockEstds, fmt=".--", label="Metropolis"
    )
    plt.errorbar(
        methas_lrs, np.abs(methas_blockEs - analytical),
        methas_blockEstds, fmt=".--", label="Metropolis-Hastings"
    )
    plt.legend()


    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel("Learning rate []")
    plt.ylabel("$|E_{{ana}} - E_{{num}}|$ [a.u.]")
    plt.savefig(f"figures/{folder}_d{d}N{N}M{M}.png", dpi=200, bbox_inches='tight')
    plt.clf()


def plot_Ms(N, d, mc_type, folder):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    avg_Es = []
    std_Es = []

    Ms, _, _ = np.loadtxt(
        f"data/{folder}/N{N}d{d}_{mc_type}_blockavg.dat"
    ).T
    for M, c in zip(Ms[::-1], colors):
        idx, E, Estd = np.loadtxt(
            f"data/{folder}/N{N}d{d}M{int(M)}_{mc_type}_grad.dat"
        ).T
        # plt.errorbar(idx, E, Estd, label=f"M={M}", color=c)
        plt.plot(idx, E, label=f"M={M}", color=c)

        avg_Es.append(np.mean(E[50:]))
        std_Es.append(np.std(E[50:]))
    avg_E = np.average(avg_Es, weights=1/(Ms[::-1]**2))
    std_E = np.average(std_Es, weights=1/(Ms[::-1]**2))

    true_val = 0.5 * N * d
    if folder == "interactions":
        true_val = 3
    plt.axhline(true_val, linestyle="--", color="k", label="analytic")

    plt.ylim(min(true_val, avg_E) - 1.1*std_E, avg_E + 1.1*std_E)
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

    plt.errorbar(
        Ms, np.abs(metE - true_val), metEstd,
        fmt=".--", label="Metropolis"
    )
    plt.errorbar(
        Ms, np.abs(methasE - true_val), methasEstd,
        fmt=".--", label="Metropolis-Hastings"
    )

    # Hardcoding, yay!
    if (N, d) == (2, 2):
        plt.ylim(0.09, 0.31)
    else:
        plt.yscale("log")

    plt.xscale("log")
    plt.xticks(Ms, [int(M) for M in Ms])

    plt.legend()
    plt.xlabel("M []")
    plt.ylabel("$|E_{{num}} - E_{{ana}}|$ [a.u.]")
    plt.savefig(f"figures/{folder}_d{d}N{N}_comp.png", dpi=200, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    
    plot_lrs(1, 1, 10, "lrComp")

    plot_Ms(1, 1, "met", "MComp")
    plot_Ms(1, 1, "methas", "MComp")
    plot_comp(1, 1, "MComp")

    plot_Ms(2, 3, "met", "MComp")
    plot_Ms(2, 3, "methas", "MComp")
    plot_comp(2, 3, "MComp")

    plot_Ms(2, 2, "met", "interactions")
    plot_Ms(2, 2, "methas", "interactions")
    plot_comp(2, 2, "interactions")
