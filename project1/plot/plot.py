import os
import numpy as np
import matplotlib.pyplot as plt


def plot_w_std(ax, xs, ys, stds, label, color, alpha=0.4):
    ax.fill_between(
        xs, ys - stds, ys + stds,
        color=color, alpha=alpha, edgecolor=color
    )
    ax.plot(xs, ys, ".-", color=color, label=label, markersize=4)


def E_anal(alpha, N, d):
    return d*N*(alpha + 1/(4*alpha))/2


def grad_E_anal(alpha, N, d):
    return d*N*(1 - 1/(4*alpha**2))/2


def plot_part(axs, data, N, d):
    alpha_ray, E_num_ray, E_err_ray, grad_num_ray, grad_err_ray = data

    E_anal_ray = E_anal(alpha_ray, N, d)
    grad_anal_ray = grad_E_anal(alpha_ray, N, d)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axs[0].axvline(0.5, color="k", linewidth=0.8, alpha=0.5)
    plot_w_std(axs[0], alpha_ray, E_num_ray, E_err_ray, "numeric", colors[0])
    axs[0].plot(alpha_ray, E_anal_ray, ".--k", markersize=4, label="analytic")
    axs[0].legend()

    axs[1].axhline(0, color="k", linewidth=0.8, alpha=0.5)
    axs[1].axvline(0.5, color="k", linewidth=0.8, alpha=0.5)
    plot_w_std(axs[1], alpha_ray, grad_num_ray,
               grad_err_ray, "numeric", colors[1])
    axs[1].plot(alpha_ray, grad_anal_ray, ".--k",
                markersize=4, label="analytic")
    axs[1].legend()


def comp_E(N, d):
    met_data = np.loadtxt(f"data/d{d}N{N}_met.dat").T
    methas_data = np.loadtxt(f"data/d{d}N{N}_methas.dat").T

    fig, axs = plt.subplots(
        2, 2, tight_layout=True, sharex=True, sharey='row',
        figsize=(10, 6), gridspec_kw={"hspace": 0.025, "wspace": 0.05}
    )
    met_axs, methas_axs = axs.T

    plot_part(met_axs, met_data, N, d)
    plot_part(methas_axs, methas_data, N, d)

    met_axs[0].set_ylabel(
        "$\\bar{{E}}$ [$\\hbar\\omega_{{ho}}$]")
    met_axs[1].set_ylabel(
        "$\\frac{{d\\bar{{E}}}}{{d\\alpha}}$ [$\\hbar\\omega_{{ho}}$]")

    met_axs[1].set_xlabel("$\\alpha$ []")
    methas_axs[1].set_xlabel("$\\alpha$ []")

    met_axs[0].set_title("Metropolis")
    methas_axs[0].set_title("Metropolis-Hastings")

    met_axs[0].locator_params(axis='y', nbins=6)
    met_axs[1].locator_params(axis='both', nbins=6)
    methas_axs[1].locator_params(axis='x', nbins=6)

    # the met gradient can become very large, crop y-axis in that case to better showcase methas
    _, max_y = met_axs[1].get_ylim()
    min_met_y = np.min(met_data[3])
    min_anal_y = np.min(grad_E_anal(met_data[0], N, d))
    if min_met_y < 2*min_anal_y:
        met_axs[1].set_ylim(2*min_anal_y, max_y)

    plt.suptitle(
        f"Comparison between numeric and analytic results (d={d}, N={N})")
    plt.savefig(f"plot/d{d}N{N}.png", dpi=200)
    plt.close(fig)


def grad_comp_E(N, d, isEliptical=False):
    if isEliptical:
        alpha_ray, E_num_ray, E_err_ray = np.loadtxt(
            f"data/eliptical_d{d}N{N}_methas.dat").T
    else:
        alpha_ray, E_num_ray, E_err_ray = np.loadtxt(
            f"data/grad_d{d}N{N}_methas.dat").T

    maxDiff = np.max(np.abs(alpha_ray-0.5))
    hq_alpha_ray = np.linspace(0.5-maxDiff, 0.5+maxDiff, 100)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(tight_layout=True)

    if not isEliptical:
        plt.plot(hq_alpha_ray, E_anal(hq_alpha_ray, N, d), "k")
    plot_w_std(plt, alpha_ray, E_num_ray, E_err_ray, "numeric", colors[0])

    if isEliptical:
        plt.savefig(f"plot/eliptical_d{d}N{N}.png", dpi=200)
    else:
        plt.savefig(f"plot/grad_d{d}N{N}.png", dpi=200)
    plt.close()


def calibrate_comp_E(Nd_pairs, mctype):
    plt.figure(tight_layout=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    min_mags = []
    Nds = []

    for i, (N, d) in enumerate(Nd_pairs):
        magnitude_ray, E_diff_ray, E_err_ray = np.loadtxt(
            f"data/calibrate_d{d}N{N}_{mctype}.dat").T

        sort_idxs = np.argsort(magnitude_ray)
        magnitude_ray = magnitude_ray[sort_idxs]
        E_diff_ray = E_diff_ray[sort_idxs]
        E_err_ray = E_err_ray[sort_idxs]

        min_idx = np.argmin(E_diff_ray)
        min_E_diff = E_diff_ray[min_idx]
        min_mag = magnitude_ray[min_idx]
        plot_w_std(plt, magnitude_ray, E_diff_ray, E_err_ray, f"d{d}N{N}", colors[i], alpha=0.02)
        plt.plot(min_mag, min_E_diff, "o", color=colors[i])
        plt.plot(min_mag, min_E_diff, ".k")

        min_mags.append(min_mag)
        Nds.append(N*d)
    plt.legend()

    plt.xscale("log")
    plt.yscale("log")

    if mctype == "met":
        plt.xlabel("stepsize []")
    else:
        plt.xlabel("timestep []")
    plt.ylabel(
        "$\\left<\\frac{{ E_{{num}} - E_{{anal}} }}{{ E_{{anal}} }}\\right>$ []")

    plt.savefig(f"plot/calibrate_{mctype}.png", dpi=200)
    plt.clf()


    plt.plot(Nds, min_mags, ".-")
    if mctype == "met":
        plt.plot([Nds[0], Nds[-1]], [4., 4.], "k--")
        plt.ylabel("stepsize []")
    else:
        plt.plot([Nds[0], Nds[-1]], [0.0127, 0.0127], "k--")
        plt.ylabel("timestep []")

    plt.xlabel("Nd []")
    plt.savefig(f"plot/calibrate_{mctype}_mincomp.png", dpi=200)

    plt.close()


def plot_dist():
    N_list = [1, 10, 100, 500]
    d_list = [1, 2, 3]

    for N in N_list:
        for d in d_list:
            comp_E(N, d)


def plot_grad():
    grad_comp_E(1, 1, False)
    grad_comp_E(500, 3, False)

    grad_comp_E(10, 3, True)
    grad_comp_E(50, 3, True)
    grad_comp_E(100, 3, True)


def plot_calibrate():
    Nd_pairs = [(1, 1), (100, 2), (500, 3)][::-1]
    calibrate_comp_E(Nd_pairs, "met")
    calibrate_comp_E(Nd_pairs, "methas")


def main():
    # plot_dist()
    plot_grad()
    # plot_calibrate()


if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    main()
