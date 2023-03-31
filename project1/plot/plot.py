import os
import numpy as np
import matplotlib.pyplot as plt


def plot_w_std(
        ax, xs, ys, stds, label, color, alpha=0.4,
        markersize=4, marker=".", linestyle="-"):
    ax.fill_between(
        xs, ys - stds, ys + stds,
        color=color, alpha=alpha, edgecolor=color
    )
    ax.plot(
        xs, ys, color=color, label=label,
        markersize=markersize, linestyle=linestyle,
        marker=marker
    )


def E_anal(alpha, N, d):
    return d*N*(alpha + 1/(4*alpha))/2


def grad_E_anal(alpha, N, d):
    return d*N*(1 - 1/(4*alpha**2))/2


def plot_part(axs, data, N, d):
    alpha_ray, E_num_ray, E_err_ray, grad_num_ray, grad_err_ray = data

    # I forgot to divide by sqrt(walkerCount)
    E_err_ray /= np.sqrt(8)
    grad_err_ray /= np.sqrt(8)

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
    met_data = np.loadtxt(f"data/full/d{d}N{N}_met.dat").T
    methas_data = np.loadtxt(f"data/full/d{d}N{N}_methas.dat").T

    fig, axs = plt.subplots(
        2, 2, tight_layout=True, sharex=True, sharey='row',
        figsize=(10, 6), gridspec_kw={"hspace": 0.025, "wspace": 0.05}
    )
    met_axs, methas_axs = axs.T

    plot_part(met_axs, met_data, N, d)
    plot_part(methas_axs, methas_data, N, d)

    met_axs[0].set_ylabel(r"$\left<E\right>$ [$\hbar\omega_{ho}$]")
    met_axs[1].set_ylabel(
        r"$\frac{d\left<E\right>}{d\alpha}}$ [$\hbar\omega_{ho}$]")

    met_axs[1].set_xlabel("$\\alpha$ []")
    methas_axs[1].set_xlabel("$\\alpha$ []")

    met_axs[0].set_title("Metropolis")
    methas_axs[0].set_title("Metropolis-Hastings")

    met_axs[0].locator_params(axis='y', nbins=6)
    met_axs[1].locator_params(axis='both', nbins=6)
    methas_axs[1].locator_params(axis='x', nbins=6)

    plt.suptitle(f"d={d}, N={N}")
    plt.savefig(f"plot/full/d{d}N{N}.png", dpi=200, transparent=True)
    plt.close(fig)


def grad_comp_E(N, d, isEliptical=False):
    if isEliptical:
        alpha_ray, E_num_ray, E_err_ray = np.loadtxt(
            f"data/grad/eliptical_d{d}N{N}_methas.dat").T
    else:
        alpha_ray, E_num_ray, E_err_ray = np.loadtxt(
            f"data/grad/d{d}N{N}_methas.dat").T

    # I forgot to divide by sqrt(walkerCount)
    E_err_ray /= np.sqrt(8)

    maxDiff = np.max(np.abs(alpha_ray-0.5))
    hq_alpha_ray = np.linspace(0.5-maxDiff, 0.5+maxDiff, 100)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(tight_layout=True)

    if not isEliptical:
        plt.plot(hq_alpha_ray, E_anal(hq_alpha_ray, N, d), "k")
        plot_w_std(plt, alpha_ray, E_num_ray, E_err_ray,
                   "", colors[0], markersize=8)
    else:
        avg_E = np.average(E_num_ray, weights=1/E_err_ray)
        std_E = np.std(E_num_ray)

        idxs = np.where(np.abs(E_num_ray - avg_E) < std_E)

        plt.scatter(alpha_ray[idxs], E_num_ray[idxs], c=E_err_ray[idxs])
        plt.colorbar(label=r"$\sigma_{\left<\left<E\right>\right>}$")

        min_E = np.average(E_num_ray[idxs][-10:],
                           weights=1/E_err_ray[idxs][-10:])
        opt_alpha = np.average(
            alpha_ray[idxs][-10:], weights=1/E_err_ray[idxs][-10:])
        plt.axhline(min_E, color="k", linestyle="--")
        plt.axvline(opt_alpha, color="k", linestyle="--")

        yticks = np.round(plt.yticks()[0], decimals=4-int(np.log10(min_E)))
        close_idx = np.argmin(np.abs(yticks - min_E))
        yticks[close_idx] = np.round(min_E, decimals=4)
        plt.yticks(yticks, yticks)

        _, xmax = plt.xlim()
        xticks = np.round(
            np.linspace(opt_alpha, xmax, 4),
            decimals=4
        )
        plt.xticks(xticks, xticks)

        print(opt_alpha)

    plt.xlabel("$\\alpha$ []")
    plt.ylabel("$\\left<{{E}}\\right>$ [$\\hbar\\omega_{{ho}}$]")
    plt.title(f"d={d}, N={N}")

    if isEliptical:
        plt.savefig(
            f"plot/grad/eliptical_d{d}N{N}.png", dpi=200, transparent=True)
    else:
        plt.savefig(f"plot/grad/d{d}N{N}.png", dpi=200, transparent=True)
    plt.close()


def calibrate_comp_E(mctype):
    plt.figure(tight_layout=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ["-", "--"]

    min_mags = []
    min_diffs = []
    min_errs = []

    Nd_pairs = []
    for N in [1, 10, 100, 500]:
        for d in [1, 2, 3]:
            Nd_pairs.append((N, d))

    for i, (N, d) in enumerate(Nd_pairs):
        magnitude_ray, E_diff_ray, E_err_ray = np.loadtxt(
            f"data/calibrate/d{d}N{N}_{mctype}.dat").T

        # I forgot to divide by sqrt(walkerCount)
        E_err_ray /= np.sqrt(128)

        sort_idxs = np.argsort(magnitude_ray)
        magnitude_ray = magnitude_ray[sort_idxs]
        E_diff_ray = E_diff_ray[sort_idxs]
        E_err_ray = E_err_ray[sort_idxs]

        min_idx = np.argmin(E_diff_ray)
        min_E_diff = E_diff_ray[min_idx]
        min_mag = magnitude_ray[min_idx]
        plot_w_std(
            plt, magnitude_ray, E_diff_ray, E_err_ray, f"d{d}N{N}",
            colors[i//2], alpha=0, markersize=8, linestyle=linestyles[i % 2]
        )
        plt.plot(min_mag, min_E_diff, "o", color=colors[i//2])
        plt.plot(min_mag, min_E_diff, ".k")

        min_mags.append(min_mag)
        min_diffs.append(min_E_diff)
        min_errs.append(E_err_ray[min_idx])
    plt.legend(ncol=2)

    plt.yscale("log")

    if mctype == "met":
        plt.xlabel("stepsize []")
    else:
        plt.xlabel("timestep []")
    plt.ylabel(r"$\left<\frac{ E_{num} - E_{anal} }{ E_{anal} }\right>$ []")

    plt.savefig(f"plot/calibrate/{mctype}.png", dpi=200, transparent=True)
    plt.clf()

    plt.scatter(min_mags, min_diffs, c=10000*np.array(min_errs))
    plt.colorbar(
        label=r"$10^{4}\cdot\left<\sigma_{\left<\left<E_{num}\right>\right>}\right>$")
    plt.yscale("log")

    ymin, ymax = plt.ylim()
    ycenter = np.exp((np.log(ymax) + np.log(ymin))/2)

    optimal_magnitude = min_mags[np.argmin(min_diffs)]
    if mctype == "met":
        plt.axvline(optimal_magnitude, color="k",
                    linestyle="--", label="Chosen stepsize")
        plt.text(optimal_magnitude*1.01, ycenter, f"{optimal_magnitude:.5f}")
        plt.xlabel("stepsize []")
    else:
        plt.axvline(optimal_magnitude, color="k",
                    linestyle="--", label="Chosen timestep")
        plt.text(optimal_magnitude*1.01, ycenter, f"{optimal_magnitude:.5f}")
        plt.xlabel("timestep []")
    plt.ylabel(r"$\left<\frac{ E_{num} - E_{anal} }{ E_{anal} }\right>$ []")

    plt.legend()

    plt.savefig(f"plot/calibrate/{mctype}_mincomp.png",
                dpi=200, transparent=True)

    plt.close()

def onebody_plot(N):
    plt.figure(tight_layout=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    x_J, rho_J, err_J = np.loadtxt(f"data/onebody/N{N}_Jastrow.dat").T
    x_nJ, rho_nJ, err_nJ = np.loadtxt(f"data/onebody/N{N}_noJastrow.dat").T

    max_rho = np.max(rho_J)
    idxs = np.where(rho_J > max_rho/100)
    xlim = np.max(x_J[idxs])

    plot_w_std(plt, x_J, rho_J, err_J, "with interactions", colors[0])
    plot_w_std(plt, x_nJ, rho_nJ, err_nJ, "without interactions", colors[1])
    plt.xlim(-xlim, xlim)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    plt.xlabel("$x$ []")
    plt.ylabel(r"$\rho$ []")
    plt.title(f"N={N}")
    plt.legend()
    plt.savefig(f"plot/onebody/N{N}", dpi=200)
    plt.close()

def plot_calibrate():
    calibrate_comp_E("met")
    calibrate_comp_E("methas")


def plot_dist():
    N_list = [1, 10, 100, 500]
    d_list = [1, 2, 3]

    for N in N_list:
        for d in d_list:
            comp_E(N, d)


def plot_grad():
    grad_comp_E(1, 1, False)
    grad_comp_E(100, 2, False)
    grad_comp_E(500, 3, False)

    grad_comp_E(10, 3, True)
    grad_comp_E(50, 3, True)
    grad_comp_E(100, 3, True)


def plot_onebody():
    onebody_plot(10)
    onebody_plot(50)
    onebody_plot(100)

def main():
    plot_calibrate()
    plot_dist()
    plot_grad()
    plot_onebody()


if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    main()
