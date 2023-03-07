import os
import numpy as np
import matplotlib.pyplot as plt

def plot_w_std(ax, xs, ys, stds, label, color, alpha=0.4):
    ax.fill_between(
        xs, ys - stds, ys + stds,
        color=color, alpha=alpha, edgecolor=color
    )
    ax.plot(xs, ys, ".-", color=color, label=label, markersize=4)

def E_anal(alpha, N, d, m, omega):
    return d*N*(alpha/m + (m*omega**2)/(4*alpha))/2

def grad_E_anal(alpha, N, d, m, omega):
    return d*N*(1/m - (m*omega**2)/(4*alpha**2))/2

def plot_part(axs, data, N, d, m, omega):
    alpha_ray, E_num_ray, E_err_ray, grad_num_ray, grad_err_ray = data

    E_anal_ray = E_anal(alpha_ray, N, d, m, omega)
    grad_anal_ray = grad_E_anal(alpha_ray, N, d, m, omega)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axs[0].axvline(0.5, color="k", linewidth=0.8, alpha=0.5)
    plot_w_std(axs[0], alpha_ray, E_num_ray, E_err_ray, "numeric", colors[0])
    axs[0].plot(alpha_ray, E_anal_ray, ".--k", markersize=4, label="analytic")
    axs[0].legend()

    axs[1].axhline(0, color="k", linewidth=0.8, alpha=0.5)
    axs[1].axvline(0.5, color="k", linewidth=0.8, alpha=0.5)
    plot_w_std(axs[1], alpha_ray, grad_num_ray, grad_err_ray, "numeric", colors[1])
    axs[1].plot(alpha_ray, grad_anal_ray, ".--k", markersize=4, label="analytic")
    axs[1].legend()


def comp_E(N, d, m, omega):
    met_data = np.loadtxt(f"data/d{d}N{N}_met.dat").T
    methas_data = np.loadtxt(f"data/d{d}N{N}_methas.dat").T

    fig, axs = plt.subplots(
        2, 2, sharex=True, sharey='row', figsize=(10, 6),
        gridspec_kw={"hspace": 0.025, "wspace": 0.05}
    )
    met_axs, methas_axs = axs.T
    
    plot_part(met_axs, met_data, N, d, m, omega)
    plot_part(methas_axs, methas_data, N, d, m, omega)

    met_axs[0].set_ylabel("$\\bar{{E}}$", rotation=0, fontsize=16, labelpad=15)
    met_axs[1].set_ylabel("$\\frac{{d\\bar{{E}}}}{{d\\alpha}}$", rotation=0, fontsize=20, labelpad=15)

    met_axs[1].set_xlabel("$\\alpha$ []")
    methas_axs[1].set_xlabel("$\\alpha$ []")

    met_axs[0].set_title("Metropolis")
    methas_axs[0].set_title("Metropolis-Hastings")

    plt.locator_params(axis='both', nbins=10)

    # the met gradient can become very large, crop y-axis in that case to better showcase methas
    _, max_y = met_axs[1].get_ylim()
    min_met_y = np.min(met_data[3])
    min_anal_y = np.min(grad_E_anal(met_data[0], N, d, m, omega))
    if min_met_y < 2*min_anal_y:
        met_axs[1].set_ylim(2*min_anal_y, max_y)

    plt.suptitle(f"Comparison between numeric and analytic results (d={d}, N={N})")
    plt.savefig(f"plot/d{d}N{N}.png", dpi=200)
    plt.close(fig)

def main():
    m = 1
    omega = 1

    # N_list = [1, 10, 100, 500]
    # d_list = [1, 2, 3]
    N_list = [1]
    d_list = [1]

    for N in N_list:
        for d in d_list:
            comp_E(N, d, m, omega)

if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    main()
