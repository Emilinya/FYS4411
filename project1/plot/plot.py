import os
import numpy as np
import matplotlib.pyplot as plt

def plot_w_std(xs, ys, stds, label, color, alpha=0.4):
    plt.fill_between(
        xs, ys - stds, ys + stds,
        color=color, alpha=alpha, edgecolor=color
    )
    plt.plot(xs, ys, ".-", color=color, label=label, markersize=4)

def E_anal(alpha, N, d, m, omega):
    return d*N*(alpha/m + (m*omega**2)/(4*alpha))/2


def compE(N, d, m, omega, mctype):
    alpha_ray, E_num_ray, err_ray = np.loadtxt(f"data/d{d}N{N}_{mctype}.dat").T
    E_anal_ray = E_anal(alpha_ray, N, d, m, omega)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(tight_layout=True)
    plot_w_std(alpha_ray, E_num_ray, err_ray, "numeric", colors[0])
    plt.plot(alpha_ray, E_anal_ray, ".--k", markersize=4, label="analytic")
    plt.xlabel("$\\alpha$ []")
    plt.ylabel("$\\left<E\\right>$")
    plt.title(f"Comparison between numeric and analytic result (d={d}, N={N})")
    plt.legend()
    plt.savefig(f"plot/d{d}N{N}_{mctype}.png", dpi=200)
    plt.clf()

    # plt.plot(alpha_ray, err_ray, ".--", markersize=4)
    # plt.xlabel("$\\alpha$ []")
    # plt.ylabel("$\\sigma_E$")
    # plt.title(f"Standard deviation as a function of $\\alpha$")
    # plt.savefig(f"plot/d{d}N{N}_std.png", dpi=200)
    # plt.clf()

def main():
    m = 1
    omega = 1

    for file in os.listdir("data"):
        try:
            name, extension = file.split(".")
            vals, mctype = name.split("_")
            dn, N = vals.split("N")
            d = int(dn[1:])
            N = int(N)
            compE(N, d, m, omega, mctype)
        except:
            pass

if __name__ == "__main__":
    main()
