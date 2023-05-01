import numpy as np 
import glob

# code from http://compphysics.github.io/ComputationalPhysics2/doc/pub/week9/html/week9.html
def block(x):
    # preliminaries
    n = len(x)
    d = int(np.log2(n))
    s, gamma = np.zeros(d), np.zeros(d)
    mu = np.mean(x)

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in np.arange(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*np.sum((x[0:(n-1)]-mu)*(x[1:n]-mu))
        # estimate variance of x
        s[i] = np.var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (np.cumsum(((gamma/s)**2*2**np.arange(1, d+1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = np.array([
        6.634897, 9.210340, 11.344867, 13.276704, 15.086272, 16.811894,
        18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967,
        27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306,
        36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820,
        44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181
    ])

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0, d):
        if (M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")
    return mu, np.sqrt(s[k]/2**(d-k))

if __name__ == "__main__":
    folder_stats = {
        "interactions": {
            "NdMs": [("N2d2", [1, 2, 5, 10, 20, 40])],
        },
        "lrComp": {
            "NdMs": [("N1d1", [10])],
        },
        "MComp": {
            "NdMs": [
                ("N1d1", [1, 2, 5, 10]),
                ("N2d3", [1, 2, 5, 10])
            ],
        }
    }

    exit()
    for folder, data in folder_stats.items():
        for Nd, Ms in data["NdMs"]:
            for mc_type in ["met", "methas"]:
                if folder != "lrComp":
                    with open(f"data/{folder}/{Nd}_{mc_type}_blockavg.dat", "w") as datafile:
                        for M in Ms:
                            print(f"{folder}/{Nd}M{M}_{mc_type}")
                            blockE, blockStd = block(np.loadtxt(
                                f"data/{folder}/{Nd}M{M}_{mc_type}_samples.dat"
                            ).T.flatten())
                            datafile.write(f"{M} {blockE} {blockStd}\n")
                elif folder == "lrComp":
                    for M in Ms:
                        with open(f"data/lrComp/{Nd}M{M}_{mc_type}_blockavg.dat", "w") as datafile:
                            for lr_file in glob.glob(f"data/lrComp/{Nd}M{M}*_{mc_type}.dat"):
                                print(lr_file)
                                if "blockavg" in lr_file:
                                    continue
                                lr = float(lr_file.split("=")[-1].split("_")[0])
                                blockE, blockStd = block(np.loadtxt(lr_file).T.flatten())
                                datafile.write(f"{lr} {blockE} {blockStd}\n")


