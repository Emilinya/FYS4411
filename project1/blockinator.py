import numpy as np 

# analytical mean energy
def E_anal(alpha, N, d):
    return d*N*(alpha + 1/(4*alpha))/2

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

# standard deviation of the mean of the means
def calc_mystd(data):
    means = np.mean(data, axis=0)

    return np.mean(means), np.std(means) / np.sqrt(means.size)

def scistr(num):
    scale = int(np.log10(num))
    if scale == 0:
        return f"${num:.2f}$"
    return f"${num/(10**scale):.2f}\\cdot 10^{{{scale}}}$"

def analyze(datafile, alpha):
    data = np.loadtxt(datafile)
    
    mymean, mystd = calc_mystd(data)
    print(datafile)
    print(" ", "    my mean and std:", mymean, mystd)

    samplemean, meanstd = np.mean(data), np.std(data) / np.sqrt(data.size)
    print(" ", "sample mean and std:", samplemean, meanstd)

    flat_data = data.T.flatten()
    blockmean, blockstd = block(flat_data)
    print(" ", " block mean and std:", blockmean, blockstd)

    np.random.shuffle(flat_data)
    sblockmean, sblockstd = block(flat_data)
    print(" ", "sblock mean and std:", sblockmean, sblockstd)

    if "eliptical" in datafile:
        dN = datafile.split("/")[-1].split("_")[1]
    else:
        dN = datafile.split("/")[-1].split("_")[0]
    d, N = int(dN[1]), int(dN[3:])

    block_scistr = scistr(blockstd)
    ndigits = 2 - int(block_scistr.split("{")[-1][:-2])

    return f"""\
${d}, {N}$ & ${alpha:.3g}$ & ${E_anal(alpha, N, d):.{ndigits}f}$ & ${samplemean:.{ndigits}f}$ & \
{scistr(meanstd)} & {scistr(mystd)} & {block_scistr} \\\\\
"""

np.random.seed(1337)

tlist = []
tlist.append(analyze("data/samples/d1N1_methas.dat", 0.4))
tlist.append(analyze("data/samples/d2N100_methas.dat", 0.4))
tlist.append(analyze("data/samples/d3N500_methas.dat", 0.4))
for s in tlist:
    print(s)

print()

tlist = []
tlist.append(analyze("data/samples/eliptical_d3N10_methas.dat", 0.4977))
tlist.append(analyze("data/samples/eliptical_d3N50_methas.dat", 0.4901))
tlist.append(analyze("data/samples/eliptical_d3N100_methas.dat", 0.4845))
for s in tlist:
    print(s)
