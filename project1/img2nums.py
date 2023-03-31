import numpy as np
import matplotlib.pyplot as plt

points = [
    (136.5, 979), (141, 954), (145, 935), (159, 890), (181, 836), (225, 758),
    (359, 603), (581, 434), (803, 307), (1025, 202)
]

def coord2val(coords):
    def linprox(x, x1, x2, y1, y2):
        a = (y1 - y2) / (x1 - x2)
        b = (x1*y2 - x2*y1) / (x1-x2)

        return a*x + b

    tr_coords = (1025.5, 25.5)
    tr_vals = (20000, 12)

    bl_coords = (136.5, 1020)
    bl_vals = (0, 2)

    x, y = coords
    vx = linprox(x, bl_coords[0], tr_coords[0], bl_vals[0], tr_vals[0])
    vy = linprox(y, bl_coords[1], tr_coords[1], bl_vals[1], tr_vals[1])
    
    return vx, vy 

xs = []
ys = []
for point in points:
    vx, vy = coord2val(point)
    xs.append(vx)
    ys.append(vy)
xs = np.array(xs)
ys = np.array(ys)

poly = np.polynomial.Polynomial.fit(np.log(xs+20), np.log(ys), 4)

def approx(x):
    return np.exp(poly(np.log(x+20)))
approx = np.vectorize(approx)

xs, ys = xs[:5], ys[:5]
large_xs = np.linspace(np.min(xs), np.max(xs), 1000)

plt.rcParams['font.size'] = '14'
plt.figure(tight_layout=True)
plt.plot(large_xs, approx(large_xs), "k--", label="approximation")
plt.plot(xs, ys, "o", label="data")
plt.legend()
plt.xlabel("$N$")
plt.ylabel(r"$\frac{E}{N}$", rotation=0, fontsize=18, labelpad=10)
plt.savefig("plot/approximation.png", dpi=200, transparent=True)

print(approx(10), abs(approx(10) - 2.4398) / approx(10) * 100)
print(approx(50), abs(approx(50) - 2.5452) / approx(10) * 100)
print(approx(100), abs(approx(100) - 2.6627) / approx(10) * 100)
