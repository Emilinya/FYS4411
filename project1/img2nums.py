import numpy as np
import matplotlib.pyplot as plt

# estimated pixel coordinates of data points
points = [
    (136.5, 979), (141, 954), (145, 935), (159, 890), (181, 836), (225, 758),
    (359, 603), (581, 434), (803, 307), (1025, 202)
]

def coord2val(coords):
    def lerp(x, x1, x2, y1, y2):
        a = (y1 - y2) / (x1 - x2)
        b = (x1*y2 - x2*y1) / (x1-x2)

        return a*x + b

    # estimated pixel coordinates with corresponding values
    tr_coords = (1025.5, 25.5)
    tr_vals = (20000, 12)

    bl_coords = (136.5, 1020)
    bl_vals = (0, 2)

    # use linear interpolation to convert cooridinates to values
    x, y = coords
    vx = lerp(x, bl_coords[0], tr_coords[0], bl_vals[0], tr_vals[0])
    vy = lerp(y, bl_coords[1], tr_coords[1], bl_vals[1], tr_vals[1])
    
    return vx, vy 

vs = []
for point in points:
    vs.append(coord2val(point))
xs, ys = np.array(vs).T

# create polynomial approximation
poly = np.polynomial.Polynomial.fit(np.log(xs+20), np.log(ys), 4)
def approx(x):
    return np.exp(poly(np.log(x+20)))

# only plot first 5 data points
xs, ys = xs[:5], ys[:5]
large_xs = np.linspace(np.min(xs), np.max(xs), 1000)

# create figure
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
