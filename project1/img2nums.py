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

poly = np.polynomial.Polynomial.fit(np.log(xs[1:]+100), np.log(ys[1:]), 4)
print(poly)
def approx(x):
    return np.exp(poly(np.log(x+100)))
approx = np.vectorize(approx)

xs, ys = xs[:4], ys[:4]

large_xs = np.linspace(np.min(xs), np.max(xs), 1000)
plt.plot(large_xs, approx(large_xs), "k--")
plt.plot(xs, ys, "o")
plt.savefig("temp.png")

print(approx(10))
print(approx(50))
print(approx(100))
