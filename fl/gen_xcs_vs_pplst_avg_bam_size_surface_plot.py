import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

GRID_SIZES = (4, 8, 12)
SLIP_PROBS = (0, 0.1, 0.3, 0.5)

XCS_DATA = {
    (4, 0): 63.8,
    (4, 0.1): 68.9,
    (4, 0.3): 83.4,
    (4, 0.5): 108.2,
    (8, 0): 261.9,
    (8, 0.1): 236.0,
    (8, 0.3): 272.2,
    (8, 0.5): 321.0,
    (12, 0): 539.8,
    (12, 0.1): 527.3,
    (12, 0.3): 697.6,
    (12, 0.5): 790.4
}
PPLST_DATA = {
    (4, 0): 6.0,
    (4, 0.1): 6.7,
    (4, 0.3): 6.5,
    (4, 0.5): 6.5,
    (8, 0): 17.8,
    (8, 0.1): 17.4,
    (8, 0.3): 18.4,
    (8, 0.5): 18.4,
    (12, 0): 33.1,
    (12, 0.1): 34.5,
    (12, 0.3): 35.2,
    (12, 0.5): 34.1
}

fig = plt.figure()
ax = plt.axes(projection='3d')

vmin = min(list(XCS_DATA.values()) + list(PPLST_DATA.values()))
vmax = max(list(XCS_DATA.values()) + list(PPLST_DATA.values()))
# XCS
X, Y = np.meshgrid(GRID_SIZES, SLIP_PROBS)
Z = np.array(list(XCS_DATA.values())).reshape(
    (len(GRID_SIZES), len(SLIP_PROBS))).T
ax.plot_surface(X, Y, Z, alpha=0.5)

# PPLST
Z = np.array(list(PPLST_DATA.values())).reshape(
    (len(GRID_SIZES), len(SLIP_PROBS))).T
ax.plot_surface(X, Y, Z, alpha=0.5)

ax.elev = 20
ax.azim = 0

ax.set_title('Surface plot')
plt.show()
