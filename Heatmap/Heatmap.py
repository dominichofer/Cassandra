import numpy as np
import numpy.random
import matplotlib.pyplot as plt

# Generate some test data
x = np.random.normal(0, 20, 5_000_000)
x = np.clip(x, -64, 64)
x = np.around(x / 2, decimals=0) * 2

y = x + np.random.normal(0, 6.2, np.size(x))
y = np.clip(y, -64, 64)

r_squared = np.corrcoef(x, y)[0,1]**2
std = numpy.std(x - y)

heatmap, xedges, yedges = np.histogram2d(x, y, bins=(65, 129), range=((-64,64),(-64,64)))

plt.text(35, -55, '$R^2$={:.3f}\n$\sigma$={:.3f}'.format(r_squared, std))
plt.set_cmap('gist_heat_r')
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
plt.show()
