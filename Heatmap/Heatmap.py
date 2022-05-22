import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import csv

# Generate some test data
#x = np.random.normal(0, 20, 5_000_000)
#x = np.clip(x, -64, 64)
#x = np.around(x / 2, decimals=0) * 2

#y = x + np.random.normal(0, 6.2, np.size(x))
#y = np.clip(y, -64, 64)

for block in range(10):
    with open(f'G:\\Reversi\\{block}.log', newline='') as csvfile:
        data = list(csv.reader(csvfile))
        x = np.array([float(x) for x,y in data])
        y = np.array([float(y) for x,y in data])

    counts, bins = np.histogram(x, bins=65, range=(-32,32))
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()

    r_squared = np.corrcoef(x, y)[0,1]**2
    std = numpy.std(x - y)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(65, 65), range=((-32,32),(-32,32)))

    plt.text(-20, 20, '$R^2$={:.3f}\n$\sigma$={:.3f}'.format(r_squared, std), color='white')
    plt.set_cmap('gist_ncar')
    plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
    plt.show()
