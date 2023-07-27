import csv
import numpy as np
import matplotlib.pyplot as plt
from statistics import variance
from scipy.optimize import curve_fit

def f1(x, a, b, c, e, f):
    D, d, E = x
    sigma = a * E + b * D + c * d
    return sigma * sigma + e * sigma + f

def f2(x, a, b, c, e, f):
    D, d, E = x
    return (np.exp(a * d) + b) * pow(D - d, c) * (e * E + f)

def f3(x, a, b, c):
    D, d, E = x
    return (np.exp(a * d) + 1) * pow(D - d, b) * c

x_D = []
x_d = []
x_E = []
y = []
with open('G:\it2.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=';'):
        D, d, E, sd = row
        D = int(D)
        d = int(d)
        E = int(E)
        sd = float(sd)
        x_D.append(D)
        x_d.append(d)
        x_E.append(E)
        y.append(sd)

print(len(y))
popt1, _ = curve_fit(f1, (x_D, x_d, x_E), y)
popt2, _ = curve_fit(f2, (x_D, x_d, x_E), y, (-0.2, 1, 0.25, 0, 2))
popt3, _ = curve_fit(f3, (x_D, x_d, x_E), y)
print(popt1)
print(popt2)
print(popt3)

print(1.0 - variance([f1((D, d, E), *popt1) - yy for D, d, E, yy in zip(x_D, x_d, x_E, y)]) / variance(y))
print(1.0 - variance([f2((D, d, E), *popt2) - yy for D, d, E, yy in zip(x_D, x_d, x_E, y)]) / variance(y))
print(1.0 - variance([f3((D, d, E), *popt3) - yy for D, d, E, yy in zip(x_D, x_d, x_E, y)]) / variance(y))