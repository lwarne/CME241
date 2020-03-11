#utilitymapping.py

import numpy as np
import matplotlib.pyplot as plt

gamma = .01

x = np.linspace(10,260,1000)
util = (1 - np.exp(-gamma * x))/gamma
plt.plot(x,util)
plt.show()