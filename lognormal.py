#lognormal.py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import lognorm

sigma_sq = .05
t = 5
sigma = np.sqrt(sigma_sq) * np.sqrt(t)
print(sigma)
mu = .05
r = 2
#plot log norm
x = np.linspace(0,r,100)

plt.figure("pdf")
plt.plot(x,lognorm.pdf(x,s = sigma))

l = [.0005,.001,.01,.05,.5,.95,.99,.999,.9995]
change = [ lognorm.ppf(p,s = sigma) for p in l ]

for p,w in enumerate(change):
    print("{} : {}".format(l[p],w))

plt.show()
