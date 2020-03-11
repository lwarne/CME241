import numpy as np
from scipy.stats import norm

mu = 0.08
sigma = .1
pr = -1000
s = 0.0
#simple for loop
for r in np.linspace(-1,1,17):
    print("cdf P(R<{}) = {}".format(r, norm.cdf(r, mu, sigma)))
    diff = norm.cdf(r, mu, sigma)- norm.cdf(pr, mu, sigma)
    print("diff P(R<{}) - P(R<{}): {}".format(r,pr, diff))
    s += diff
    pr = r

print("sum: {}".format(s))