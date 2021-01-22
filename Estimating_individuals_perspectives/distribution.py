from scipy.optimize import fsolve
import scipy.integrate as integrate
import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self):
        self.distribution_type = "vonmises"
        self.params = []
        self.target = []
        self.results = []

    def vonmises(self, angle):
        self.distribution_type = "vonmises"
        # https://math.stackexchange.com/questions/1574927/what-is-the-equation-for-a-bessel-function-of-order-zero
        def I(x):
            ans = (1.0 / np.pi) * integrate.quad(lambda t: np.exp(x * np.cos(t)), 0.0, np.pi)[0]
            return ans

        # https://en.wikipedia.org/wiki/Von_Mises_distribution
        def pdf(x, k):
            ans = np.exp(k * np.cos(x)) / (2.0 * np.pi * I(k))
            return ans

        def cdf(angle, k):
            ans = integrate.quad(lambda t: pdf(t, k), -np.pi, angle)[0]
            return ans

        def solve(z):
            k = z[0]
            c1 = cdf(angle, k) - 0.9
            return [c1]

        # Initit K value
        z0 = [0.001]
        kappa = fsolve(solve,z0)[0]
        print(kappa)
        self.params = [kappa]

    def plot(self):
        if(self.distribution_type=="vonmises"):
            kappa = self.params[0]
            fig, ax = plt.subplots(1, 1)
            x = np.linspace(vonmises.ppf(0.01, kappa), vonmises.ppf(0.99, kappa), 100)
            ax.plot(x, vonmises.pdf(x, kappa), 'r-', lw=5, alpha=0.6, label='vonmises pdf')
            ax.set_xlim([-np.pi-0.1, np.pi+0.1])
            ax.set_ylim([0.0, 1.0])

            if self.target is not []:
                for target in self.target:
                    ax.axvline(x=target, label='line at x = {}'.format(target))
            plt.show()

    def target_probability(self, a1, a2, opposite=False):
        if (self.distribution_type == "vonmises"):
            # https://math.stackexchange.com/questions/1574927/what-is-the-equation-for-a-bessel-function-of-order-zero
            def I(x):
                return (1.0 / np.pi) * integrate.quad(lambda t: np.exp(x * np.cos(t)), 0.0, np.pi)[0]

            # https://en.wikipedia.org/wiki/Von_Mises_distribution
            def pdf(x, k):
                return np.exp(k * np.cos(x)) / (2.0 * np.pi * I(k))

            def cdf(angle, k):
                return integrate.quad(lambda t: pdf(t, k), -np.pi, angle)

            kappa = self.params[0]
            self.target = [a1, a2]
            t1 = cdf(a1,kappa)
            t2 = cdf(a2,kappa)
            self.results = [t1, t2]
            ans = t2[0]-t1[0]
            if opposite:
                ans = 1-ans

            print(self.results)
            print(ans)




#dist = Distribution()
#dist.vonmises(np.pi/3)
#dist.target_probability(-np.pi, np.pi)
#dist.target_probability(-np.pi/2, np.pi/2,True)
#dist.target_probability(-np.pi/2, np.pi/2)
#dist.plot()

