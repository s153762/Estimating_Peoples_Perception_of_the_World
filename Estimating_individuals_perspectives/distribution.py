from scipy.optimize import fsolve
from scipy.optimize import least_squares
import scipy.integrate as integrate
import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt
import time

class Distribution:
    def __init__(self):
        self.distribution_type = "vonmises"
        self.params = []
        self.target = []
        self.results = []
        self.distribution_time = 0
        self.parafoveal = np.pi/6

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
            c1 = vonmises.cdf(angle, k) - 0.9#cdf(angle, k) - 0.9
            return [c1]

        # Initit K value
        z0 = [0.001]

        start_time = time.time()
        kappa = fsolve(solve, z0)[0]
        self.params = [kappa]
        if kappa < 0:
            kappa = least_squares(solve, z0, bounds=((0), (1000)))
            self.params = kappa['x']
        self.distribution_time += time.time() - start_time



    def plot(self, output, frame_number):
        if(self.distribution_type=="vonmises"):
            kappa = self.params[0]
            fig, ax = plt.subplots(1, 1)
            x = np.linspace(vonmises.ppf(0.01, kappa), vonmises.ppf(0.99, kappa), 100)
            ax.plot(x, vonmises.pdf(x, kappa), 'r-', lw=5, alpha=0.6, label='vonmises pdf')
            ax.set_xlim([-np.pi-0.1, np.pi+0.1])
            ax.set_ylim([0.0, 1.0])

            if self.target is not []:
                for target in self.target:
                    angle  = target-self.parafoveal if target == min(self.target) else target+self.parafoveal
                    ax.axvline(x=angle, label='line at x = {}'.format(target))
            plt.xlabel("Angles (Degrees)")
            plt.ylabel("Estimated Probability")
            plt.title("Method 3a Distribution")
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

            start_time = time.time()
            kappa = self.params[0]
            self.target = [a1, a2]
            a_min = min(self.target)-self.parafoveal
            a_max = max(self.target)+self.parafoveal
            t1 = vonmises.cdf(a_min,kappa)
            t2 = vonmises.cdf(a_max,kappa)
            self.results = [t1, t2]
            ans = np.abs(t2-t1)
            if opposite:
                ans = 1-ans
            self.distribution_time += time.time() - start_time
            #print(self.results)
            #print(ans)
            return ans




#dist = Distribution()
#dist.vonmises(np.pi/3)
#dist.target_probability(-np.pi, np.pi)
#dist.target_probability(-np.pi/2, np.pi/2,True)
#dist.target_probability(-np.pi/2, np.pi/2)
#dist.plot()

