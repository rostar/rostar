'''
Created on 30 Sep 2013

@author: Q
'''

from quaducom.meso.scm.analytical.hui_pho_ibn_smi_model import SFC_Hui
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
import numpy as np
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
from etsproxy.traits.api import Float, Array, cached_property, Property

    
class AnalyticalFragmentLength(SFC_Hui):
    
    x = Array
    s = Float

    x_fragments = Property(depend_on='x')
    def _get_x_fragments(self):
        return np.linspace(afl.x[0], 2 * afl.x[-1], 2 * len(self.x)-1)

    hui_pdf = Property(depends_on='s,x')
    @cached_property
    def _get_hui_pdf(self):
        pdf = self.p_x(self.s, self.x)
        norm = np.trapz(pdf, self.x)
        pdf = pdf / norm
        return pdf

    hui_cdf = Property(depends_on='s,x')
    @cached_property
    def _get_hui_cdf(self):
        pdf = self.hui_pdf
        cdf = cumtrapz(pdf, self.x)
        return np.hstack((0.0,cdf))

    fragment_length_pdf = Property(depends_on='s,x')
    @cached_property
    def _get_fragment_length_pdf(self):
        pdf = self.hui_pdf
        convolved = np.convolve(pdf,pdf)
        norm = np.trapz(convolved, self.x_fragments)
        return convolved / norm
    
    fragment_length_cdf = Property(depends_on='s,x')
    @cached_property
    def _get_fragment_length_cdf(self):
        return np.hstack((0.0,cumtrapz(self.fragment_length_pdf, self.x_fragments)))
    
    def montecarlo(self,N):
        ppf = MFnLineArray(xdata=self.hui_cdf, ydata=self.x)
        fragment_lengths = []
        for i in range(N):
            fragment_length = ppf.get_value(np.random.rand(1)) + ppf.get_value(np.random.rand(1))
            fragment_lengths.append(fragment_length / 2.)
        return np.array(fragment_lengths)

if __name__ == '__main__':
    #plot_cracks_example()
    
    afl = AnalyticalFragmentLength(l0=1., d=0.007, tau=0.1, sigma0=2200., s=4.0,
                          rho=5.0, x=np.linspace(0.0, 5.0, 500))
    plt.hist(afl.montecarlo(100000), cumulative=True, normed=True, bins=200)
    plt.plot(afl.x_fragments / 2., afl.fragment_length_cdf, label='convolved')
    plt.plot(afl.x, afl.hui_cdf, label='original')
    #plt.plot(afl.montecarlo(np.linspace(0.0001,0.9999,100)), np.linspace(0.0001,0.9999,100), label='ppf')
    plt.legend(loc='best')
    plt.show()