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

    x_cbs = Property(depend_on='x')
    def _get_x_cbs(self):
        return np.linspace(self.x[0], 2 * self.x[-1], 2 * len(self.x)-1)

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

    cbs_length_pdf = Property(depends_on='s,x')
    @cached_property
    def _get_cbs_length_pdf(self):
        pdf = self.hui_pdf
        convolved = np.convolve(pdf,pdf)
        norm = np.trapz(convolved, self.x_cbs)
        return convolved / norm
    
    cbs_length_cdf = Property(depends_on='s,x')
    @cached_property
    def _get_cbs_length_cdf(self):
        return np.hstack((0.0,cumtrapz(self.cbs_length_pdf, self.x_cbs)))
    
    def montecarlo(self,N):
        ppf = MFnLineArray(xdata=self.hui_cdf, ydata=self.x)
        fragment_lengths = []
        for i in range(N):
            fragment_length = ppf.get_value(np.random.rand(1)) + ppf.get_value(np.random.rand(1))
            fragment_lengths.append(fragment_length / 2.)
        return np.array(fragment_lengths)

if __name__ == '__main__':
    afl = AnalyticalFragmentLength(l0=1., d=0.007, tau=0.1, sigma0=2200., s=4.0,
                          rho=5.0, x=np.linspace(0.0, 3.0, 500))
    plt.hist(afl.montecarlo(10000), cumulative=True, normed=True, bins=100, label='simul_CB_lengths')
    plt.plot(afl.x_cbs / 2., afl.cbs_length_cdf, label='convolved')
    plt.plot(afl.x, afl.hui_cdf, label='original')
    plt.legend(loc='best')
    plt.show()