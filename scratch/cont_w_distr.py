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


class AnalyticalCracks(SFC_Hui):
    c = Float
    x = Array
    s = Float
    
    def w_func(self,x1,x2):
        a1 = np.maximum(0., self.s - self.c * x1)
        x1min = (self.s-a1)/self.c
        w1 = (self.s + a1) / 2. * x1min
        a2 = np.maximum(0., self.s - self.c * x2)
        x2min = (self.s - a2)/self.c
        w2 = (self.s + a2) / 2. * x2min
        return w1 + w2
    
    cached_pdfs = Property(depends_on='s')
    @cached_property
    def _get_cached_pdfs(self):
        pdf = self.p_x(self.s, self.x)
        norm = np.trapz(pdf, self.x)
        pdf = pdf / norm
        jpdf = pdf.reshape(1,len(self.x)) * pdf.reshape(len(self.x),1)
        w_arr = self.w_func(self.x.reshape(1, len(self.x)), self.x.reshape(len(self.x),1))
        return norm, pdf, jpdf, w_arr

    def jpdf(self, x1, x2):
        return self.p_x(s, x1) * self.p_x(s, x2) / self.cached_pdfs[0]**2
    
    def Pw(self, w):
        mask = self.cached_pdfs[3] < w
        dx2 = (self.x[-1]-self.x[-2])**2
        return np.sum(self.cached_pdfs[2] * mask * dx2)
    
    def CDF_w(self, w_arr):
        cdfs = []
        for w in w_arr:
            cdfs.append(self.Pw(w))
        return cdfs

if __name__ == '__main__':
    ac = AnalyticalCracks(l0=1., d=0.007, tau=0.1, sigma0=2200., s=2.0,
                          rho=5.0, c=1.0, x=np.linspace(0.0, 5.0, 500))
    for s in [1.2, 2.0, 3.0]:
        ac.s = s
        pdf_x = ac.p_x(s, ac.x)
        pdf_x = pdf_x / np.trapz(pdf_x, ac.x)
        cdf_x = np.hstack((0., cumtrapz(pdf_x, ac.x)))
        sf = MFnLineArray(xdata=cdf_x + 1e-12 * ac.x, ydata=ac.x)
        rand_vals1 = sf.get_values(np.random.rand(10000))
        rand_vals2 = sf.get_values(np.random.rand(10000))
        cracks = ac.w_func(rand_vals1, rand_vals2)
        plt.hist(cracks, bins=40, normed=True, label=str(s), cumulative=True)
        w_arr = np.linspace(0.0,7.0,100)
        cdf_w = ac.CDF_w(w_arr)
        plt.plot(w_arr, cdf_w, color='black', lw=2)
    plt.ylim(0,1.2)
    plt.legend()
    plt.show()
    