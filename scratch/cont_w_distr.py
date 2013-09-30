'''
Created on 30 Sep 2013

@author: Q
'''

from quaducom.meso.scm.analytical.hui_pho_ibn_smi_model import SFC_Hui


if __name__ == '__main__':
    import numpy as np
    from scipy.integrate import cumtrapz
    from matplotlib import pyplot as plt
    sfc = SFC_Hui(l0=1., d=0.007, tau=0.1, sigma0=2200., rho=5.0)
    
    for rho in np.array([300.]):
        sfc.rho = rho
        x = np.linspace(0.3, 1.5, 500)
        pdf_x = sfc.p_x(50., x)
        muL = np.trapz(x**2*pdf_x, x)
        print muL
        cdf_x = np.hstack((0., cumtrapz(pdf_x, x)))
#         s = np.linspace(0.01, 1.0, 200)
#         pdf_s = sfc.p_s(s, 1.0)
#         cdf_s = np.hstack((0., cumtrapz(pdf_s, s)))
        plt.plot(x, pdf_x, label=str(rho))
        plt.plot(x, pdf_x * x, label='lengths')
    plt.legend()
    plt.show()
    