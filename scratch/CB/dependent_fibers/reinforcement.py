'''
Created on 23.10.2012

An instance or a list of instances of the Reinforcement class
can be used by the composite crack bridge model.

@author: Q
'''

import numpy as np
from stats.spirrid.rv import RV
from etsproxy.traits.api import HasTraits, cached_property, \
    Float, Property, Int
from types import FloatType
from util.traits.either_type import EitherType


class Reinforcement(HasTraits):

    r = EitherType(klasses=[FloatType, RV])
    V_f = Float
    E_f = Float
    xi = EitherType(klasses=[FloatType, RV])
    tau = EitherType(klasses=[FloatType, RV])
    n_int = Int

    depsf_arr = Property(depends_on='r, V_f, E_F, xi, tau, n_int')
    @cached_property
    def _get_depsf_arr(self):
        weights = 1.0
        if isinstance(self.tau, RV):
            tau = self.tau.ppf(
                np.linspace(.5 / self.n_int, 1. - .5 / self.n_int, self.n_int))
            weights *= 1. / self.n_int
        else:
            tau = self.tau
        if isinstance(self.r, RV):
            r = self.r.ppf(
                np.linspace(.5 / self.n_int, 1. - .5 / self.n_int, self.n_int))
            weights *= 1. / self.n_int
        else:
            r = self.r

        if isinstance(tau, np.ndarray) and isinstance(r, np.ndarray):
            r = r.reshape(1, self.n_int)
            tau = tau.reshape(self.n_int, 1)
            
        return 2. * tau / r / self.E_f, weights
    