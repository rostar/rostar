from traits.api import HasTraits, Array, Int, \
    Property, cached_property, Function, List, \
    Instance, Tuple
import pymc
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
import os
# 10% 4.5s


def data_reader(campaigns):
    '''
    reads the TCXX_campaign_XX.txt files and returns numpy arrays

    iput:   string - campaign combination file name  

    output: (2D array, 1D array) - (test parameters and number of cracks,
                                    chamber parameters)
    '''
    campaign_data_lst = []
    chamber_data_lst = []
    for campaign in campaigns:
        # loading the output file as a numpy array
        # 'data_pack/'
        data_pack_dir = '/home/rostislav/Documents/MATLAB/data_pack/'
        campaign_data = np.loadtxt(
            data_pack_dir + 'TC_campaign_combinations/' + campaign + '.txt', skiprows=8)
        campaign_data_lst.append(campaign_data)
        chamber_data = np.loadtxt(
            data_pack_dir + 'TCs/' + campaign[0:4] + '.txt', skiprows=13)
        chamber_data_lst.append(chamber_data)
    return campaign_data_lst, chamber_data_lst


class RegressionModel(HasTraits):
    '''
    Bayesian inference of model parameters for
    liquid fuel rocket engine reliability analysis
    '''

    # tested or virtually generated data
    campaign_data = List(Array)
    chamber_data = List(Array)

    cycles = Property(Array, depends_on='campaign_data')

    @cached_property
    def _get_cycles(self):
        cycles_lst = [np.hstack((1, np.diff(campaign[:, 0])))
                      for campaign in self.campaign_data]
        return cycles_lst

    firing_times = Property(Array, depends_on='campaign_data')

    @cached_property
    def _get_firing_times(self):
        firing_times_lst = [campaign[:, 1]
                            for campaign in self.campaign_data]
        return firing_times_lst

    cumulative_test_times = Property(List(Array), depends_on='campaign_data')

    @cached_property
    def _get_cumulative_test_times(self):
        return [np.cumsum(firing_times) for firing_times in self.firing_times]

    # extracting pressure values from the campaign_data array
    pressures = Property(List(Array), depends_on='campaign_data')

    @cached_property
    def _get_pressures(self):
        pressure_lst = [campaign[:, 2] for campaign in self.campaign_data]
        return pressure_lst

    # extracting mix ratio values from the campaign_data array
    mix_ratios = Property(List(Array), depends_on='campaign_data')

    @cached_property
    def _get_mix_ratios(self):
        mix_ratio_lst = [campaign[:, 3] for campaign in self.campaign_data]
        return mix_ratio_lst

    # number of cracks per time interval
    cumulative_cracks = Property(List(Array), depends_on='campaign_data')

    @cached_property
    def _get_cumulative_cracks(self):
        cracks_lst = [campaign[:, 4] for campaign in self.campaign_data]
        return cracks_lst

    # auxiliary array of indices of inspection times
    inspection_idxs = Property(List(Int), depends_on='campaign_data')

    @cached_property
    def _get_inspection_idxs(self):
        indices_lst = []
        for campaign in self.campaign_data:
            cycles = campaign[-1, 0]
            indices = np.zeros(cycles)
            for test_i in np.arange(cycles):
                indices[test_i] = np.where(
                    campaign[:, 0] == test_i + 1)[0][-1]
            indices_lst.append(indices.astype(int))
        return indices_lst

    inspection_times = Property(List(Array), depends_on='campaign_data')

    @cached_property
    def _get_inspection_times(self):
        inspection_times_lst = [self.cumulative_test_times[i][
            self.inspection_idxs[i]] for i in range(len(self.campaign_data))]
        return inspection_times_lst

    inspected_cracks = Property(Array, depens_on='campaign_data')

    @cached_property
    def _get_inspected_cracks(self):
        inspected_cracks_1D = np.array([])
        for i in range(len(self.campaign_data)):
            inspected_cracks_1D = np.hstack(
                (inspected_cracks_1D, self.cumulative_cracks[i][self.inspection_idxs[i]]))
        return inspected_cracks_1D

    inspected_cracks_list = Property(List(Array), depends_on='campaign_data')

    @cached_property
    def _get_inspected_cracks_list(self):
        inspected_cracks_lst = []
        for i in range(len(self.campaign_data)):
            inspected_cracks_lst.append(
                self.cumulative_cracks[i][self.inspection_idxs[i]])
        return inspected_cracks_lst

    n_channels = Int  # number of individuals (cooling channels)

    # parameters for the MCMC algorithm
    mcmc_iterations = Int
    mcmc_burn_in = Int
    mcmc_thinning = Int

    # Bayesian inference components
    priors = List
    log_likelihood = Function

    # log likelihood function for model parameters given observed data
    pymc_log_likelihood = Property()

    def _get_pymc_log_likelihood(self):
        f = pymc.stochastic(self.log_likelihood, observed=True)
        return f

    # instance of the inference model class "pymc.Model"
    model = Property(Instance(pymc.Model))

    @cached_property
    def _get_model(self):
        return pymc.Model(self.pymc_log_likelihood, self.priors)

    samples = Property(depends_on='priors,log_likelihood,mcmc_iterations,\
                                mcmc_thinning,mcmc_burn_in')

    @cached_property
    def _get_samples(self):
        '''set the Markov chain Monte Carlo with specified
        No. of iterations, burn-in length and thinning'''
        iterations = self.mcmc_iterations
        burn_in = self.mcmc_burn_in
        thinning = self.mcmc_thinning
        map_ = pymc.MAP(self.model)
        map_.fit()
        sampling_engine = pymc.MCMC(self.model)
        sampling_engine.sample(iter=iterations, burn=burn_in, thin=thinning)
        return sampling_engine

    def plotting(self):
        '''plot the results of the sampling procedure'''
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        pymc.Matplot.plot(self.samples)
        plt.show()


def survival_probability_model_MCMC(rm, model, process):
    '''postprocessor for MCMC samples
    rm: instance of a RegressionModel class
    model: string 'PH' or 'AFT'
    process: string 'regression' or 'validation' - affects the plotting '''

    N_samples = len(rm.samples.trace('beta_pressure')[:])

    m_ifr_samples = rm.samples.trace('m_ifr')[:][np.newaxis, :]
    s_ifr_samples = rm.samples.trace('s_ifr')[:][np.newaxis, :]
    p_coeff_samples = rm.samples.trace('beta_pressure')[:][np.newaxis, :]
    mr_coeff_samples = rm.samples.trace('beta_mix_ratio')[:][np.newaxis, :]
    ft_coeff_samples = rm.samples.trace('beta_firing_times')[:][np.newaxis, :]
    cycle_coeff_samples = rm.samples.trace('beta_cycle')[:][np.newaxis, :]
    p_mr_coeff_samples = rm.samples.trace('beta_p_mr')[:][np.newaxis, :]
    p_ft_coeff_samples = rm.samples.trace('beta_p_ft')[:][np.newaxis, :]
    mr_ft_coeff_samples = rm.samples.trace('beta_mr_ft')[:][np.newaxis, :]
    p2_coeff_samples = rm.samples.trace('beta_p2')[:][np.newaxis, :]
    mr2_coeff_samples = rm.samples.trace('beta_mr2')[:][np.newaxis, :]
    ft2_coeff_samples = rm.samples.trace('beta_ft2')[:][np.newaxis, :]

    ### EVALUATION OF THE MEAN CDF FUNCTION ###
    # Due to thinning, the weakly dependent MCMC samples become close to independent.
    # Therefore, the Monte Carlo method can be directly applied for the
    # estimation of the CDF mean value

    # adding covariate values at time 0 and reshaping the arrays
    for j in range(len(rm.campaign_data)):
        jth_pressure = np.hstack((rm.pressures[j][0], rm.pressures[j]))
        jth_mix_ratio = np.hstack((rm.mix_ratios[j][0], rm.mix_ratios[j]))
        jth_firing_times = np.hstack(
            (rm.firing_times[j][0], rm.firing_times[j]))
        jth_cycle = np.hstack((rm.cycles[j][0], rm.cycles[j]))
#         jth_covar_effect = np.exp(p_coeff_samples * (jth_pressure[:, np.newaxis] - p_os_samples) +
#                                   mr_coeff_samples * (jth_mix_ratio[:, np.newaxis] - mr_os_samples) +
#                                   ft_coeff_samples * (jth_firing_times[:, np.newaxis] - ft_os_samples) +
#                                   cycle_coeff_samples *
#                                   jth_cycle[:, np.newaxis]
#                                   )
        jth_covar_effect = np.exp(p_coeff_samples * jth_pressure[:, np.newaxis] +
                                  mr_coeff_samples * jth_mix_ratio[:, np.newaxis] +
                                  ft_coeff_samples * jth_firing_times[:, np.newaxis] +
                                  cycle_coeff_samples *
                                  jth_cycle[:, np.newaxis] +
                                  p_mr_coeff_samples *
                                  jth_pressure[:, np.newaxis] *
                                  jth_mix_ratio[:, np.newaxis] +
                                  p_ft_coeff_samples * jth_pressure[:, np.newaxis] * jth_firing_times[:, np.newaxis] +
                                  mr_ft_coeff_samples *
                                  jth_mix_ratio[:, np.newaxis] *
                                  jth_firing_times[:, np.newaxis] +
                                  p2_coeff_samples * jth_pressure[:, np.newaxis]**2 +
                                  mr2_coeff_samples * jth_mix_ratio[:, np.newaxis]**2 +
                                  ft2_coeff_samples *
                                  jth_firing_times[:, np.newaxis]**2
                                  )
        jth_time_variable = np.hstack(
            (1e-5, rm.cumulative_test_times[j]))[:, np.newaxis]

        if model == 'PH':
            baseline_hazard = m_ifr_samples / s_ifr_samples * (jth_time_variable /
                                                               s_ifr_samples) ** (m_ifr_samples - 1)
            hazard = baseline_hazard * jth_covar_effect
            cumhazard = cumtrapz(
                hazard, jth_time_variable, initial=0.0, axis=0)

        if model == 'AFT':
            time_variable_aux = np.hstack(
                (1e-5, 1e-5, rm.cumulative_test_times[j]))[:, np.newaxis]
            accel_time = np.cumsum(
                jth_covar_effect * np.diff(time_variable_aux, axis=0), axis=0)
            cumhazard = (accel_time / s_ifr_samples)**m_ifr_samples

        survival = np.exp(-cumhazard)

        # Monte Carlo mean
        CDF_mean = np.sum(1 - survival, axis=1) / N_samples
        lower_idx = np.rint(N_samples * 0.05)
        upper_idx = np.rint(N_samples * 0.95)
        survival_sorted = np.sort(survival)

        ### plotting ###
        if process == 'regression':
            #             for i in np.arange(N_samples):
            #                 plt.plot(
            # jth_time_variable, 1 - survival[:, i], lw=0.5, color='grey')
            plt.plot(jth_time_variable, CDF_mean,
                     lw=1, color='black')
            plt.plot(rm.inspection_times[j], rm.inspected_cracks_list[j] /
                     rm.n_channels, 'bo')
#             plt.plot(jth_time_variable, 1 -
#                      survival_sorted[:, lower_idx], color='black', lw=2, ls='dashed')
#             plt.plot(jth_time_variable, 1 - survival_sorted[:, upper_idx], color='black',
#                      lw=2, ls='dashed')

        elif process == 'validation':
            plt.plot(jth_time_variable, CDF_mean,
                     color='red', lw=2, label='prediction')
            plt.plot(jth_time_variable, 1 -
                     survival_sorted[:, lower_idx], color='red', lw=2, ls='dashed')
            plt.plot(jth_time_variable, 1 - survival_sorted[:, upper_idx], color='red',
                     lw=2, ls='dashed')
            plt.plot(rm.inspection_times[j], rm.inspected_cracks_list[j] /
                     rm.n_channels, 'ro')

        plt.legend(loc='best')
        plt.xlabel('time [s]')
        plt.ylabel('failure probability [-]')
        if model == 'PH':
            title = 'Survival Analysis - PH model'
        elif model == 'AFT':
            title = 'Survival Analysis - AFT model'
        plt.ylim(0, 1)
        plt.title(title)

    # sq_err_CDF = np.sum(
    #    (CDF_mean[rm.inspection_idxs] - rm.inspected_cracks_list[j] / rm.n_channels)**2)
    #sq_err_scatter = np.sum((CDF_mean - fract5)**2 + (CDF_mean - fract95)**2)
    return 1, 1  # sq_err_CDF, sq_err_scatter


def run_example(model, regression_campaigns, validation_campaign):
    # generate data
    #generated_data = data_generator(regression_campaigns)
    campaign_data, chamber_data = data_reader(regression_campaigns)
    n_channels = 300
    rm = RegressionModel(campaign_data=campaign_data,
                         chamber_data=chamber_data,
                         n_channels=n_channels,
                         mcmc_iterations=300000,
                         mcmc_burn_in=250000,
                         mcmc_thinning=100
                         )

    # Define prior distribution instances from the pymc module.
    # e.g. pymc.Uniform('distribution_name', lower=Float, upper=Float)
    # or pymc.Normal('distribution_name', mu=Float, tau=Float), where mu
    # and tau are the central moments.
    #### MODEL PARAMETERS ####
    prior_m_ifr = pymc.Uniform('m_ifr', lower=.1, upper=10.)
    prior_s_ifr = pymc.Uniform(
        's_ifr', lower=5000., upper=10000000.)
    #### COVARIATES COEFFICIENTS ####
    prior_beta_pressure = pymc.Uniform(
        'beta_pressure', lower=-.5, upper=.5)
    prior_beta_mix_ratio = pymc.Uniform(
        'beta_mix_ratio', lower=-2., upper=5.)
    prior_beta_firing_time = pymc.Uniform(
        'beta_firing_times', lower=-.5, upper=.5)
    prior_beta_cycle = pymc.Uniform(
        'beta_cycle', lower=-50.0, upper=50.)
    prior_beta_p_mr = pymc.Uniform(
        'beta_p_mr', lower=-.05, upper=.05)
    prior_beta_p_ft = pymc.Uniform(
        'beta_p_ft', lower=-.05, upper=.05)
    prior_beta_mr_ft = pymc.Uniform(
        'beta_mr_ft', lower=-.02, upper=.02)
    prior_beta_p2 = pymc.Uniform(
        'beta_p2', lower=-.01, upper=.01)
    prior_beta_mr2 = pymc.Uniform(
        'beta_mr2', lower=-.05, upper=.05)
    prior_beta_ft2 = pymc.Uniform(
        'beta_ft2', lower=-.002, upper=.002)

    rm.priors = [prior_m_ifr, prior_s_ifr, prior_beta_pressure, prior_beta_mix_ratio,
                 prior_beta_firing_time, prior_beta_cycle, prior_beta_p_mr, prior_beta_p_ft, prior_beta_mr_ft,
                 prior_beta_p2, prior_beta_mr2, prior_beta_ft2]

    def likelihood(value=rm.inspected_cracks, beta=rm.priors):
        prior_m_ifr, prior_s_ifr, prior_beta_pressure, prior_beta_mix_ratio,\
            prior_beta_firing_time, prior_beta_cycle, prior_beta_p_mr, prior_beta_p_ft, prior_beta_mr_ft,\
            prior_beta_p2, prior_beta_mr2, prior_beta_ft2 = beta
        loglike = 0.0
        split_values = np.split(
            value, np.cumsum([len(interval) for interval in rm.inspected_cracks_list]))[:-1]
        for i in range(len(rm.campaign_data)):
            ith_value = split_values[i]
            ith_time_variable = np.hstack((1e-5, rm.cumulative_test_times[i]))
#             ith_covar_effect = np.exp(prior_beta_pressure * (rm.pressures[i] - prior_beta_offset_pressure) +
#                                       prior_beta_mix_ratio * (rm.mix_ratios[i] - prior_beta_offset_mix_ratio) +
#                                       prior_beta_firing_time * (rm.firing_times[i] - prior_beta_offset_firing_time) +
#                                       prior_beta_cycle * rm.cycles[i])
            ith_covar_effect = np.exp(prior_beta_pressure * rm.pressures[i] +
                                      prior_beta_mix_ratio * rm.mix_ratios[i] +
                                      prior_beta_firing_time * rm.firing_times[i] +
                                      prior_beta_cycle * rm.cycles[i] +
                                      prior_beta_p_mr * rm.pressures[i] * rm.mix_ratios[i] +
                                      prior_beta_p_ft * rm.pressures[i] * rm.firing_times[i] +
                                      prior_beta_mr_ft * rm.mix_ratios[i] * rm.firing_times[i] +
                                      prior_beta_p2 * rm.pressures[i]**2 +
                                      prior_beta_mr2 * rm.mix_ratios[i]**2 +
                                      prior_beta_ft2 * rm.firing_times[i]**2)

            # loglikelihood function for parameter inference - PH model
            if model == 'PH':
                baseline_hazard = (prior_m_ifr / prior_s_ifr) * (ith_time_variable /
                                                                 prior_s_ifr)**(prior_m_ifr - 1)  # \
#                     + (prior_m_wo / prior_s_wo) * \
#                     (ith_time_variable / prior_s_wo)**(prior_m_wo - 1)
                hazard = baseline_hazard * \
                    np.hstack((ith_covar_effect[0], ith_covar_effect))
                cumhazard = cumtrapz(hazard, ith_time_variable, initial=0.)
            # loglikelihood function for parameter inference - ATF model
            elif model == 'AFT':
                accel_time = np.hstack(
                    (0.0, np.cumsum(ith_covar_effect * np.diff(ith_time_variable))))
                cumhazard = (accel_time / prior_s_ifr)**prior_m_ifr
            survival = np.exp(-cumhazard)
            k = ith_value[-1]
            aux = 1e-10 * np.ones_like(rm.inspection_idxs[i])
            ith_loglike = (rm.n_channels - k) * (-cumhazard[-1]) + np.sum(
                np.diff(np.hstack((0.0, ith_value))) * np.log(aux - np.diff(np.hstack((1.0, survival[1:][rm.inspection_idxs[i]])))))
            if np.isinf(ith_loglike):
                ith_loglike = np.nan_to_num(ith_loglike) / 10.
            loglike += ith_loglike
        return loglike

    rm.log_likelihood = likelihood

    rm.plotting()

    ####################################
    #### REGRESSION POSTPROCESSING ####
    ####################################

    plt.figure()
    survival_probability_model_MCMC(
        rm=rm, model=model, process='regression')

    #################################
    ### PREDICTION AND VALIDATION ###
    #################################
    validation_test_campaign = data_reader(
        validation_campaign)
    rm.campaign_data = validation_test_campaign[0]
    rm.chamber_data = validation_test_campaign[1]

    return survival_probability_model_MCMC(
        rm=rm, model=model, process='validation')


### MAIN ###
if __name__ == '__main__':
    from random import shuffle

    TC_number = []
    for i in range(80):
        if i < 9:
            TC_number.append(str(0) + str(i + 1))
        else:
            TC_number.append(str(i + 1))

    TC_number = ['30']

    campaigns_number = []
    for j in range(1):
        if j < 9:  # and j > 6:
            campaigns_number.append(str(0) + str(j + 1))
        elif j == 10:
            campaigns_number.append(str(j + 1))
        else:
            pass

    campaigns_lst = []
    for tc in TC_number:
        for camp in campaigns_number:
            campaigns_lst.append('TC' + tc + '_campaign_' + camp)
    # shuffle(campaigns_lst)
    campaigns_lst = [
        'TC30_campaign_01', 'TC30_campaign_02', 'TC30_campaign_03', 'TC30_campaign_07', 'TC30_campaign_08', 'TC30_campaign_09']
    for i in [9]:
        print campaigns_lst[0:i + 1]
        ith_CDF_err, ith_scatter = run_example(
            model='PH', regression_campaigns=campaigns_lst[0:i + 1], validation_campaign=['TC30_campaign_10'])
    plt.show()
