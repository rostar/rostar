from traits.api import HasTraits, Array, Int, \
    Property, cached_property, Function, List, \
    Instance, Tuple
import pymc
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
import os

def data_reader(campaigns, replicates=False):
    data_pack_dir = '/Users/rostislavrypl/git/rostar/survival_probability/data_pack/'
    if replicates == True:
        # loading the output file as a numpy array
        cracks = np.zeros((17, 80))
        for i in range(80):
            if i < 9:
                i = str(0) + str(i + 1)
            else:
                i = str(i + 1)
            campaign_data = np.loadtxt(
                data_pack_dir + 'rep80xTC30/' + 'TC30_rep' + i + '_campaign_10.txt', skiprows=8)
            cracks[:, int(i) - 1] = campaign_data[:, 4]

        chamber_data = np.loadtxt(
            data_pack_dir + 'rep80xTC30/TC30_rep01.txt', skiprows=13)
        return [campaign_data], [chamber_data], cracks

    else:
        campaign_data_lst = []
        chamber_data_lst = []
        for campaign in campaigns:
            # loading the output file as a numpy array
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
        firing_times_lst = [campaign[:, 1] for campaign in self.campaign_data]
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

# global chamber parameters
    Rm = Property(List(Array), depends_on='chamber_data')

    @cached_property
    def _get_Rm(self):
        return [chamber[0] for chamber in self.chamber_data]

    sRm = Property(List(Array), depends_on='chamber_data')

    @cached_property
    def _get_sRm(self):
        return [chamber[1] for chamber in self.chamber_data]

    A = Property(List(Array), depends_on='chamber_data')

    @cached_property
    def _get_A(self):
        return [chamber[2] for chamber in self.chamber_data]

    sA = Property(List(Array), depends_on='chamber_data')

    @cached_property
    def _get_sA(self):
        return [chamber[3] for chamber in self.chamber_data]

    CD = Property(List(Array), depends_on='chamber_data')

    @cached_property
    def _get_CD(self):
        return [chamber[4] for chamber in self.chamber_data]

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


def survival_probability_model_MCMC(rm, model, process, statistics=None, no_of_regression_campaigns=None):
    '''postprocessor for MCMC samples
    rm: instance of a RegressionModel class
    model: string 'PH' or 'AFT'
    process: string 'regression' or 'validation' - affects the plotting '''

    N_samples = len(rm.samples.trace('beta_pressure')[:])

    m_ifr_samples = rm.samples.trace('m_ifr')[:][np.newaxis, :]
    s_ifr_samples = rm.samples.trace('s_ifr')[:][np.newaxis, :]
    m_wo_samples = rm.samples.trace('m_wo')[:][np.newaxis, :]
    s_wo_samples = rm.samples.trace('s_wo')[:][np.newaxis, :]
    p_coeff_samples = rm.samples.trace('beta_pressure')[:][np.newaxis, :]
    mr_coeff_samples = rm.samples.trace('beta_mix_ratio')[:][np.newaxis, :]
    ft_coeff_samples = rm.samples.trace('beta_firing_times')[:][np.newaxis, :]
    cycle_coeff_samples = rm.samples.trace('beta_cycle')[:][np.newaxis, :]
    Rm_coeff_samples = rm.samples.trace('beta_Rm')[:][np.newaxis, :]
    sRm_coeff_samples = rm.samples.trace('beta_sRm')[:][np.newaxis, :]
    A_coeff_samples = rm.samples.trace('beta_A')[:][np.newaxis, :]
    sA_coeff_samples = rm.samples.trace('beta_sA')[:][np.newaxis, :]
    CD_coeff_samples = rm.samples.trace('beta_CD')[:][np.newaxis, :]

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
        jth_covar_effect = np.exp(p_coeff_samples * jth_pressure[:, np.newaxis] +
                                  mr_coeff_samples * jth_mix_ratio[:, np.newaxis] +
                                  ft_coeff_samples * jth_firing_times[:, np.newaxis] +
                                  cycle_coeff_samples * jth_cycle[:, np.newaxis] +
                                  Rm_coeff_samples * rm.Rm[j] + sRm_coeff_samples * rm.sRm[j] +
                                  A_coeff_samples * rm.A[j] + sA_coeff_samples * rm.sA[j] +
                                  CD_coeff_samples * rm.CD[j]
                                  )
        jth_time_variable = np.hstack(
            (1e-5, rm.cumulative_test_times[j]))[:, np.newaxis]

        if model == 'PH':
            baseline_hazard = m_ifr_samples / s_ifr_samples * (jth_time_variable /
                                                               s_ifr_samples) ** (m_ifr_samples - 1) + \
                            m_wo_samples / s_wo_samples * (jth_time_variable /
                                                               s_wo_samples) ** (m_wo_samples - 1)
            hazard = baseline_hazard * jth_covar_effect
            cumhazard = cumtrapz(
                hazard, jth_time_variable, initial=0.0, axis=0)

        if model == 'AFT':
            time_variable_aux = np.hstack(
                (1e-5, 1e-5, rm.cumulative_test_times[j]))[:, np.newaxis]
            accel_time = np.cumsum(
                jth_covar_effect * np.diff(time_variable_aux, axis=0), axis=0)
            cumhazard = (accel_time / s_ifr_samples)**m_ifr_samples + (accel_time / s_wo_samples)**m_wo_samples

        survival = np.exp(-cumhazard)

        # Monte Carlo mean
        CDF_mean = np.sum(1 - survival, axis=1) / N_samples
        
        lower_idx = np.rint(N_samples * 0.05)
        upper_idx = np.rint(N_samples * 0.95)
        survival_sorted = np.sort(survival)
        fract5 = 1 - survival_sorted[:, lower_idx]
        fract95 = 1 - survival_sorted[:, upper_idx]


    if process == 'validation':
        plt.figure()
        if statistics is not None:
            for g in range(statistics.shape[1]):
                if g == 0:
                    plt.plot(rm.inspection_times[j], statistics[:, g] / rm.n_channels, marker='.', color='grey', lw=0, label='individual experiments')
                else:
                    plt.plot(rm.inspection_times[j], statistics[:, g] / rm.n_channels, marker='.', color='grey', lw=0,)
            plt.plot(rm.inspection_times[j], np.mean(statistics, axis=1) / rm.n_channels, color='black', marker = 'o', lw=0, label='mean experiment data')
                
            plt.plot(jth_time_variable, 1 - survival_sorted[:, lower_idx], color='red', lw=2, ls='dashed')
            plt.plot(jth_time_variable, 1 - survival_sorted[:, upper_idx], color='red', lw=2, ls='dashed', label = '0.05 and 0.95 fractiles')
        plt.plot(jth_time_variable, CDF_mean, color='red', lw=2, label='prediction')
        plt.xlabel('time [s]')
        plt.ylabel('fraction of failed channels')
        plt.title(str(no_of_regression_campaigns) + ' regression campaigns')
        plt.legend(loc='best')
        plt.draw()
   
    sq_err_CDF = np.sum((CDF_mean[rm.inspection_idxs] - np.mean(statistics, axis=1)/rm.n_channels)**2)
    sq_err_scatter = np.sum((CDF_mean - fract5)**2 + (CDF_mean - fract95)**2)
    return sq_err_CDF, sq_err_scatter, rm

def run_example(model, regression_campaigns, validation_campaign):
    # generate data
    #generated_data = data_generator(regression_campaigns)
    campaign_data, chamber_data = data_reader(regression_campaigns)
    n_channels = 300
    rm = RegressionModel(campaign_data=campaign_data,
                         chamber_data=chamber_data,
                         n_channels=n_channels,
                         mcmc_iterations=30000,
                         mcmc_burn_in=25000,
                         mcmc_thinning=10,
                         )

    # Define prior distribution instances from the pymc module.
    # e.g. pymc.Uniform('distribution_name', lower=Float, upper=Float)
    # or pymc.Normal('distribution_name', mu=Float, tau=Float), where mu
    # and tau are the central moments.
    #### MODEL PARAMETERS ####
    prior_m_ifr = pymc.Uniform('m_ifr', lower=.1, upper=10.)
    prior_s_ifr = pymc.Uniform(
        's_ifr', lower=100., upper=10000000.)
    prior_m_wo = pymc.Uniform('m_wo', lower=10., upper=100.)
    prior_s_wo = pymc.Uniform(
        's_wo', lower=100., upper=10000000.)
    #### COVARIATES COEFFICIENTS ####
    prior_beta_pressure = pymc.Uniform(
        'beta_pressure', lower=0.0, upper=.5)
    prior_beta_mix_ratio = pymc.Uniform(
        'beta_mix_ratio', lower=0.0, upper=5.)
    prior_beta_firing_time = pymc.Uniform(
        'beta_firing_times', lower=0.0, upper=.5)
    prior_beta_cycle = pymc.Uniform(
        'beta_cycle', lower=0.0, upper=10.)
    prior_beta_Rm = pymc.Uniform('beta_Rm', lower=-15., upper=15.)
    prior_beta_sRm = pymc.Uniform('beta_sRm', lower=-30., upper=30.)
    prior_beta_A = pymc.Uniform('beta_A', lower=-15., upper=0.5)
    prior_beta_sA = pymc.Uniform('beta_sA', lower=-0.5, upper=100.)
    prior_beta_CD = pymc.Uniform('beta_CD', lower=-.001, upper=.001)
    rm.priors = [prior_m_ifr, prior_s_ifr, prior_m_wo, prior_s_wo, prior_beta_pressure, prior_beta_mix_ratio,
                 prior_beta_firing_time, prior_beta_cycle, prior_beta_Rm, prior_beta_sRm,
                 prior_beta_A, prior_beta_sA, prior_beta_CD]

    def likelihood(value=rm.inspected_cracks, beta=rm.priors):
        prior_m_ifr, prior_s_ifr, prior_m_wo, prior_s_wo, prior_beta_pressure, prior_beta_mix_ratio,\
            prior_beta_firing_time, prior_beta_cycle, prior_beta_Rm, prior_beta_sRm,\
            prior_beta_A, prior_beta_sA, prior_beta_CD = beta
        loglike = 0.0
        for i in range(len(rm.campaign_data)):
            ith_value = rm.inspected_cracks_list[i]
            ith_time_variable = np.hstack((1e-5, rm.cumulative_test_times[i]))
            ith_covar_effect = np.exp(prior_beta_pressure * rm.pressures[i] +
                                      prior_beta_mix_ratio * rm.mix_ratios[i] +
                                      prior_beta_firing_time * rm.firing_times[i] +
                                      prior_beta_cycle * rm.cycles[i] +
                                      prior_beta_Rm * rm.Rm[i] + prior_beta_sRm * rm.sRm[i] +
                                      prior_beta_A * rm.A[i] + prior_beta_sA * rm.sA[i] +
                                      prior_beta_CD * rm.CD[i])

            # loglikelihood function for parameter inference - PH model
            if model == 'PH':
                baseline_hazard = (prior_m_ifr / prior_s_ifr) * (ith_time_variable /
                                prior_s_ifr)**(prior_m_ifr - 1) + (prior_m_wo / prior_s_wo) * (ith_time_variable / prior_s_wo)**(prior_m_wo - 1)
                hazard = baseline_hazard * \
                    np.hstack((ith_covar_effect[0], ith_covar_effect))
                cumhazard = cumtrapz(hazard, ith_time_variable, initial=0.)
            # loglikelihood function for parameter inference - ATF model
            elif model == 'AFT':
                accel_time = np.hstack(
                    (0.0, np.cumsum(ith_covar_effect * np.diff(ith_time_variable))))
                cumhazard = (accel_time / prior_s_ifr)**prior_m_ifr + (accel_time / prior_s_wo)**prior_m_wo
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

    print 'mean_m_ifr = ', np.mean(rm.samples.trace('m_ifr')[:])
    print 'mean_s_ifr = ', np.mean(rm.samples.trace('s_ifr')[:])
    print 'mean_m_wo = ', np.mean(rm.samples.trace('m_wo')[:])
    print 'mean_s_wo = ', np.mean(rm.samples.trace('s_wo')[:])
    print 'mean_mix_ratio_coeff = ', np.mean(rm.samples.trace('beta_mix_ratio')[:])
    print 'mean_pressure_coeff = ', np.mean(rm.samples.trace('beta_pressure')[:])
    print 'mean_firing_times_coeff = ', np.mean(rm.samples.trace('beta_firing_times')[:])
    print 'mean_cycles_coeff = ', np.mean(rm.samples.trace('beta_cycle')[:])
    #rm.plotting()

    ####################################
    #### REGRESSION POSTPROCESSING ####
    ####################################

    #plt.figure()
    #survival_probability_model_MCMC(
    #    rm=rm, model=model, process='regression')

    #################################
    ### PREDICTION AND VALIDATION ###
    #################################
    validation_test_campaign = data_reader(
        validation_campaign, replicates=True)
    
    rm.campaign_data = validation_test_campaign[0]
    rm.chamber_data = validation_test_campaign[1]

    return survival_probability_model_MCMC(
        rm=rm, model=model, process='validation', statistics=validation_test_campaign[2], no_of_regression_campaigns=len(regression_campaigns)-1)
  

### MAIN ###
if __name__ == '__main__':
    from random import shuffle
    #from random import seed
    #seed(True)
    
    TC_number = []
    for i in range(80):
        if i<9:
            TC_number.append(str(0) + str(i+1))
        else:
            TC_number.append(str(i+1))

    campaigns_number = []
    for j in range(10):
        if j<9 and j>6:
            campaigns_number.append(str(0) + str(j+1))
        elif j==10:
            campaigns_number.append(str(j+1))

    campaigns_lst = []
    for tc in TC_number:
        for camp in campaigns_number:
            campaigns_lst.append('TC'+tc+'_campaign_'+camp)
    shuffle(campaigns_lst)
    
    CDF_err = []
    scatter = []
    
    m_ifr_lst = []
    s_ifr_lst = []
    m_wo_lst = []
    s_wo_lst = []
    beta_pressure_lst = []
    beta_mix_ratio_lst = []
    beta_firing_time_lst = []
    beta_cycle_lst = []
    beta_Rm_lst = []
    beta_sRm_lst = []
    beta_A_lst = []
    beta_sA_lst = []
    beta_CD_lst = []
    
    regression_campaigns = [1,2,3,10,30]
    for i in regression_campaigns:
        ith_CDF_err, ith_scatter, rm = run_example(
        model='PH', regression_campaigns=campaigns_lst[0:i+1],
        validation_campaign=['TC30_campaign_10'])
        CDF_err.append(ith_CDF_err)
        scatter.append(ith_scatter)
        samples = rm.samples
        m_ifr_lst.append(samples.trace('m_ifr')[:])
        s_ifr_lst.append(samples.trace('s_ifr')[:])
        m_wo_lst.append(samples.trace('m_wo')[:])
        s_wo_lst.append(samples.trace('s_wo')[:])
        beta_pressure_lst.append(samples.trace('beta_pressure')[:])
        beta_mix_ratio_lst.append(samples.trace('beta_mix_ratio')[:])
        beta_firing_time_lst.append(samples.trace('beta_firing_times')[:])
        beta_cycle_lst.append(samples.trace('beta_cycle')[:])
        beta_Rm_lst.append(samples.trace('beta_Rm')[:])
        beta_sRm_lst.append(samples.trace('beta_sRm')[:])
        beta_A_lst.append(samples.trace('beta_A')[:])
        beta_sA_lst.append(samples.trace('beta_sA')[:])
        beta_CD_lst.append(samples.trace('beta_CD')[:])

    plt.figure()
    plt.plot(regression_campaigns, np.array(CDF_err)/np.max(np.array(CDF_err)),label='CDF error',lw=2,color='black')
    plt.plot(regression_campaigns, np.array(scatter)/np.max(np.array(scatter)),label='scatter',lw=2,color='black', ls='dashed')
    plt.xlabel('number of regression campaigns')
    plt.ylabel('normalized CDF_error and scatter')
    plt.legend(loc='best')
    plt.title('learning by campaigns 8,9,10')
    
    coefficients_lst = [m_ifr_lst, s_ifr_lst, m_wo_lst, s_wo_lst, beta_pressure_lst, beta_mix_ratio_lst, beta_firing_time_lst,
                        beta_cycle_lst, beta_Rm_lst, beta_sRm_lst, beta_A_lst, beta_sA_lst, beta_CD_lst]
    #initial values
    plt.figure()
    plot_container = []
    initial_var = []
    for prior in rm.priors:
        upper = prior._parents['upper']
        lower = prior._parents['lower']
        var = (upper - lower)**2/12.
        initial_var.append(var)
    initial_var = np.array(initial_var)
    plot_container.append(initial_var)

    for i in range(len(regression_campaigns)):
        var = []
        for coefficient in coefficients_lst:
            ith_trace = coefficient[i]
            var.append(np.var(ith_trace))
        var = np.array(var)/initial_var
        plot_container.append(var)

    plot_container[0] = np.ones_like(initial_var)
    plt.plot([0] + regression_campaigns, plot_container)
    plt.legend([prior.__name__ for prior in rm.priors])
    plt.title('standardized variance of coefficients')
    plt.xlabel('number of campaigns')
    plt.ylabel('standardized variance')
    plt.show()