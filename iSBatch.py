import numpy as np
import warnings
import scipy.integrate as integrate
import scipy.stats as st
from scipy.optimize import curve_fit
from collections import Counter
import sys
from enum import IntEnum
import warnings
import itertools

class CRStrategy(IntEnum):
    ''' Enumeration class to hold the types of Checkpoint/Restart
        strategies available to the application '''

    NeverCheckpoint = 0
    AlwaysCheckpoint = 1
    AdaptiveCheckpoint = 2


class LimitStrategy(IntEnum):
    ''' Enumeration class to hold the types of strategies for
        limiting the allowed number of submissions '''

    ThresholdBased = 0
    AverageBased = 1


class StaticCheckpointMemoryModel():
    ''' Default checkpoint model, defined by a static checkpoint/restart '''

    def __init__(self, checkpoint_cost=1, restart_cost=1):
        self.C = checkpoint_cost
        self.R = restart_cost

    def get_checkpoint_time(self, ts):
        return self.C

    def get_restart_time(self, ts):
        return self.R


class DynamicCheckpointMemoryModel():
    ''' Dynamic checkpoint model, defined by read/write bandwidths and Csize,
    an array (checkpoint_size, time) that defines the valability of each size.
    Example: [(c1,0), (c2,10)] means the application checkpoint size is c1 for
    the first 10 time units and c2 for the remaining time '''

    def __init__(self, checkpoint_size, write_bandwidth=1,
                 read_bandwidth=1):
        self.wbw = write_bandwidth
        self.rbw = read_bandwidth

        self.size = checkpoint_size
        # if the checkpoint size is a value create a list of one element
        if type(checkpoint_size) is int or  type(checkpoint_size) is float:
            self.size = [(0, checkpoint_size)]

        # check sequence correctness
        assert(self.size[0][0] == 0), "The C size needs to start from ts 0"
        assert(all(self.size[i][0] < self.size[i + 1][0] for i in range(
            len(self.size)-1))), "Incorrect ts in the checkpoint size sequence"
        assert(all(i[1] > 0 for i in self.size)), "Negative checkpoint size"

    def __get_size(self, ts):
        # find the last entry in the checkpoint_size list
        # that has the timestamp < the given ts
        C = next((i for i in reversed(self.size) if i[0] <= ts), None)
        return C[1]

    def get_checkpoint_time(self, ts):
        size = self.__get_size(ts)
        return self.wbw * size

    def get_restart_time(self, ts):
        size = self.__get_size(ts)
        return self.rbw * size


class ClusterCosts():
    ''' Class for storing the costs of running on the cluster
        For a job of actual length t, a reservation of lenth t1
        will cost alpha * t + beta * min(t, t1) + gamma '''

    def __init__(self, reservation_cost=1, utilization_cost=1, deploy_cost=0,
                 checkpoint_memory_model=None):
        # default pay what you reserve (AWS model) (alpha 1 beta 0 gamma 0)
        # pay what you use (HPC model) would be alpha 1 beta 1 gamma 0
        self.alpha = reservation_cost
        self.beta = utilization_cost
        self.gamma = deploy_cost
        self.checkpoint_memory_model = checkpoint_memory_model
        if checkpoint_memory_model is None:
            self.checkpoint_memory_model = StaticCheckpointMemoryModel()

    def get_checkpoint_time(self, ts):
        return self.checkpoint_memory_model.get_checkpoint_time(ts)

    def get_restart_time(self, ts):
        return self.checkpoint_memory_model.get_restart_time(ts)


class ResourceParameters():
    interpolation_model = None
    resource_discretization = -1
    verbose = False
    CR_strategy = CRStrategy.NeverCheckpoint
    request_upper_limit = None
    request_lower_limit = None
    request_increment_limit = 0
    submissions_limit = None
    submissions_limit_strategy = LimitStrategy.ThresholdBased

class ResourceEstimator():
    ''' Class used to generate the sequence of resource requests
        needed to be used for application submissions '''

    def __init__(self, past_runs, params=ResourceParameters()):
        # Set all initial data
        self.params = params
        self.fit_model = None
        self.discrete_data = None
        self.default_interpolation = True
        self.discretization = -1
        self.adjust_discrete_data = False
        assert (len(past_runs) > 0), "Invalid log provided"
        self.__set_workload(past_runs)

        if params.resource_discretization > 0:
            assert(params.resource_discretization > 2), \
                'The discretization needs at least 3 points'
            self.discretization = params.resource_discretization
            self.adjust_discrete_data = True

        if params.interpolation_model is not None:
            self.set_interpolation_model(params.interpolation_model)
            self.default_interpolation = False
        elif len(past_runs) < 100:
            if self.discretization == -1:
                self.discretization = 500
            self.set_interpolation_model(
                DistInterpolation(discretization=self.discretization))

        if self.discretization == -1:
            self.discretization = len(set(past_runs))

    ''' Private functions '''

    def __set_workload(self, past_runs):
        self.data = past_runs
        self.best_fit = None

    def __adjust_discrete_data(self, discrete_data, cdf):
        ''' Adjust the discrete_data / cdf according to the discretization '''

        if self.discretization == len(cdf):
            return (discrete_data, cdf)

        if self.discretization < len(cdf):
            idx = np.random.choice(np.arange(len(cdf)),
                                   self.discretization)
            newdata = [discrete_data[i] for i in idx]
            newcdf = [cdf[i] for i in idx]
            # the largest value needs to be included in the selected data
            if discrete_data[-1] not in newdata:
                newdata[-1] = discrete_data[-1]
                newcdf[-1] = cdf[-1]

        if self.discretization > len(cdf):
            idx = np.random.choice(np.arange(len(cdf) - 1),
                                   abs(self.discretization - len(cdf)))
            add_elements = Counter(idx)  # where add_elements[idx]=count
            newdata = discrete_data
            newcdf = cdf
            for idx in add_elements:
                step = (discrete_data[idx + 1] - discrete_data[idx]) /\
                       (add_elements[idx] + 1)
                newdata += [discrete_data[idx] + i * step
                            for i in range(add_elements[idx])]
                step = (cdf[idx + 1] - cdf[idx]) / (add_elements[idx] + 1)
                newcdf += [cdf[idx] + i * step
                           for i in range(add_elements[idx])]

        newdata = [x for _, x in sorted(zip(newcdf, newdata))]
        newcdf.sort()
        return (newdata, newcdf)

    def __compute_discrete_cdf(self):
        assert (self.data is not None),\
            'Data needs to be set to compute the discrete CDF'

        discrete_data = list(Counter(self.data).keys())
        cdf = list(Counter(self.data).values())
        cdf = [x for _, x in sorted(zip(discrete_data, cdf))]
        discrete_data.sort()
        cdf = [i * 1. / len(cdf) for i in cdf]
        for i in range(1, len(cdf)):
            cdf[i] += cdf[i-1]
        # normalize the cdf
        for i in range(len(cdf)):
            cdf[i] /= cdf[-1]

        # adjust the discrete data accordin to the discretization
        if self.adjust_discrete_data:
            discrete_data, cdf = self.__adjust_discrete_data(
                discrete_data, cdf)

        self.discrete_data = discrete_data
        self.cdf = cdf
        return discrete_data, cdf

    def __compute_best_fit(self):
        if self.fit_model is None:
            return -1

        # set dicrete data and cdf to the original ones
        ddata, dcdf = self.__compute_discrete_cdf()

        best_fit = self.fit_model[0].get_empty_fit()
        best_i = -1
        for i in range(len(self.fit_model)):
            fit = self.fit_model[i].get_best_fit(
                ddata, dcdf)
            if fit[2] < best_fit[2]:
                best_fit = fit
                best_i = i
        self.best_fit = best_fit
        self.best_fit_index = best_i
        return best_i

    def __get_limits(self):
        limits = []
        warnings.simplefilter("error", category=RuntimeWarning)
        if self.params.request_lower_limit != None:
            if self.params.request_lower_limit < 0:
                warnings.warn("Error ! Sequence limits cannot be negative",
                              RuntimeWarning, stacklevel=2)
            limits.append(self.params.request_lower_limit)
        if self.params.request_upper_limit != None:
            if self.params.request_upper_limit < 0:
                warnings.warn("Error ! Sequence limits cannot be negative",
                              RuntimeWarning, stacklevel=2)
            # if only the upper limit is set
            if len(limits) == 0:
                # add the min data as the lower limit
                limits.append(min(self.data))
            limits.append(self.params.request_upper_limit)
        return limits

    def __get_interpolation_cdf(self, all_data):
        if self.best_fit is None:
            self.__compute_best_fit()
        limits = self.__get_limits()
        self.discrete_data, self.cdf = self.fit_model[
            self.best_fit_index].get_discrete_cdf(all_data, self.best_fit)
        return self.discrete_data, self.cdf

    def __get_sequence_type(self):
        ''' Function returns sequence_type, parameters '''

        if self.params.CR_strategy == CRStrategy.AdaptiveCheckpoint:
            warnings.warn("Warning! The adaptive CR strategy has high " \
                          "complexity. Expect large run times.", stacklevel=2)
        if self.params.submissions_limit is not None:
            return LimitedSequence, (self.params.CR_strategy,
                                     self.params.submissions_limit_strategy,
                                     self.params.submissions_limit,
                                     self.params.request_increment_limit)
        if self.params.CR_strategy == CRStrategy.AdaptiveCheckpoint:
            return CheckpointSequence, (self.params.request_increment_limit, )
        if self.params.CR_strategy == CRStrategy.AlwaysCheckpoint:
            return AllCheckpointSequence, (self.params.request_increment_limit, )
        # by default return request times when checkpoint is not availabe
        return RequestSequence, (self.params.request_increment_limit, )

    def __trim_according_to_limits(self, data=[], cdf=[]):
        if len(data) == 0:
            data = self.discrete_data
            cdf = self.cdf
        idx = range(len(data))
        if self.params.request_upper_limit != None:
            idx = [i for i in idx if data[i] <=\
                   self.params.request_upper_limit]
        if self.params.request_lower_limit != None:
            idx = [i for i in idx if data[i] >=\
                   self.params.request_lower_limit]
        discrete_data = [data[i] for i in idx]
        cdf = [cdf[i] for i in idx]
        return discrete_data, cdf

    ''' Functions used for debuging or printing purposes '''
    
    # Function that returns the best fit
    def _get_best_fit(self):
        if self.best_fit is None:
            self.__compute_best_fit()
        return self.best_fit

    # Function that computes the cdf
    def _compute_cdf(self):
        # if all runs have the same execution time
        if all(elem == self.data[0] for elem in self.data):
            self.discrete_data = [self.data[0]]
            self.cdf = [1]
            return

        if self.fit_model is not None:
            self.__get_interpolation_cdf(self.data)
            valid = self._check_cdf_validity(self.cdf)
            if valid:
                return

        self.__compute_discrete_cdf()

    def _get_cdf(self):
        self._compute_cdf()
        return self.discrete_data, self.cdf

    # Function to check if the cdf is [0,1] and strictly increasing
    def _check_cdf_validity(self, cdf):
        test = all(elem >= 0 and elem <= 1 for elem in cdf)
        if not test:
            return False
        return all(cdf[i - 1] <= cdf[i] for i in range(1, len(cdf)))

    ''' Public functions '''

    def set_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model = [interpolation_model]
        else:
            self.fit_model = interpolation_model
        for model in self.fit_model:
            if model.discrete_steps < (self.discretization - 1):
                model.discrete_steps = self.discretization - 1
        self.best_fit = None
        if len(self.fit_model) == 0:
            self.fit_model = None
            return -1

    def set_CR_strategy(self, CR_strategy):
        self.params.CR_strategy = CR_strategy

    def compute_request_sequence(self, cluster_cost=None):
        if cluster_cost == None:
            cluster_cost = ClusterCosts()
        self._compute_cdf()
        sequence_type, params = self.__get_sequence_type()
        discrete_data, cdf = self.__trim_according_to_limits()
        if len(cdf) < 100:
            warnings.warn("Warning! Sequence is computed based on only" \
                          "%d elements. It is recommended to " \
                          "increase the discretization value." %(len(cdf)),
                          stacklevel=2)
        handler = sequence_type(discrete_data, cdf,
                                cluster_cost, params=params)
        sequence = handler.compute_request_sequence()
        # compute the expected average submissions
        self._avg_submissions = handler.check_avegage_submissions()
        return sequence

    def compute_sequence_cost(self, sequence, data, cluster_cost=None):
        if cluster_cost == None:
            cluster_cost = ClusterCosts()
        handler = LogDataCost(sequence)
        return (handler.compute_cost(data, cluster_cost),
                self._avg_submissions)

# -------------
# Classes for defining how the interpolation will be done
# -------------


class InterpolationModel():
    # define the format of the return values for the get_best_fit functions
    def get_empty_fit(self):
        return (-1, -1, np.inf)

    def discretize_data(self, data, discrete_steps):
        upper_limit = max(data)
        lower_limit = min(data)
        step = (upper_limit - lower_limit) / discrete_steps
        return np.unique(
            [lower_limit + i * step for i in range(discrete_steps)]
            + [upper_limit])


class FunctionInterpolation(InterpolationModel):
    # function could be any function that takes one parameter (e.g. log, sqrt)
    def __init__(self, function, order=1, discretization=500):
        self.fct = function
        self.order = order
        self.discrete_steps = discretization - 1

    def get_discrete_cdf(self, data, best_fit):
        all_data = self.discretize_data(data, self.discrete_steps)
        all_cdf = [max(0, min(1, np.polyval(best_fit, self.fct(d))))
                   for d in all_data]
        # make sure the cdf is always increasing
        for i in range(1, len(all_cdf)):
            if all_cdf[i] < all_cdf[i-1]:
                all_cdf[i] = all_cdf[i-1]
        return all_data, all_cdf

    # fitting the function a + b * fct
    def get_best_fit(self, x, y):
        try:
            params = np.polyfit(self.fct(x), y, self.order)
        except:
            return self.get_empty_fit()
        err = np.sum((np.polyval(params, self.fct(x)) - y)**2)
        return (self.order, params, err)

    def get_cdf(self, x, params):
        return np.polyval(params, self.fct(x))


class PolyInterpolation(InterpolationModel):

    def __init__(self, max_order=10, discretization=500):
        self.max_order = max_order
        self.discrete_steps = discretization - 1

    def get_discrete_cdf(self, data, best_fit):
        all_data = self.discretize_data(data, self.discrete_steps)
        all_cdf = [max(0, min(1, np.polyval(best_fit[1], d)))
                   for d in all_data]
        # make sure the cdf is always increasing
        for i in range(1, len(all_cdf)):
            if all_cdf[i] < all_cdf[i-1]:
                all_cdf[i] = all_cdf[i-1]
        return all_data, all_cdf

    def get_best_fit(self, x, y):
        empty = self.get_empty_fit()
        best_err = empty[2]
        best_params = empty[1]
        best_order = empty[0]
        for order in range(1, self.max_order):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    params = np.polyfit(x, y, order)
                except:
                    break

                err = np.sum((np.polyval(params, x) - y)**2)
                if err < best_err:
                    best_order = order
                    best_params = params
                    best_err = err

        return (best_order, best_params, best_err)


class DistInterpolation(InterpolationModel):
    def __init__(self, list_of_distr=[], discretization=500):
        self.distr = list_of_distr
        self.discrete_steps = discretization - 1

    def get_discrete_cdf(self, data, best_fit):
        arg = best_fit[1][:-2]
        loc = best_fit[1][-2]
        scale = best_fit[1][-1]
        all_data = self.discretize_data(data, self.discrete_steps)
        all_cdf = [best_fit[0].cdf(d, loc=loc, scale=scale, *arg)
                   for d in all_data]
        return all_data, all_cdf

    def get_best_fit(self, x, y):
        dist_list = self.distr
        if len(dist_list) == 0:
            # list of distributions to check
            dist_list = [
                st.alpha, st.beta, st.cosine, st.dgamma, st.dweibull, st.exponnorm,
                st.exponweib, st.exponpow, st.genpareto, st.gamma, st.halfnorm,
                st.invgauss, st.invweibull, st.laplace, st.loggamma, st.lognorm,
                st.lomax, st.maxwell, st.norm, st.pareto,  # st.pearson3,st.rice,
                st.truncexpon, st.truncnorm, st.uniform, st.weibull_min,
                st.weibull_max]

        # Best holders
        best_distribution = -1
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # estimate distribution parameters from data
        for distribution in dist_list:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(x)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse
            except Exception:
                pass

        return (best_distribution, best_params, best_sse)

# -------------
# Classes for computing the sequence of requests
# -------------


class DefaultRequests():
    ''' Default class for generating the sequence of requests given 
    an application behavior and system properties '''

    def __init__(self, discrete_values, cdf_values,
                 cluster_cost, params=0):
        self._alpha = cluster_cost.alpha
        self._beta = cluster_cost.beta
        self._gamma = cluster_cost.gamma
        self.makespan_increment_limit = params[0]

        assert (len(discrete_values) > 0), "Invalid input"
        assert (len(discrete_values) == len(cdf_values)), "Invalid cdf"

        self.discret_values = discrete_values
        self._cdf = cdf_values
        self.upper_limit = max(self.discret_values)
        self._E = {}
        self._request_sequence = []

        self._sumF = self.get_discrete_sum_F()
        self._sumFV = self.compute_FV()
        self._index_sequence = []

    def compute_F(self, vi):
        fi = self._cdf[vi]
        if vi > 0:
            fi -= self._cdf[vi-1]
        return fi / self._cdf[-1]

    def compute_FV(self):
        FV = 0
        for i in range(len(self.discret_values)):
            FV += (self.discret_values[i] * self.compute_F(i))
        return FV

    # Compute sumF[i] as sum_k=i,n f[k]
    def get_discrete_sum_F(self):
        sumF = (len(self.discret_values) + 1) * [0]
        for k in range(len(self.discret_values) - 1, -1, -1):
            sumF[k] = self.compute_F(k) + sumF[k + 1]
        return sumF

    def compute_E_value(self, i):
        if i in self._E:
            return self._E[i]
        E_val = self.compute_E_table(i)
        self._E[i] = E_val
        return E_val

    def update_best_makespan(self, makespan, min_makespan, j, next_j):
        # if the new makespan is infinity do not update
        if makespan == np.inf:
            return False
        # if the new makespan is better than the chosen one, update
        if min_makespan == -1 or min_makespan >= makespan:
            step = abs(self.discret_values[j] - self.discret_values[next_j])
            # if the increment exceeds the limit do not update
            if step < self.makespan_increment_limit:
                return False
            return True
        return False

    def check_avegage_submissions(self):
        if len(self._index_sequence) == 0:
            return -1
        # t the sequence of reservations, avg = 1 + P(X>t1) + P(X>t2) + ...
        avg = 1
        for j in self._index_sequence:
            avg += self._sumF[j + 1]
        return avg

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        ic = -1
        il = -1
        th = self.threshold
        idx = self._E_index[(ic, il)][th]
        E_val = self._E[(ic, il)][idx]
        already_compute = 0
        self._index_sequence = []
        while E_val[1] < len(self.discret_values) - 2:
            self._request_sequence.append(
                (self.discret_values[E_val[1]] - already_compute, E_val[2]))
            self._index_sequence.append(E_val[1])
            ic = (1 - E_val[2]) * ic + (E_val[1] + 1) * E_val[2]
            il = E_val[1] + 1
            if self.th_strategy == LimitStrategy.AverageBased:
                th = th - int(round(
                    self._sumF[il + 1] * self.th_precision))
            else:
                th -= 1
            if E_val[2] == 1:
                already_compute = self.discret_values[E_val[1]]
            idx = self._E_index[(ic, il)][th]
            E_val = self._E[(ic, il)][idx]

        self._request_sequence.append(
            (self.discret_values[E_val[1]] - already_compute, 0))
        self._index_sequence.append(E_val[1])
        return self._request_sequence


class RequestSequence(DefaultRequests):
    ''' Sequence that optimizes the total makespan of a job for discret
    values (instead of a continuous space) '''

    def __init__(self, discrete_values, cdf_values,
                 cluster_cost, params=[0]):

        super().__init__(discrete_values, cdf_values, cluster_cost,
                         params=params)
        E_val = self.compute_E_value(0)
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def makespan_init_value(self, i, j):
        init = float(self._alpha * self.discret_values[j] + self._gamma) \
            * self._sumF[i]
        init += self._beta * self.discret_values[j] * self._sumF[j + 1]
        return init

    def compute_E_table(self, first):
        self._E[len(self.discret_values)] = (
            self._beta * self._sumFV, len(self.discret_values) - 1, 0)
        for i in range(len(self.discret_values) - 1, first - 1, -1):
            min_makespan = -1
            min_request = -1
            for j in range(i, len(self.discret_values)):
                makespan = self.makespan_init_value(i, j)
                makespan += self._E[j + 1][0]
                next_j = self._E[j + 1][1]
                if self.update_best_makespan(makespan, min_makespan, i, next_j):
                    min_makespan = makespan
                    min_request = j
            self._E[i] = (min_makespan, min_request, 0)
        return self._E[first]

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        j = 0
        E_val = self.compute_E_value(j)
        while E_val[1] < len(self.discret_values) - 1:
            self._request_sequence.append(
                (self.discret_values[E_val[1]], E_val[2]))
            j = E_val[1] + 1
            E_val = self.compute_E_value(j)

        if self.discret_values[E_val[1]] != self.upper_limit:
            self._request_sequence.append(
                (self.discret_values[E_val[1]], E_val[2]))
            self._request_sequence.append((self.upper_limit, 0))
        else:
            self._request_sequence.append(
                (self.discret_values[E_val[1]], 0))

        return self._request_sequence


class CheckpointSequence(DefaultRequests):
    ''' Sequence that optimizes the total makespan of a job when the
    application or system is capable of taking checkpoints '''

    def __init__(self, discrete_values, cdf_values,
                 cluster_cost, params=[0]):

        super().__init__(discrete_values, cdf_values, cluster_cost,
                         params=params)
        self.CR = cluster_cost.checkpoint_memory_model
        E_val = self.compute_E_value((0, 0))
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def makespan_init_value(self, ic, il, j, delta, R):
        vic = self.discret_values[ic]
        if R == 0:
            vic = 0

        C = self.CR.get_checkpoint_time(self.discret_values[j])
        init = (self._alpha * (R + self.discret_values[j] - vic +
                               delta * C) + self._beta * R + self._gamma) \
            * self._sumF[il + 1]
        init += self._beta * ((1 - delta) * (self.discret_values[j] - vic)
                              + delta * C) * self._sumF[j + 1]
        return init

    def compute_E(self, ic, il, R):
        min_makespan = -1
        min_request = -1
        for j in range(il, len(self.discret_values) - 1):
            # makespan with checkpointing the last sequence (delta = 1)
            makespan = self.makespan_init_value(ic, il, j, 1, R)
            makespan += self._E[(j + 1, j + 1)][0]
            next_j = self._E[(j + 1, j + 1)][1]
            if self.update_best_makespan(makespan, min_makespan, il, next_j):
                min_makespan = makespan
                min_request = j
                min_delta = 1

            # makespan without checkpointing the last sequence (delta = 0)
            makespan = self.makespan_init_value(ic, il, j, 0, R)
            makespan += self._E[(ic, j + 1)][0]
            next_j = self._E[(ic, j + 1)][1]
            if self.update_best_makespan(makespan, min_makespan, il, next_j):
                min_makespan = makespan
                min_request = j
                min_delta = 0

        self._E[(ic, il)] = (min_makespan, min_request, min_delta)

    def compute_E_table(self, first):
        for ic in range(len(self.discret_values) - 1, -1, -1):
            self._E[(ic, len(self.discret_values) - 1)] = (
                self._beta * self._sumFV, len(self.discret_values) - 1, 0)

        for il in range(len(self.discret_values) - 2, -1, -1):
            for ic in range(il, 0, -1):
                if (ic, il) in self._E:
                    continue
                R = self.CR.get_restart_time(self.discret_values[il])
                self.compute_E(ic, il, R)
            self.compute_E(0, il, 0)

        return self._E[first]

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        ic = 0
        il = 0
        E_val = self._E[(ic, il)]
        already_compute = 0
        while E_val[1] < len(self.discret_values) - 1:
            if E_val[1] == -1:
                break
            self._request_sequence.append(
                (self.discret_values[E_val[1]] - already_compute, E_val[2]))
            ic = (1 - E_val[2]) * ic + (E_val[1] + 1) * E_val[2]
            il = E_val[1] + 1
            if E_val[2] == 1:
                already_compute = self.discret_values[E_val[1]]
            E_val = self._E[(ic, il)]

        self._request_sequence.append(
            (self.discret_values[E_val[1]] - already_compute, 0))
        return self._request_sequence


class AllCheckpointSequence(CheckpointSequence):
    ''' Sequence that optimizes the total makespan of a job when forcing
    the application or system to take checkpoints at the end of each
    reservation '''

    def compute_E(self, i, R):
        min_makespan = -1
        min_request = -1
        for j in range(i, len(self.discret_values)):
            makespan = self.makespan_init_value(i, i, j, 1, R)
            makespan += self._E[(j + 1, j + 1)][0]
            next_j = self._E[(j + 1, j + 1)][1]
            if self.update_best_makespan(makespan, min_makespan, i, next_j):
                min_makespan = makespan
                min_request = j

        self._E[(i, i)] = (min_makespan, min_request, 1)

    def compute_E_table(self, first):
        # the last reservation will not have to take a checkpoint
        self._E[(len(self.discret_values), len(self.discret_values))] = (
            self._beta * self._sumFV, len(self.discret_values) - 1, 0)

        for i in range(len(self.discret_values) - 1, 0, -1):
            if (i, i) in self._E:
                continue
            R = self.CR.get_restart_time(self.discret_values[i])
            self.compute_E(i, R)
        self.compute_E(0, 0)

        return self._E[first]

# -------------
# Classes for computing the sequence of requests
# when the number of submissions is limited
# -------------


class LimitedSequence(DefaultRequests):
    ''' Sequence that optimizes the total makespan of a job using a
    maxim number of submissions (the checkpoint strategy can be
    either one defined by the CRStrategy class). The limit strategy
    can either be ThBased or AvgBased '''

    def __init__(self, discrete_values, cdf_values,
                 cluster_cost, params=[]):

        super(LimitedSequence, self).__init__(
            discrete_values, cdf_values, cluster_cost,
            params=[params[3]])
        self._E_index = {}

        assert (len(params) > 3), "Not enough parameters provided"
        self.threshold = params[2]
        assert (self.threshold >= 1), "Invalid submission limit (< 1)"
        self.th_precision = 10
        self.th_strategy = params[1]
        self.CRstrategy = params[0]
        self.CR = cluster_cost.checkpoint_memory_model
        if self.threshold == 1:
            E_val = (1, len(self.discret_values) - 1, 0)
            self._E[(-1, -1)] = []
            self._E[(-1, -1)].append(E_val)
            self._E_index[(-1, -1)] = {1: 0}
        else:
            if self.th_strategy == LimitStrategy.ThresholdBased:
                self.threshold = int(np.floor(self.threshold))
                E_val = self.compute_E_threshold()
            else:
                self.threshold = int(
                    round(self.threshold * self.th_precision))
                E_val = self.compute_E_average()
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def makespan_with_checkpoint(self, ic, il, j, restart_cost):
        vic = self.discret_values[ic]
        if restart_cost == 0:
            vic = 0

        C = self.CR.get_checkpoint_time(self.discret_values[j])
        cost = self._alpha * (restart_cost + C + self.discret_values[j] - vic)
        cost = (cost + self._gamma) * self._sumF[il + 1]
        cost += self._beta * C * self._sumF[j + 1]
        return cost

    def makespan_no_checkpoint(self, ic, il, j, restart_cost):
        vic = self.discret_values[ic]
        if restart_cost == 0:
            vic = 0

        C = self.CR.get_checkpoint_time(self.discret_values[j])
        cost = self._alpha * (restart_cost + self.discret_values[j] - vic)
        cost = (cost + self._gamma) * self._sumF[il + 1]
        cost += self._beta * self._sumF[j + 1] * \
                (self.discret_values[j] - vic)
        return cost

    def add_element_in_E(self, reservation, val, k):
        idx = -1
        if reservation not in self._E:
            self._E[reservation] = []
            self._E_index[reservation] = {}
        if val not in self._E[reservation]:
            self._E[reservation].append(val)
            self._E_index[reservation][k] = len(self._E[reservation]) - 1
        else:
            idx = [i for i in range(len(self._E[reservation]))
                   if val == self._E[reservation][i]][0]
            self._E_index[reservation][k] = idx

    def compute_E(self, ic, il, R, k):
        min_makespan = np.inf
        min_request = -1
        min_delta = 0
        th_next = k - 1
        for j in range(il + 1, len(self.discret_values)):
            if self.th_strategy == LimitStrategy.AverageBased:
                th_next = k - int(round(
                    self._sumF[j + 1] * self.th_precision))
            # we cannot exceed the threshold number of submission
            if th_next < 0:
                continue
            # makespan with checkpointing the last sequence (delta = 1)
            if self.CRstrategy != CRStrategy.NeverCheckpoint:
                makespan = self.makespan_with_checkpoint(ic, il, j, R)
                idx = self._E_index[(j, j)][th_next]
                makespan += self._E[(j, j)][idx][0]
                next_j = self._E[(j, j)][idx][1]
                if self.update_best_makespan(
                    makespan, min_makespan, il, next_j):
                    min_makespan = makespan
                    min_request = j
                    min_delta = 1

            # makespan without checkpointing the last sequence (delta = 0)
            if self.CRstrategy != CRStrategy.AlwaysCheckpoint:
                makespan = self.makespan_no_checkpoint(ic, il, j, R)
                idx = self._E_index[(ic, j)][th_next]
                makespan += self._E[(ic, j)][idx][0]
                next_j = self._E[(ic, j)][idx][1]
                if self.update_best_makespan(
                    makespan, min_makespan, il, next_j):
                    min_makespan = makespan
                    min_request = j
                    min_delta = 0

        self.add_element_in_E(
            (ic, il), (min_makespan, min_request, min_delta), k)

    def initialize_threshold_E(self):
        th = self.threshold
        for k in range(max(0, th - len(self.discret_values)), th + 1):
            if self.CRstrategy == CRStrategy.AdaptiveCheckpoint:
                for ic in range(len(self.discret_values) - 1, -2, -1):
                    idx = (ic, len(self.discret_values) - 1)
                    self.add_element_in_E(
                        idx, (self._beta * self._sumFV,
                              len(self.discret_values) - 2, 0), k)
            if self.CRstrategy == CRStrategy.AlwaysCheckpoint:
                idx = (len(self.discret_values) - 1,
                       len(self.discret_values) - 1)
                self.add_element_in_E(
                    idx, (self._beta * self._sumFV,
                          len(self.discret_values) - 2, 0), k)
            if self.CRstrategy == CRStrategy.NeverCheckpoint:
                idx = (-1, len(self.discret_values) - 1)
                self.add_element_in_E(
                    idx, (self._beta * self._sumFV,
                          len(self.discret_values) - 2, 0), k)

    def compute_E_threshold(self):
        th = self.threshold
        self.initialize_threshold_E()
        for il in range(len(self.discret_values) - 2, -1, -1):
            R = self.CR.get_restart_time(self.discret_values[il])
            for k in range(max(0, th - il - 1), th + 1):
                if self.CRstrategy == CRStrategy.AdaptiveCheckpoint:
                    for ic in range(il, -1, -1):
                        if (ic, il) in self._E and k in self._E_index[(ic, il)]:
                            continue
                        self.compute_E(ic, il, R, k)
                if self.CRstrategy == CRStrategy.AlwaysCheckpoint:
                    if (il, il) in self._E and k in self._E_index[(il, il)]:
                        continue
                    self.compute_E(il, il, R, k)
                if self.CRstrategy != CRStrategy.AlwaysCheckpoint:
                    if (-1, il) in self._E and k in self._E_index[(-1, il)]:
                        continue
                    self.compute_E(-1, il, 0, k)

        self.compute_E(-1, -1, 0, th)
        idx = self._E_index[(-1, -1)][th]
        return self._E[(-1, -1)][idx]

    def initialize_average_E(self):
        th = self.threshold
        for il in range(len(self.discret_values) - 1, -1, -1):
            startk = int(round(self._sumF[il + 1] * th * \
                           (len(self.discret_values) - il)))
            for k in range(startk, th + 1):
                if self.CRstrategy == CRStrategy.AdaptiveCheckpoint:
                    for ic in range(il, -2, -1):
                        self.add_element_in_E(
                            (ic, il), (self._beta * self._sumFV,
                                       len(self.discret_values) - 2, 0), k)
                if self.CRstrategy == CRStrategy.AlwaysCheckpoint:
                    self.add_element_in_E(
                        (il, il), (self._beta * self._sumFV,
                                   len(self.discret_values) - 2, 0), k)
                if self.CRstrategy == CRStrategy.NeverCheckpoint:
                    self.add_element_in_E(
                        (-1, il), (self._beta * self._sumFV,
                                  len(self.discret_values) - 2, 0), k)

    def compute_E_average(self):
        th = self.threshold
        self.initialize_average_E()
        for il in range(len(self.discret_values) - 2, -1, -1):
            endk = min(th, int(round(self._sumF[il + 1] * th * \
                             (len(self.discret_values) - il)))) + 1
            R = self.CR.get_restart_time(self.discret_values[il])
            if self.CRstrategy == CRStrategy.AdaptiveCheckpoint:
                for ic in range(il, -1, -1):
                    for k in range(0, endk):
                        if (ic, il) in self._E and k in self._E_index[(ic, il)]:
                            continue
                        self.compute_E(ic, il, R, k)
            if self.CRstrategy == CRStrategy.AlwaysCheckpoint:
                for k in range(0, endk):
                    if (il, il) in self._E and k in self._E_index[(il, il)]:
                        continue
                    self.compute_E(il, il, R, k)
            if self.CRstrategy != CRStrategy.AlwaysCheckpoint:
                for k in range(0, endk):
                    if (-1, il) in self._E and k in self._E_index[(-1, il)]:
                        continue
                    self.compute_E(-1, il, 0, k)

        self.compute_E(-1, -1, 0, th)
        idx = self._E_index[(-1, -1)][th]
        return self._E[(-1, -1)][idx]


# -------------
# Classes for defining how the cost is computed
# -------------


class SequenceCost():
    def compute_cost(self, data):
        return -1


class LogDataCost(SequenceCost):

    def __init__(self, sequence):
        # Sequences need to use a multi information format
        # if not provided assume a never checkpoint model
        if not isinstance(sequence[0], tuple):
            self.sequence = [(i, 0) for i in sequence]
        else:
            self.sequence = sequence

    def __compute_instance_cost(self, time, cluster_cost):
        cost = 0
        compute_time = 0
        # cost of reservation: alpha * t + beta min(t, reservation) + gamma
        for reservation in self.sequence:
            cost += cluster_cost.alpha * reservation[0]
            cost += cluster_cost.beta * (
                min(time - compute_time, reservation[0]))
            cost += cluster_cost.gamma
            # stop when the reservation is bigger than the execution time
            if compute_time + reservation[0] >= time:
                break
            if reservation[1] == 1:
                compute_time += reservation[0]
        return cost

    def compute_cost(self, data, cluster_cost):
        cost = 0
        for instance in data:
            cost += self.__compute_instance_cost(instance, cluster_cost)
        return cost / len(data)
