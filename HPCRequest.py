import numpy as np
import warnings
import scipy.integrate as integrate
import scipy.stats as st
from scipy.optimize import curve_fit
import sys

class Workload():
    def __init__(self, data, cost_model=None, interpolation_model=None,
                 verbose=False):
        self.verbose = verbose
        self.best_fit = None
        self.fit_model = None
        self.discrete_data = None
        self.discrete_cdf = None
        if len(data) > 0:
            self.set_workload(data)
        if interpolation_model is not None:
            self.set_interpolation_model(interpolation_model)

    def set_workload(self, data):
        self.data = data
        self.compute_discrete_cdf()
        self.lower_limit = min(data)
        self.upper_limit = max(data)
        self.best_fit = None

    def set_interpolation_model(self, interpolation_model):
        if not isinstance(interpolation_model, list):
            self.fit_model = [interpolation_model]
        else:
            self.fit_model = interpolation_model
        self.best_fit = None
        best_fit = self.compute_best_cdf_fit()
        return best_fit
    
    # best_fit has the format returned by the best_fit functions in the
    # interpolation model: [distr, params] or [order, params], ['log', params]
    def set_best_fit(self, best_fit):
        self.best_fit = best_fit

    def get_best_fit(self):
        if self.best_fit is None:
            self.compute_best_fit()
        return self.best_fit

    def compute_discrete_cdf(self):
        assert (self.data is not None),\
            'Data needs to be set to compute the discrete CDF'

        if self.discrete_cdf is not None and self.discrete_data is not None:
            return self.discrete_data, self.discrete_cdf

        discret_data = sorted(self.data)
        cdf = [1 for _ in self.data]
        todel = []
        for i in range(len(self.data) - 1):
            if discret_data[i] == discret_data[i + 1]:
                todel.append(i)
                cdf[i + 1] += cdf[i]
        todel.sort(reverse=True)
        for i in todel:
            del discret_data[i]
            del cdf[i]
        cdf = [i * 1. / len(cdf) for i in cdf]
        for i in range(1, len(cdf)):
            cdf[i] += cdf[i-1]
        # normalize the cdf
        for i in range(len(cdf)):
            cdf[i] /= cdf[-1]

        self.discrete_data = discret_data
        self.discrete_cdf = cdf
        return discret_data, cdf

    def compute_best_cdf_fit(self):
        assert (len(self.fit_model)>0), "No fit models available"

        best_fit = self.fit_model[0].get_empty_fit()
        best_i = -1
        for i in range(len(self.fit_model)):
            fit = self.fit_model[i].get_best_fit(
                self.discrete_data, self.discrete_cdf)
            if fit[2] < best_fit[2]:
                best_fit = fit
                best_i = i
        self.best_fit = best_fit
        self.best_fit_index = best_i
        return best_i

    def get_interpolation_cdf(self, all_data, best_fit):
        if self.best_fit is None:
            self.compute_best_fit()
        self.discrete_data, self.discrete_cdf = self.fit_model[
            self.best_fit_index].get_discrete_cdf(all_data, best_fit)
        return self.discrete_data, self.discrete_cdf
    
    def compute_cdf(self, data=None, fit=None):
        if data is None:
            data = self.data
        if fit is None:
            fit = self.best_fit
        if self.fit_model is not None and len(self.fit_model)>0: 
            self.get_interpolation_cdf(data, fit)
        else:
            self.compute_discrete_cdf()
        return self.discrete_data, self.discrete_cdf

    def compute_request_sequence(self, max_request=-1):
        self.compute_cdf()
        if max_request == -1:
            max_request = max(self.discrete_data)
        handler = RequestSequence(max_request, self.discrete_data,
                                  self.discrete_cdf)
        return handler.compute_request_sequence()

    def compute_sequence_cost(self, sequence, data):
        handler = LogDataCost(sequence)
        return handler.compute_cost(data)

#-------------
# Classes for defining how the interpolation will be done
#-------------

class InterpolationModel():
    # define the format of the return values for the get_best_fit functions
    def get_empty_fit(self):
        return (-1, -1, np.inf)


class FunctionInterpolation(InterpolationModel):
    # function could be any function that takes one parameter (e.g. log, sqrt)
    def __init__(self, function, order=1):
        self.fct = function
        self.order = order

    def get_discrete_cdf(self, data, best_fit):
        all_data = np.unique(data)
        all_cdf = [max(0, min(1, np.polyval(best_fit, self.fct(d)))) for d in all_data]
        # make sure the cdf is always increasing
        for i in range(1,len(all_cdf)):
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

    def __init__(self, max_order=10):
        self.max_order = max_order

    def get_discrete_cdf(self, data, best_fit):
        all_data = np.unique(data)
        all_cdf = [max(0, min(1, np.polyval(best_fit, d))) for d in all_data]
        # make sure the cdf is always increasing
        for i in range(1,len(all_cdf)):
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
    def __init__(self, data, list_of_distr=[]):
        self.distr = list_of_distr
        self.data = data
    
    def get_discrete_cdf(self, data, best_fit):
        arg = best_fit[1][:-2]
        loc = best_fit[1][-2]
        scale = best_fit[1][-1]
        all_data = np.unique(data)
        all_cdf = [best_fit[0].cdf(d, loc=loc, scale=scale, *arg) for d in all_data]
        return all_data, all_cdf

    def get_best_fit(self, x, y): 
        dist_list = self.distr
        if len(dist_list)==0:
            # list of distributions to check
            dist_list = [        
                st.alpha,st.beta,st.cosine,st.dgamma,st.dweibull,st.exponnorm,
                st.exponweib,st.exponpow,st.genpareto,st.gamma,st.halfnorm,
                st.invgauss,st.invweibull,st.laplace,st.loggamma,st.lognorm,
                st.lomax,st.maxwell,st.norm,st.pareto,st.pearson3,st.rayleigh,
                st.rice,st.truncexpon,st.truncnorm,st.uniform,st.weibull_min,
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
                    params = distribution.fit(self.data)

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

#-------------
# Classes for computing the sequence of requests
#-------------

class RequestSequence():
    ''' Sequence that optimizes the total makespan of a job for discret
    values (instead of a continuous space) '''

    def __init__(self, max_value, discrete_values, probability_values):
        self.discret_values = discrete_values
        self.__prob = probability_values
        self.upper_limit = max_value
        self._E = {}
        self._request_sequence = []
        
        self.__sumF = self.get_discret_sum_F()
        E_val = self.compute_E_value(0)
        self.__t1 = self.discret_values[E_val[1]]
        self.__makespan = E_val[0]

    def compute_F(self, vi):
        fi = self.__prob[vi]
        if vi > 0:
            fi -= self.__prob[vi-1]
        return fi / self.__prob[-1]

    def get_discret_sum_F(self):
        sumF = (len(self.discret_values) + 1) * [0]
        for k in range(len(self.discret_values) - 1, -1, -1):
            sumF[k] = self.compute_F(k) + sumF[k + 1]
        return sumF

    def __compute_E_table(self, i):
        if i == len(self.discret_values):
            return (0, len(self.discret_values))

        min_makespan = -1
        min_request = -1
        for j in range(i, len(self.discret_values)):
            makespan = float(self.__sumF[i] * self.discret_values[j])
            if j + 1 in self._E:
                makespan += self._E[j + 1][0]
            else:
                E_val = self.__compute_E_table(j + 1)
                makespan += E_val[0]
                self._E[j + 1] = E_val

            if min_request == -1 or min_makespan > makespan:
                min_makespan = makespan
                min_request = j
        return (min_makespan, min_request)

    def __compute_E_table_iter(self, first):
        self._E[len(self.discret_values)] = (0, len(self.discret_values))
        for i in range(len(self.discret_values) - 1, first - 1, -1):
            if i in self._E:
                continue
            min_makespan = 0
            min_request = len(self.discret_values)
            for j in range(i, len(self.discret_values)):
                makespan = float(self.__sumF[i] * self.discret_values[j])
                makespan += self._E[j + 1][0]

                if min_makespan == 0 or min_makespan >= makespan:
                    min_makespan = makespan
                    min_request = j
            self._E[i] = (min_makespan, min_request)
        return self._E[first]

    def compute_request_sequence(self):
        if len(self._request_sequence) > 0:
            return self._request_sequence
        j = 0
        E_val = self.compute_E_value(j)
        while E_val[1] < len(self.discret_values):
            self._request_sequence.append((self.discret_values[E_val[1]], ))
            j = E_val[1] + 1
            E_val = self.compute_E_value(j)

        if self._request_sequence[-1][0] != self.upper_limit:
            self._request_sequence.append((self.upper_limit, ))
        
        return self._request_sequence

    def compute_E_value(self, i):
        if i in self._E:
            return self._E[i]
        if len(self.discret_values)<600:
            E_val = self.__compute_E_table(i)
        else:
            E_val = self.__compute_E_table_iter(i)
        self._E[i] = E_val
        return E_val

#-------------
# Classes for defining how the cost is computed
#-------------

class SequenceCost():
    def compute_cost(self, data):
        return -1

class LogDataCost(SequenceCost):

    def __init__(self, sequence):
        # if entries in the sequence use a multi information format
        # extract only the execution time
        if not isinstance(sequence[0], tuple):
            self.sequence = sequence
        else:
            self.sequence = [i[0] for i in sequence]

    def compute_cost(self, data):
        cost = 0
        for instance in data:
            # get the sum of all the values in the sequences <= walltime
            cost += sum([i for i in self.sequence if i < instance])
            # add the first reservation that is >= current walltime
            idx = 0
            if len(self.sequence) > 1:
                idx_list = [i for i in range(1,len(self.sequence)) if
                            self.sequence[i-1] < instance and
                            self.sequence[i] >= instance]
                if len(idx_list) > 0:
                    idx = idx_list[0]
            cost += self.sequence[idx]
        cost = cost / len(data)
        return cost