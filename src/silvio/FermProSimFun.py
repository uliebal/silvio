# Fermentation Process Simulator
# noinspection PySingleQuotedDocstring
import os
from math import ceil
from random import uniform
import matplotlib.pyplot as plt
from datetime import datetime
import json

import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.optimize import Bounds, minimize


class MonodModel:
    '''
    The 'Monod_Model' class stores all information about the bioprocess model its properties.
        Attributes:
            OperationMode: str, Mode of operation ('batch', 'fedbatch' or 'continuous')
            Params: dict, contains model parameters for monod-like kinetics, e.g. µmax
            Conditions: dict, contains starting conditions for simulated fermentation, like [Substrate]
    '''
    # Class attributes
    __Organism = 'E. coli'                          # Alternatively: 'Pput'
    __OperationMode = 'batch'                       # Alternatively: 'fedbatch', 'continuous'
    __Results = dict()

    def __init__(self):
        # Instance attributes
        self.__Params = dict(
                {
                        'u0':       0,                            # Initial growth rate [h^-1]
                        'umax':     round(uniform(0.5, 1.1), 3),  # maximal growth rate [h^-1] (default 0.5 - 1.1)
                        'duration': 24,                           # Process Duration [h] as Integer
                        'Ks':       round(uniform(7, 10), 3),
                        # Monod substrate affinity constant (default 7 - 10) [g/L]
                        'Yx':       round(uniform(0.4, 0.6), 3),
                        # Yield coefficient for growth on glucose (default 0.4 - 0.6) [g/g]
                        'k1':       round(uniform(0.05, 0.2), 3),
                        # Production rate of Product (default 0.05 - 0.2) [# h^-1]

                        # sources: https://bionumbers.hms.harvard.edu/bionumber.aspx?id=105318,
                        # https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=3&id=111049
                })
        self.__hiddenParams = dict(
                {
                        'u0':       0,                            # Initial growth rate [h^-1]
                        'umax':     round(uniform(0.5, 1.1), 3),  # maximal growth rate [h^-1] (default 0.5 - 1.1)
                        'duration': 24,                           # Process Duration [h] as Integer
                        'Ks':       round(uniform(7, 10), 3),
                        'Yx':       round(uniform(0.4, 0.6), 3),
                        'k1':       round(uniform(0.05, 0.2), 3),
                })
        self.__Conditions = dict(
                {
                        'S0': round(uniform(19, 21), 3),  # Initial substrate concentration [g/L]
                        'P0': 0,  # Initial product concentration [g/L]
                        'X0': round(uniform(0.05, 0.3), 3)  # Initial biomass concentration [g/L]
                })
        self.var_ModelingCount = 0
        self.var_ExpCount = 0
        self.__optimal_S = round(uniform(15, 30), 3)
        self.var_Organism = self.__Organism
        self.var_OperationMode = self.__OperationMode
        self.var_Params = self.__Params
        self.var_Conditions = self.__Conditions
        self.__TimeCreated = datetime.now().strftime('%d_%m_%Y_%I_%M_%S_%f')
        self.__ModelName = f'Monod_Model_{self.__TimeCreated}'
        self.__Description = f'Monod_Model instance for a {self.var_Organism}-{self.var_OperationMode} microbial ' \
                             f'production process. \nCurrent parameters: {self.var_Params}'

    def __str__(self):
        '''
        Use print(Monod_Model) for a quick description of the model instance.
        '''
        return self.__Description

    # Instance methods
    def get_start_params(self, hidden_params=False):
        '''
        This function calculates and returns the initial numeric derivatives of process parameters.
            Output:
                start_params: dict, contains initial process derivatives

        Test for complete set of params
        >>> TestModelBatch = MonodModel()
        >>> len(TestModelBatch.get_start_params()) == 3
        True

        Test for all fedbatch params
        >>> TestModel_Fedbatch = Monod_Model()
        >>> TestModel_Fedbatch.var_OperationMode = 'fedbatch'
        >>> len(TestModelBatch.get_start_params()) == 4
        True

        Test for correct sign of initial derivatives
        >>> init_params = TestModelBatch.get_start_params()
        >>> init_params['rX0'] >= 0 and init_params['rS0'] <= 0 and init_params['rP0'] >= 0
        True
        '''
        if hidden_params:
            u0, Yx, k1 = self.__hiddenParams['u0'], self.__hiddenParams['Yx'], self.__hiddenParams['k1']
        else:
            u0, Yx, k1 = self.var_Params['u0'], self.var_Params['Yx'], self.var_Params['k1']
        X0, S0 = self.var_Conditions['X0'], self.var_Conditions['S0']

        if self.var_OperationMode == 'batch':
            rX0 = u0 * X0                       # Initial
            rS0 = -(rX0 / (Yx / S0))            # Startwert Änderungsrate S
            rP0 = (k1 * u0) * X0                # Startwert Änderungsrate P
        else:
            rX0 = 0
            rS0 = 0
            rP0 = 0                             # TODO: Fedbatch & Konti Ergänzen!

        start_params = {
                'rX0': rX0,
                'rS0': rS0,
                'rP0': rP0
                        }

        return start_params

    def calculate_monod(self, hidden_params=False):
        '''
        Calculates monod kinetics for current model instance.
            Output:
                monod_result: dict, Result of kinetics (X, S, P, µ, rX, rS, rP) as Lists

        Test correct length of model output
        >>> TestModelBatch = Monod_Model()
        >>> monod_result = TestModelBatch.calculate_monod()
        >>> len(monod_result) == 7
        True

        >>> len(monod_result['X']) == TestModelBatch.var_Params['duration']
        True

        >>> max(monod_result['u']) <= TestModelBatch.var_Params['umax']
        True
        '''
        # Get start parameters
        params = self.get_start_params(hidden_params)
        rX, rS, rP = [params['rX0']], [params['rS0']], [params['rP0']]
        if hidden_params:
            duration, umax, Ks, = self.var_Params['duration'], self.__hiddenParams['umax'], self.__hiddenParams['Ks']
            Yx, k1, u = self.__hiddenParams['Yx'], self.__hiddenParams['k1'], [self.__hiddenParams['u0']]
        else:
            duration, umax, Ks,  = self.var_Params['duration'], self.var_Params['umax'], self.var_Params['Ks']
            Yx, k1, u = self.var_Params['Yx'], self.var_Params['k1'], [self.var_Params['u0']]
        S, P, X = [self.var_Conditions['S0']], [self.var_Conditions['P0']], [self.var_Conditions['X0']]

        for j in range(1, duration):
            new_u = umax * S[j - 1] / (Ks + S[j - 1])       # Change of µ
            if new_u >= 0:
                u.append(new_u)
            else:
                u.append(0)

            new_rX = u[j - 1] * X[j - 1]                    # Derivative of Biomass
            if new_rX >= 0:
                rX.append(new_rX)
            else:
                rX.append(0)
            X.append(X[j - 1] + rX[j])                      # New [Biomass]

            new_rS = -(rX[j - 1] / Yx)                      # Derivative of substrate
            if new_rS <= 0:
                rS.append(new_rS)
            else:
                rS.append(0)
            new_S = S[j - 1] + rS[j]
            if new_S < 0:
                S.append(0)                                 # New [Substrate]
            else:
                S.append(new_S)

            new_rP = (k1 * u[j]) * X[j]                     # Derivative of product
            if new_rP >= 0:
                rP.append(new_rP)
            else:
                rP.append(0)

            P.append(P[j - 1] + rP[j])                      # New [Product]

        monod_result = {
                'X': X,
                'S': S,
                'P': P,
                'u': u,
                'rX': rX,
                'rS': rS,
                'rP': rP
                }

        if not hidden_params:
            self.Results = monod_result

        return monod_result

    def plot_results(self, offline_results: pd.DataFrame = None):
        '''
        Returns plot of current model instance results (X,S,P vs. Time).
        Raises AttributeError if no results are stored inside the model when the method is called.

        >>> TestModelBatch = Monod_Model()
        >>> TestModelBatch.plot_results()
        Traceback (most recent call last):
        ...
        AttributeError: No Results yet! Call Monod_Model.calculate_monod() before plotting.
        '''
        if not hasattr(self, 'Results'):
            raise AttributeError('No Results yet! Call Monod_Model.calculate_monod() before plotting.')

        time = range(1, self.var_Params['duration']+1)
        X, S, P = self.Results['X'], self.Results['S'], self.Results['P']

        plt.plot(time, X, 'r', time, S, 'g', time, P, 'b')
        if offline_results is not None:
            off_time = offline_results.index + 1
            off_X, off_S, off_P = offline_results['X'].values, offline_results['S'].values, offline_results['P'].values
            plt.plot(off_time.values, off_X, 'r', off_time.values, off_S, 'g', off_time.values, off_P, 'b', linestyle='', marker='X')

        plt.legend(['Biomass [g/L]', 'Substrate [g/L]', 'Product [g/L]'])
        plt.ylabel('Biomass, Substrate & Product Concentration [g/L]')

        plt.xlabel('Process Duration [h]')
        plt.title(self.__Description)
        plt.show()
        return plt

    def calc_new_mu(self, step: int):
        '''
        Calculates new µ value depending on different inhibition terms.
        Raises AttributeError if mode of operation or inhibition term is unsupported.

        >>> TestModelBatch = Monod_Model()
        >>> TestModelBatch.var_OperationMode = 'test'
        >>> TestModelBatch.calc_new_mu(1)
        Traceback (most recent call last):
        ...
        AttributeError: Unsupported mode of operation. Check to see if Model.var_OperationMode is one of "batch", "fedbatch", or "continuous".
        '''
        self.var_Paramsparams = self.get_start_params()
        duration, umax, Ks,  = self.var_Params['duration'], self.var_Params['umax'], self.var_Params['Ks']
        # Yx, k1, u = self.var_Params['Yx'], self.var_Params['k1'], [self.var_Params['u0']]
        # TODO: Anpassen auf iteratives Aufrufen der Monod_Calculation
        S, P, X = [self.var_Conditions['S0']], [self.var_Conditions['P0']], [self.var_Conditions['X0']]

        if self.var_OperationMode == 'batch':
            new_u = umax * S[step - 1] / (Ks + S[step - 1])
        elif self.var_OperationMode == 'fedbatch':
            new_u = 0
        elif self.var_OperationMode == 'continuous':
            new_u = 0
        else:
            raise AttributeError('Unsupported mode of operation. Check to see if Model.var_OperationMode is one of '
                                 '"batch", "fedbatch", or "continuous".')

        return new_u

    def results_to_csv(self, experiments_ID: int = 0):
        '''
        Writes calculated results to .csv file in dir /csv_files
        :param experiments_ID:
        :return:
        '''
        pathname = os.path.relpath('model_results')
        if not os.path.isdir(pathname):
            os.mkdir(pathname)
        df = pd.DataFrame(self.Results)
        filename = os.path.join(pathname, f'Experiment_{experiments_ID}_{self.__ModelName}.csv')
        df.to_csv(filename)

    def to_json(self, suffix: str = None):
        '''
        Serializes model results to .json.
        '''
        pathname = os.path.relpath('json_files')
        if not os.path.isdir(pathname):
            os.mkdir(pathname)

        if suffix:
            filename = f'{self.__ModelName}_{suffix}.json'
        else:
            filename = f'{self.__ModelName}.json'
        path_file_name = os.path.join(pathname, filename)

        with open(path_file_name, 'w') as outfile:
            json.dump(self, outfile, indent=4, default=lambda o: o.__dict__)

        return path_file_name

    def from_json(self, json_path: str):
        '''
        Deserializes model from .json.
        '''
        with open(json_path, 'r') as file:
            readout = json.load(file)
            new_model = MonodModel()
            new_model.var_Params = readout['var_Params']
            new_model.var_Conditions = readout['var_Conditions']
            new_model.var_Organism = readout['var_Organism']
            new_model.var_OperationMode = readout['var_OperationMode']
            new_model.Results = readout['Results']
            new_model.__Description = readout['_MonodModel__Description']
        return new_model

    def Make_SubstrateGrowthExp(self, substrate_values: list, experiments_ID: int):
        '''

        '''
        ncols = 2
        nrows = ceil(len(substrate_values)/ncols)
        for value in substrate_values:
            assert type(value) == int or float, 'Substrate value is not a number!'
            self.var_Conditions['S0'] = value
            self.calculate_monod()
            plt = self.plot_results()
            self.to_json(suffix=str(experiments_ID))
            self.results_to_csv(experiments_ID)
            self.var_ExpCount += 1

        return plt

    def offline_samples(self, experiments_ID: int = 0):
        '''
        Simulates manual sampling of process with samples at discrete timesteps.
        :return:
        '''

        results = pd.DataFrame(self.calculate_monod(hidden_params=True))
        sampling_times = [0, 2, 4, 6, 8, 20, 23]
        offline_values = results.iloc[sampling_times][['X', 'S', 'P']]

        pathname = os.path.join('..','Data','FermProSim_samples')
        if not os.path.isdir(pathname):
            os.mkdir(pathname)

        mu, sigma = 0, 0.1
        noise = np.random.normal(mu, sigma, [offline_values.shape[0], offline_values.shape[1]])
        offline_values = offline_values + noise
        offline_values[offline_values < 0] = np.nan

        filename = os.path.join(pathname, f'Experiment_{experiments_ID}_{self.__ModelName}.csv')
        offline_values.to_csv(filename)

        time = np.array(offline_values.index)
        X, S, P = offline_values['X'].values, offline_values['S'].values, offline_values['P'].values

        plt.plot(time, X, 'r', time, S, 'g', time, P, 'b', linestyle='', marker='X')
        plt.legend(['Biomass [g/L]', 'Substrate [g/L]', 'Product [g/L]'])
        plt.ylabel('Biomass, Substrate & Product Concentration [g/L]')

        plt.xlabel('Process Duration [h]')
        plt.title(self.__Description)
        plt.show()

        return filename

    def load_offline_values(self, experiment_name):
        '''

        :param experiments_ID:
        :return:
        '''
        return pd.read_csv(experiment_name, index_col=0)

    def get_optimal_X(self):
        save_old_S = self.var_Conditions['S0']
        self.var_Conditions['S0'] = self.__optimal_S
        opt_res = self.calculate_monod(hidden_params=True)
        optimal_X = round(max(opt_res['X']), 1)
        self.var_Conditions['S0'] = save_old_S
        print(f'Desired biomass for this process is {optimal_X} g/L.')
        self.__optimal_X = optimal_X
        return optimal_X

    def calc_rmse(self, param_list=None):
        '''

        :param offline_values:
        :return:
        '''
        if param_list is not None:
            self.set_params(param_list, count=False)
        self.calculate_monod()
        calc_X, calc_S, calc_P = self.Results['X'], self.Results['S'], self.Results['P']
        right_values = self.calculate_monod(hidden_params=True)
        real_X, real_S, real_P = right_values['X'], right_values['S'], right_values['P']

        rmse = sklearn.metrics.mean_squared_error([calc_X, calc_S, calc_P], [real_X, real_S, real_P], squared=False)
        return rmse

    def fit_model(self, param_list: list):
        '''

        :param param_list:
        :param offline_values:
        :return:
        '''
        # umax, Ks, Yx, k1
        bounds = Bounds([0.5, 7, 0.4, 0.05], [1.1, 10, 0.6, 0.2])

        optimizer = minimize(self.calc_rmse, param_list, bounds=bounds)
        print(f'Model optimized in {optimizer.nit} steps.')

        return [optimizer.x[0], optimizer.x[1], optimizer.x[2], optimizer.x[3]]

    def set_params(self, param_list: list, count=True):
        '''

        :param count:
        :param param_list:
        :return:
        '''
        self.var_Params['umax'] = param_list[0]
        self.var_Params['Ks'] = param_list[1]
        self.var_Params['Yx'] = param_list[2]
        self.var_Params['k1'] = param_list[3]
        if count:
            self.var_ModelingCount += 1

    def set_conditions(self, condition_list: list, count=True):
        '''

        :param condition_list:
        :param count:
        :return:
        '''
        self.var_Conditions['S0'] = condition_list[0]
        self.var_Conditions['X0'] = condition_list[1]
        if count:
            self.var_ExpCount += 1

    def get_num_experiments(self):
        '''

        :return:
        '''

        print(f'Model parameters were changed {self.var_ModelingCount} times.\n'
              f'{self.var_ExpCount} Experiments performed.')

    def plot_linear_fit(self, offline_results: pd.DataFrame):
        '''

        :param offline_results:
        :return:
        '''
        if not hasattr(self, 'Results'):
            self.calculate_monod()

        X, S, P = self.Results['X'], self.Results['S'], self.Results['P']
        time = range(1, self.var_Params['duration'] + 1)
        plt.plot(time, X, 'r', time, S, 'g', time, P, 'b')
        off_time = offline_results.index + 1
        offline_results.fillna(0, inplace=True)
        off_X, off_S, off_P = offline_results['X'], offline_results['S'], offline_results['P']
        off_dict = {'X': [], 'S': [], 'P': []}
        for i in range(0, self.var_Params['duration']):
            if i in offline_results.index.to_list():
                off_dict['X'].append(off_X[i])
                off_dict['S'].append(off_S[i])
                off_dict['P'].append(off_P[i])
            else:
                off_dict['X'].append(np.nan)
                off_dict['S'].append(np.nan)
                off_dict['P'].append(np.nan)
        off_df = pd.DataFrame(off_dict)
        off_df.interpolate(inplace=True)
        rmse = sklearn.metrics.mean_squared_error([X, S, P], [off_df['X'], off_df['S'], off_df['P']], squared=False)

        plt.plot(off_time.values, off_X.values, 'r', off_time.values, off_S.values, 'g', off_time.values, off_P.values, 'b', linestyle='--', marker='X')

        plt.legend(['Biomass [g/L]', 'Substrate [g/L]', 'Product [g/L]'])
        plt.ylabel('Biomass, Substrate & Product Concentration [g/L]')

        plt.xlabel('Process Duration [h]')
        plt.title(f'RMSE of linear fit: {round(rmse, 3)}')
        plt.show()

        return rmse

    def get_max_biomass(self):
        if not hasattr(self, 'Results'):
            self.calculate_monod()

        X = self.Results['X']
        max_X = max(X)
        print(f'Max. biomass in current setup is: {round(max_X, 1)} g/L.')

        mae_X = sklearn.metrics.mean_absolute_error([max_X], [self.__optimal_X])
        print(f'Absolute error of current and desired biomass: {round(mae_X, 3)}')

        return max_X
