"""
An experiment that uses hosts which implements all currently existing modules.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as spi

from silvio import (
    DATADIR, # config
    Experiment, ExperimentException, Host, HostException, # base classes
    alldef, coalesce, first, Generator, # utilities
    DataOutcome, DataWithPlotOutcome, combine_data, # outcome
    GrowthBehaviour, # modules
)
# additional non-renamed utilities
from silvio.extensions.utils.visual import Help_Progressbar
from silvio.extensions.utils.laboratory import ErrorRate


class GrowthExperiment (Experiment) :
    """
    Growth experiments on Monod equation with temperature, substrate and sampling times.
    """

    budget: int

    suc_rate: float

    # Keep track of how many hosts were created
    host_counter: int



    def __init__ ( self, seed:Optional[int] = None, equipment_investment:int = 0, max_budget:int = 10000  ) :

        if equipment_investment > max_budget :
            raise ExperimentException("Investment cost is higher than maximal budget.")

        super().__init__(seed=seed)
        self.budget = max_budget - equipment_investment
        self.suc_rate = ErrorRate(equipment_investment, max_budget)
        self.host_counter = 0



    def create_host ( self, name:Optional[str] ) -> GroHost:
        self.host_counter += 1
        seed = self.rnd_gen.pick_seed() # The experiment provides stable seed generation for hosts.
        new_host = None

        chosen_name = coalesce( name, 'ecol' + str(self.host_counter) )
        new_host = self.build_host( name=chosen_name, seed=seed )

        self.bind_host(new_host)
        return new_host



    def build_host ( self, name:str, seed:int ) -> GroHost :
        gen = Generator( seed )
        host = GroHost( name=name, seed=seed )
        host.make(
            opt_growth_temp= gen.pick_integer(25, 40), # unit: degree celsius, source: https://application.wiley-vch.de/books/sample/3527335153_c01.pdf
            max_biomass= gen.pick_integer(30, 100), # unit: in gDCW/l, source (german): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjzt_aJ9pzpAhWGiqQKHb1jC6MQFjABegQIAhAB&url=https%3A%2F%2Fwww.repo.uni-hannover.de%2Fbitstream%2Fhandle%2F123456789%2F3512%2FDissertation.pdf%3Fsequence%3D1&usg=AOvVaw2XfGH11P9gK2F2B63mY4IM
            Ks = round(gen.pick_uniform(7, 10), 3),
            Yxs = round(gen.pick_uniform(.4, .6), 2),
            k1 = round(gen.pick_uniform(.05, .2), 3),
            umax = round(gen.pick_uniform(.5, 1.1), 3),
        )
        host.sync()
        return host


    def find_host_or_abort ( self, host_name:str ) -> GroHost :
        """
        Find a host by a given name or abort with an exception.
        Throws: ExperimentException
        """
        host:Optional[GroHost] = first( self.hosts, lambda h: h.name == host_name )
            # Find the first host matching the name.
        if host is None :
            raise ExperimentException("Experiment has not found the host '{}'".format(host_name))
        return host

    def spend_budget_or_abort ( self, amount:int ) -> None :
        """
        Each operation may use up a certain amount of the budget.
        When the budget is not sufficient those operations will all fail.
        By using error handling via raised exceptions we don't need to check each method for a
        success flag on their return value.
        Throws: ExperimentException
        """
        if amount > self.budget :
            raise ExperimentException("Experiment has surpassed its budget. No more operations are allowed.")
        self.budget -= amount

    def simulate_monod( self, host_name:str, temps:List[float], samplevec:List[float], substrates:List[float], test=False ) -> DataOutcome :
        """ Simulate a growth under multiple temperatures and return expected biomasses over time. 
        Args:
            host_name: name of the host
            temps: list of temperatures
            samplevec: array of (total sampling time, sample number) with shape (len(temps),2)
            substrates: list of substrate concentrations
        
        Returns:
            GrowthOutcome: dataframe with the expected biomass over time
        """
        host = self.find_host_or_abort(host_name)
        # Equipment failure can prematurely end the simulation.
        if self.rnd_gen.pick_uniform(0,1) < self.suc_rate :
            return DataOutcome( None, 'Experiment failed, bad equipment.' )

        if test == 'Test':
            import random
            samples = random.randint(8, 12)
            time = np.linspace(0, 10, samples)
            mu = [.5,.2]
            biomass = [.01*np.exp(mu*time) for mu in mu]
            Result = pd.DataFrame(np.vstack((time,biomass)).T)
            DataReturn = DataOutcome(Result)
            DataPlot = GrowthOutcome(Result)
            DataPlot.make_plot()
        elif test == 'Monod':
            Params={'mumax': host.growth.umax, 'Ks': host.growth.Ks, 'Yxs': host.growth.Yxs, 'max_biomass': host.growth.max_biomass}
            Initials={'X0': 0.01, 'S0': float(substrates[0])}
            t=np.linspace(0, samplevec[0], samplevec[1])
            BaseCost = 100
            TotalCost = round(BaseCost * calculate_ExpPriceFactor(samplevec[1]))
            self.spend_budget_or_abort( TotalCost )
            MonodSol = add_noise(solve_MonodEqn(Params, Initials, t), self.suc_rate)
            Result = round(pd.DataFrame(np.vstack((t, MonodSol.T)).T, columns=['t', 'X', 'S']),3)
            DataReturn = DataOutcome(Result)
            DataPlot = GrowthOutcome(Result)
            DataPlot.make_plot()
        elif test == 'Monod2':
            SampleNumber = samplevec[1] if samplevec[1] < len(range(samplevec[1])) else len(range(samplevec[1]))
            # calculating the experiment price based on the number of samples
            BaseCost = 100
            TotalCost = round(BaseCost * calculate_ExpPriceFactor(SampleNumber))
            self.spend_budget_or_abort( TotalCost )
            Variables = {'S': 0.5, 'P':0, 'X':.1, 'T': 25}
            Params = {'Ks': 0.5, 'Yxs': 0.5, 'k1': 0.1, 'umax': 0.5}
            allDat = pd.DataFrame.from_dict(makeMonod(Variables, Params, samplevec[0]))
            # selecting data according to the sampling rate, i.e. using every nth row
            ChooseRowAprx = np.round(np.linspace(0, samplevec[0]-1, SampleNumber))
            Result = allDat.iloc[ChooseRowAprx, :]
            DataReturn = DataOutcome(Result[['t', 'X', 'S']])
            DataPlot = GrowthOutcome(Result[['t', 'X', 'S']])
            DataPlot.make_plot()

        # save DataReturn to csv
        DataReturn.export_data('ExperimentGrowthSubstrateRate.csv')
        return DataReturn



    def print_status ( self ) -> None :
        print("Experiment:")
        print("  budget = {}".format( self.budget ))
        print("  failure rate = {}".format( round(self.suc_rate, 2) ))
        print("  hosts = [ {} ]".format( " , ".join([h.name for h in self.hosts]) ))
        # Could display the status of each host if wanted.



class GroHost (Host) :

    growth: GrowthBehaviour


    def make ( self, opt_growth_temp:int, max_biomass:int , Ks:float, Yxs:float, k1:float, umax:float) -> None :

        if not alldef( opt_growth_temp, max_biomass, Ks, Yxs, k1, umax ) :
            raise HostException("Host not initialized. Reason: incomplete arguments.")

        # Setup GrowthBehaviour module
        self.growth = GrowthBehaviour()
        self.opt_growth_temp = opt_growth_temp
        self.max_biomass = max_biomass

        self.growth.make2(
            opt_growth_temp=opt_growth_temp, max_biomass=max_biomass, Ks=Ks, Yxs=Yxs, k1=k1, umax=umax
        )
        self.growth.bind2( host=self )



    def copy ( self, ref:GroHost ) -> None :

        # Setup GrowthBehaviour module using the ref
        self.growth = GrowthBehaviour()
        self.growth.copy( ref=ref.growth )
        self.growth.bind2( host=self )



    def sync ( self ) -> None :
        self.sync_modules([ self.growth ])




    def print_status ( self ) -> None :
        print("Host [{}]:".format( self.name ))
        print("  seed plus counter = {} + {}".format( self.rnd_seed, self.rnd_counter ))
        print("  optimal growth temperature = {}".format( self.growth.opt_growth_temp ))
        print("  max biomass = {}".format( self.growth.max_biomass ))
        print("  Event History: {} events".format(len(self.event_log)))
        for el in self.event_log :
            print("  - {}".format(el))



class GrowthOutcome ( DataWithPlotOutcome ) :

    def make_plot ( self ) -> plt.Figure :
        """
        Plotting with pyplot is unfortunately unintuitive. You cannot display a single figure by
        using the object-oriented API. When you do `plt.subplots` (or create a plot by any other
        means) it will be stored in the global context. You can only display things from the glbbal
        context, and displaying it will remove it from there.
        """
        Time, Biomass = self.value.iloc[:,0], self.value.iloc[:,1:]
        LnBiomass = np.log(Biomass)

        fig, ax = plt.subplots()
        for Exp,X in Biomass.iteritems(): # LnBiomass.iteritems(), Biomass.iteritems()
            ax.scatter(Time, X, label=Exp)
        ax.legend()
        return fig

def makeMonod(Variables, Params, Duration):
        # Get start parameters
    # params = self.get_start_params(hidden_params)
    umax, Ks, Yxs, k1, u = Params['umax'], Params['Ks'], Params['Yxs'], Params['k1'], [0]
    S, P, X = [Variables['S']], [Variables['P']], [Variables['X']]
    # Intial rates
    rX = [u[0] * X[0]]
    rS = [-(rX[0] / (Yxs / S[0]))]
    rP = [(k1 * u[0]) * X[0]]
    # return {'rX': rX, 'rS': rS, 'rP': rP}
    t = [0]
    for j in range(1, Duration):
        new_u = round(umax * S[j - 1] / (Ks + S[j - 1]),3)       # Change of µ
        if new_u >= 0:
            u.append(new_u)
        else:
            u.append(0)

        new_rX = round(u[j - 1] * X[j - 1], 3)                    # Derivative of Biomass
        if new_rX >= 0:
            rX.append(new_rX)
        else:
            rX.append(0)
        X.append(round(X[j - 1] + rX[j], 3))                      # New [Biomass]

        new_rS = round(-(rX[j - 1] / Yxs), 3)                      # Derivative of substrate
        if new_rS <= 0:
            rS.append(new_rS)
        else:
            rS.append(0)
        new_S = S[j - 1] + rS[j]
        if new_S < 0:
            S.append(0)                                 # New [Substrate]
        else:
            S.append(new_S)

        new_rP = round((k1 * u[j]) * X[j], 3)                     # Derivative of product
        if new_rP >= 0:
            rP.append(new_rP)
        else:
            rP.append(0)

        P.append(round(P[j - 1] + rP[j], 3))                      # New [Product]
        t.append(j)
    monod_result = {
            't': t,
            'X': X,
            'S': S,
            'P': P,
            'u': u,
            'rX': rX,
            'rS': rS,
            'rP': rP
            }
    return monod_result

def calculate_ExpPriceFactor(SampleAmount, PlotExample=False):
    '''logit function to generate the price-factor for amount of sampling
        https://nathanbrixius.wordpress.com/2016/06/04/functions-i-have-known-logit-and-sigmoid/
        increasing sampling amount will increase the price until a certain point
        after that the price will remain constant until a certain point
        after that the price will increase again
        the input needs to be between 0 and 1, otherwise the function will return an error
        therefore we need to normalize the input with a saturation function
        the saturation function is a tanh function
        the saturation function will be used to normalize the input

        Args:
            SampleAmount (float): amount of sampling
            PlotExample (bool): if True, the function will plot an example of the function
        Returns:
            PriceFactor (float): price factor for the amount of sampling
        '''
    myTanh = lambda x: np.tanh(x/20) #+ np.tanh(1/20)
    myLogit = lambda x: (np.log(x/(1-x))**3 - np.log(myTanh(1)/(1-myTanh(1)))**3 + 1)/26
    PriceFactor = myLogit(myTanh(SampleAmount))
    if PlotExample:
        x = np.linspace(1,SampleAmount,SampleAmount)
        y = myLogit(myTanh(x))
        plt.plot(x,y, label='logit function', marker='o', linestyle='None')
        plt.xlabel('Sampling amount')
        plt.ylabel('Price factor')
        plt.show()

    return PriceFactor

def add_noise(data, noise):
    """
    Add noise to data.
    """
    # Each column get a different noise based on the average value of the column
    noise = noise * np.mean(data, axis=0)  
    # ensure that there are no negative values
    return np.abs(data + np.random.normal(0, noise, data.shape))

def MonodEqnODE(MonodVars, t, mumax, Ks, Yxs, max_biomass):
    """
    Monod bioprocess model with ODEs. Based on the model described in:
    Hemmerich et al., doi:10.1002/elsc.202000088, Fig. 6

    Args:
        MonodVars (list): list of variables [X, S]
        t (float): time
        mumax (float): maximum specific growth rate
        Ks (float): half-saturation constant
        Yxs (float): yield coefficient

    Returns:
        list: list of derivatives [dX_dt, dS_dt]
    """
    # Unpack y and z
    X, S = MonodVars

    mu = mumax * S / (Ks + S)
    # Compute biomass growth rate with  a Verhulst growth function
    dX_dt = mu * X * (1 - X / max_biomass)

    # Compute substrate consumption rate
    dS_dt = -mu * X  * (1 - X / max_biomass) / Yxs

    # Return the result as a NumPy array
    return np.array([dX_dt, dS_dt])

def solve_MonodEqn(Params, Initials, t):
    """
    Solve the Monod bioprocess model with ODEs.

    Args:
        Params (dict): dictionary of parameters
        Initials (dict): dictionary of initial conditions
        t (list): list of time points

    Returns:
        list: list of solutions [X, S]
    """
    # Unpack parameters
    mumax, Ks, Yxs, max_biomass = Params['mumax'], Params['Ks'], Params['Yxs'], Params['max_biomass']
    # Initial condition
    MonodVars_0 = [Initials['X0'], Initials['S0']]

    # Parameters
    args = (mumax, Ks, Yxs, max_biomass)

    # Integrate ODES
    MonodSol = spi.odeint(MonodEqnODE, MonodVars_0, t, args=args)

    # Return the result as a NumPy array
    return MonodSol
