"""
An experiment that uses hosts which implements all currently existing modules.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from silvio import (
    DATADIR, # config
    Experiment, ExperimentException, Host, HostException, # base classes
    alldef, coalesce, first, Generator, # utilities
    DataOutcome, DataWithPlotOutcome, combine_data, # outcome
    GrowthBehaviour, # modules
)
# additional non-renamed utilities
from silvio.extensions.utils.visual import Help_Progressbar



class GrowthExperiment (Experiment) :
    """
    Growth experiments on Monod equation with temperature, substrate and sampling times.
    """

    # budget: int

    suc_rate: float

    # Keep track of how many hosts were created
    host_counter: int



    def __init__ ( self, seed:Optional[int] = None ) :

        super().__init__(seed=seed)
        self.suc_rate = 0.12 # 12% error rate
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
            Yx = round(gen.pick_uniform(.4, .6), 2),
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


    def simulate_monod( self, temps:List[float], samplevec:List[float], substrates:List[float], test=False ) -> GrowthOutcome :
        """ Simulate a growth under multiple temperatures and return expected biomasses over time. 
        Args:
            host_name: name of the host
            temps: list of temperatures
            samplevec: list of sampling times
            substrates: list of substrate concentrations
        
        Returns:
            GrowthOutcome: dataframe with the expected biomass over time
        """
        if test == 'Test':
            import random
            samples = random.randint(8, 12)
            time = np.linspace(0, 10, samples)
            mu = [.5,.2]
            biomass = [.01*np.exp(mu*time) for mu in mu]
            Result = pd.DataFrame(np.vstack((time,biomass)).T)
            DataHandle = GrowthOutcome(value=Result)
            DataHandle.make_plot()
        elif test == 'Monod':
            Variables = {'S': 0.5, 'P':0, 'X':.1, 'T': 25}
            Params = {'Ks': 0.5, 'Yx': 0.5, 'k1': 0.1, 'umax': 0.5}
            Duration = 10
            Result = pd.DataFrame.from_dict(makeMonod(Variables, Params, Duration))
            DataHandle = GrowthOutcome(value=Result[['t', 'X', 'S']])
            DataHandle.make_plot()
            # host = self.find_host_or_abort(host_name)
            # growth = host.growth
            # growth.make_monod(temps, samplevec, substrates)
            # growth.simulate()
            # Result = GrowthOutcome(value=growth.biomass)
            # Result.make_plot()

        return Result



    def print_status ( self ) -> None :
        print("Experiment:")
        print("  failure rate = {}".format( round(self.suc_rate, 2) ))
        print("  hosts = [ {} ]".format( " , ".join([h.name for h in self.hosts]) ))
        # Could display the status of each host if wanted.



class GroHost (Host) :

    growth: GrowthBehaviour


    def make ( self, opt_growth_temp:int, max_biomass:int , Ks:float, Yx:float, k1:float, umax:float) -> None :

        if not alldef( opt_growth_temp, max_biomass, Ks, Yx, k1, umax ) :
            raise HostException("Host not initialized. Reason: incomplete arguments.")

        # Setup GrowthBehaviour module
        self.growth = GrowthBehaviour()
        self.opt_growth_temp = opt_growth_temp
        self.max_biomass = max_biomass

        self.growth.make2(
            opt_growth_temp=opt_growth_temp, max_biomass=max_biomass, Ks=Ks, Yx=Yx, k1=k1, umax=umax
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
    umax, Ks, Yx, k1, u = Params['umax'], Params['Ks'], Params['Yx'], Params['k1'], [0]
    S, P, X = [Variables['S']], [Variables['P']], [Variables['X']]
    # Intial rates
    rX = [u[0] * X[0]]
    rS = [-(rX[0] / (Yx / S[0]))]
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

        new_rS = round(-(rX[j - 1] / Yx), 3)                      # Derivative of substrate
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
