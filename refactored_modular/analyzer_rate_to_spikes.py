# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor
#  license agreements; and to You under the Apache License, Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
#
# ------------------------------------------------------------------------------

# TODO: 
# copied: rate_spike.py from NEST-TVB usecase.
# implement properly as elephant science part
# use both methods as Elephant plugin example!

from elephant.spike_train_generation import homogeneous_poisson_process, inhomogeneous_poisson_process
import numpy as np
from neo import AnalogSignal
from quantities import Hz

from EBRAINS_InterscaleHUB.refactored_modular.interscale_transformer_base import ScienceAnalyzerBaseClass
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class AnalyzerRateToSpikes(ScienceAnalyzerBaseClass):
    '''Implements the abstract base class for analyzing the data.'''

    def __init__(self, configurations_manager=None, log_settings=None):
        # TODO Discuss whether the logging is not necessary
        try:
            self._log_settings = log_settings
            self._configurations_manager = configurations_manager
            self.__logger = self._configurations_manager.load_log_configurations(
                                            name="Analyzer--Rate_to_Spikes",
                                            log_configurations=self._log_settings,
                                            target_directory=DefaultDirectories.SIMULATION_RESULTS)
            self.__logger.info("initialized")
        except Exception:
            # continue withour logger
            pass

    def analyze(self, data, time_start, time_stop, variation=False):
        '''
        Wrapper for computing the spikes from rate. It generate the spike
        train with homogenous or inhomogenous Poisson generator.

        Parameters
        ----------
        data : Any
            an array or a float of quantities to be analyzed.

        
        time_start: int
           time to start the computation (spike train)

        time_stop: int
           time to stop the computation (spike train)

        variation : bool
            boolean for variation of rate

        Returns
        ------
        rate: numpy array
                the resultant one or multiple spike trains.
        '''
        
        return self.__rates_to_spikes(data, time_start, time_stop, variation)

    def __rates_to_spikes(self, rates, t_start, t_stop, variation):
        """helper  function to compute the spikes from rate."""
        if variation:
            # the case where the variation of the rate is include
            # We generate the inhomogenous poisson
            if len(rates.shape) == 1:
                # the case where we have only one rate
                signal = AnalogSignal(rates, t_start=t_start, sampling_period=(t_stop-t_start)/rates.shape[-1])
                result = [inhomogeneous_poisson_process(signal,as_array=True)]
                return np.array(result)
            else :
                # the case where we have multiple rates
                result = []
                for rate in rates:
                    signal = AnalogSignal(rate, t_start=t_start, sampling_period=(t_stop - t_start) / rates.shape[-1])
                    result.append(inhomogeneous_poisson_process(signal,as_array=True))
                return np.array(result)
        else:
            # the case we have only the rate
            # We generate the homogenous poisson
            if len(rates.shape) ==0:
                # the case where we have only one rate
                result = np.array([homogeneous_poisson_process(rate=rates, t_start=t_start, t_stop=t_stop, as_array=True)])
            else:
                # the case where we have multiple rates
                result = []
                for rate in rates:
                    result.append(homogeneous_poisson_process(rate=rate, t_start=t_start, t_stop=t_stop, as_array=True))
            return np.array(result)

if __name__=='__main__':
    from quantities import ms,Hz
    rates_to_spikes_analyzer = AnalyzerRateToSpikes()
    resultant_spikes = rates_to_spikes_analyzer.analyze([7942.65518188, 7168.64013672, 6612.2756958,  4990.57312012, 5077.53219604,
    7417.4659729,  7284.71984863, 6751.57318115, 5528.10173035, 5102.45170593,
    6506.46362305, 6908.81881714, 4290.93399048, 5575.27732849, 7972.46932983,
    9518.87893677, 7561.74240112, 7813.84735107, 8878.12805176, 6734.36965942,
    7277.06069946, 7547.31292725, 9538.67263794, 6232.00950623, 8005.1651001,
    5534.51652527, 7441.41998291, 7747.13668823, 8784.91744995, 9481.22253418,
    7909.23614502, 6691.65420532, 7793.50280762, 8774.40795898, 7772.98965454]*Hz, 7.1*ms ,10.5*ms,variation=True)
    print(resultant_spikes)
 
