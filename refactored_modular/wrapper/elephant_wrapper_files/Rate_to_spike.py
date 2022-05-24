# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------


#TODO raw copy paste of both rate to spike implementations (by Lionel)
#TODO "choose" one
import numpy as np

from quantities import ms, Hz
from neo.core import AnalogSignal

from elephant.spike_train_generation import inhomogeneous_poisson_process,
homogeneous_poisson_process

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories

# NOTE Former rate_to_spikes_transformer.py subclass
class RatetoSpikes():
    '''Transforms the rates into spikes train.'''

    def __init__(self, param, configurations_manager, log_settings):
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="Transformer--Rate_to_Spikes",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)     
        # number of spike generators
        self.nb_spike_generator = param['nb_neurons']
        self.path = param['path'] + "/transformation/"
        
        # variable for saving values
        self.save_spike = bool(param['save_spikes'])
        if self.save_spike:
            self.save_spike_buf = None
        self.save_rate = bool(param['save_rate'])
        if self.save_rate:
            self.save_rate_buf = None
        
        # number of synapsis
        self.nb_synapse = int(param["nb_brain_synapses"])
        self.__logger.info("Initialised")
        
    def transform(self,count,time_step,rate):
        """
        implements the abstract method for the transformation of the
        rate to spikes.

        Parameters
        ----------
        count : int
            counter of the number of time of the transformation (identify the
            timing of the simulation)

        time_step: int
           time interval for the signal

        rate: Any  # e.g. 1D numpy array
            Spikes rate

        Returns
        ------
            spikes_train: list
                returns spikes train if data is transformed successfully
        """
        rate *= self.nb_synapse  # rate of poisson generator ( due property of poisson process)
        rate += 1e-12
        rate = np.abs(rate)  # avoid rate equals to zeros
        signal = AnalogSignal(rate * Hz, t_start=(time_step[0] + 0.1) * ms,
                              sampling_period=(time_step[1] - time_step[0]) / rate.shape[-1] * ms)
        self.__logger.debug(f"rate: {rate}, signal: {signal}, time_step: {time_step}")
        spikes_train = []
        # generate individual spike trains
        for i in range(self.nb_spike_generator[0]):
            spikes_train.append(np.around(np.sort(inhomogeneous_poisson_process(signal, as_array=True)), decimals=1))
        return spikes_train
    
    
#NOTE former analyzer_rate_to_spike.py subclass    
class AnalyzerRateToSpikes():
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
