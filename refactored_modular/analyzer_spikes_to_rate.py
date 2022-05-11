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

from elephant.statistics import mean_firing_rate
import numpy as np
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

    def analyze(self, data, time_start, time_stop, window=0.0):
        '''
        Wrapper for computing the rate from spike train. It generate the spike
        train with homogenous or inhomogenous Poisson generator.

        Parameters
        ----------
        data : Any
            an array or a float of quantities to be analyzed (one spike train
            or multiple spike train)

        
        time_start: int
           time to start the computation (rate)

        time_stop: int
           time to stop the computation (rate)

        window : float
            the window to compute the rate

        Returns
        ------
        rate: list
                rates or variation of rates
        '''
        return self.__spikes_to_rate(data, time_start, time_stop, window)

    def __spikes_to_rate(self, spikes,t_start,t_stop, windows):
        """helper  function to compute the rate from spikes."""

        if windows == 0.0:
            #case without variation of rate
            if len(spikes[0].shape) ==0:
                # with only one rate
                result = [mean_firing_rate(spiketrain=spikes,t_start=t_start,t_stop=t_stop).rescale(Hz)]
            else:
                # with multiple rate
                result = []
                for spike in spikes:
                    result.append(mean_firing_rate(spiketrain=spike,t_start=t_start,t_stop=t_stop).rescale(Hz))
            return np.array(result)
        else:
            # case with variation of rate
            rate = []
            for time in np.arange(t_start,t_stop,windows):
                t_start_window = time*t_start.units
                t_stop_window = t_start_window+windows
                if len(spikes[0].shape) == 0:
                    # for only one spike train
                    result = [mean_firing_rate(spiketrain=spikes, t_start=t_start_window, t_stop=t_stop_window).rescale(Hz)]
                else:
                    # for multiple spike train
                    result = []
                    for spike in spikes:
                        result.append(mean_firing_rate(spiketrain=spike, t_start=t_start_window, t_stop=t_stop_window).rescale(Hz))
                rate.append(result)
            return np.array(rate)
