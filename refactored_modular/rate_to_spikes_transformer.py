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
# ------------------------------------------------------------------------------
import numpy as np
from quantities import ms, Hz

from neo.core import AnalogSignal

from elephant.spike_train_generation import inhomogeneous_poisson_process

from EBRAINS_InterscaleHUB.refactored_modular.interscale_transformer_base import InterscaleTransformerBaseClass
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class RatetoSpikes(InterscaleTransformerBaseClass):
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
