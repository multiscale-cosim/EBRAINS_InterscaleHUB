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


from EBRAINS_InterscaleHUB.refactored_modular.elephant_delegator import ElephantDelegator
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class Transformer:
    '''
    Transforms the data to change the scales. It wraps the functionality of the libraries
    such as ELEPHANT for transformation.

    NOTE this wrapper class exposes only the functionality that is supported by
    InterscaleHub.
    '''
    def __init__(self, param, configurations_manager, log_settings):
        """
        """
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="InterscaleTransformer",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        self.__elephant_delegator = ElephantDelegator(param, configurations_manager, log_settings)
        self.__logger.info("Initialised")
    
    def spike_to_spiketrains(self, count, data_size, data_buffer):
        """Transforms the data from one format to another .

        Parameters
        ----------
        count: int
            counter of the number of time of the transformation
            (identify the timing of the simulation)

        data_size : int
            size of the data to be read from the buffer for transformation

        data_buffer: MPI shared memory window
            buffer contains id of devices, id of neurons and spike times

        Returns
        ------
            returns the spike trains from spikes
        """
        return self.__elephant_delegator.spike_to_spiketrains(count, data_size, data_buffer)

    def rate_to_spikes(self, time_step, data_buffer):
        """Transforms the data from one format to another .

        Parameters
        ----------
        time_step: [int, int]
            time start and time stops of the current simulation step
        
        data_buffer: MPI shared memory window
            buffer contains the rate to be converted into spikes

        Returns
        ------
            returns the spike trains from rate
        """
        return self.__elephant_delegator.rate_to_spikes(time_step, data_buffer)
    
    
    
    
    
