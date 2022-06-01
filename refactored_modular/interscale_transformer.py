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


class InterscaleTransformer:
    '''
    Class for transformation of data to change the scales.
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
        
        # TODO discuss what parameters are required for (usecase specific) transformation
        # TODO validate if it transforms the data otherwise return ERROR as response
        # NOTE Followings are taken from rate_to_spike and spike_to_rate functions

        Parameters
        ----------
        data : Any
            Data to be transformed

        count: int
            counter of the number of time of the transformation
            (identify the timing of the simulation)

        buffer: int
            buffer contains id of devices, id of neurons and spike times
        size : int
            size of the data to be read from the buffer for transformation

        Returns
        ------
            returns the data transformed into required format
        """
        return self.__elephant_delegator.spike_to_spiketrains(count, data_size, data_buffer)
    

    def rate_to_spikes(self, time_step, data_buffer):
        """Transforms the data from one format to another .
        
        # TODO discuss what parameters are required for (usecase specific) transformation
        # TODO validate if it transforms the data otherwise return ERROR as response
        # NOTE Followings are taken from rate_to_spike and spike_to_rate functions

        Parameters
        ----------
        data : Any
            Data to be transformed

        count: int
            counter of the number of time of the transformation
            (identify the timing of the simulation)

        buffer: int
            buffer contains id of devices, id of neurons and spike times
        size : int
            size of the data to be read from the buffer for transformation

        Returns
        ------
            returns the data transformed into required format
        """
        return self.__elephant_delegator.rate_to_spikes(time_step, data_buffer)
    
    
    
    
    
