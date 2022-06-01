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


from EBRAINS_InterscaleHUB.Interscale_hub.elephant_delegator import ElephantDelegator
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories

class Analyzer:
    '''
    Main class for analysis of data. It wraps the functionality of the libraries
    such as ELEPHANT for analysis.

    NOTE this wrapper class exposes only the functionality that is supported by
    InterscaleHub.
    '''
    def __init__(self, param, configurations_manager, log_settings):
        """
        """
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="Analyzer",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        self.__elephant_delegator = ElephantDelegator(param, configurations_manager, log_settings)
        self.__logger.info("Initialized")
    
    def spiketrains_to_rate(self, count, spike_trains):
        """analyzes the data for a given time interval and returns the results.

        Parameters
        ----------
        count : int
            counter of the number of time of the transformation (identify the
            timing of the simulation)

        spike_trains: list
            list of spike trains to be converted into rate

        Returns
        ------
             times, rate: numpy array, float
                tuple of interval and the rate for the interval if data is
                transformed successfully
        """
        return self.__elephant_delegator.spiketrains_to_rate(count, spike_trains)
