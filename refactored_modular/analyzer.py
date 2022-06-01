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

class Analyzer:
    '''
    Main class for analysis of data.
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
        self.__logger.info("Initialised")

    
    def spiketrains_to_rate(self, count, spiketrains):
        """analyzes the data for a given time interval and returns the results.
        
        # TODO Discuss how to handle and call the available Analysis wrappers
        # TODO Validate if it analyze the data otherwise return ERROR as response
        # TODO First usecase functions are rate to spike and spike to rate 

        Parameters
        ----------
        data : Any
            Data to be analyzed

        time_start: int
           time to start the analysis

        time_stop: int
           time to stop the analysis

        variation : bool
            boolean for variation of rate

        windows: float
            the window to compute rate

        Returns
        ------
            returns the analyzed data
        """
        return self.__elephant_delegator.spiketrains_to_rate(count, spiketrains)
