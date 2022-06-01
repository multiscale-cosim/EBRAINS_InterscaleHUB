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


#TODO raw copy paste of all calls to Elephant functions --> refactor!
#TODO overlap of transformation and analysis --> clean separation

import numpy as np

from EBRAINS_InterscaleHUB.refactored_modular.delegation import (
    elephant_plugin,
    spike_rate_conversion
    )
    '''
    Rate_to_spike,
    Spike_to_spiketrain,
    Spiketrain_to_rate
    )
    '''
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class ElephantDelegator:
    '''
    NOTE: some functionalities only had on attribute/method, e.g. rate_to_spike.
    -> new Class "spike_rate_conversion" contains all related functionalities.
    '''
    def __init__(self, param, configurations_manager, log_settings):
        '''
        
        '''
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="ElephantDelegator",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        # init members
        self.spike_rate_conversion = spike_rate_conversion.SpikeRateConversion(
                                        param, 
                                        configurations_manager, 
                                        log_settings)
        self.elephant_plugin = elephant_plugin.ElephantPlugin(
                                        configurations_manager, 
                                        log_settings)
        # dir member methods
        self.spikerate_methods = [f for f in dir(SpikeRateConversion) if not f.startswith('_')]
        self.plugin_methods = [f for f in dir(ElephantPlugin) if not f.startswith('_')]
        self.__logger.info("Initialised")

    
    def __getattr__(self, func):
        '''
        '''
        def elephant_method(*args):
            if func in self.spikerate_methods:
                return getattr(self.spike_rate_conversion, func)(*args)
            elif func in self.plugin_methods:
                return getattr(self.elephant_plugin, func)(*args)
            else:
                raise AttributeError
        return method
        
