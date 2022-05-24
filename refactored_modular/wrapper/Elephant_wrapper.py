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

from EBRAINS_InterscaleHUB.refactored_modular.wrapper.elephant_wrapper.elephant_wrapper_files import (
    Elephant_plugin,
    Rate_to_spike,
    Spike_to_spiketrain,
    Spiketrain_to_rate
    )
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories

#TODO make <x> separate files for each class
#here: create instance/object of these classes and expose their functionality, e.g. 'transform_rate_to_spike'
# they call the transform of the rate to spike classes and expose it to interscale_transform

class ElephantWrapper:
    '''
    
    '''
    def __init__(self, configurations_manager, log_settings):
        '''
        
        '''
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="ElephantWrapper",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self.__logger.info("Initialised")
    
    
    def rate_to_spike():
        """Converts from rate to spike with functionality
        provided by Elephant analysis.
        
        Parameters
        ----------
        

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError
    
    
    def spike_to_spiketrain():
        """Converts from spikes to spiketrains with functionality
        provided by Elephant analysis.
        
        Parameters
        ----------
        

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError
    
    
    def spiketrain_to_rate():
        """Converts from spiketrains to rate with functionality
        provided by Elephant analysis.
        
        Parameters
        ----------
        

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError
        
