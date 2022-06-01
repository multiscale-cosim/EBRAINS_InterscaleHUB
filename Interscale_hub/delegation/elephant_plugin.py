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
# import numpy as np

# NOTE plugin to be merged in Elephant main !?
# import elephant
# TODO fork, clone and properly import the plugin
# https://github.com/ojoenlanuca/online_elephant/blob/master/online_statistics.py
# import online_elephant 

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class ElephantPlugin:
    '''
    
    '''
    def __init__(self, configurations_manager, log_settings):
        '''
        
        '''
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="Elephant -- ElephantPlugin",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self.__logger.info("Initialised")

    def online_statistics():
        '''
        TODO: expose the available statistic modules/functions 
        '''
        raise NotImplementedError
    
    def online_unitary_events():
        '''
        TODO: expose the unitary event function
        '''
        raise NotImplementedError
