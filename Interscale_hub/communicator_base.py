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
from abc import ABC, abstractmethod

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class BaseCommunicator(ABC):
    '''
    Abstract Base Class which abstracts the
    1) data exchange with applications/simulators
    2) transformation of the data to required scale
    '''
    def __init__(self, configurations_manager, log_settings,
                 communicator_name, data_buffer_manager, mediator):
        '''Base Class initializer to setting up the variables common to all child
        classes'''
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self._logger = self._configurations_manager.load_log_configurations(
                        name=communicator_name,
                        log_configurations=self._log_settings,
                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        # variables commonly used across the child classes
        self._mediator = mediator
        self._data_buffer_manager = data_buffer_manager
        self._comm_receiver = None
        self._comm_sender = None
        self._num_sending = 0
        self._num_receiving = 0

    @abstractmethod
    def start(self,  intra_communicator, inter_communicator):
        """Starts the pivot operations.
        
        Parameters
        ----------
        intra_communicator : MPI Intracommunicator
            for communicating within a group

        inter_communicator : MPI Intercommunicator
            communicates between the groups

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """Stops the pivot operations.

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError
