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
        Abstract Base Class which abstracts the data exchange with
        applications/simulators
    '''
    def __init__(self, configurations_manager, log_settings,
                 communicator_name,
                 data_buffer_manager,
                 intra_comm,
                 receiver_inter_comm,
                 sender_inter_comm,
                 sender_group_ranks,
                 receiver_group_ranks,
                 root_transformer_rank):
        '''
        Base Class initializer to setting up the variables common to all child
        classes
        '''
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self._logger = self._configurations_manager.load_log_configurations(
                        name=communicator_name,
                        log_configurations=self._log_settings,
                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        # variables commonly used across the child classes
        self._data_buffer_manager = data_buffer_manager
        self._group_of_ranks_for_sending = sender_group_ranks
        self._group_of_ranks_for_receiving = receiver_group_ranks
        self._root_transformer_rank = root_transformer_rank
        self._receiver_inter_comm = receiver_inter_comm
        self._sender_inter_comm = sender_inter_comm
        self._intra_comm = intra_comm
        self._my_rank = self._intra_comm.Get_rank()

    @abstractmethod
    def send(self):
        """sends the data
        
        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError

    @abstractmethod
    def receive(self):
        """receives the data

        Returns
        ------
            return code as int to indicate an un/successful termination.
        """
        raise NotImplementedError
