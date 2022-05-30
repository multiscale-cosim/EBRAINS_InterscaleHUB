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
from abc import ABC, abstractmethod
from mpi4py import MPI

from EBRAINS_InterscaleHUB.refactored_modular.communicator_nest_to_tvb import CommunicatorNestTvb 
from EBRAINS_InterscaleHUB.refactored_modular.communicator_tvb_to_nest import CommunicatorTvbNest 
from EBRAINS_InterscaleHUB.refactored_modular.Analyzer import Analyzer                                  
from EBRAINS_InterscaleHUB.refactored_modular.interscale_transformer import InterscaleTransformer       
from EBRAINS_InterscaleHUB.refactored_modular.interscalehub_buffer import InterscaleHubBufferManager    
from EBRAINS_InterscaleHUB.refactored_modular.interscaleHub_mediator import InterscaleHubMediator
from EBRAINS_InterscaleHUB.refactored_modular.intercomm_manager import  IntercommManager as icm
from EBRAINS_InterscaleHUB.refactored_modular.interscalehub_enums import DATA_EXCHANGE_DIRECTION

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class InterscaleHubBaseManager(ABC):
    '''
    Abstract Base Class which
    1) Interacts with InterscaleHub Facade to steer the execution
    2) Manages the InterscaleHub functionality.
    '''
    def __init__(self, parameters, direction, configurations_manager, log_settings):
        '''
        Init params, create buffer, open ports, accept connections
        '''
        
        self.__log_settings = log_settings
        self.__configurations_manager = configurations_manager
        self.__logger = self.__configurations_manager.load_log_configurations(
                                        name="InterscaleHub -- Base Manager",
                                        log_configurations=self.__log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        # 1) param stuff, create IntercommManager
        # MPI and IntercommManager
        self.__intra_comm = MPI.COMM_WORLD  # INTRA communicator
        self.__root = 0 # hardcoded!
        self.__intercomm_manager = icm.IntercommManager(
            self.__intra_comm,
            self.__root,
            self.__configurations_manager,
            self.__log_settings)
        
        self.__path = self.__parameters['path']

        # instances for mediation
        # Data Buffer Manager
        self.__interscalehub_buffer_manager = InterscaleHubBufferManager(
            self.__configurations_manager,
            self.__log_settings)
        self.__interscalehub_buffer = None
        # InterscaleHub Transformer
        self.__interscale_transformer = InterscaleTransformer()
        # Analyzer
        self.__analyzer = Analyzer()
        # Mediator
        self.__mediator = InterscaleHubMediator(
            self.__configurations_manager,
            self.__log_settings,
            self.__interscale_transformer,
            self.__analyzer,
            self.__interscalehub_buffer_manager)
        
        # Simulators Managers
        # Case a: NEST to TVB Manager
        if direction == DATA_EXCHANGE_DIRECTION.NEST_TO_TVB:
            self.__nest_tvb_communicator = CommunicatorNestTvb(
                self.__configurations_manager,
                self.__log_settings,
                self.__interscalehub_buffer_manager,
                self.__mediator)
        # Case b: TVB to NEST Manager
        elif direction == DATA_EXCHANGE_DIRECTION.TVB_TO_NEST:
            self.__tvb_nest_communicator = CommunicatorTvbNest(
                self.__configurations_manager,
                self.__log_settings,
                parameters,
                self.__interscalehub_buffer_manager,
                self.__mediator)

        # TODO: set via XML settings.
        # NOTE consider the scenario when handling the data larger than the buffer size
        self.__max_events = 1000000  # max. expected number of events per step

        self.__parameters = parameters.get_param(direction)
        self.__transformer_id = 0  # NOTE: hardcoded
        self.__id_proxy_nest_region = self.__parameters['id_nest_region']
        self.__logger.info("initialized")
    
    def __get_mpi_shared_memory_buffer(self, buffer_size):
        '''
        Creates shared memory buffer for MPI One-sided-Communication.
        This is wrapper to buffer manager function which creates the mpi
        shared memory buffer.
        '''
        
        # create an MPI shared memory buffer
        self.__interscalehub_buffer =\
            self.__interscalehub_buffer_manager.create_mpi_shared_memory_buffer(
                 buffer_size,
                 self.__intra_comm)
        return self.__interscalehub_buffer
    
    def __set_up_connection(self, path_to_port_file):
        '''
        Open ports and register connection details.
        Accept connection on ports and create INTER communicators.
        
        MVP: register = write port details to file.
        MVP: Two connections 
            - input = incoming simulation data
            - output = outgoing simulation data
        '''
        return self.__intercomm_manager.open_port_accept_connection(
            path_to_port_file)
    
    @abstractmethod
    def start(self):
        '''
        1) init pivot objects depending on the usecase (direction)
        2) start pivot with INTRA communicator (M:N mapping)
        '''
        raise NotImplementedError
        
    @abstractmethod
    def stop(self):
        '''
        Receive stop command.
        Call stop on the pivot operation loop (receiving and sending)
        
        TODO: add error handling and fail checks
        '''
        raise NotImplementedError
