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

from EBRAINS_InterscaleHUB.managers.general.buffer_manager import BufferManager
from EBRAINS_InterscaleHUB.managers.general.intercomm_manager import IntercommManager
from EBRAINS_InterscaleHUB.common.interscalehub_enums import DATA_EXCHANGE_DIRECTION
from EBRAINS_RichEndpoint.application_companion.common_enums import INTERCOMM_TYPE
from EBRAINS_InterscaleHUB.common.interscalehub_enums import DATA_EXCHANGE_DIRECTION
from EBRAINS_InterscaleHUB.common.interscalehub_enums import DATA_BUFFER_TYPES
from EBRAINS_InterscaleHUB.common.interscalehub_enums import DATA_BUFFER_STATES
from EBRAINS_InterscaleHUB.common.interscalehub_utils import info_log_message, debug_log_message

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class BaseManager(ABC):
    """
    Abstract Base Class which
    1) Interacts with InterscaleHub Facade to steer the execution
    2) Manages the InterscaleHub functionality.
    """
    def __init__(self, configurations_manager, log_settings,
                 receiver_group_ranks,
                 sender_group_ranks,
                 buffer_size,
                 parameters,
                 sci_params,
                 direction):
        """
        Init params, setup mpi groups, create buffers, initialize default
        settings, data channel setup
        """
        # TODO Revisit variable names/ remove all interscalehub prefixes
        
        # STEP 1) init phase
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self._logger = self._configurations_manager.load_log_configurations(
            name="InterscaleHub -- Base Manager",
            log_configurations=self._log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        # 1.1) create MPI world and INTRA communicator for InterscaleHub
        self._intra_comm = MPI.COMM_WORLD
        self._root = 0  # hardcoded!
        self._my_rank = self._intra_comm.Get_rank()

        # NOTE the log level is set to INFO only for the rank 0
        info_log_message(self._my_rank,
                         self._logger,
                         "STEP 1: initializing parameters...")

        # 1.2) Sci-Params initialization
        self._parameters = parameters
        self._sci_params = sci_params
        self._direction = direction
        
        # Manager object to create INTER communicator to communicate with other
        # applications (e.g simulators)
        self._intercomm_manager = IntercommManager(
            self._intra_comm,
            self._root,
            self._configurations_manager,
            self._log_settings)
        
        # 1.3) objects needed by Mediator
        # Data Buffer Manager
        self._data_buffer_manager = BufferManager(
            self._configurations_manager,
            self._log_settings)
        self._interscalehub_buffer = None
        
        # 1.4) class variables
        self._path = self._parameters['path']
        self._databuffer_input = None
        self._buffer_size = buffer_size
        # INTER = between applications
        self._receiver_inter_comm = None
        self._sender_inter_comm = None
        # INTRA = within applications
        self._receiver_intra_comm = None
        self._sender_intra_comm = None
        self._transformer_intra_comm = None
        self._receiver_group_ranks = receiver_group_ranks  # NOTE hardcoded
        self._sender_group_ranks = sender_group_ranks  # NOTE hardcoded
        # NOTE all remaining ranks are transformers
        self._transformer_group_ranks = [x for x in range(self._intra_comm.Get_size())
                                         if x not in (self._receiver_group_ranks + self._sender_group_ranks) ]
        
        # STEP 2) setup MPI groups
        info_log_message(self._my_rank,
                         self._logger,
                         "STEP 2: setting up mpi groups...")
        self._setup_mpi_groups_and_comms()

        # STEP 3) create buffers
        info_log_message(self._my_rank,
                         self._logger,
                         "STEP 3: Creating MPI shared memory Buffer...")
        #  3.1) create input buffer
        # NOTE more buffer types (e.g. output buffer) can be created in a
        # similar way, if/when needed
        self._databuffer_input = self._get_mpi_shared_memory_buffer(
            self._buffer_size, self._intra_comm, DATA_BUFFER_TYPES.INPUT)
        
        # STEP 4) initialize buffers state
        info_log_message(self._my_rank,
                         self._logger,
                         "STEP 4: initialize buffers state...")
        # 4.1) initialize the input buffer state
        # NOTE state is set to 'READY_TO_RECEIVE' to wait until some data is
        # received from simulators
        if self._receiver_intra_comm and self._intra_comm.Get_rank() == self._receiver_group_ranks[0]:
            self._set_buffer_state(state=DATA_BUFFER_STATES.READY_TO_RECEIVE,
                                           buffer_type=DATA_BUFFER_TYPES.INPUT)
        # sync up point so that initial state of the INPUT buffer could be set
        debug_log_message(self._root,
                          self._logger,
                          "wait until the initial state is set")
        self._intra_comm.Barrier()

        # STEP 5) Data channel setup
        info_log_message(self._my_rank,
                         self._logger,
                         "STEP 5: setting up data channels...")
        self._data_channel_setup()
        
        debug_log_message(self._root, self._logger, "initialized")

    def _set_buffer_state(self, state, buffer_type):
        """helper function to set the buffer state for the given buffer_type"""
        self._data_buffer_manager.set_ready_state_at(index=-1,
                                                              state=state,
                                                              buffer_type=buffer_type)

    def _setup_mpi_groups_and_comms(self):
        """
            helper function to group mpi processes based on their functionality
        """
        if self._intra_comm.Get_rank() == self._receiver_group_ranks[0]:  
            self._receiver_intra_comm = self._setup_mpi_groups_including_ranks(self._receiver_group_ranks)

        elif self._intra_comm.Get_rank() == self._sender_group_ranks[0]:
            self._sender_intra_comm = self._setup_mpi_groups_including_ranks(self._sender_group_ranks)

        elif self._intra_comm.Get_rank() in self._transformer_group_ranks:
            self._transformer_intra_comm = self._setup_mpi_groups_including_ranks(self._transformer_group_ranks)
    
    def _get_mpi_shared_memory_buffer(self, buffer_size, comm, buffer_type):
        """
        Creates shared memory buffer for MPI One-sided-Communication.
        This is wrapper to buffer manager function which creates the mpi
        shared memory buffer.
        """
        # create an MPI shared memory buffer
        self._interscalehub_buffer = \
            self._data_buffer_manager.create_mpi_shared_memory_buffer(
                buffer_size,
                comm,
                buffer_type)
        return self._interscalehub_buffer

    def _data_channel_setup(self):
        """
        Open ports and register connection details.
        Accept connection on ports and create INTER communicators.

        MVP: register = write port details to file.
        MVP: Two connections
            - input = incoming simulation data
            - output = outgoing simulation data
        """
        # NOTE: create port files and make connection
        # In Demo example: producer/Consumer are inherited from mpi_io_extern,
        # and then they are started as threads which then call mpi_io_extern run() method
        # which then calls make_connection() method
        if self._intra_comm.Get_rank() in self._receiver_group_ranks:  
            self._receiver_inter_comm, self._input_port = self._intercomm_manager.open_port_accept_connection(
                direction=DATA_EXCHANGE_DIRECTION(self._direction).name,
                intercomm_type=INTERCOMM_TYPE.RECEIVER.name)
            # self._sender_inter_comm = None

        elif self._intra_comm.Get_rank() in self._sender_group_ranks:
            self._sender_inter_comm, self._output_port = self._intercomm_manager.open_port_accept_connection(
                direction=DATA_EXCHANGE_DIRECTION(self._direction).name,
                intercomm_type=INTERCOMM_TYPE.SENDER.name)
            # self._receiver_inter_comm = None
    
    def _setup_mpi_groups_excluding_ranks(self, ranks_to_exclude):
        """ 
            creates an MPI group communicator by reordering an existing group and taking
            only unlisted members

            Parameters
            ----------

            ranks_to_exclud: list
                list of ranks to be excluded from the group
        """

        # NOTE be careful to not use group_comm on processes which are not part
        # of the gorup
        mpi_group = self._intra_comm.group.Excl(ranks_to_exclude)
        group_comm = self._intra_comm.Create_group(mpi_group)
        info_log_message(self._my_rank,
                         self._logger,
                         msg=f"MPI group created with size: {group_comm.Get_size()}, "
                         f"ranks: {ranks_to_exclude}")
        return group_comm
    
    def _setup_mpi_groups_including_ranks(self, ranks_to_include):
        """ 
            creates an MPI group communicator by reordering an existing group and taking
            only listed members

            Parameters
            ----------
            ranks_to_include: list
                list of ranks to be included in the group
        """
        
        # NOTE be careful to not use group_comm on processes which are not part
        # of the gorup
        mpi_group = self._intra_comm.group.Incl(ranks_to_include)
        group_comm = self._intra_comm.Create_group(mpi_group)
        info_log_message(self._my_rank,
                         self._logger,
                         msg=f"MPI group created with size: {group_comm.Get_size()}, "
                         f"ranks: {ranks_to_include}")
        return group_comm
    
    @abstractmethod
    def start(self):
        """
        1) init pivot objects depending on the use case (direction)
        2) start pivot with INTRA communicator (M:N mapping)
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """
        Receive stop command.
        concludea the pivot operations and stop exchanging the data.
        
        TODO: add error handling and fail checks
        """
        raise NotImplementedError
        
    def _close_data_channels(self):
        info_log_message(self._my_rank,
                         self._logger,
                         "Stop InterscaleHub and disconnect...")
        # if self._intra_comm.Get_rank() == 0:
        if self._intra_comm.Get_rank() in self._sender_group_ranks:
            self._intercomm_manager.close_and_finalize(self._sender_inter_comm, self._output_port)
        elif self._intra_comm.Get_rank() in self._receiver_group_ranks:
            self._intercomm_manager.close_and_finalize(self._receiver_inter_comm, self._input_port)
