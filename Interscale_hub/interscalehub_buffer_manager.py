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
from mpi4py import MPI
import numpy as np

from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_BUFFER_STATES, DATA_BUFFER_TYPES

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class MetaInterscaleHubBuffer(type):
    """This metaclass ensures there exists only one instance of
    InterscaleHubBuffer class. It prevents the side-effects such as
    the creation of multiple buffers.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Case: First time instantiation.
            cls._instances[cls] = super(MetaInterscaleHubBuffer,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class InterscaleHubBufferManager(metaclass=MetaInterscaleHubBuffer):
    """
    InterscaleHub data buffer which stores the in-transit data.
    """
    def __init__(self, configurations_manager, log_settings):
        self.__logger = configurations_manager.load_log_configurations(
                    name="InterscaleHub -- Buffer",
                    log_configurations=log_settings,
                    target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        self.__databuffer_input = None
        self.__logger.debug("initialized")

    @property
    def databuffer_input(self): return self.__databuffer_input

    def get_buffer(self, buffer_type):
        if buffer_type == DATA_BUFFER_TYPES.INPUT:
            return self.databuffer_input
        elif buffer_type == DATA_BUFFER_TYPES.OUTPUT:
            return self.databuffer_output
        else:
            self.__terminate_with_error(f"unknown data buffer type. {buffer_type}")

    # NOTE be careful to not to include the HEADER and READY indecies when
    # specifying the indices to fetch the data
    # TODO fix the HEADER and READY flags to specific indexes e.g. first two indecies
    def set_ready_state_at(self, index, state, buffer_type):
        shared_memory_buffer =  self.get_buffer(buffer_type)
        shared_memory_buffer[index] = state

    def set_header_at(self, index, header, buffer_type):
        """Sets header to the given value at a given index"""
        shared_memory_buffer =  self.get_buffer(buffer_type)
        shared_memory_buffer[index] = header

    def set_custom_value_at(self, index, value, buffer_type):
        shared_memory_buffer =  self.get_buffer(buffer_type)
        shared_memory_buffer[index] = value

    def get_at(self, index, buffer_type):
        shared_memory_buffer =  self.get_buffer(buffer_type)
        return shared_memory_buffer[index]

    def get_from(self, starting_index, buffer_type):
        shared_memory_buffer =  self.get_buffer(buffer_type)
        return shared_memory_buffer[starting_index:]

    def get_upto(self, end_index, buffer_type):
        shared_memory_buffer =  self.get_buffer(buffer_type)
        return shared_memory_buffer[:end_index]

    def get_from_range(self, start, end, buffer_type):
        shared_memory_buffer =  self.get_buffer(buffer_type)
        return shared_memory_buffer[start:end]

    def create_mpi_shared_memory_buffer(self, buffer_size, intra_comm, buffer_type):
        # set unit (data) size for the memory buffer
        desired_data_size = MPI.DOUBLE.Get_size()
        
        # Case a: if rank 0 then create the shared block
        if intra_comm.Get_rank() == 0:
            buffer_bytes = desired_data_size * buffer_size
        # Case b: otherwise if rank 1-x then get a handle to it
        else:
            buffer_bytes = 0

        # create an MPI Window object that allocates memory
        self.__logger.debug("creating shared memory window")
        mpi_window = MPI.Win.Allocate_shared(buffer_bytes, desired_data_size, comm=intra_comm)
        # get the address for load/store access to window segment
        self.__logger.debug("getting buffer and data (unit) size")
        shared_buffer, actual_data_size = mpi_window.Shared_query(0)
        # check if the resulting unit (data) size is different
        if actual_data_size != desired_data_size:
            # Case a: the datasize is mismatching
            # log the exception and raise a RuntimeError to terminate with error
            self.__terminate_with_error(f"desrired datasize: {desired_data_size} "
                                        "is not equal to "
                                        f"actual_datasize: {actual_data_size}")
        # Case b: everything went good,
        # now create a 1D numpy array (buffer) whose data points to the
        # shared memory
        
        if buffer_type == DATA_BUFFER_TYPES.INPUT:
            self.__logger.debug("creating input buffer")
            self.__databuffer_input = np.ndarray(
            buffer=shared_buffer,
            dtype='d',
            shape=(buffer_size,))
            
            self.__logger.debug(f"input buffer: {self.databuffer_input}")
            return self.databuffer_input
        
        # NOTE add here if more buffer types are needed to be created
        
        else:
            self.__terminate_with_error("could not create shared memory buffer")
        
    def __terminate_with_error(self, msg):
        try:
            raise RuntimeError
        except RuntimeError:
            self.__logger.exception(msg)
            # re-raise the exception to terminate
            raise RuntimeError
