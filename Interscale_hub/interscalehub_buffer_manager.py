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

from EBRAINS_InterscaleHUB.refactored_modular.interscalehub_enums import DATA_BUFFER_STATES

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
        
        self.__mpi_shared_memory_buffer = None
        self.__logger.debug("initialized")

    @property
    def mpi_shared_memory_buffer(self): return self.__mpi_shared_memory_buffer

    def set_ready_at(self, index):
        self.__mpi_shared_memory_buffer[index] = DATA_BUFFER_STATES.READY

    def set_header_at(self, index):
        self.__mpi_shared_memory_buffer[index] = DATA_BUFFER_STATES.HEADER

    def set_custom_value_at(self, index, value):
        self.__mpi_shared_memory_buffer[index] = value

    def get_at(self, index):
        return self.mpi_shared_memory_buffer[index]

    def get_from(self, starting_index):
        return self.mpi_shared_memory_buffer[starting_index:]

    def get_upto(self, index):
        return self.mpi_shared_memory_buffer[:index]

    def get_from_range(self, start, end):
        return self.mpi_shared_memory_buffer[start:end]

    def create_mpi_shared_memory_buffer(self, buffer_size, intra_comm):
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
        mpi_window = MPI.Win.Allocate_shared(buffer_bytes, desired_data_size, intra_comm)
        # get the address for load/store access to window segment
        self.__logger.debug("getting buffer and data (unit) size")
        shared_buffer, actual_data_size = mpi_window.Shared_query(0)
        # check if the resulting unit (data) size is different
        if actual_data_size != desired_data_size:
            # Case a: the datasize is mismatching
            # log the exception and raise a RuntimeError to terminate with error
            try:
                raise RuntimeError
            except RuntimeError:
                self.__logger.exception(
                    f"desrired datasize: {desired_data_size} is not equal to "
                    f"actual_datasize: {actual_data_size}")
                # re-raise the exception to terminate
                raise RuntimeError
        else:
            # Case b: everything went good,
            # now create a 1D numpy array (buffer) whose data points to the
            # shared memory
            self.__mpi_shared_memory_buffer = np.ndarray(
                buffer=shared_buffer,
                dtype='d',
                shape=(buffer_size,))

            return self.mpi_shared_memory_buffer
