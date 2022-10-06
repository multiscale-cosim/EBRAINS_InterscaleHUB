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
#
# ------------------------------------------------------------------------------ 

import os
from mpi4py import MPI
import pathlib

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_RichEndpoint.Application_Companion.common_enums import INTEGRATED_INTERSCALEHUB_APPLICATION as INTERSCALE_HUB


class IntercommManager:
    '''
    NOTE: use the implementation from Muhammad Fahad's design.
    '''
    def __init__(self, comm, root, configurations_manager, log_settings):
        '''
        General MPI Server-Client connection.

        :param comm: the INTRA communicator of the calling application ('server') which opens and accepts the connection
        :param root: the root rank on which the 'main' connection before broadcast in done
        '''
        self.__comm = comm
        self.__root = root
        self.__info = MPI.INFO_NULL
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="IntercommManager",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self.__logger.info("Initialised")
    
    def open_port_accept_connection(self, direction, intercomm_type):
        '''
        Opens a port and writes the details to file.
        Accepts connection on the port.
        
        In some MPI implementations, information about the rank is encoded in the port infos.
        Therefore only rank 0 opens the port and broadcasts the relevant info to all other ranks.
        This is necessary to avoid conflicts in the port file -> contains MPI-Rank infos
        
        TODO: split up into 'open port' and 'accept connection'
        
        :param path_to_files: location of the files 
        :return inter_comm: newly created intercommunicator
        :return port: specific port information
        '''
        comm = MPI.COMM_SELF
        root = 0
        port = MPI.Open_port(self.__info)
        self.__logger.info(f'Rank {str(self.__comm.Get_rank())} accepting '
                           f'connection on: {port}')

        # send port info to Application Manager as a response to init command
        interscalehub_endpoint_address = self.__prepare_endpoint_address_response(
            direction, port, intercomm_type)
        print(f'{interscalehub_endpoint_address}')

        inter_comm = comm.Accept(port, self.__info, root)
        self.__logger.info(f'Simulation client is connected to '
                           f'{inter_comm.Get_rank()}')
        return inter_comm, port

    def __prepare_endpoint_address_response(self, direction, port, intercomm_type):
        '''
        helper function to format the port name to be send as a response
        '''
        interscalehub_endpoint_address = \
            {
                INTERSCALE_HUB.PID.name: os.getpid(),
                INTERSCALE_HUB.DATA_EXCHANGE_DIRECTION.name: direction,
                INTERSCALE_HUB.MPI_CONNECTION_INFO.name: port,
                INTERSCALE_HUB.INTERCOMM_TYPE.name: intercomm_type
            }
        return interscalehub_endpoint_address
    
    def close_and_finalize(self, inter_comm, port):
        inter_comm.Disconnect()
        MPI.Close_port(port) 
        self.__logger.info('Successfully disconnected and closed port')

        # Finalize not needed in mpi4py
        # source:  https://mpi4py.readthedocs.io/en/stable/overview.html
        # MPI.Finalize()
