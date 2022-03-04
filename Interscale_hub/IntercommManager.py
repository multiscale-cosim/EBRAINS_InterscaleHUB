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

from mpi4py import MPI
import logging
import sys
import pathlib

class IntercommManager:
    '''
    NOTE: use the implementation from Muhammad Fahad's design.
    '''
    def __init__(self, comm, root):
        '''
        General MPI Server-Client connection.

        :param comm: the INTRA communicator of the calling application ('server') which opens and accepts the connection
        :param root: the root rank on which the 'main' connection before broadcast in done
        '''
        self.__comm = comm
        self.__root = root
        self.__info = MPI.INFO_NULL
        
        # TODO: logger placeholder for testing
        self.__logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)
        self.__logger.setLevel(logging.DEBUG)
    
    def open_port_accept_connection(self, paths):
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
        #if self.__comm.Get_rank() == self.__root:
        # Write file configuration of the port
        port = MPI.Open_port(self.__info)
        for path in paths:
            fport = open(path, "w+")
            fport.write(port)
            fport.close()
            pathlib.Path(path + '.unlock').touch()
                # self.__logger.info("Port opened and file created:" + path +
                #             "on rank" + str(self.__comm.Get_rank()))
        #else:
        #    port = None
        #port = self.__comm.bcast(port, self.__root) # avoid issues with mpi rank information.
        self.__logger.info('Rank ' + str(self.__comm.Get_rank()) + ' accepting connection on: ' + port)
        inter_comm = comm.Accept(port, self.__info, root) 
        self.__logger.info('Simulation client connected to' + str(inter_comm.Get_rank()))
        
        return inter_comm, port


    def close_and_finalize(self, inter_comm, port):
        
        inter_comm.Disconnect()
        MPI.Close_port(port) 
        self.__logger.info('Successfully disconnected and closed port')
        # Finalize not needed in mpi4py
        # source:  https://mpi4py.readthedocs.io/en/stable/overview.html
        # MPI.Finalize()
