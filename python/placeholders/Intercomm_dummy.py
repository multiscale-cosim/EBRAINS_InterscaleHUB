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


# TODO: use the abstrat class implementation from Muhammad Fahad's design.
# TODO: oop: create class, store port info in file (hardcoded)

from mpi4py import MPI
import pathlib
import sys


def open_port_accept_connection(comm, root, info, path_to_files):
    '''
    General MPI Server-Client connection.
    Opens a port and writes the details to file.
    
    In some MPI implementations, information about the rank is encoded in the port infos.
    Therefore only rank 0 opens the port and broadcasts the relevant info to all other ranks.
    This is necessary to avoid conflicts in the port file -> contains MPI-Rank infos
    
    So a M:N connection between two MPI applications is possible.

    :param comm: the INTRA communicator of the calling application ('server') which opens and accepts the connection
    :param root: the root rank on which the 'main' connection before broadcast in done
    :param info: MPI info object
    :param path_to_files: location of the files 
    
    :return intra_comm: the newly created intra communicator between the two applications
    :return port: the port information, needed to properly close the connection after the job
    '''
    if comm.Get_rank() == root:
        port = MPI.Open_port(info)
        fport = open(path_to_files, "w+")
        fport.write(port)
        fport.close()
        pathlib.Path(path_to_files+'.unlock').touch()
    else:
        port = None
    # control print to console.
    print("InterscaleHub\n -- port opened and file created:", path_to_files,
          "\n -- I'm rank",comm.Get_rank())
    sys.stdout.flush()
    
    port = comm.bcast(port,root)
    print('InterscaleHub: Rank ' + str(comm.Get_rank()) + ' accepting connection on: ' + port)
    intra_comm = comm.Accept(port, info, root) 
    print('InterscaleHub: Simulation client connected to' + str(intra_comm.Get_rank()))
    
    return intra_comm, port


def close_and_finalize(port_send, port_receive, logger_master):
    # close port
    MPI.Close_port(port_send)
    MPI.Close_port(port_receive)
    logger_master.info('close communicator')
    # Finalize not needed in mpi4py
    # source:  https://mpi4py.readthedocs.io/en/stable/overview.html
    # MPI.Finalize()



# TODO: check if open port and accept connection can be split in two steps (NOTE: solve bcast issue)
def _accept_connection(comm, port, root, info):
    '''
    accept connection after call
    broadcast port info, accept connection on all ranks!
    necessary to avoid conflicts in the port file -> contains MPI-Rank infos
    
    :param comm: the INTRA communicator of the calling application ('server') which opens and accepts the connection
    :param port: the port information
    :param root: the root rank on which the 'main' connection before broadcast in done
    :param info: MPI info object
    
    :return intra_comm: the newly created intra communicator between the two applications
    :return port: the port information, needed to properly close the connection after the job
    '''
    port = comm.bcast(port,root)
    #logger_master.info('Transformer: Rank ' + str(comm.Get_rank()) + ' accepting connection on: ' + port)
    intra_comm = comm.Accept(port, info, root) 
    #logger_master.info('Transformer: Simulation client connected to' + str(intra_comm.Get_rank()))
    
    return intra_comm, port
