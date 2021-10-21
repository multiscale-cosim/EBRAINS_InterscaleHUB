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
# 
# Author Kim Sontheimer - k.sontheimer@fz-juelich.de


import numpy as np
from mpi4py import MPI
import os
import time

class TvbInput:
    """
    TODO: implement a proper mock and use the real simulation
    TODO: do some simple error handling for the MVP (and tests).
    """
    __path = None
    __comm = None
    __port = None
    
    def __init__(self, path):
        self.__path = path

    def get_connection_details(self):
        # Init connection from file connection
        print(self.__path)
        print("TVB INPUT: requesting connection details");sys.stdout.flush()
        while not os.path.exists(self.__path):
            print ("Port file not found yet, retry in 1 second")
            time.sleep(1)
        fport = open(self.__path, "r")
        self.__port=fport.readline()
        fport.close()
    
    def connect_to_hub(self):
        print('TVB INPUT: connecting to '+port);sys.stdout.flush()
        self.__comm = MPI.COMM_WORLD.Connect(self.__port)
        print('TVB INPUT: connected to '+port);sys.stdout.flush()
    
    def simulate(self):
        '''
        simulate the receptor of the translator for nest to TVB
        TODO: copied from Lionels tests. change to fit the MVP-mock
        '''
        status_ = MPI.Status()
        while(True):
            # send to the translator, I want the next part
            req = self.__comm.isend(True, dest=1, tag=0)
            req.wait()

            times=np.empty(2,dtype='d')
            self.__comm.Recv([times, MPI.FLOAT], source=1, tag=0)
            # get the size of the rate
            size=np.empty(1,dtype='i')
            self.__comm.Recv([size, MPI.INT], source=1, tag=0)
            # get the rate
            rates = np.empty(size, dtype='d')
            self.__comm.Recv([rates,size, MPI.DOUBLE],source=1,tag=MPI.ANY_TAG,status=status_)
            # print the summary of the data
            if status_.Get_tag() == 0:
                print("TVB INPUT :",self.__comm.Get_rank(),times,np.sum(rates));sys.stdout.flush()
            else:
                break
            if times[1] >9900:
                break
        # closing the connection at this end
        req = self.__comm.isend(True, dest=1, tag=1)
        req.wait()
        print('TVB INPUT :end');sys.stdout.flush()
        self.__comm.Disconnect()
        MPI.Close_port(port)
        print('TVB INPUT :exit');sys.stdout.flush()
        MPI.Finalize()

