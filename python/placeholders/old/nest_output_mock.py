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



# TODO: proper mock and the real simulation


import numpy as np
from mpi4py import MPI

import os
import sys
import time


class NestOutput:
    """
    TODO: implement a proper mock and use the real simulation
    """
    __path = None
    __min_delay = None
    __comm = None #intercomm
    __port = None
    
    def __init__(self,path, min_delay):
        self.__path = path
        self.__min_delay = min_delay
    
    def get_min_delay():
        return self.__min_delay
    
    def get_connection_details(self):
        # Init connection from file connection
        print(self.__path)
        print("Nest Output: requesting connection details");sys.stdout.flush()

        while not os.path.exists(self.__path):
            print ("Port file not found yet, retry in 1 second")
            time.sleep(1)
        fport = open(self.__path, "r")
        self.__port = fport.readline()
        fport.close()
    
    def connect_to_hub(self):
        print('Nest Output: connecting to '+port);sys.stdout.flush()
        self.__comm = MPI.COMM_WORLD.Connect(self.__port)
        print('Nest Output: connected to '+port);sys.stdout.flush()

    def simulate(self):
        '''
        simulate spike detector output for testing the nest to tvb translator input
        path = the path to the file for the connections
        min_delay = the time of one simulation
        TODO: copied from Lionels tests. change to fit the MVP-mock
        '''
        starting = 0.0 # the begging of each time of synchronization
        check = np.empty(1,dtype='b')
        status_ = MPI.Status() # status of the different message
        while True:
            # wait until the translator accept the connections
            self.__comm.Send([np.array([True],dtype='b'), 1, MPI.CXX_BOOL], dest=0, tag=0)
            self.__comm.Recv([check, 1, MPI.CXX_BOOL], source=MPI.ANY_SOURCE, tag=0,status=status_)
            # create random data
            size= np.random.randint(0,1000)
            times = starting+np.random.rand(size)*(self.__min_delay-0.2)
            times = np.around(np.sort(np.array(times)),decimals=1)
            id_neurons = np.random.randint(0,10,size)
            id_detector = np.random.randint(0,10,size)
            data = np.ascontiguousarray(np.swapaxes([id_detector,id_neurons,times],0,1),dtype='d')
            # send data one by one like spike generator
            self.__comm.Send([np.array([size*3],dtype='i'),1, MPI.INT], dest=status_.Get_source(), tag=0)
            self.__comm.Send([data,size*3, MPI.DOUBLE], dest=status_.Get_source(), tag=0)
            # ending the actual run
            self.__comm.Send([np.array([True],dtype='b'), 1, MPI.CXX_BOOL], dest=0, tag=1)
            #print result and go to the next run
            print("Nest Output : ",self.__comm.Get_rank(),size);sys.stdout.flush()
            starting+=self.__min_delay
            if starting > 10000:
                break
        # closing the connection at this end
        print("Nest Output : ending" );sys.stdout.flush()
        # send the signal for end the translation
        self.__comm.Send([np.array([True], dtype='b'), 1, MPI.CXX_BOOL], dest=0, tag=2)
        print("Nest Output : ending" );sys.stdout.flush()
        self.__comm.Disconnect()
        MPI.Close_port(port)
        MPI.Finalize()
        print('Nest Output : exit');sys.stdout.flush()
    

