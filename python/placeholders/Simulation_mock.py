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


import numpy as np
from mpi4py import MPI

import os
import time
import logging
import sys

class SimulationMock:
    """
    Simulator mock
    NOTE: not used yet
    TODO: extend to a proper mock 
    TODO: use the real simulators 
    """
    def __init__(self,path):
        '''
        init output simulation
        '''
        self.__path = path
        self.__min_delay = 0
        
    def get_min_delay(self):
        return self.__min_delay


    def get_connection_details(self):
        # Get connection info from file
        self.__logger.info("requesting connection details from {}".format(self.__path))
        while not os.path.exists(self.__path):
            self.__logger.info ("Port file not found yet, retry in 1 second")
            time.sleep(1)
        fport = open(self.__path, "r")
        self.__port = fport.readline()
        fport.close()


    def connect_to_hub(self):
        self.__logger.info("Connecting to {}".format(self.__port))
        self.__comm = MPI.COMM_WORLD.Connect(self.__port)
        self.__logger.info("Connected to {}".format(self.__port))


    def disconnect_from_hub(self):
        self.__comm.Disconnect()
        MPI.Close_port(self.__port)
        self.__logger.info("Disconnected and port closed")


    def simulate(self):
        '''
        mock output
        '''
        raise NotImplementedError


    def receive(self):
        '''
        mock input
        '''
        raise NotImplementedError


class NestMock:
    def __init__(self,path):
        '''
        init Nest mock simulation
        '''
        
        # TODO: logger placeholder for testing
        self.__logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)
        self.__logger.setLevel(logging.DEBUG)
        self.__logger.info("Initialise...")
        
        self.__path = path
        self.__min_delay = 100.0 # NOTE: hardcoded
        
        
    def get_min_delay(self):
        return self.__min_delay


    def get_connection_details(self):
        # Get connection info from file
        self.__logger.info("requesting connection details from {}".format(self.__path))
        while not os.path.exists(self.__path):
            self.__logger.info ("Port file not found yet, retry in 1 second")
            time.sleep(1)
        fport = open(self.__path, "r")
        self.__port = fport.readline()
        fport.close()


    def connect_to_hub(self):
        self.__logger.info("Connecting to {}".format(self.__port))
        self.__comm = MPI.COMM_WORLD.Connect(self.__port)
        self.__logger.info("Connected to {}".format(self.__port))


    def disconnect_from_hub(self):
        self.__comm.Disconnect()
        MPI.Close_port(self.__port)
        self.__logger.info("Disconnected and port closed")
    
    
    def simulate(self):
        #super().simulate()
        # NOTE: the mock NEST OUTPUT simulation
        starting = 0.0 # the begging of each time of synchronization
        status_ = MPI.Status() # status of the different message
        check = np.empty(1,dtype='b') # needed?
        while True:
            self.__logger.info("NEST_OUTPUT: wait for ready signal")
            # NOTE: seems like a handshake..needed?
            self.__comm.Send([np.array([True],dtype='b'), 1, MPI.CXX_BOOL], dest=0, tag=0)
            self.__comm.Recv([check, 1, MPI.CXX_BOOL], source=MPI.ANY_SOURCE, tag=0,status=status_)
            
            self.__logger.info("NEST_OUTPUT: simulate next step...")
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
            # results and go to the next run
            self.__logger.info("NEST_OUTPUT: Rank {} sent data of size {}".format(self.__comm.Get_rank(),size))
            
            # ending the simulation step
            self.__comm.Send([np.array([True],dtype='b'), 1, MPI.CXX_BOOL], dest=0, tag=1)
            starting+=self.__min_delay
            if starting > 10000:
                break
        # end of nest out
        self.__comm.Send([np.array([True], dtype='b'), 1, MPI.CXX_BOOL], dest=0, tag=2)
        self.__logger.info("NEST_OUTPUT: end of simulation" )


    def receive(self):
        status_ = MPI.Status() # status of the different message
        #NOTE: hardcoded...
        ids=np.arange(0,10,1) # random id of spike detector
        while(True):
            # Send start simulation
            self.__logger.info("NEST_INPUT: send ready to receive next step")
            self.__comm.Send([np.array([True], dtype='b'), MPI.CXX_BOOL], dest=1, tag=0)
            self.__logger.info("NEST_INPUT: send spike detector info")
            self.__comm.Send([np.array(10,dtype='i'), MPI.INT], dest=1, tag=0)
            self.__comm.Send([np.array(ids,dtype='i'), MPI.INT], dest=1, tag=0)
            
            # receive the number of spikes
            size=np.empty(11,dtype='i')
            self.__comm.Recv([size,11, MPI.INT], source=1, tag=ids[0],status=status_)
            self.__logger.info("NEST_INPUT ({}):receive size : {}".format(ids[0],size))
            
            # receive the spikes for updating the spike detector
            data = np.empty(size[0], dtype='d')
            self.__comm.Recv([data,size[0], MPI.DOUBLE],source=1,tag=ids[0],status=status_)
            self.__logger.info ("NEST_INPUT ({}):receive size : {}".format(ids[0],np.sum(data)))
            
            # send end of sim step
            # NOTE: why?
            self.__logger.info("NEST_INPUT: send end of simulation step")
            self.__comm.Send([np.array([True], dtype='b'), MPI.CXX_BOOL], dest=1, tag=1)

            if np.any(data > 10000):
                break

        # closing the connection at this end
        self.__logger.info("NEST_INPUT: end of simulation")
        self.__comm.Send([np.array([True], dtype='b'), MPI.CXX_BOOL], dest=1, tag=2)



class TvbMock:
    def __init__(self,path):
        '''
        init tvb mock simulation
        '''
        
        # TODO: logger placeholder for testing
        self.__logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)
        self.__logger.setLevel(logging.DEBUG)
        self.__logger.info("Initialise...")
        
        self.__path = path
        self.__min_delay = 100.0 # NOTE: hardcoded


    def get_min_delay(self):
        return self.__min_delay


    def get_connection_details(self):
        # Get connection info from file
        self.__logger.info("requesting connection details from {}".format(self.__path))
        while not os.path.exists(self.__path):
            self.__logger.info("Port file not found yet, retry in 1 second")
            time.sleep(1)
        fport = open(self.__path, "r")
        self.__port = fport.readline()
        fport.close()


    def connect_to_hub(self):
        self.__logger.info("Connecting to {}".format(self.__port))
        self.__comm = MPI.COMM_WORLD.Connect(self.__port)
        self.__logger.info("Connected to {}".format(self.__port))


    def disconnect_from_hub(self):
        self.__comm.Disconnect()
        MPI.Close_port(self.__port)
        self.__logger.info("Disconnected and port closed")
    
    
    def simulate(self):
        self.__logger.info("TVB_OUTPUT: start of simulation")
        starting = 0.0 # the beginning of each time of synchronization
        status_ = MPI.Status() # status of the different message
        while True:
            # wait for InterscaleHub ready signal
            accept = False
            self.__logger.info("TVB_OUTPUT: wait for ready signal")
            while not accept:
                req = self.__comm.irecv(source=0,tag=0)
                accept = req.wait(status_)
            self.__logger.info("TVB_OUTPUT: simulate next step")
            # TODO: the irecv above is from source 0, so 'source = status_.Get_source()' will be 0.
            # TODO: If the goal was to send from multiple TVB ranks to multiple sources, this needs some work.
            # TODO: essentially this would be an M:N coupling then
            source = status_.Get_source() # the id of the excepted source
            # create random data
            size= int(self.__min_delay/0.1 )
            rate = np.random.rand(size)*400
            data = np.ascontiguousarray(rate,dtype='d') # format the rate for sending
            shape = np.array(data.shape[0],dtype='i') # size of data
            times = np.array([starting,starting+self.__min_delay],dtype='d') # time of stating and ending step
            
            self.__logger.info("TVB_OUTPUT: sending timestep {}".format(times))
            self.__comm.Send([times,MPI.DOUBLE],dest=source,tag=0)
            
            self.__logger.info("TVB_OUTPUT: sending shape : {}".format(shape))
            self.__comm.Send([shape,MPI.INT],dest=source,tag=0)
            
            self.__logger.info("TVB_OUTPUT: sending data : {}".format(np.sum(np.sum(data))))
            self.__comm.Send([data, MPI.DOUBLE], dest=source, tag=0)
            
            starting+=self.__min_delay
            if starting > 10000:
                break
        
        accept = False
        self.__logger.info("TVB_OUTPUT: ending...sending last timestep")
        while not accept:
            req = self.__comm.irecv(source=0,tag=0)
            accept = req.wait(status_)

        self.__logger.info("TVB_OUTPUT: sending timestep : {}".format(times))
        self.__comm.Send([times, MPI.DOUBLE], dest=0, tag=1)
        
        self.__logger.info("TVB_OUTPUT: end of simulation" )
        
    
    def receive(self):
        self.__logger.info("TVB_INPUT: start receiving...")
        status_ = MPI.Status() # status of the different message
        while(True):
            self.__logger.info("TVB_INPUT: ready to receive next step")
            # send to the translator, I want the next part
            req = self.__comm.isend(True, dest=1, tag=0)
            req.wait()
            
            times=np.empty(2,dtype='d')
            self.__comm.Recv([times, MPI.FLOAT], source=1, tag=0)
            
            size=np.empty(1,dtype='i')
            self.__comm.Recv([size, MPI.INT], source=1, tag=0)
            
            
            rates = np.empty(size, dtype='d')
            self.__comm.Recv([rates,size, MPI.DOUBLE],source=1,tag=MPI.ANY_TAG,status=status_)
            
            # summary of the data
            if status_.Get_tag() == 0:
                self.__logger.info("TVB_INPUT:{} received timestep {} and rates {}"
                                   .format(self.__comm.Get_rank(),times,np.sum(rates)))
            else:
                break
            if times[1] >9900:
                break
        # end of tvb in
        req = self.__comm.isend(True, dest=1, tag=1)
        req.wait()
        self.__logger.info("TVB_INPUT: received end signal")
