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
import sys
import time

class SimulationMock:
    """
    Abstract class: simulator mock
    Send and receive: 
    TODO: extend/implement a proper mock 
    TODO: use the real simulators ?
    TODO: add documentation and logging
    """
    
    def __init__(self,path):
        '''
        init output simulation
        '''
        self.__path = path
        self.__min_delay = 100.0 # NOTE: hardcoded
        
    def get_min_delay(self):
        return self.__min_delay
    
    def get_connection_details(self):
        # Get connection info from file
        print(self.__path)
        print(self.__name + ": requesting connection details");sys.stdout.flush()

        while not os.path.exists(self.__path):
            print ("Port file not found yet, retry in 1 second")
            time.sleep(1)
        fport = open(self.__path, "r")
        self.__port = fport.readline()
        fport.close()
    
    def connect_to_hub(self):
        print(self.__name + ": connecting to " + self.__port);sys.stdout.flush()
        self.__comm = MPI.COMM_WORLD.Connect(self.__port)
        print(self.__name + ": connected to " + self.__port);sys.stdout.flush()
        
    def disconnect_from_hub(self):
        self.__comm.Disconnect()
        MPI.Close_port(self.__port)
        print(self.__name + ": disconnected and port closed");sys.stdout.flush()

    def simulate(self):
        '''
        mock output
        '''
        starting = 0.0 # the begging of each time of synchronization
        status_ = MPI.Status() # status of the different message
        check = np.empty(1,dtype='b') # needed?
        
    def receive(self):
        '''
        mock input
        '''
        status_ = MPI.Status() # status of the different message


class NestMock(SimulationMock):
    '''
    Implementation of the abstract class
    '''
    
    def __init__(self,path, min_delay):
        super().__init__(path, min_delay)
        self.__name = "Nest Output"
        
    def get_min_delay(self):
        super().get_min_delay()
        
    def get_connection_details(self):
        super().get_connection_details()
        
    def connect_to_hub(self):
        super().connect_to_hub()
    
    def disconnect_from_hub(self):
        super().disconnect_from_hub()
        
    def simulate(self):
        super().simulate()
        # NOTE: the mock NEST OUTPUT simulation
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
            print(self.__name, "-- Rank: ",self.__comm.Get_rank(),"Size: "size);sys.stdout.flush()
            starting+=self.__min_delay
            if starting > 10000:
                break
        # end of nest out
        print(self.__name, ": ending" );sys.stdout.flush()
        self.__comm.Send([np.array([True], dtype='b'), 1, MPI.CXX_BOOL], dest=0, tag=2)
        print(self.__name, ": sent end signal" );sys.stdout.flush()
        
    def receive(self):
        super().receive()
        # TODO: check this (tvb-nest direction)
        ids=np.arange(0,10,1) # random id of spike detector
        print(ids);sys.stdout.flush()
        while(True):
            # Send start simulation
            comm.Send([np.array([True], dtype='b'), MPI.CXX_BOOL], dest=1, tag=0)
            # NOTE: hardcoded...
            comm.Send([np.array(10,dtype='i'), MPI.INT], dest=1, tag=0)
            # send ID of spike generator
            comm.Send([np.array(ids,dtype='i'), MPI.INT], dest=1, tag=0)
            # receive the number of spikes for updating the spike detector
            size=np.empty(11,dtype='i')
            comm.Recv([size,11, MPI.INT], source=1, tag=ids[0],status=status_)
            print ("Nest_Input (" + str(ids[0]) + ") :receive size : " + str(size));sys.stdout.flush()
            # receive the spikes for updating the spike detector
            data = np.empty(size[0], dtype='d')
            comm.Recv([data,size[0], MPI.DOUBLE],source=1,tag=ids[0],status=status_)
            print ("Nest_Input (" + str(id) + ") : " + str(np.sum(data)));sys.stdout.flush()
            # printing value and exist
            print ("Nest_Input: Before print ");sys.stdout.flush()
            if ids[0] == 0:
                print ("Nest_Input:" + str([ids[0], data,np.sum(data)]) );sys.stdout.flush()
            print ("Nest_Input: debug end of loop");sys.stdout.flush()
            #send ending the the run of the simulation
            print("Nest_Input: Debug before send");sys.stdout.flush()
            comm.Send([np.array([True], dtype='b'), MPI.CXX_BOOL], dest=1, tag=1)
            print("Nest_Input: Debug after  send");sys.stdout.flush()

            print ("Nest_Input: before break");sys.stdout.flush()
            # print ("Nest_Input: before break" + str(data > 10000));sys.stdout.flush()
            if np.any(data > 10000):
                break
            

        # closing the connection at this end
        print('Nest_Input : Disconnect')
        comm.Send([np.array([True], dtype='b'), MPI.CXX_BOOL], dest=1, tag=2)
                

class TvbMock(SimulationMock):
    '''
    Implementation of the abstract class
    '''
    def __init__(self,path, min_delay):
        super().__init__(path, min_delay)
        self.__name = "TVB Input"
        
    def get_connection_details(self):
        super().get_connection_details()
        
    def connect_to_hub(self):
        super().connect_to_hub()
    
    def disconnect_from_hub(self):
        super().disconnect_from_hub()
    
    def simulate(self):
        super().simulate()
        # TODO: check this (tvb-nest direction)
        while True:
            # wait until the translator accept the connections
            accept = False
            print("TVB_OUTPUT :wait acceptation");sys.stdout.flush()
            while not accept:
                req = comm.irecv(source=0,tag=0)
                accept = req.wait(status_)
            print("TVB_OUTPUT :accepted");sys.stdout.flush()
            # TODO: the irecv above is from source 0, so 'source = status_.Get_source()' will be 0.
            # TODO: If the goal was to send from multiple TVB ranks to multiple sources, this needs some work.
            # TODO: essentially this would be an M:N coupling then
            source = status_.Get_source() # the id of the excepted source
            # create random data
            size= int(min_delay/0.1 )
            rate = np.random.rand(size)*400
            data = np.ascontiguousarray(rate,dtype='d') # format the rate for sending
            shape = np.array(data.shape[0],dtype='i') # size of data
            times = np.array([starting,starting+min_delay],dtype='d') # time of stating and ending step
            print("TVB_OUTPUT :send time : " +str(times));sys.stdout.flush()
            comm.Send([times,MPI.DOUBLE],dest=source,tag=0)
            print("TVB_OUTPUT :send shape : " +str(shape));sys.stdout.flush()
            comm.Send([shape,MPI.INT],dest=source,tag=0)
            print("TVB_OUTPUT :send data : " +str(np.sum(np.sum(data))));sys.stdout.flush()
            print("TVB_OUTPUT :send data array : ", data.shape);sys.stdout.flush()
            comm.Send([data, MPI.DOUBLE], dest=source, tag=0)
            # print result and go to the next run
            starting+=min_delay
            if starting > 10000:
                break
        print("TVB_OUTPUT :ending" );sys.stdout.flush()
        accept = False
        print("TVB_OUTPUT :wait acceptation");sys.stdout.flush()
        while not accept:
            req = comm.irecv(source=0,tag=0)
            accept = req.wait(status_)
        print("TVB_OUTPUT :ending 2" );sys.stdout.flush()
        comm.Send([times, MPI.DOUBLE], dest=0, tag=1)
        
    
    def receive(self):
        super().receive()
        # NOTE: the mock TVB INPUT
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
                print(self.__name, ": ",self.__comm.Get_rank(),times,np.sum(rates));sys.stdout.flush()
            else:
                break
            if times[1] >9900:
                break
        # end of tvb in
        print(self.__name, ": ending" );sys.stdout.flush()
        req = self.__comm.isend(True, dest=1, tag=1)
        req.wait()
        print(self.__name, ": received end signal" );sys.stdout.flush()
