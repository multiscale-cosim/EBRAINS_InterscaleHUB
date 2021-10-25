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


# TODO: multiplexing, exact pivot and transformer tasks, multiple transformers?, second pivot?

from mpi4py import MPI
import numpy as np
import copy

from placeholders.parameter import Parameter
import placeholders.Intercomm_dummy as ic
import Interscale_hub.pivot as piv


class InterscaleHub:
    '''
    InterscaleHub for connecting the (two) simulators in cosim.
    
    MVP: init, start, stop functionality
    MVP: NEST-TVB cosim showcase
    '''
    
    def __init__(self, param, direction):
        '''
        Init params, create buffer, open ports, accept connections
        '''
        # 1) param stuff
        self._init_params(param,direction)
        
        
        # 2) create buffer in self.__databuffer
        self._create_buffer()
        
        # 3) open ports and xreate intercomms
        # self.__input_comm, self.__input_port
        # and 
        # self.__output_comm, self.__output_port 
        self._open_ports_accept_connections()


    def start(self):
        '''
        InterscaleHub:
        1) receive
        2) pivot data from buffer (
        '''
        # start -> pivot, transform, analysis, transform, pivot
        # pivot = split receiving ranks and transformer/sending ranks
        # 
        # stop -> loop with either interrupt or waiting for normal end/stop.
        if self.__direction == 1:
            pivot = piv.NestTvbPivot(
                self.__param, 
                self.__input_comm, 
                self.__output_comm, 
                self.__databuffer)
        elif self.__direction == 2:
            pivot = piv.TvbNestPivot(
                self.__param, 
                self.__input_comm, 
                self.__output_comm, 
                self.__databuffer)
            
        if self.__comm.Get_rank() == 0: # Receiver from input sim, rank 0
            pivot._receive()
        else: #  Science/analyse and sender to TVB, rank 1-x
            pivot._send()
        

    def stop(self):
        
        # TODO: see above -> set global variable and interrupt loop?
        # Disconnect and close ports
        print('InterscaleHUB: disconnect communicators and close ports...')
        self.__input_comm.Disconnect()
        self.__output_comm.Disconnect()
        MPI.Close_port(self.__input_port)
        MPI.Close_port(self.__output_port) 

    
    def _create_buffer(self):
        '''
        Create shared memory buffer. MPI One-sided-Communication.
        MVP: datasize ist MPI.Double, buffersize is set with param init
        '''
        if self.__comm.Get_rank() == 0:
            bufbytes = self.__datasize * self.__buffersize
        else: 
            bufbytes= 0
        # rank 0: create the shared block
        # rank 1-x: get a handle to it
        win = MPI.Win.Allocate_shared(bufbytes, self.__datasize, comm=self.__comm)
        buf, self.__datasize = win.Shared_query(0)
        assert self.__datasize == MPI.DOUBLE.Get_size()
        # create a 1D numpy array (buffer) whose data points to the shared mem
        self.__databuffer = np.ndarray(buffer=buf, dtype='d', shape=(self.__buffersize,))
        
    
    def _open_ports_accept_connections(self):
        '''
        Open port and 'register connection details'
        i.e. write them to file.
        Accept connection on ports and create INTER communicators.
        
        input = incoming simulation data
        output = outgoing simulation data
        '''
        self.__input_comm, self.__input_port = ic.open_port_accept_connection(self.__comm, self.__root, self.__info, self.__input_path)
        self.__output_comm, self.__output_port = ic.open_port_accept_connection(self.__comm, self.__root, self.__info, self.__output_path)
        
        
    def _init_params(self, p, direction):
        '''
        Init MPI, buffer parameter and science parameter. 
        The science parameter are taken from the TVB-NEST implementation
        in the co-sim github (refactored usecase from Lionel).
        # TODO: MPI and buffer init needs to be here, but all parameter are passed through by the 
        Launcher->Orchestrator->AppCompanion
        '''
        # MPI
        self.__comm = MPI.COMM_WORLD  # INTRA communicator
        self.__info = MPI.INFO_NULL
        self.__root = 0 # hardcoded!
        
        # Buffer
        # TODO: needs to be a global cosim setting. more information needed!
        max_events = 1000000 # max. expected number of events per step
        self.__datasize = MPI.DOUBLE.Get_size()
        
        # science parameter
        self.__direction = direction
        # TODO: used as global param dict for now and passed all the way to pivot._analyse()
        # TODO: align this with the rest of the implementation and below param init
        self.__param = p.get_param(direction)
        
        # nest to tvb
        if direction == 1:
            self.__buffersize = max_events * 3 # 3 doubles per event
            self.__input_path = p.get_nest_path()
            self.__output_path = p.get_tvb_path()
            
            self.synch=self.__param['synch']                # time of synchronization between 2 run
            self.dt=self.__param['resolution']              # the resolution of the integrator
            self.shape = (int(self.synch/self.dt),1) # the shape of the buffer/histogram
            self.hist = np.zeros(self.shape)         # the initialisation of the histogram
            self.width = int(self.__param['width']/self.__param['resolution']) # the window of the average in time
            self.synch = self.__param['synch']                          # synchronize time between simulator
            self.buffer = np.zeros((self.width,))                  #initialisation/ previous result for a good result
            self.coeff = 1 / ( self.__param['nb_neurons'] * self.__param['resolution'] ) # for the mean firing rate in in KHZ
        # tvb to nest
        elif direction == 2:
            self.__buffersize = 2 + max_events # 2 doubles: [start_time,end_time] of simulation step
            self.__input_path = p.get_tvb_path()
            self.__output_path = p.get_nest_path()
            
            self.percentage_shared = self.__param['percentage_shared']  # percentage of shared rate between neurons
            self.nb_spike_generator = self.__param['nb_spike_generator']         # number of spike generator
            self.nb_synapse = self.__param['nb_synapses']               # number of synapses by neurons
            self.function_translation = self.__param['function_select'] # choose the function for the translation
    
    
    def _temp_protocol_translation():
        '''
        TODO: temporary translation of protocol behaviour
        from current NEST i/o to cosim 
        and from current TVB i/o to cosim.
        '''
        raise NotImplementedError
