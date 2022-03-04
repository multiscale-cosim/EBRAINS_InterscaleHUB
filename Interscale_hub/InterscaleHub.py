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
import numpy as np
import logging
import sys
import os
import time

from Interscale_hub.parameter import Parameter
import Interscale_hub.pivot as piv
import Interscale_hub.IntercommManager as icm


class InterscaleHub:
    '''
    InterscaleHub for connecting cosim applications (two simulators).
    MVP: Expose INIT, START, STOP functionality
    
    Init:
    - Parameter reading and initialisation
    - Buffer creation, MPI shared memory, layout depending on the parameter
    - Open MPi ports (write to file) and accept connections
    - create (two) MPI intercommunicators, one for each applications
    
    Start:
    - initialise the pivot operation
    - start receive and send (data channels)
    - TODO: multiplexing 
    - proper M:N mapping of MPI ranks in the Pivot-operation
        - How many MPI ranks on the sending simulation (M ranks)
        - How many MPI ranks on the InterscaleHub (N ranks)
        -> This contains: parallel buffer access, transformation, analysis and sending
    - M:N:O mapping -> How many MPI ranks on the receiving simulation (O ranks)
    - multiple transformers, second pivot?
    
    Stop:
    - Call stop on the pivot operation (the receiving and sending loop).
    - NOTE: This is currently not bound to the simulation, i.e. the actual simulation has stopped
    
    
    MVP: NEST-TVB cosim showcase
    '''
    def __init__(self, param, direction):
        '''
        Init params, create buffer, open ports, accept connections
        '''
        
        # TODO: logger placeholder for testing
        self.__logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)
        self.__logger.setLevel(logging.DEBUG)
        self.__logger.info("Initialise...")
        
        # 1) param stuff, create IntercommManager
        self.__logger.debug("Init Params...")
        self._init_params(param,direction)
        
        # 2) create buffer in self.__databuffer
        self.__logger.debug("Creating Buffer...")
        self._create_buffer()
        self.__logger.info("Buffer created...")
        
        # 3) Data channel setup
        self._data_channel_setup(direction)
        self.__logger.info("data channels open and ready...")


    def start(self):
        '''
        1) init pivot objects depending on the usecase (direction)
        2) start pivot with INTRA communicator (M:N mapping)
        '''
        self.__logger.info("Start data transfer and usecase science...")
        # TODO: use enums
        if self.__direction == 1:
            self.__pivot = piv.NestTvbPivot(
                self.__comm,
                self.__param,
                self.__input_comm, 
                self.__output_comm, 
                self.__databuffer)
        elif self.__direction == 2:
            self.__pivot = piv.TvbNestPivot(
                self.__comm,
                self.__param, 
                self.__input_comm, 
                self.__output_comm, 
                self.__databuffer)
        self.__pivot.start(self.__comm)
        

    def stop(self):
        '''
        Receive stop command.
        Call stop on the pivot operation loop (receiving and sending)
        
        TODO: add error handling and fail checks
        '''
        self.__logger.info("Stop InterscaleHub and disconnect...")
        self.__pivot.stop()
        # time.sleep(5)
        if self.__direction == 1:
                if self.__comm.Get_rank() == 0:
                    self.__ic.close_and_finalize(self.__input_comm, self.__input_port)
                else:
                    self.__ic.close_and_finalize(self.__output_comm, self.__output_port)

        elif self.__direction == 2:
                if self.__comm.Get_rank() == 0:
                    self.__ic.close_and_finalize(self.__output_comm, self.__output_port)
                else:
                    self.__ic.close_and_finalize(self.__input_comm, self.__input_port)

        


    
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
        self.__logger.debug("allocating shared...")
        win = MPI.Win.Allocate_shared(bufbytes, self.__datasize, comm=self.__comm)
        self.__logger.debug("buf, and datasize...")
        buf, self.__datasize = win.Shared_query(0)
        # TODO: add error handling and fail checks
        assert self.__datasize == MPI.DOUBLE.Get_size()
        # create a 1D numpy array (buffer) whose data points to the shared mem
        self.__databuffer = np.ndarray(buffer=buf, dtype='d', shape=(self.__buffersize,))
        
    
    def _data_channel_setup(self, direction):
        '''
        Open ports and register connection details.
        Accept connection on ports and create INTER communicators.
        
        MVP: register = write port details to file.
        MVP: Two connections 
            - input = incoming simulation data
            - output = outgoing simulation data
        '''
        # NEST-to-TVB
        if self.__direction == 1:
                if self.__comm.Get_rank() == 0:
                    self.__input_comm, self.__input_port = self.__ic.open_port_accept_connection(self.__input_path)
                    self.__output_comm = None
                else:
                    self.__output_comm, self.__output_port = self.__ic.open_port_accept_connection(self.__output_path)
                    self.__input_comm = None

        # TVB-to-NEST
        elif self.__direction == 2:
                if self.__comm.Get_rank() == 0:
                    self.__output_comm, self.__output_port = self.__ic.open_port_accept_connection(self.__output_path)
                    self.__input_comm = None
                else:
                    self.__input_comm, self.__input_port = self.__ic.open_port_accept_connection(self.__input_path)
                    self.__output_comm = None
    
    def get_ids_of_nodes_to_be_connected(self, path, direction):
        
        id_transformer = 0
        if self.__direction  == 1:    
            # get information from NEST
            while not os.path.exists(path + 'nest/spike_detector.txt.unlock'):
                # self.__logger.info("DEBUG==>1 PATH: " + path)
                self.__logger.info("spike detector ids not found yet, retry in 1 second")
                time.sleep(1)
            spike_detector = np.loadtxt(path + 'nest/spike_detector.txt', dtype=int)
            # case of one spike detector
            try:
                spike_detector = np.array([int(spike_detector)])
            except:
                pass

            id_spike_detector = spike_detector[id_transformer]
            path_to_spike_detector = [path + "transformation/spike_detector/" + str(id_spike_detector) + ".txt"]
            # TVB_recev_file = "/transformation/send_to_tvb/" + str(id_proxy[id_transformer]) + ".txt"
            # id_spike_detector = os.path.splitext(os.path.basename(path + file_spike_detector))[0]
            return path_to_spike_detector

        elif self.__direction == 2:
            while not os.path.exists(path + 'nest/spike_generator.txt.unlock'):
                # self.__logger.info("DEBUG==>2 PATH: " + path)
                self.__logger.info("spike generator ids not found yet, retry in 1 second")
                time.sleep(1)
            spike_generator = np.loadtxt(path + 'nest/spike_generator.txt', dtype=int)
            # case of one spike generator
            try:
                if len(spike_generator.shape) < 2:
                    spike_generator = np.expand_dims(spike_generator, 0)
                print("extension :", spike_generator.shape)
            except:
                pass

            self.__logger.info("spike_generator : " + str(spike_generator))
            print(spike_generator[id_transformer])
            id_first_spike_generator = spike_generator[id_transformer][0]
            nb_spike_generator = len(spike_generator[id_transformer])
            path_to_spike_generators = []
            for i in range(nb_spike_generator):
                # write file with port and unlock
                running_path = os.path.join(path + "transformation/spike_generator/",
                                                str(id_first_spike_generator + i) + ".txt")
                path_to_spike_generators.append(running_path)
            # create path for receive from TVB
            return path_to_spike_generators
    
    
    def _init_params(self, p, direction):
        '''
        Init MPI stuff, buffer parameter and USECASE parameter.
        
        The USECASE parameter are taken from the TVB-NEST implementation
        in the co-sim github (refactored usecase from Lionel).
        
        MPI and buffer initialisation is done here.
        
        TODO: rework after defining the interfaces. 
        -> parameter (config and science) passing and handling.
        
        :param p: Parameters are passed through by the Launcher->Orchestrator->AppCompanion
        :param direction: hardcoded 1 for NEST->TVB or 2 for TVB->NEST
        '''
        # MPI and IntercommManager
        self.__comm = MPI.COMM_WORLD  # INTRA communicator
        self.__root = 0 # hardcoded!
        self.__ic = icm.IntercommManager(self.__comm, self.__root)
        
        # Buffer
        # TODO: needs to be a global cosim setting. more information needed!
        max_events = 1000000 # max. expected number of events per step
        self.__datasize = MPI.DOUBLE.Get_size()
        
        # USECASE parameter
        # TODO: self.__param used as global dict for now and passed all the way to pivot._analyse()
        # align this with the rest of the implementation and below param init
        self.__direction = direction
        self.__param = p.get_param(direction)
        path = self.__param['path']
        id_transformer = 0
        id_proxy = self.__param['id_nest_region']
        # nest to tvb
        if self.__direction == 1:
            self.__buffersize = max_events * 3 # 3 doubles per event
            # NOTE input and output are connected to the same port
            # self.__input_path = p.get_nest_to_tvb_port()
            # self.__output_path = p.get_nest_to_tvb_port()
            
            self.synch=self.__param['time_synchronization']                # time of synchronization between 2 run
            self.dt=self.__param['resolution']              # the resolution of the integrator
            self.shape=(2, int(self.__param['time_synchronization'] / self.__param['resolution'])) # the shape of the buffer/histogram
            self.width = int(self.__param['width']/self.__param['resolution']) # the window of the average in time
            self.buffer = np.zeros((self.width,))                  #initialisation/ previous result for a good result
            # path_to_spike_detectors
            self.__input_path = self.get_ids_of_nodes_to_be_connected(path, direction)  
            # path to send_to_tvb
            self.__output_path= [
                path + "transformation/send_to_tvb/" + str(id_proxy[id_transformer]) + ".txt"]  # NOTE id_transformer is 0

        # tvb to nest
        elif self.__direction == 2:
            self.__buffersize = 2 + max_events # 2 doubles: [start_time,end_time] of simulation step
            # self.__buffersize = (2, 2)
            # self.percentage_shared = self.__param['percentage_shared']  # percentage of shared rate between neurons
            # self.nb_spike_generator = self.__param['nb_spike_generator']         # number of spike generator
            self.nb_synapse = self.__param['nb_brain_synapses']               # number of synapses by neurons
            # self.function_translat
            # path to receive_from_tvb
            self.__input_path= [
                path + "transformation/receive_from_tvb/" + str(id_proxy[id_transformer]) + ".txt"]  # NOTE id_transformer is 0
            # path to spike_gernerators
            self.__output_path = self.get_ids_of_nodes_to_be_connected(path, direction)

     
        # NOTE: create port files and make connection
        # In Demo example: producer/Consumer are inhertied from mpi_io_extern,
        # and then they are started as threads which then call mpi_io_extern run() method
        # which then calls make_connection() method
       
       ######################################################
       # NOTE usecase specific param settings (from old code)
       ######################################################
        # # nest to tvb
        # if direction == 1:
        #     self.__buffersize = max_events * 3 # 3 doubles per event
        #     self.__input_path = p.get_nest_path()
        #     self.__output_path = p.get_tvb_path()
            
        #     self.synch=self.__param['synch']                # time of synchronization between 2 run
        #     self.dt=self.__param['resolution']              # the resolution of the integrator
        #     self.shape = (int(self.synch/self.dt),1) # the shape of the buffer/histogram
        #     self.hist = np.zeros(self.shape)         # the initialisation of the histogram
        #     self.width = int(self.__param['width']/self.__param['resolution']) # the window of the average in time
        #     self.synch = self.__param['synch']                          # synchronize time between simulator
        #     self.buffer = np.zeros((self.width,))                  #initialisation/ previous result for a good result
        #     self.coeff = 1 / ( self.__param['nb_neurons'] * self.__param['resolution'] ) # for the mean firing rate in in KHZ
        # # tvb to nest
        # elif direction == 2:
        #     self.__buffersize = 2 + max_events # 2 doubles: [start_time,end_time] of simulation step
        #     self.__input_path = p.get_tvb_path()
        #     self.__output_path = p.get_nest_path()
            
        #     self.percentage_shared = self.__param['percentage_shared']  # percentage of shared rate between neurons
        #     self.nb_spike_generator = self.__param['nb_spike_generator']         # number of spike generator
        #     self.nb_synapse = self.__param['nb_synapses']               # number of synapses by neurons
        #     self.function_translation = self.__param['function_select'] # choose the function for the translation
    
        #######################################################
    
    
    
    def _temp_protocol_translation():
        '''
        TODO: temporary translation of protocol behaviour
        from current NEST i/o to cosim 
        and from current TVB i/o to cosim.
        '''
        raise NotImplementedError
