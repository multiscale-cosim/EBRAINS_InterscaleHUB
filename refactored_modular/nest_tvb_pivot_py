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
import time
import numpy as np

from EBRAINS_InterscaleHUB.refactored_modular.pivot_base import PivotBaseClass
from EBRAINS_InterscaleHUB.refactored_modular import interscalehub_utils
from EBRAINS_InterscaleHUB.Interscale_hub.transformer import spiketorate

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_RichEndpoint.Application_Companion.common_enums import Response

# NestTvbPivot and TvbNestPivot classes:
# TODO: proper abstraction -> extract the usecase details from the general implementation
# -> Init, start, stop are pretty much the same every time
# -> incoming (receive) and outgoing (send) loops (M:N mapping)
# -> the analyse (method) should be 
#   a) pivot, as raw data to cosim data 
#   b) transform (might be trivial) and 
#   c) analysis (might be trivial)

# TODO: rework on the receive and send loops (both, general coding style and usecase specifics)

class NestTvbPivot(PivotBaseClass):
    '''
    Implements the PivotBaseClass for abstracting the pivot operations and
    the underlying communication protocol. This class provides wrappers
    for receving the data from NEST simulator and sending it to TVB simulator
    after processing/transforming to the required format.
    '''
    def __init__(self, intracomm, param, comm_receiver, comm_sender, databuffer, configurations_manager, log_settings):
        '''
        '''
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="NestTvbPivot",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        # Parameter for transformation and analysis
        self.__param = param
        # INTERcommunicator
        if intracomm.Get_rank() == 0:
            self.__comm_receiver = comm_receiver
            self.__num_sending = self.__comm_receiver.Get_remote_size()
        else:
            self.__comm_sender = comm_sender
            self.__num_receiving = self.__comm_sender.Get_remote_size()

        # How many Nest ranks are sending, how many Tvb ranks are receiving
        self.__databuffer = databuffer
        self.__logger.info("Initialised")
    
    
    def start(self, intracomm):
        '''
        Starts the pivot operation.
        
        M:N mapping of MPI ranks, receive data, further process data.
        Receive on rank 0, do the rest on rest of the ranks.
        '''
        if intracomm.Get_rank() == 0: # Receiver from input sim, rank 0
            self._receive()
        else: #  Science/analyse and sender to TVB, rank 1-x
            self._send()


    def stop(self):
        '''
        TODO: proper execution of stop command
        '''
        self.__stop = True


    def _receive(self):
        '''
        Receive data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        # The last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        self.__logger.info("setting up buffers")
        self.__databuffer[-1] = 1 # set buffer to 'ready to receive from nest'
        self.__databuffer[-2] = 0 # marks the 'head' of the buffer
        # It seems the 'check' variable is used to receive tags from NEST, i.e. ready for send...
        # change this in the future, also mentioned in the FatEndPoint solution from Wouter.
        check = np.empty(1,dtype='b')
        shape = np.empty(1, dtype='i')    
        count = 0
        status_ = MPI.Status()
        self.__logger.info("reading from buffer")
        
        ###########################################################
        #TODO Refactor to move this functionality to appropriate location
        
        #NOTE As per protocol, it should be the response message of 'init'
        # command, and should return the PID and the port information

        import os
        from EBRAINS_RichEndpoint.Application_Companion.common_enums import INTEGRATED_SIMULATOR_APPLICATION as SIMULATOR
        pid_and_local_minimum_step_size = \
        {SIMULATOR.PID.name: os.getpid(),
        SIMULATOR.LOCAL_MINIMUM_STEP_SIZE.name: 0.0}
        print(f'{pid_and_local_minimum_step_size}')
        ###########################################################
        # self.__logger.info("NESTtoTVB -- consumer/receiver -- Rank:"+str(self.__comm_receiver.Get_rank()))
        while True:
            head_ = 0 # head of the buffer, reset after each iteration            
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # IF all ranks send always the same tag in one iteration (simulation step)
            # then this works. But it should be handled differently!!!!
            self.__comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=0, tag=MPI.ANY_TAG, status=status_)
            
            status_rank_0 = status_.Get_tag()
            for i in range(1, self.__num_sending):
                # new: We do not care which source sends first, give MPI the freedom to send in whichever order.
                # self.__comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
                # self.__logger.info("checking status")
                self.__comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=i, tag=MPI.ANY_TAG, status=status_)
                if status_rank_0 != status_.Get_tag():
                    # Case: the state of the NEST is different between the ranks
                    # Log the exception with traceback
                    interscalehub_utils.log_exception(
                        log_message="Abnormal state : the state of Nest is different between rank. Tag received: ",
                        mpi_tag_received=status_.Get_tag())
                    # Terminate with Error
                    return Response.ERROR

            if status_.Get_tag() == 0:
                # wait until ready to receive new data (i.e. the sender has cleared the buffer)
                while self.__databuffer[-1] != 1: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                for source in range(self.__num_sending):
                    # send 'ready' to the nest rank
                    # self.__logger.info("send ready")
                    self.__comm_receiver.Send([np.array(True,dtype='b'),MPI.BOOL],dest=source,tag=0)
                    # receive package size info
                    # self.__logger.info("DEBUG 121 ====> receiving size in NEST_TVB_PIVOT")
                    self.__comm_receiver.Recv([shape, 1, MPI.INT], source=source, tag=0, status=status_)
                    # self.__comm_receiver.Recv([shape, 1, MPI.INT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
                    # NEW: receive directly into the buffer
                    self.__comm_receiver.Recv([self.__databuffer[head_:], MPI.DOUBLE], source=source, tag=0, status=status_)
                    head_ += shape[0] # move head 
                # Mark as 'ready to do analysis'
                self.__databuffer[-1] = 0
                # important: head_ is first buffer index WITHOUT data.
                self.__databuffer[-2] = head_
                # continue receiving the data
                continue
            elif status_.Get_tag() == 1:
                # increment the count and continue receiving the data
                count += 1
                continue
            elif status_.Get_tag() == 2:
                # NOTE: simulation ended
                # everything goes fine, terminate the loop and respond with OK
                return Response.OK
            else:
               # A 'bad' MPI tag is received,
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    log_message="bad mpi tag :",
                    mpi_tag_received=status_.Get_tag())
                # terminate with Error
                return Response.ERROR
    
    
    def _send(self):
        '''
        Send data to TVB (multiple MPI ranks possible).
        Replaces the former 'send' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        count=0 # simulation/iteration step
        status_ = MPI.Status()
        # self.__logger.info("NESTtoTVB -- producer/sender -- Rank:"+str(self.__comm_sender.Get_rank()))
        while True:
            # TODO: this communication has the 'rank 0' problem described in the beginning
            accept = False
            #logger.info("Nest to TVB : wait to send " )
            while not accept:
                req = self.__comm_sender.irecv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
                accept = req.wait(status_)
            #logger.info(" Nest to TVB : send data status : " +str(status_.Get_tag()))
            if status_.Get_tag() == 0:
                # wait until the receiver has cleared the buffer, i.e. filled with new data
                while self.__databuffer[-1] != 0: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                # TODO: All science/analysis here. Move to a proper place.
                times,data = self._transform(count)
                # Mark as 'ready to receive next simulation step'
                self.__databuffer[-1] = 1
                
                ### OLD Code
                #logger.info("Nest to TVB : send data :"+str(np.sum(data)) )
                # time of sim step
                self.__comm_sender.Send([times, MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                # send the size of the rate
                size = np.array(int(data.shape[0]),dtype='i')
                self.__comm_sender.Send([size,MPI.INT], dest=status_.Get_source(), tag=0)
                # send the rates
                self.__comm_sender.Send([data,MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                # increment the count
                count+=1  
                # continue sending the data
                continue
                ### OLD Code end
            elif status_.Get_tag() == 1:
                # NOTE: simulation ended
                # everything goes fine, terminate the loop and respond with OK
                return Response.OK
            else:
                # A 'bad' MPI tag is received,
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    log_message="bad mpi tag :",
                    mpi_tag_received=status_.Get_tag())
                # terminate with Error
                return Response.ERROR

    
    def _transform(self, count):
        '''
        This step contains some pivoting, transformation and analysis.
        TODO: encapsulate
        :param count: Simulation iteration/step
        :return times, data: simulation times and the calculated rates
        '''
        #store: Python object, create the histogram 
        #analyse: Python object, calculate rates
        spikerate = spiketorate(self.__param)
        times, data = spikerate.spike_to_rate(count, self.__databuffer[-2], self.__databuffer)

        '''
        store = store_data(self.__param)
        analyse = analyse_data(self.__param)
        
        # TODO: Step 1 and 2 can be merged into one step. Buffer is no longer filled rank by rank.
        # Make this parallel with the INTRA communicator (should be embarrassingly parallel).
        # Step 1) take all data from buffer and create histogram
        # second to last index in databuffer denotes how much data there is
        self.__logger.info("NESTtoTVBPivot -- transform -- buffer head:"+str(self.__databuffer[-2]))
        store.add_spikes(count, self.__databuffer[:int(self.__databuffer[-2])])
        # Step 2) take the resulting histogram
        data_to_analyse = store.return_data()
        # Step 3) Analyse this data, i.e. calculate rates?
        times,data = analyse.analyse(count, data_to_analyse)
        '''
        return times, data