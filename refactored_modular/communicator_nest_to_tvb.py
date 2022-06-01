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

from EBRAINS_InterscaleHUB.refactored_modular.communicator_base import BaseCommunicator
from EBRAINS_InterscaleHUB.refactored_modular import interscalehub_utils
from EBRAINS_InterscaleHUB.refactored_modular.interscalehub_enums import DATA_BUFFER_STATES

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


class CommunicatorNestTvb(BaseCommunicator):
    '''
    Implements the BaseCommunicator. It 
    1) Receives the data from NEST
    2) Transforms it to the required format such as to 'rate'
    3) Sends the transformed data to TVB
    '''
    def __init__(self, configurations_manager, log_settings,
                 data_buffer_manager, mediator):
        '''
        '''
        # initialize the common settings such as logger, data buffer, etc.
        super().__init__(configurations_manager,
                         log_settings,
                         __name__,
                         data_buffer_manager,
                         mediator)
        self.__logger.info("Initialized")
      
    def start(self, intra_communicator, inter_comm_receiver, inter_comm_sender):
        '''
        implements the abstract method to start
        1) receiving the data
        2) transforming the data to required scale
        3) sending the transformed data
        
        M:N mapping of MPI ranks, receive data, further process data.
        Receive on rank 0, do the rest on rest of the ranks.
        '''
        # Rank-0 will receive the data
        if intra_communicator.Get_rank() == 0:
            # set inter_communicator for receiving the data
            self.__comm_receiver = inter_comm_receiver
            self.__num_sending = self.__comm_receiver.Get_remote_size()
            return self._receive()

        # Rank-1 will transform and send the data
        elif intra_communicator.Get_rank() == 1:
            # set inter_communicator for sending the data
            self.__comm_sender = inter_comm_sender
            self.__num_receiving = self.__comm_sender.Get_remote_size()  # TODO discuss if it is needed
            return self._send()

    def stop(self):
        '''
        TODO: proper execution of stop command
        '''
        # self.__stop = True
        raise NotImplementedError

    def _receive(self):
        '''
        Receives data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        # The last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        self.__logger.info("setting up buffers")
        
        # set buffer to 'ready to receive from nest'
        # self.__databuffer[-1] = 1
        self.__data_buffer_manager.set_ready_at(index=-1)
        
        # marks the 'head' of the buffer
        # self.__databuffer[-2] = 0
        self.__data_buffer_manager.set_head_at(index=-2)
        
        # It seems the 'check' variable is used to receive tags from NEST,
        # i.e. ready for send...
        # change this in the future, also mentioned in the FatEndPoint solution
        # from Wouter.
        check = np.empty(1,dtype='b')
        shape = np.empty(1, dtype='i')    
        count = 0
        status_ = MPI.Status()
        self.__logger.info("reading from buffer")
        
        ###########################################################
        # TODO Refactor to move this functionality to appropriate location
        
        # NOTE As per protocol, it should be the response message of 'init'
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
            running_head = 0  # head of the buffer, reset after each iteration
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # IF all ranks send always the same tag in one iteration (simulation step)
            # then this works. But it should be handled differently!!!!
            self.__comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=0, tag=MPI.ANY_TAG, status=status_)
            
            status_rank_0 = status_.Get_tag()
            for i in range(1, self.__num_sending):
                # new: We do not care which source sends first, give MPI the freedom to send in whichever order.
                # self.__comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
                self.__comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=i, tag=MPI.ANY_TAG, status=status_)
                # Check if     the state of the NEST is different between the ranks
                if status_rank_0 != status_.Get_tag():
                    # Log the exception with traceback
                    interscalehub_utils.log_exception(
                        log_message="Abnormal state : the state of Nest is "
                                    "different between rank. Tag received: ",
                        mpi_tag_received=status_.Get_tag())
                    # Terminate with Error
                    return Response.ERROR

            if status_.Get_tag() == 0:
                # wait until ready to receive new data (i.e. the sender has cleared the buffer)
                
                # TODO: use MPI, remove the sleep
                # # while self.__databuffer[-1] != 1:
                while self.__data_buffer_manager.get_at(index=-1) != DATA_BUFFER_STATES.READY:
                    time.sleep(0.001)
                    pass

                for source in range(self.__num_sending):
                    # send 'ready' to the nest rank
                    # self.__logger.info("send ready")
                    self.__comm_receiver.Send([np.array(True,dtype='b'),MPI.BOOL],dest=source,tag=0)
                    # receive package size info
                    self.__comm_receiver.Recv([shape, 1, MPI.INT], source=source, tag=0, status=status_)
                    # self.__comm_receiver.Recv([shape, 1, MPI.INT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
                    # NEW: receive directly into the buffer
                    # self.__comm_receiver.Recv([self.__databuffer[head_:], MPI.DOUBLE], source=source, tag=0, status=status_)
                    data_buffer = self.__data_buffer_manager.get_from(
                                    starting_index=running_head)
                            
                    self.__comm_receiver.Recv([data_buffer, MPI.DOUBLE],
                                              source=source,
                                              tag=0,
                                              status=status_)
                    running_head += shape[0]  # move running head
                # Mark as 'ready to do analysis'
                # self.__databuffer[-1] = 0
                self.__data_buffer_manager.set_header_at(index=-1)

                # important: head_ is first buffer index WITHOUT data.
                # self.__databuffer[-2] = head_
                self.__data_buffer_manager.set_custom_value_at(
                                                        index=-2,
                                                        value=running_head)
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
        count = 0  # simulation/iteration step
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
                # TODO: use MPI, remove the sleep
                # while self.__databuffer[-1] != 0:
                while self.__data_buffer_manager.get_at(index=-1) != DATA_BUFFER_STATES.HEAD:
                    time.sleep(0.001)
                    pass
                
                # NOTE: calling the mediator which calls the corresponding transformer functions
                # times,data = mediator.spike_to_rate(self.__databuffer, count)
                # TODO: change to inject the buffer in the wrapper method of mediator
                # times, data = spikerate.spike_to_rate(count, self.__databuffer[-2], self.__databuffer)
                times, data = self.__mediator.spike_to_rate(
                    count,
                    buffer_size=self.__data_buffer_manager.get_at(index=-2),
                    data_buffer=self.__data_buffer_manager.mpi_shared_memory_buffer)

                # Mark as 'ready to receive next simulation step'
                # self.__databuffer[-1] = 1
                self.__data_buffer_manager.set_ready_at(index=-1)
                
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
                count += 1
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

'''    
    def _transform(self, count):
        #store: Python object, create the histogram 
        #analyse: Python object, calculate rates
        spikerate = spiketorate(self.__param)
        times, data = spikerate.spike_to_rate(count, self.__databuffer[-2], self.__databuffer)

        return times, data
'''
