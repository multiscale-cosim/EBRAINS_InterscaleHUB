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
import datetime

from EBRAINS_InterscaleHUB.Interscale_hub.communicator_base import BaseCommunicator
from EBRAINS_InterscaleHUB.Interscale_hub import interscalehub_utils
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_BUFFER_STATES, DATA_BUFFER_TYPES

from EBRAINS_RichEndpoint.application_companion.common_enums import Response

# TODO refactor to perform receive, send and transform operations by their
# corresponding mpi gorups. For ref. look at communicator TVB to NEST class


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
        
        # self._logger = self._configurations_manager.load_log_configurations(
        #                 name=__name__,
        #                 log_configurations=self._log_settings,
        #                 target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self._logger.info("Initialized")
      
    def start(self, intra_communicator, inter_comm_receiver,
              inter_comm_sender,
              mpi_com_group_senders,
              mpi_com_group_receivers,
              mpi_com_group_transformers,
              group_of_ranks_for_sending,
              group_of_ranks_for_receiving,
              group_of_ranks_for_transformation):
        '''
        implements the abstract method to start
        1) receiving the data
        2) transforming the data to required scale
        3) sending the transformed data
        
        M:N mapping of MPI ranks, receive data, further process data.
        Receive on rank 0, do the rest on rest of the ranks.
        '''
        my_rank =  intra_communicator.Get_rank()
        self._intra_comm = intra_communicator
        self._group_of_ranks_for_sending = group_of_ranks_for_sending
        self._group_of_ranks_for_receiving = group_of_ranks_for_receiving
        self._group_of_ranks_for_transformation = group_of_ranks_for_transformation
        self._root_sending_rank = self._group_of_ranks_for_sending[0]
        self._root_transformer_rank = self._group_of_ranks_for_transformation[0]
        
        # # Rank-0 will receive the data
        # if intra_communicator.Get_rank() == 0:
        #     # set inter_communicator for receiving the data
        #     self._comm_receiver = inter_comm_receiver
        #     self._num_sending = self._comm_receiver.Get_remote_size()
        #     return self._receive()

        # # Rank-1 will transform and send the data
        # elif intra_communicator.Get_rank() == 1:
        #     # set inter_communicator for sending the data
        #     self._comm_sender = inter_comm_sender
        #     return self._send()
        
        # Case a, if rank is in group of senders
        if my_rank in group_of_ranks_for_sending:
            # set inter_communicator for sending the data
            self._comm_sender = inter_comm_sender
            self._num_receiving = self._comm_sender.Get_remote_size()
            self._logger.debug(f"num_receiving:{self._num_receiving}")
            self._mpi_com_group_senders = mpi_com_group_senders
            return self._send()
        
        # Case b, if rank is in group of transformers
        elif my_rank in group_of_ranks_for_transformation:
            self._mpi_com_group_transformers = mpi_com_group_transformers
            return self._transform()
        
        # Case c, if rank is in group of receivers
        elif my_rank in group_of_ranks_for_receiving:
            # set inter_communicator for receiving the data
            self._comm_receiver = inter_comm_receiver
            self._num_sending = self._comm_receiver.Get_remote_size()
            self._logger.debug(f"num_sending:{self._num_sending}")
            self._mpi_com_group_receivers = mpi_com_group_receivers
            return self._receive()
        
        else:  # default, unknown group
            self._logger.critical(f"my_rank: {my_rank}, unknown group of ranks")
            return Response.ERROR

    def stop(self):
        '''
        TODO: proper execution of stop command
        '''
        # self.__stop = True
        try:
            raise NotImplementedError
        except NotImplementedError:
            self._logger.exception("stop() is not implemented yet")
            return Response.OK

    def _receive(self):
        '''
        Receives data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        # The last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        self._logger.info("setting up buffers")


        # set buffer to 'ready to receive from nest'
        # self.__databuffer[-1] = 1
        # self._data_buffer_manager.set_ready_state_at(index=-1,
        #                                              state = DATA_BUFFER_STATES.READY_TO_RECEIVE,
        #                                              buffer_type=DATA_BUFFER_TYPES.INPUT)

        # marks the 'head' of the buffer (NOTE set 0 to start)
        # self._data_buffer_manager.set_header_at(index=-2,
        #                                         header=0,
        #                                         buffer_type=DATA_BUFFER_TYPES.INPUT)
        
        # It seems the 'check' variable is used to receive tags from NEST,
        # i.e. ready for send...
        # change this in the future, also mentioned in the FatEndPoint solution
        # from Wouter.
        check = np.empty(1,dtype='b')
        # TODO Discuss with Kim if shape here is size --> rename
        size = np.empty(1, dtype='i')    
        count = 0
        status_ = MPI.Status()
        self._logger.info("reading from buffer")
        while True:
            self._logger.debug(f"__DEBUG__ _receive() start receiving loop, time:{datetime.datetime.now()}")
            raw_data_end_index = 0  # head of the buffer, reset after each iteration
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # IF all ranks send always the same tag in one iteration (simulation step)
            # then this works. But it should be handled differently!!!!
            self._comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=0, tag=MPI.ANY_TAG, status=status_)

            status_rank_0 = status_.Get_tag()
            for i in range(1, self._num_sending):
                # new: We do not care which source sends first, give MPI the freedom to send in whichever order.
                # self._comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
                self._comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=i, tag=MPI.ANY_TAG, status=status_)
                # Check if the state of the NEST is different between the ranks
                if status_rank_0 != status_.Get_tag():
                    # Log the exception with traceback
                    interscalehub_utils.log_exception(
                        log_message="Abnormal state : the state of Nest is "
                                    "different between rank. Tag received: ",
                        mpi_tag_received=status_.Get_tag())
                    # Terminate with Error
                    return Response.ERROR

            if status_.Get_tag() == 0:
                # TODO 
                #       1. use MPI, remove the sleep and refactor while loop
                #       to soemthing more efficient
                counter = 0
                while self._data_buffer_manager.get_at(index=-1, buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.READY_TO_RECEIVE:
                    # wait until ready to receive new data (i.e. the
                    # Transformers has cleared the buffer)
                    counter += 1
                    time.sleep(0.001)
                    pass
                
                self._logger.debug(f"__DEBUG__ while loop counter until buffer state is ready:{counter}")

                for source in range(self._num_sending):
                    # send 'ready' to the nest rank
                    # self._logger.info("send ready")
                    self._comm_receiver.Send([np.array(True,dtype='b'),MPI.BOOL],dest=source,tag=0)
                    # receive package size info
                    self._comm_receiver.Recv([size, 1, MPI.INT], source=source, tag=0, status=status_)
                    # self._comm_receiver.Recv([shape, 1, MPI.INT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
                    # NEW: receive directly into the buffer
                    # self._comm_receiver.Recv([self.__databuffer[head_:], MPI.DOUBLE], source=source, tag=0, status=status_)
                    data_buffer = self._data_buffer_manager.get_from(
                                    starting_index=raw_data_end_index,
                                    buffer_type=DATA_BUFFER_TYPES.INPUT)
                            
                    self._comm_receiver.Recv([data_buffer, MPI.DOUBLE],
                                              source=source,
                                              tag=0,
                                              status=status_)
                    # running_head += shape[0]  # move running head
                    raw_data_end_index += size[0]  # move running size
                
                # Mark as 'ready to do analysis/transform'
                self._data_buffer_manager.set_ready_state_at(index=-1,
                                                             state=DATA_BUFFER_STATES.READY_TO_TRANSFORM,
                                                             buffer_type=DATA_BUFFER_TYPES.INPUT)

                # set the header (i.e. the last index where the data ends)
                self._data_buffer_manager.set_header_at(index=-2,
                                                        header=raw_data_end_index,
                                                        buffer_type=DATA_BUFFER_TYPES.INPUT)

                # continue receiving the data
                self._logger.debug(f"__DEBUG__ _receive() start receiving loop ends, time:{datetime.datetime.now()}")
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
        status_received_transformer = MPI.Status()
        check = np.empty(1,dtype='b')
        # initialize with state as waiting for receivers group to put the data
        # in INPUT buffer
        # self._data_buffer_manager.set_ready_state_at(index=-1,
        #                                              state = DATA_BUFFER_STATES.WAIT,
        #                                              buffer_type=DATA_BUFFER_TYPES.INPUT)
        # self._logger.info("NESTtoTVB -- producer/sender -- Rank:"+str(self._comm_sender.Get_rank()))
        while True:
            self._logger.debug(f"__DEBUG__ start sending loop, count: {count}, time:{datetime.datetime.now()}")
            # TODO: this communication has the 'rank 0' problem described in the beginning
            accept = False
            #logger.info("Nest to TVB : wait to send " )
            while not accept:
                req = self._comm_sender.irecv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
                accept = req.wait(status_)
            #logger.info(" Nest to TVB : send data status : " +str(status_.Get_tag()))

            ###############################################
            # send tag to transformers
            # TODO Discuss with Kim if we rather send tag directly as data
            if self._intra_comm.Get_rank() == self._root_sending_rank:
                self._intra_comm.Send([check, 1, MPI.CXX_BOOL], dest=self._root_transformer_rank, tag=status_.Get_tag())
            ###############################################

            if status_.Get_tag() == 0:
            #     # wait until the receiver has cleared the buffer, i.e. filled with new data
            #     # TODO: use MPI, remove the sleep
            #     # counter = 0
            #     while self._data_buffer_manager.get_at(index=-1,
            #                                            buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.READY_TO_TRANSFORM:
            #         # wait until the transformer has filled the buffer with
            #         # new data
            #         # counter += 1
            #         time.sleep(0.001)
            #         pass

                # self._logger.debug(f"__DEBUG__ while loop counter until buffer state is ready:{counter}")
                
                
                ###############################################
                # TODO add a blocking receive call to receive the spike trains
                # from transformers
                if self._intra_comm.Get_rank() == self._root_sending_rank:
                    # self._intra_comm.Recv([size_spike_trains, 1, MPI.INT], self._root_transformer_rank, tag=MPI.ANY_TAG(), status=status_received_transformer)
                    # spike_trains = [[] for _ in range(size_spike_trains)]
                    times, data = self._intra_comm.recv(source=self._root_transformer_rank, tag=0, status=status_received_transformer)

                ###############################################
                # NOTE: calling the mediator which calls the corresponding transformer functions
                # times,data = mediator.spike_to_rate(self.__databuffer, count)
                # TODO: change to inject the buffer in the wrapper method of mediator
                # times, data = spikerate.spike_to_rate(count, self.__databuffer[-2], self.__databuffer)
                # times, data = self._mediator.spikes_to_rate(
                #     count,
                #     size_at_index=-2,
                #     buffer_type=DATA_BUFFER_TYPES.INPUT)

                    # buffer_size=self._data_buffer_manager.get_at(index=-2),
                    # data_buffer=self._data_buffer_manager.mpi_shared_memory_buffer)

                # Mark as 'ready to receive next simulation step'
                # self._data_buffer_manager.set_ready_state_at(
                #     index=-1,
                #     state=DATA_BUFFER_STATES.READY_TO_RECEIVE,
                #     buffer_type=DATA_BUFFER_TYPES.INPUT)
                
                ### OLD Code
                #logger.info("Nest to TVB : send data :"+str(np.sum(data)) )
                # time of sim step
                self._comm_sender.Send([times, MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                # send the size of the rate
                size = np.array(int(data.shape[0]),dtype='i')
                self._comm_sender.Send([size,MPI.INT], dest=status_.Get_source(), tag=0)
                # send the rates
                self._comm_sender.Send([data,MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                # increment the count
                count += 1
                # continue sending the data
                self._logger.debug(f"__DEBUG__ start sending loop ends, time:{datetime.datetime.now()}")
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

    def _transform(self):
        """
        transforms the data from input buffer and puts it into output buffer
        """
        check = np.empty(1,dtype='b')
        status_ = MPI.Status()
        count = 0

        # NOTE root_transformer_rank = rank id BEFORE group creation
        # translated_root_rank = rank id WITHIN the new group (shifted by size of the other groups)

        # TODO Discuss with Kim
        # 1. if we could move it to start function, and 
        # 2. why not making root_transformer_rank = translated_root_rank
        translated_root_rank = self._root_transformer_rank - (
            len(self._group_of_ranks_for_sending) + len(self._group_of_ranks_for_receiving))
        
        while True:
            # Check if the simulation is still running
            ###############################################
            # TODO 
            # 1. receive tag from sender group
            tag = None            
            self._logger.debug(f"__DEBUG__ _transform() start loop, count: {count}, time:{datetime.datetime.now()}")

            if self._intra_comm.Get_rank() == self._root_transformer_rank:
                self._intra_comm.Recv([check, 1, MPI.CXX_BOOL], source=self._root_sending_rank, status=status_)
                tag = status_.Get_tag()
            
            # # NOTE root_transformer_rank = rank id BEFORE group creation
            # # translated_root_rank = rank id WITHIN the new group (shifted by size of the other groups)
            # translated_root_rank = self._root_transformer_rank - (
            #     len(self._group_of_ranks_for_sending) + len(self._group_of_ranks_for_receiving))
            tag = self._mpi_com_group_transformers.bcast(tag, root=translated_root_rank)

            if tag == 0:
                # Case a, simulation is running, do transformation

                # STEP 1. Wait until recceivers receive data in Input Buffer
                sleep_loop_counter = 0
                while self._data_buffer_manager.get_at(index=-1,
                                                        buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.READY_TO_TRANSFORM:
                    # wait until the transformer has filled the buffer with
                    # new data
                    sleep_loop_counter +=1
                    time.sleep(0.001)
                    pass

                # self._logger.debug(f"__DEBUG__ _transform() while loop counterfor INPUT until buffer state is ready:{sleep_loop_counter}")
                
                # STEP 2. Transform the data
                # print("\n"*2)
                # print(f"__DEBUG 1__ _transform() time before spikes_to_rate conversion: {datetime.datetime.now()}")
                # NOTE the results are gathered to only the root_transformer_rank
                times, data = self._mediator.spikes_to_rate(
                    count,
                    size_at_index=-2,
                    buffer_type=DATA_BUFFER_TYPES.INPUT,
                    comm=self._mpi_com_group_transformers,
                    root_transformer_rank=translated_root_rank)
                # print(f"__DEBUG 2__ _transform() time after spikes_to_rate conversion: {datetime.datetime.now()}")
                # print("\n"*2)
                # NOTE input buffer is already marked as 'ready to receive next
                # simulation step' in self._mediator.spikes_to_rate()


                # STEP 3. put the transformed data in Output Buffer
                ###############################################
                # TODO send the spike trains to sender
                if self._intra_comm.Get_rank() == self._root_transformer_rank:
                    # self._intra_comm.Send([len(spike_trains),1,MPI.INT], self._root_sending_rank, tag=MPI.ANY_TAG)
                    self._intra_comm.send((times, data), self._root_sending_rank, tag=0)
                ###############################################
                
                # wait until root transformer rank sends the data
                self._logger.debug("waiting for root to finish with sending"
                                   " the data")
                self._mpi_com_group_transformers.Barrier()

                # continue transformation
                self._logger.debug(f"__DEBUG__ _transform() start loop ends, time:{datetime.datetime.now()}")
                count += 1
                continue

            elif tag == 1:
                # everything goes fine, terminate the loop and respond with OK
                # self._logger.debug('TVB_to_NEST: End of transform function')
                # self._mpi_com_group_transformers.Barrier()
                 # Mark as 'ready to do analysis/transformation'
                
                # self._data_buffer_manager.set_ready_state_at(index=-1,
                #                                             state=DATA_BUFFER_STATES.TERMINATE,
                #                                             buffer_type=DATA_BUFFER_TYPES.INPUT)
                # self._mpi_com_group_transformers.Barrier()
                self._logger.debug('NEST_TO_TVB: End of transform function')
                return Response.OK
            else:
                # A 'bad' MPI tag is received,
                # log the exception with traceback
                # self._mpi_com_group_transformers.Barrier()
                interscalehub_utils.log_exception(
                    logger=self._logger,
                    log_message="bad mpi tag :",
                    mpi_tag_received=tag)
                # terminate with Error
                return Response.ERROR