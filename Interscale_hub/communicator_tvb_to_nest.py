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
from mpi4py import MPI
import time
import datetime
import numpy as np

from EBRAINS_InterscaleHUB.Interscale_hub.communicator_base import BaseCommunicator
from EBRAINS_InterscaleHUB.Interscale_hub import interscalehub_utils
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_BUFFER_STATES, DATA_BUFFER_TYPES

from EBRAINS_RichEndpoint.application_companion.common_enums import Response

# NOTE TODO its not fault tolerate when one of the simulators termiantes abruptly
# for any reason such as mybe throws some exception.


# NOTE TODO check termination order for receive(), send() and transform()


class CommunicatorTvbNest(BaseCommunicator):
    '''
    Implements the BaseCommunicator Class for implementing the operations such
    as the underlying communication protocol, and computation
    (analysis/transformation). 
    
    This class provides the methods to
    1) receive 'data' from TVB,
    2) transform 'data'
    3) send the 'transformed data' to NEST
    '''
    def __init__(self, configurations_manager, log_settings,
                 data_buffer_manager, mediator):
         # initialize the common settings such as logger, data buffer, etc.
        super().__init__(configurations_manager,
                         log_settings,
                         __name__,
                         data_buffer_manager,
                         mediator)
        
        self._logger.info("Initialized")

    def start(self, intra_communicator, inter_comm_receiver,
              inter_comm_sender, id_first_spike_detector,
              mpi_com_group_senders,
              mpi_com_group_receivers,
              mpi_com_group_transformers,
              group_of_ranks_for_sending,
              group_of_ranks_for_receiving,
              group_of_ranks_for_transformation
              ):
        '''
        Start the operations
        M:N mapping of MPI ranks, receive data, further process data.
        
        MVP: receive on rank 0, do the rest on rank 1.
        '''
        my_rank =  intra_communicator.Get_rank()
        self._intra_comm = intra_communicator
        self._group_of_ranks_for_sending = group_of_ranks_for_sending
        self._group_of_ranks_for_receiving = group_of_ranks_for_receiving
        self._group_of_ranks_for_transformation = group_of_ranks_for_transformation
        self._root_sending_rank = self._group_of_ranks_for_sending[0]
        self._root_transformer_rank = self._group_of_ranks_for_transformation[0]
        
        # Case a, if rank is in group of senders
        if my_rank in group_of_ranks_for_sending:
            # set inter_communicator for sending the data
            self._comm_sender = inter_comm_sender
            self._num_receiving = self._comm_sender.Get_remote_size()
            self._logger.debug(f"num_receiving:{self._num_receiving}")
            self._mpi_com_group_senders = mpi_com_group_senders
            return self._send(id_first_spike_detector)
        
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
        Receive data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        # The last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        # set buffer to 'ready to receive from tvb'
        # self._data_buffer_manager.set_ready_state_at(index=-1,
        #                                              state = DATA_BUFFER_STATES.READY_TO_RECEIVE,
        #                                              buffer_type=DATA_BUFFER_TYPES.INPUT)
        
        # marks the 'head' of the buffer (NOTE set 0 to start)
        if self._mpi_com_group_receivers.Get_rank() == self._group_of_ranks_for_receiving[0]:
            self._data_buffer_manager.set_header_at(index=-2,
                                                header=0,
                                                buffer_type=DATA_BUFFER_TYPES.INPUT)
        # sync up point to set the buffer header
        self._logger.debug("sync up point to set up the buffer head")
        self._mpi_com_group_receivers.Barrier()
        
        # init placeholder for incoming data
        size = np.empty(1, dtype='i') # size of the rate-array
        status_ = MPI.Status()
        count = 0
        while True:
            count += 1
            self._logger.debug(f"__DEBUG__ _receive() start receiving loop, count: {count}, time:{datetime.datetime.now()}")

            # NOTE: Check communication protocol between simulators and transformers!
            requests=[]
            for rank in range(self._num_sending):
                requests.append(self._comm_receiver.isend(True,dest=rank,tag=0))
            MPI.Request.Waitall(requests)
            # NOTE: works for now, needs rework if multiple ranks are used on TVB side
            # we receive from "ANY_SOURCE", but only check the status_ of the last receive...
            
            # get the starting and ending time of the simulation step
            data_buffer = self._data_buffer_manager.get_from_range(start=0, end=2,
                                                             buffer_type=DATA_BUFFER_TYPES.INPUT)
            
            self._comm_receiver.Recv([data_buffer, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_)
            # check status.tag 
            # Case a, simulation is running
            if status_.Get_tag() == 0:
                # TODO 
                #       1. use MPI, remove the sleep and refactor while loop
                #       to soemthing more efficient
                counter = 0

                while self._data_buffer_manager.get_at(index=-1,
                                                    buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.READY_TO_RECEIVE:
                    # wait until ready to receive new data (i.e. the
                    # Transformers has cleared the buffer)
                    counter += 1
                    time.sleep(0.001)
                    pass
                
                # self._logger.debug(f"__DEBUG__ _receive() while loop counter until buffer state is ready:{counter}")

                # Get the size/shape of the data
                self._comm_receiver.Recv([size, 1, MPI.INT], source=status_.Get_source(), tag=0, status=status_)
                
                # data buffer to receive the data
                data_buffer = self._data_buffer_manager.get_from(starting_index=2,
                                                                 buffer_type=DATA_BUFFER_TYPES.INPUT)
                # receive data
                self._comm_receiver.Recv([data_buffer, MPI.DOUBLE], source=status_.Get_source(), tag=0, status=status_)
                
                # Mark as 'ready to do analysis/transformation'
                self._data_buffer_manager.set_ready_state_at(index=-1,
                                                            state=DATA_BUFFER_STATES.READY_TO_TRANSFORM,
                                                            buffer_type=DATA_BUFFER_TYPES.INPUT)

                # set the header (i.e. the last index where the data ends)

                # NOTE because the first two values are always the time steps,
                # and the data starts from index 2
                raw_data_end_index = size+2
                self._data_buffer_manager.set_header_at(index=-2,
                                                    header=raw_data_end_index,
                                                    buffer_type=DATA_BUFFER_TYPES.INPUT)

                # synchronize
                # continue receiving the data
                self._logger.debug(f"__DEBUG__ _receive() start receiving loop ends, time:{datetime.datetime.now()}")
                continue
            
            # Case b, simulation is ended
            elif status_.Get_tag() == 1:
                self._logger.debug(f"__DEBUG__ _receive() tag ==1 ")
                # everything goes fine, terminate the loop and respond with OK
                # self._logger.info('TVB_to_NEST: End of receive function')
                # counter = 0
                # while self._data_buffer_manager.get_at(index=-1,
                #                                     buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.TERMINATE:
                #     # wait until ready to receive new data (i.e. the
                #     # Transformers has cleared the buffer)
                #     counter += 1
                #     time.sleep(0.001)
                #     pass
                self._logger.info('TVB_to_NEST: End of receive function')
                return Response.OK

            # Case c, A 'bad' MPI tag is received
            else:
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    log_message="bad mpi tag :",
                    mpi_tag_received=status_.Get_tag())
                # terminate with Error
                return Response.ERROR

    def _send(self, id_first_spike_detector):
        '''
        Send data to NEST (multiple MPI ranks possible).
        Replaces the former 'send' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        status_ = MPI.Status()
        # NOTE: hardcoded...
        check = np.empty(1,dtype='b')
        size_list = np.empty(1, dtype='i')
        # id_first_spike_detector = self.__parameters['id_first_spike_detector']
        # self._data_buffer_manager.set_ready_state_at(index=-1,
        #                                              state = DATA_BUFFER_STATES.WAIT,
        #                                              buffer_type=DATA_BUFFER_TYPES.OUTPUT)
        # spike_trains = None
        count = 0
        status_received_transformer = MPI.Status()
        while True:
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # IF all ranks send always the same tag in one iteration (simulation step)
            # then this works. But it should be handled differently!!!!
            count += 1
            self._logger.debug(f"__DEBUG__ start sending loop, count: {count}, time:{datetime.datetime.now()}")
            for rank in range(self._num_receiving):
                self._comm_sender.Recv([check, 1, MPI.CXX_BOOL], source=rank, tag=MPI.ANY_TAG, status=status_)

            ###############################################
            # TODO send tag to transformers
            
            if self._intra_comm.Get_rank() == self._root_sending_rank:
                self._intra_comm.Send([check, 1, MPI.CXX_BOOL], dest=self._root_transformer_rank, tag=status_.Get_tag())
            
            ###############################################
            if status_.Get_tag() == 0:
                # TODO 
                #       1. use MPI, remove the sleep and refactor while loop
                #       to soemthing more efficient
                
                ###############################################
                # TODO add a blocking receive call to receive the spike trains
                # from transformers
                if self._intra_comm.Get_rank() == self._root_sending_rank:
                    # self._intra_comm.Recv([size_spike_trains, 1, MPI.INT], self._root_transformer_rank, tag=MPI.ANY_TAG(), status=status_received_transformer)
                    # spike_trains = [[] for _ in range(size_spike_trains)]
                    spike_trains = self._intra_comm.recv(source=self._root_transformer_rank, tag=0, status=status_received_transformer)
                
                # in case if there are more than one sender ranks
                # self._comm_sender.bcast(spike_trains, self._root_sending_rank)
                # self._logger.debug(f"__DEBUG__ spike_trains: {spike_trains}")
                ###############################################
                
                ### OLD code starts
                # TODO revisit it
                for rank in range(self._num_receiving):
                    self._comm_sender.Recv([size_list, 1, MPI.INT], source=rank, tag=0, status=status_)
                    if size_list[0] != 0:
                        list_id = np.empty(size_list, dtype='i')
                        self._comm_sender.Recv([list_id, size_list, MPI.INT], source=status_.Get_source(), tag=0, status=status_)

                        # Select the good spike train and send it
                        # TODO: create lists, append to lists, nested loops
                        # this is slow and will be a bottleneck when we scale up.
                        data = []
                        shape = []
                        # TODO check the times for this loop
                        for i in list_id:
                            shape += [spike_trains[i-id_first_spike_detector].shape[0]]
                            data += [spike_trains[i-id_first_spike_detector]]

                        send_shape = np.array(np.concatenate(([np.sum(shape)],shape)), dtype='i')
                        # firstly send the size of the spikes train
                        self._logger.debug("sending size of train")
                        self._comm_sender.Send([send_shape, MPI.INT], dest=status_.Get_source(), tag=list_id[0])
                        
                        # secondly send the spikes train
                        data = np.concatenate(data).astype('d')
                        self._logger.debug("sending train")
                        self._comm_sender.Send([data, MPI.DOUBLE], dest=rank, tag=list_id[0])
                ### OLD code end
                self._logger.debug(f"__DEBUG__ start sending loop ends, time:{datetime.datetime.now()}")
                # synchronize
            elif status_.Get_tag() == 1:
                # NOTE: one sim step? inconsistent with receiving side
                # continue sending data
                self._logger.debug("__DEBUG__ _send(tag == 1)")
                continue
            elif status_.Get_tag() == 2:
                # everything goes fine, terminate the loop and respond with OK
                self._logger.debug('TVB_to_NEST: End of send function')
                return Response.OK
            else:
                # A 'bad' MPI tag is received,
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    logger=self._logger,
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
        translated_root_rank = self._root_transformer_rank - (
            len(self._group_of_ranks_for_sending) + len(self._group_of_ranks_for_receiving))
        while True:
            # Check if the simulation is still running
            ###############################################
            # TODO 
            # 1. receive tag from sender group
            tag = None
            count += 1
            
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
                counter = 0
                while self._data_buffer_manager.get_at(index=-1,
                                                        buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.READY_TO_TRANSFORM:
                    # wait until the transformer has filled the buffer with
                    # new data
                    counter +=1
                    time.sleep(0.001)
                    pass

                # self._logger.debug(f"__DEBUG__ _transform() while loop counterfor INPUT until buffer state is ready:{counter}")
                
                # STEP 2. Transform the data
                # print("\n"*2)
                # print(f"__DEBUG 1__ _transform() time before rate_to_spikes conversion: {datetime.datetime.now()}")
                # NOTE the results are gathered to only the root_transformer_rank
                spike_trains = self._mediator.rate_to_spikes(
                    buffer_type=DATA_BUFFER_TYPES.INPUT,
                    comm=self._mpi_com_group_transformers,
                    root_transformer_rank=translated_root_rank)
                # print(f"__DEBUG 2__ _transform() time after rate_to_spikes conversion: {datetime.datetime.now()}")
                # print("\n"*2)
                # NOTE input buffer is already marked as 'ready to receive next
                # simulation step' in self._mediator.rate_to_spikes()


                # STEP 3. put the transformed data in Output Buffer
                ###############################################
                # TODO send the spike trains to sender
                if self._intra_comm.Get_rank() == self._root_transformer_rank:
                    # self._intra_comm.Send([len(spike_trains),1,MPI.INT], self._root_sending_rank, tag=MPI.ANY_TAG)
                    self._intra_comm.send(spike_trains, self._root_sending_rank, tag=0)
                ###############################################

                # sync up point
                # wait until root transformer rank sends the data
                self._logger.debug("waiting for root to finish with sending"
                                   " the data")
                self._mpi_com_group_transformers.Barrier()

                # continue transformation
                self._logger.debug(f"__DEBUG__ _transform() start loop ends, time:{datetime.datetime.now()}")

                continue

            elif tag == 1:
                    # NOTE: one sim step? inconsistent with receiving side
                    # continue sending data
                    # self._mpi_com_group_transformers.Barrier()
                    continue
            elif tag == 2:
                # everything goes fine, terminate the loop and respond with OK
                # self._logger.debug('TVB_to_NEST: End of transform function')
                self._mpi_com_group_transformers.Barrier()
                 # Mark as 'ready to do analysis/transformation'
                
                # self._data_buffer_manager.set_ready_state_at(index=-1,
                #                                             state=DATA_BUFFER_STATES.TERMINATE,
                #                                             buffer_type=DATA_BUFFER_TYPES.INPUT)
                # self._mpi_com_group_transformers.Barrier()
                self._logger.debug('TVB_to_NEST: End of transform function')
                return Response.OK
            else:
                # A 'bad' MPI tag is received,
                # log the exception with traceback
                self._mpi_com_group_transformers.Barrier()
                interscalehub_utils.log_exception(
                    logger=self._logger,
                    log_message="bad mpi tag :",
                    mpi_tag_received=tag)
                # terminate with Error
                return Response.ERROR