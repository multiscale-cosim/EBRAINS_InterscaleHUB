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

from EBRAINS_InterscaleHUB.communicators.base_communicator import BaseCommunicator
from EBRAINS_InterscaleHUB.common import interscalehub_utils
from EBRAINS_InterscaleHUB.common.interscalehub_enums import DATA_BUFFER_STATES, DATA_BUFFER_TYPES

from EBRAINS_RichEndpoint.application_companion.common_enums import Response


class NestCommunicator(BaseCommunicator):
    '''
    Implements the BaseCommunicator. It 
    1) Receives the data from NEST
    2) Transforms it to the required format such as to 'rate'
    3) Sends the transformed data to TVB
    '''
    def __init__(self,
                 configurations_manager,
                 log_settings,
                 data_buffer_manager,
                 intra_comm,
                 receiver_inter_comm,
                 sender_inter_comm,
                 sender_group_ranks,
                 receiver_group_ranks,
                 root_transformer_rank,
                 spike_detector_ids):
        # initialize the common settings such as logger, data buffer, etc.
        super().__init__(configurations_manager,
                         log_settings,
                         __name__,
                         data_buffer_manager,
                         intra_comm,
                         receiver_inter_comm,
                         sender_inter_comm,
                         sender_group_ranks,
                         receiver_group_ranks,
                         root_transformer_rank
                         )
        self.__spike_detector_ids = spike_detector_ids
        
        interscalehub_utils.info_log_message(rank=self._my_rank,
                                             logger=self._logger,
                                             msg="Initialized")

    def __check_nest_status(self, comm, num_remote_ranks, status_nest):
        """
            helper function for first handshake with NEST and checks if NEST
            is ready to receive/send
        """
        check = np.empty(1,dtype='b')
        comm.Recv([check, 1, MPI.CXX_BOOL], source=0, tag=MPI.ANY_TAG, status=status_nest)
        status_rank_0 = status_nest.Get_tag()
        for rank in range(1,num_remote_ranks):
            comm.Recv([check, 1, MPI.CXX_BOOL], source=rank, tag=MPI.ANY_TAG, status=status_nest)
            # Check if the state of the NEST is different between the ranks
            if status_rank_0 != status_nest.Get_tag():
                # Log the exception with traceback
                interscalehub_utils.log_exception(
                    log_message="Abnormal state : the state of Nest is "
                                "different between rank. Tag received: ",
                    mpi_tag_received=status_nest.Get_tag())
                # Terminate with Error
                return Response.ERROR
        
        # everything went well
        return status_nest
    
    def __send_simulation_status_to_transformers(self,
                                                 root_rank,
                                                 is_simulation_running):
        """sends the current simulation staus to transformers"""
        if self._intra_comm.Get_rank() == root_rank:
            self._intra_comm.send(is_simulation_running,
                                  dest=self._root_transformer_rank,
                                  tag=0)
    
    def receive(self):
        '''
            Receives data from NEST on rank 0 and puts it into the INPUT buffer
        '''
        # NOTE The last two buffer indices are used for setting up buffer
        # states and last index of data received (i.e. size of data)
        self._num_sending = self._receiver_inter_comm.Get_remote_size()
        root_receiving_rank = self._group_of_ranks_for_receiving[0]
        size = np.empty(1, dtype='i')    
        status_nest = MPI.Status()
        self._logger.info("start receiving from NEST")
        while True:
            raw_data_end_index = 0  # head of the buffer, reset after each iteration
            status_nest = self.__check_nest_status(self._receiver_inter_comm,
                                                   self._num_sending,
                                                   status_nest)
            if status_nest == Response.ERROR:
                # something went wrong
                # NOTE a specific exception is already logged with traceback
                # return with ERROR to terminate
                return Response.ERROR

            # Test, check the current status of simulation
            # Case a, simulaiton is still running
            if status_nest.Get_tag() == 0:
                # Case one-way communication i.e. only receiving from simulator
                if not self._group_of_ranks_for_sending:
                    # send the current simulation staus to transformers
                    self.__send_simulation_status_to_transformers(
                        root_rank=root_receiving_rank,
                        is_simulation_running=True)
                # NOTE consider using MPI, remove the sleep and refactor
                # while loop to something more efficient

                # wait until Transformer communciator set the buffer state
                while self._data_buffer_manager.get_at(
                    index=-1,
                    buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.READY_TO_RECEIVE:
                    time.sleep(0.001)
                    continue

                # Recevie the data from all NEST ranks

                # NOTE the following 3 MPI calls are matching the protocol of
                # mpi_backend_io in NEST
                for source in range(self._num_sending):
                    # i) send 'ready' to the nest rank
                    self._receiver_inter_comm.Send([np.array(True,dtype='b'),MPI.BOOL],dest=source,tag=0)
                    # ii) receive package size info
                    self._receiver_inter_comm.Recv([size, 1, MPI.INT], source=source, tag=0, status=status_nest)
                    # get the buffer portion to receive the next data package
                    data_buffer = self._data_buffer_manager.get_from(
                                    starting_index=raw_data_end_index,
                                    buffer_type=DATA_BUFFER_TYPES.INPUT)
                    # iii) receive the data in the buffer
                    self._receiver_inter_comm.Recv([data_buffer, MPI.DOUBLE],
                                                   source=source,
                                                   tag=0,
                                                   status=status_nest)
                    # move index
                    raw_data_end_index += size[0]
                
                # set the header to the last index where the data ends
                self._data_buffer_manager.set_header_at(index=-2,
                                                        header=raw_data_end_index,
                                                        buffer_type=DATA_BUFFER_TYPES.INPUT)
                
                # Mark as 'ready to do analysis/transform'
                self._data_buffer_manager.set_ready_state_at(index=-1,
                                                             state=DATA_BUFFER_STATES.READY_TO_TRANSFORM,
                                                             buffer_type=DATA_BUFFER_TYPES.INPUT)

                # continue next iteration
                continue
            
            # Case b, NEST is not ready to send the data yet
            elif status_nest.Get_tag() == 1:
                # continue next iteration
                continue

            # Case c, simulation is finished
            elif status_nest.Get_tag() == 2:
                # Case one-way communication i.e. only receiving from simulator
                if not self._group_of_ranks_for_sending:
                    # send the current simulation staus to transformers
                    self.__send_simulation_status_to_transformers(
                        root_rank=root_receiving_rank,
                        is_simulation_running=False)

                # everything goes fine
                self._logger.info('NEST: End of receive function')
                # terminate the loop and respond with OK
                return Response.OK
                
            # Case d,  A 'bad' MPI tag is received,
            else:
                # Case one-way communication i.e. only receiving from simulator
                if not self._group_of_ranks_for_sending:
                    # send the current simulation staus to transformers
                    self.__send_simulation_status_to_transformers(
                        root_rank=root_receiving_rank,
                        is_simulation_running=False)
                
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    log_message="bad mpi tag :",
                    mpi_tag_received=status_nest.Get_tag())
                # terminate with Error
                return Response.ERROR
    
    def send(self):
        '''
            Sends data to NEST
        '''
        # NOTE Refactoring is needed to have multiple MPI ranks possible
        self._num_receiving = self._sender_inter_comm.Get_remote_size()
        root_sending_rank = self._group_of_ranks_for_sending[0]
        num_spike_recorders = np.empty(1, dtype='i')
        status_nest = MPI.Status()
        status_transformer = MPI.Status()
        self._logger.info("start sending data to NEST")
        while True:
            status_nest = self.__check_nest_status(self._sender_inter_comm,
                                                   self._num_receiving,
                                                   status_nest)
            if status_nest == Response.ERROR:
                # something went wrong
                # NOTE a specific exception is already logged with traceback
                # return with ERROR to terminate
                return Response.ERROR
         
            # Test, check the current status of simulation
            # Case a, simualtion is still running
            if status_nest.Get_tag() == 0:
                # send the current simulation staus to transformers
                self.__send_simulation_status_to_transformers(
                    root_rank=root_sending_rank,
                    is_simulation_running=True)

                # wait to receive transformed data from transformers
                if self._intra_comm.Get_rank() == root_sending_rank:
                    spike_trains = self._intra_comm.recv(source=self._root_transformer_rank, tag=0, status=status_transformer)
                
                # send the data to all NEST ranks
                # NOTE the following 4 MPI calls are matching the protocol of
                # mpi_backend_io in NEST
                for rank in range(self._num_receiving):
                    # i) receive the number of spike recorders
                    self._sender_inter_comm.Recv([num_spike_recorders, 1, MPI.INT], source=rank, tag=0, status=status_nest)
                    if num_spike_recorders[0] != 0:
                        spike_recorder_ids = np.empty(num_spike_recorders, dtype='i')
                        # ii) receive the spike recorder ids
                        self._sender_inter_comm.Recv([spike_recorder_ids, num_spike_recorders, MPI.INT], source=status_nest.Get_source(), tag=0, status=status_nest)

                        # put the spike trains into the correct list index
                        data = []
                        shape = []
                        # NOTE this could be a bottleneck when we scale up
                        for i in spike_recorder_ids:
                            shape += [spike_trains[i-self.__spike_detector_ids].shape[0]]
                            data += [spike_trains[i-self.__spike_detector_ids]]
                        shape_of_spike_trains = np.array(np.concatenate(([np.sum(shape)],shape)), dtype='i')

                        # iii) send the list of shapes of the spike trains
                        self._sender_inter_comm.Send([shape_of_spike_trains, MPI.INT], dest=status_nest.Get_source(), tag=spike_recorder_ids[0])
                        
                        # iv) send the spike trains
                        data = np.concatenate(data).astype('d')
                        self._sender_inter_comm.Send([data, MPI.DOUBLE], dest=rank, tag=spike_recorder_ids[0])
                # continue next iteration
                continue

            # Case b, NEST is not ready yet
            elif status_nest.Get_tag() == 1:
                # continue next iteration
                continue
            
            # Case c, simulaiton is finished
            elif status_nest.Get_tag() == 2:
                # send the current simulation staus to transformers
                self.__send_simulation_status_to_transformers(
                    root_rank=root_sending_rank,
                    is_simulation_running=False)
                # everything goes fine, terminate the loop and respond with OK
                self._logger.info('NEST: End of send function')
                return Response.OK
            
            # Case d, A 'bad' MPI tag is received,
            else:
                # send the current simulation staus to transformers
                self.__send_simulation_status_to_transformers(
                    root_rank=root_sending_rank,
                    is_simulation_running=False)
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    logger=self._logger,
                    log_message="bad mpi tag :",
                    mpi_tag_received=status_nest.Get_tag())
                # terminate with Error
                return Response.ERROR
