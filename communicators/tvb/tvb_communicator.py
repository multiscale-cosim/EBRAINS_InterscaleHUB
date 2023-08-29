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


class TVBCommunicator(BaseCommunicator):
    '''
    Implements the BaseCommunicator. It ...
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
                 root_transformer_rank):
       
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
        
        interscalehub_utils.info_log_message(rank=self._my_rank,
                                             logger=self._logger,
                                             msg="Initialized")

    def receive(self):
        '''
            Receives data from TVB on rank 0 and puts it into the INPUT buffer.
        '''
        size = np.empty(1, dtype='i') # size of the rate-array
        status_tvb = MPI.Status()
        self._num_sending = self._receiver_inter_comm.Get_remote_size()
        self._logger.info("start receiving from TVB")
        while True:
            # send ready status to TVB (see TVB MPI wrapper)
            requests=[]
            for rank in range(self._num_sending):
                requests.append(self._receiver_inter_comm.isend(True,dest=rank,tag=0))
            MPI.Request.Waitall(requests)

            # NOTE the following 3 MPI calls are matching the protocol of
            # TVB MPI wrapper

            # 1) get the starting and ending time of the simulation step, and
            # current stauts of simulation
            simulation_step = self._data_buffer_manager.get_from_range(
                    start=0,
                    end=2,
                    buffer_type=DATA_BUFFER_TYPES.INPUT)
            self._receiver_inter_comm.Recv([simulation_step, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_tvb)

            # Test, check the current status of simulation
            # Case a, simulation is still running
            if status_tvb.Get_tag() == 0:
                # wait until Transformer communciator set the buffer state
            
                # NOTE consider using MPI, remove the sleep and refactor
                # while loop to something more efficient            
                while self._data_buffer_manager.get_at(
                    index=-1, buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.READY_TO_RECEIVE:
                    time.sleep(0.001)
                    continue

                # 2) Get the size/shape of the data
                self._receiver_inter_comm.Recv([size, 1, MPI.INT], source=status_tvb.Get_source(), tag=0, status=status_tvb)
                
                # 3) receive data
                # data buffer to receive the data
                simulation_step = self._data_buffer_manager.get_from(starting_index=2,
                                                                 buffer_type=DATA_BUFFER_TYPES.INPUT)
                self._receiver_inter_comm.Recv([simulation_step, MPI.DOUBLE], source=status_tvb.Get_source(), tag=0, status=status_tvb)
                # set the header (i.e. the last index where the data ends)
                # NOTE because the first two values are always the time steps,
                # and the data starts from index 2, so increase the size by 2
                raw_data_end_index = size+2
                self._data_buffer_manager.set_header_at(index=-2,
                                                    header=raw_data_end_index,
                                                    buffer_type=DATA_BUFFER_TYPES.INPUT)
                
                # Mark as 'ready to do analysis/transformation'
                self._data_buffer_manager.set_ready_state_at(index=-1,
                                                            state=DATA_BUFFER_STATES.READY_TO_TRANSFORM,
                                                            buffer_type=DATA_BUFFER_TYPES.INPUT)

                # continue next iteration
                continue
            
            # Case b, simulation is ended
            elif status_tvb.Get_tag() == 1:
                # everything goes fine, terminate the loop and respond with OK
                self._logger.info('TVB_to_NEST: End of receive function')
                return Response.OK

            # Case c, A 'bad' MPI tag is received
            else:
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    log_message="bad mpi tag :",
                    mpi_tag_received=status_tvb.Get_tag())
                # terminate with Error
                return Response.ERROR
            
    def send(self):
        '''
            Sends data to TVB
        '''
        status_tvb = MPI.Status()
        status_transformer = MPI.Status()
        check = np.empty(1,dtype='i')
        root_sending_rank = self._group_of_ranks_for_sending[0]
        while True:
            # get the current status of the simulation
            req = self._sender_inter_comm.irecv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
            req.wait(status_tvb)  # wait until TVB is ready to receive

            # Test, check the current status of simulation
            # Case a, simualtion is still running
            if status_tvb.Get_tag() == 0:
                # send tag (received from TVB) to transformers to let it determine
                # if the simulaiton is still running
                if self._intra_comm.Get_rank() == root_sending_rank:
                    self._intra_comm.send(True, dest=self._root_transformer_rank, tag=0)
            
                # wait to receive translated data from transformers
                if self._intra_comm.Get_rank() == root_sending_rank:
                    times, data = self._intra_comm.recv(source=self._root_transformer_rank, tag=0, status=status_transformer)

                # send the data to TVB
                # NOTE the following 3 MPI calls are matching the protocol of
                # TVB MPI Wrapper

                # i) send the (start and end) time of simulation step
                self._sender_inter_comm.Send([times, MPI.DOUBLE], dest=status_tvb.Get_source(), tag=0)
                
                # ii)send the size of the data
                size = np.array(int(data.shape[0]),dtype='i')
                self._sender_inter_comm.Send([size,MPI.INT], dest=status_tvb.Get_source(), tag=0)
                
                # iii) send the data
                self._sender_inter_comm.Send([data,MPI.DOUBLE], dest=status_tvb.Get_source(), tag=0)
                
                # continue next iteration
                continue
            
            # Case b, simulation is ended
            elif status_tvb.Get_tag() == 1:
                # send the current simulation staus to transformers
                if self._intra_comm.Get_rank() == root_sending_rank:
                    self._intra_comm.send(False, dest=self._root_transformer_rank, tag=0)
                # everything goes fine, terminate the loop and respond with OK
                return Response.OK
            
            # Case c, A 'bad' MPI tag is received
            else:
                # send the current simulation staus to transformers
                if self._intra_comm.Get_rank() == root_sending_rank:
                    self._intra_comm.send(False, dest=self._root_transformer_rank, tag=0)
                # log the exception with traceback
                interscalehub_utils.log_exception(
                    log_message="bad mpi tag :",
                    mpi_tag_received=status_tvb.Get_tag())
                # terminate with Error
                return Response.ERROR