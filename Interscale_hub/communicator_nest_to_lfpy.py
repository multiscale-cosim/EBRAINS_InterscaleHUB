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

from EBRAINS_InterscaleHUB.Interscale_hub.communicator_base import BaseCommunicator
from EBRAINS_InterscaleHUB.Interscale_hub import interscalehub_utils
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_BUFFER_STATES

from EBRAINS_RichEndpoint.application_companion.common_enums import Response

from science.models.lfpykernels_PotjansDiesmann import PotjansDiesmannKernels


class CommunicatorNestLFPY(BaseCommunicator):
    '''
    Implements the BaseCommunicator. It 
    1) Receives the data from NEST
    2) Calculates LFP signals through the kernel approach
    3) Plots results, and saves to file
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
        
        self._logger.info("Initialized")

        self._logger.info("Initiating kernels")

    def start(self, intra_communicator, inter_comm_receiver,
              inter_comm_sender, spike_detectors_ids):
        '''
        implements the abstract method to start
        1) receiving the data
        2) transforming the data to required scale
        3) sending the transformed data
        
        M:N mapping of MPI ranks, receive data, further process data.
        Receive on rank 0, do the rest on rest of the ranks.
        '''

        # Just a placeholder, will be received as input:
        # spike_recorder_ids = np.arange(7718, 7727)
        # This is maybe not the best place to put this call?
        self._logger.debug(f"spike_detectors_ids:{spike_detectors_ids}")
        self.PD_kernels = PotjansDiesmannKernels(spike_detectors_ids)

        # Rank-0 will receive the data
        if intra_communicator.Get_rank() == 0:
            # set inter_communicator for receiving the data
            self._comm_receiver = inter_comm_receiver
            self._num_sending = self._comm_receiver.Get_remote_size()
            return self._receive()

        # NOTE its, one-way communication at the moment
        # Rank-1 will transform and send the data
        elif intra_communicator.Get_rank() == 1:
            # set inter_communicator for sending the data
            self._comm_sender = inter_comm_sender
            return self._transform()

    def stop(self):
        '''
        TODO: proper execution of stop command
        '''
        # self.__stop = True

        self.PD_kernels.save_final_results()
        self.PD_kernels.plot_final_results()

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
        self._data_buffer_manager.set_ready_at(index=-1)

        # marks the 'head' of the buffer
        # self.__databuffer[-2] = 0
        self._data_buffer_manager.set_header_at(index=-2)

        # It seems the 'check' variable is used to receive tags from NEST,
        # i.e. ready for send...
        # change this in the future, also mentioned in the FatEndPoint solution
        # from Wouter.
        check = np.empty(1, dtype='b')
        shape = np.empty(1, dtype='i')

        # NOTE count is used to calculate t_start, t_stop inside
        # spike to rate conversion function. It is important to start it
        # from -1 for one way simulaiton because InterscaelHub recevies the
        # data after NEST simulate one step.
        count = -1
        status_ = MPI.Status()
        self._logger.info("reading from buffer")

        running_head = 0  # head of the buffer, reset after each iteration
        while True:
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
                # wait until ready to receive new data (i.e. the sender has cleared the buffer)

                # TODO: use MPI, remove the sleep
                # # while self.__databuffer[-1] != 1:
                while self._data_buffer_manager.get_at(index=-1) != DATA_BUFFER_STATES.READY:
                    time.sleep(0.001)
                    pass
                for source in range(self._num_sending):
                    # send 'ready' to the nest rank
                    # self._logger.info("send ready")
                    self._comm_receiver.Send([np.array(True, dtype='b'), MPI.BOOL], dest=source, tag=0)
                    # receive package size info
                    self._comm_receiver.Recv([shape, 1, MPI.INT], source=source, tag=0, status=status_)

                    # self._comm_receiver.Recv([shape, 1, MPI.INT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
                    # NEW: receive directly into the buffer
                    # self._comm_receiver.Recv([self.__databuffer[head_:], MPI.DOUBLE], source=source, tag=0, status=status_)
                    data_buffer = self._data_buffer_manager.get_from_range(
                        start=running_head, end=running_head + shape[0])
                    self._comm_receiver.Recv([data_buffer, MPI.DOUBLE],
                                             source=source,
                                             tag=0,
                                             status=status_)

                    ########################################################
                    # NOTE will be changed later to handle by rank =< 1
                    ########################################################
                    self._logger.debug(f"data received")
                    #self._logger.info(f"count: {count}, buffer now:{data_buffer}")

                    # times, data = self._mediator.spikes_to_rate(count,size_at_index=-2)
                    # self._logger.debug(f"data after transformation: times: {times}, data: {data}")

                    data_ = data_buffer.reshape(int(len(data_buffer)/3), 3)
                    self.PD_kernels.update(data_)
                    # NOTE here put the call to compute mediator.compute_lfpy()

                    self._logger.debug(f"data transformed!")
                    ########################################################

                # Mark as 'ready to receive next simulation step'
                # self.__databuffer[-1] = 1
                self._data_buffer_manager.set_ready_at(index=-1)
                # important: head_ is first buffer index WITHOUT data.
                # self.__databuffer[-2] = head_
                self._data_buffer_manager.set_custom_value_at(
                    index=-2,
                    value=running_head)
                # continue receiving the data
                count += 1
                running_head += shape[0]
                continue
            elif status_.Get_tag() == 1:
                # increment the count and continue receiving the data
                # count += 1
                # self._data_buffer_manager.set_ready_at(index=-1)
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

    
    def _transform(self):
        '''
        Transforms data to LFP signals (multiple MPI ranks possible).
        '''
        count = 0  # simulation/iteration step
        # status_ = MPI.Status()
        # self._logger.info("NESTtoTVB -- producer/sender -- Rank:"+str(self._comm_sender.Get_rank()))
        while True:
            # TODO: this communication has the 'rank 0' problem described in the beginning
            accept = False
            # while self._data_buffer_manager.get_at(index=-1) != DATA_BUFFER_STATES.HEADER:
            #     time.sleep(0.001)
            #     pass
            # NOTE: calling the mediator which calls the corresponding transformer functions
            # times,data = mediator.spike_to_rate(self.__databuffer, count)
            # TODO: change to inject the buffer in the wrapper method of mediator
            # times, data = spikerate.spike_to_rate(count, self.__databuffer[-2], self.__databuffer)
            self.__logger.debug("Transforming data")
            times, data = self._mediator.spikes_to_rate(count, size_at_index=-2)
            # TODO add call to LFPy kernel here
            self.__logger.debug(f"setting buffer ready")
            self._data_buffer_manager.set_ready_at(index=-1)
            self.__logger.debug(f"setting buffer ready")
            # size = np.array(int(data.shape[0]),dtype='i')
            count += 1
            continue