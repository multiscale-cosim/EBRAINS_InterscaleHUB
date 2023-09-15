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

from EBRAINS_InterscaleHUB.common.interscalehub_utils import info_log_message
from EBRAINS_InterscaleHUB.common.interscalehub_utils import wait_until_buffer_ready
from EBRAINS_InterscaleHUB.common.interscalehub_utils import debug_log_message
from EBRAINS_InterscaleHUB.common.interscalehub_enums import DATA_BUFFER_STATES, DATA_BUFFER_TYPES
from EBRAINS_InterscaleHUB.translator.translator import Translator

from EBRAINS_RichEndpoint.application_companion.common_enums import Response
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class TransformerCommunicator:
    def __init__(self,
                 configurations_manager,
                 log_settings,
                 intra_comm,
                 transformer_intra_comm,
                 sender_group_ranks,
                 receiver_group_ranks,
                 transformer_group_ranks,
                 data_buffer_manager,
                 parameters,
                 sci_params,
                 translation_function_id,
                 translation_function):
        
        # intialize parameters
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self._logger = self._configurations_manager.load_log_configurations(
            name="InterscaleHub -- Transfomer",
            log_configurations=self._log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        self._intra_comm = intra_comm
        self._transformer_intra_comm = transformer_intra_comm
        self._transformer_group_ranks = transformer_group_ranks
        self._sender_group_ranks = sender_group_ranks
        self._translation_function_id = translation_function_id
        self._translation_function = translation_function
        self._data_buffer_manager = data_buffer_manager
        # TODO refactor to get parameters from a signle source (see Base Manager)
        self._parameters = parameters
        self._sci_params = sci_params
        self._my_rank = self._intra_comm.Get_rank()
        # NOTE root_transformer_rank = rank id BEFORE group creation
        # translated_root_rank = rank id WITHIN the new group (shifted by size of the other groups)
        self._root_transformer_rank = transformer_group_ranks[0]
        if sender_group_ranks:
            self._root_sending_rank = sender_group_ranks[0]
        if receiver_group_ranks:
            self._root_receiver_rank = receiver_group_ranks[0]
        self._translated_root_rank = self._root_transformer_rank - (
            len(self._sender_group_ranks) + len(receiver_group_ranks))
        
        # Translator
        self._translator = Translator(
            self._configurations_manager,
            self._log_settings,
            self._parameters,
            self._sci_params)
        
        info_log_message(self._my_rank, self._logger, "initialized")

    def __get_data(self, buffer_type):
        '''converts rate to spike trains'''
        raw_data_end_index = int(self._data_buffer_manager.get_at(index=-2, buffer_type=buffer_type))
        received_data = self._data_buffer_manager.get_from_range(
            start=0,
            end=raw_data_end_index,
            buffer_type=buffer_type)
       
        return received_data
        
    def __set_buffer_ready(self, buffer_type, state):
        self._data_buffer_manager.set_ready_state_at(index=-1,
                                                     state=state,
                                                     buffer_type=buffer_type)

    def __is_simulation_running(self):
        """helper function to determine whether simulation is still running"""
        check = None
        status_ = MPI.Status()
        # Case a, two-way communication with simulator
        if self._sender_group_ranks and self._intra_comm.Get_rank() == self._root_transformer_rank:
            check = self._intra_comm.recv(source=self._root_sending_rank,
                                          tag=MPI.ANY_TAG,
                                          status=status_)

        # Case b, one-way communication i.e. only receiving data from the simulator
        elif not self._sender_group_ranks and self._intra_comm.Get_rank() == self._root_transformer_rank:
            check = self._intra_comm.recv(source=self._root_receiver_rank,
                                          tag=MPI.ANY_TAG,
                                          status=status_)
         
        return check

    def transform(self):
        """
            transforms the data from input buffer and sends it to Senders group
        """
        count = 0
        info_log_message(self._my_rank, self._logger, "start transformation")
        while True:
            # receive current simulation status from Sender group
            is_simulation_running = None 
            # broadcast the current simulation status
            self._logger.debug(f"broadcasting: is simulation running?")
            is_simulation_running = self._transformer_intra_comm.bcast(
                self.__is_simulation_running() , root=self._translated_root_rank)

            # Test, check the current status of simulation
            # Case a, simulation is still running
            if is_simulation_running:
                self._logger.debug("waiting until data is received")
                wait_until_buffer_ready(self._data_buffer_manager,
                                        DATA_BUFFER_TYPES.INPUT,
                                        DATA_BUFFER_STATES.READY_TO_TRANSFORM)
                
                # STEP 2. Transform the data
                # get data from INPUT buffer
                raw_data = self.__get_data(buffer_type=DATA_BUFFER_TYPES.INPUT)
                #  wait until all transformers get the data from buffer
                self._logger.debug("waiting unitl data is fetched from buffer")
                self._transformer_intra_comm.Barrier()
                # NOTE Mark the input buffer as
                # 'ready to receive next simulation step'
                if self._data_buffer_manager.get_at(index=-1, 
                                              buffer_type=DATA_BUFFER_TYPES.INPUT) != DATA_BUFFER_STATES.TERMINATE:

                    self.__set_buffer_ready(buffer_type=DATA_BUFFER_TYPES.INPUT,
                                            state=DATA_BUFFER_STATES.READY_TO_RECEIVE)

                # STEP 3. translate the data
                # NOTE the results are gathered to only the root_transformer_rank
                self._logger.debug("translating the data")
                translated_data = self._translator.translate(
                    self._translation_function_id,
                    self._translation_function,
                    count,
                    raw_data,
                    self._transformer_intra_comm,
                    self._translated_root_rank)
                
                # STEP 4. send the translated data to Senders group
                if self._sender_group_ranks and self._intra_comm.Get_rank() == self._root_transformer_rank:
                    self._intra_comm.send(translated_data,
                                          self._root_sending_rank,
                                          tag=0)
                # wait until root transformer rank sends the data
                self._logger.debug("waiting for root to finish with sending")
                self._transformer_intra_comm.Barrier()

                # continue next iteration
                count += 1
                continue
            
            # Case b, simulation is finished
            else:
                # terminate the loop and respond with OK
                info_log_message(self._transformer_intra_comm.Get_rank(),
                                 self._logger,
                                 'concluding transformation')
                return Response.OK