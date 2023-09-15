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
# ------------------------------------------------------------------------------
from EBRAINS_InterscaleHUB.managers.usecase_specific.base_manager import BaseManager
from EBRAINS_InterscaleHUB.communicators.nest.nest_communicator import NestCommunicator
from EBRAINS_InterscaleHUB.communicators.tvb.tvb_communicator import TVBCommunicator
from EBRAINS_InterscaleHUB.communicators.transformer.transformer_communicator import TransformerCommunicator
from EBRAINS_InterscaleHUB.common.interscalehub_enums import DATA_EXCHANGE_DIRECTION, TRANSLATION_FUNCTION_ID
from EBRAINS_InterscaleHUB.common.interscalehub_utils import info_log_message, debug_log_message

from EBRAINS_RichEndpoint.application_companion.common_enums import Response
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers.xml2class_parser import Xml2ClassParser
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories



class TvbNestManager(BaseManager):
    """
    Implements the InterscaleHubBaseManager to
    1) Interact with InterscaleHub Facade to steer the execution
    2) Manage the InterscaleHub functionality.
    """
    # NOTE two different sources of parameters TODO refactor
    # TODO Refactor comments/ docstrings
    def __init__(self, parameters, configurations_manager, log_settings,
                 direction, sci_params_xml_path_filename=''):
        """
          implements BaseManger
        
        :param parameters
        :param direction
        :param configurations_manager:
        :param log_settings:
        :param sci_params_xml_path_filename:
        """
        self.__log_settings = log_settings
        self.__configurations_manager = configurations_manager
        self.__logger = self.__configurations_manager.load_log_configurations(
                                        name=f"InterscaleHub -- {DATA_EXCHANGE_DIRECTION(direction).name} Manager",
                                        log_configurations=self.__log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)

        # 1) Initialize usecase parameters based on direction e.g. TVB to NEST
        # or NEST to TVB
        # TODO refactor to get parameters from a signle source
        self.__parameters = parameters
        self.__sci_params = Xml2ClassParser(sci_params_xml_path_filename, self.__logger)
        self.__direction = direction
        buffer_size = None
        self.__translation_function = None

        # TODO get these settings via XML configurations file
        # NOTE Refactoring of communication protocols and data management is
        # needed when more than one ranks are used for
        # intercommunication (sending, receiving) with simulators
        if self.__direction == DATA_EXCHANGE_DIRECTION.TVB_TO_NEST:
            receiver_group_ranks = [1]  # NOTE hardcoded rank-1
            sender_group_ranks = [0]  # NOTE hardcoded rank-0
            self.__translation_function_id = TRANSLATION_FUNCTION_ID.RATE_TO_SPIKES
            # set buffer size
            buffer_size = self.__sci_params.max_events + self.__sci_params.tvb_buffer_size_factor
        elif self.__direction == DATA_EXCHANGE_DIRECTION.NEST_TO_TVB:
            receiver_group_ranks = [0]  # NOTE hardcoded rank-0
            sender_group_ranks = [1]  # NOTE hardcoded rank-1
            self.__translation_function_id = TRANSLATION_FUNCTION_ID.SPIKE_TO_RATES
            # set buffer size
            buffer_size = self.__sci_params.max_events * self.__sci_params.nest_buffer_size_factor

        # 2) initialize the base class parameters
        super().__init__(self.__configurations_manager,
                         self.__log_settings,
                         receiver_group_ranks,
                         sender_group_ranks,
                         buffer_size,
                         self.__parameters,
                         self.__sci_params,
                         self.__direction
                         )
        
        info_log_message(self._my_rank,
                         self._logger,
                         "initialized")

    def start(self, *args, **kwargs):
            """
            implementation of abstract method to start transformation and
            exchanging the data with TVB and NEST.
            """
            info_log_message(self._my_rank,
                             self._logger,
                             "Start data transfer and use case science...")

            # STEP 1) initialize usecase parameters
            root_transformer_rank = self._transformer_group_ranks[0]

            # NOTE TODO change it to keyword arguments to receive a
            # dictionary of parameters from adapter
            spike_detector_ids = args[0]
            
            # STEP 2) initialize Communicators
            self.__nest_communicator = NestCommunicator(
                self.__configurations_manager,
                self.__log_settings,
                self._data_buffer_manager,
                self._intra_comm,
                self._receiver_inter_comm,
                self._sender_inter_comm,
                self._sender_group_ranks,
                self._receiver_group_ranks,
                root_transformer_rank,  # root transformer rank
                spike_detector_ids
                )
            
            self.__tvb_communicator = TVBCommunicator(
               self.__configurations_manager,
               self.__log_settings,
               self._data_buffer_manager,
               self._intra_comm,
               self._receiver_inter_comm,
               self._sender_inter_comm,
               self._sender_group_ranks,
               self._receiver_group_ranks,
               root_transformer_rank  # root transformer rank
               )

            self.__transformer_communicator = TransformerCommunicator(
                self.__configurations_manager,
                self.__log_settings,
                self._intra_comm,
                self._transformer_intra_comm,
                self._sender_group_ranks,
                self._receiver_group_ranks,
                self._transformer_group_ranks,
                self._data_buffer_manager,
                self.__parameters,
                self.__sci_params,
                self.__translation_function_id,
                self.__translation_function
            )
            
            my_rank = self._intra_comm.Get_rank()
           
            # STEP 3) start exchanging the data 
            # Case a, if rank is in group of senders
            if self.__direction == DATA_EXCHANGE_DIRECTION.TVB_TO_NEST:
                if my_rank in self._sender_group_ranks:
                    # set inter_communicator for sending the data
                    # self._num_receiving = self._sender_inter_comm.Get_remote_size()
                    # self._logger.debug(f"num_receiving:{self._num_receiving}")
                    response = self.__nest_communicator.send()
                      
                elif my_rank in self._receiver_group_ranks:
                    # set inter_communicator for receiving the data
                    # self._num_sending = self._receiver_inter_comm.Get_remote_size()
                    # self._logger.debug(f"num_sending:{self._num_sending}")
                    response = self.__tvb_communicator.receive()
                
                elif my_rank in self._transformer_group_ranks:
                    response = self.__transformer_communicator.transform()
                
            elif self.__direction == DATA_EXCHANGE_DIRECTION.NEST_TO_TVB:
                if my_rank in self._sender_group_ranks:
                    # set inter_communicator for receiving the data
                    # self._num_receiving = self._sender_inter_comm.Get_remote_size()
                    # self._logger.debug(f"num_sending:{self._num_sending}")
                    response = self.__tvb_communicator.send()

                elif my_rank in self._receiver_group_ranks:
                    # self._num_sending = self._receiver_inter_comm.Get_remote_size()
                    # self._logger.debug(f"num_receiving:{self._num_receiving}")
                    response = self.__nest_communicator.receive()
    
                elif my_rank in self._transformer_group_ranks:
                    response = self.__transformer_communicator.transform()

            # finish with execution
            debug_log_message(
                0,
                self._logger,
                f"rank: {self._my_rank} - conluding the data exchange")

            # TODO add functionality to act accordingly when response is ERROR
            # For example, consider signalling other MPI groups somehow to quit
            # forcefully if ERROR is received as a response
            
            # send the response that is received from the data exchange
            # operation
            return response

                
    def stop(self):
        """Closes the data channels"""
        self._close_data_channels()
        return Response.OK
