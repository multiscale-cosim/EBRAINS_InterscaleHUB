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
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_BUFFER_STATES, DATA_BUFFER_TYPES

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class InterscaleHubMediator:
    def __init__(self,  configurations_manager, log_settings,
                 transformer, analyzer, data_buffer_manager):
        self.__logger = configurations_manager.load_log_configurations(
                        name="InterscaleHub -- Mediator",
                        log_configurations=log_settings,
                        target_directory=DefaultDirectories.SIMULATION_RESULTS)

        self.__transformer = transformer
        self.__analyzer = analyzer
        self.__data_buffer_manager = data_buffer_manager

        self.__logger.info("initialized")

    def rate_to_spikes(self, buffer_type, comm, root_transformer_rank):
        '''converts rate to spike trains'''
        # NOTE the first two indexes are always the time steps
        time_step = self.__data_buffer_manager.get_upto(end_index=2, buffer_type=buffer_type)
        raw_data_end_index = int(self.__data_buffer_manager.get_at(index=-2, buffer_type=buffer_type))
        rates = self.__data_buffer_manager.get_from_range(
            start=2,
            end=raw_data_end_index,
            buffer_type=buffer_type)
        if comm.Get_rank() == root_transformer_rank:
            self.__logger.info(f"__DEBUG__ time_step: {time_step}, "
                               f"raw_data_end_index:{raw_data_end_index}, "
                               f"rates:{rates}")
        
        # NOTE Mark the input buffer as 'ready to receive next simulation step'
        # TODO set it from communicator class instead
        comm.Barrier()
        self.__data_buffer_manager.set_ready_state_at(index=-1,
                                            state=DATA_BUFFER_STATES.READY_TO_RECEIVE,
                                            buffer_type=DATA_BUFFER_TYPES.INPUT)
        spike_trains = self.__transformer.rate_to_spikes(time_step, rates, comm, root_transformer_rank)
        self.__logger.debug(f'spikes after conversion: {spike_trains}')
        return spike_trains

    def spikes_to_rate(self, count, size_at_index, buffer_type):
        '''
        Two step conversion from spikes/spike events to firing rates.
        '''
        # TODO refactor buffer indexing and buffer access inside analyzer and transformer
        buffer_size = self.__data_buffer_manager.get_at(index=size_at_index, buffer_type=buffer_type)
        data_buffer = self.__data_buffer_manager.get_buffer(buffer_type=buffer_type)
        # 1) spike to spike_trains in transformer
        spike_trains = self.__transformer.spike_to_spiketrains(count, buffer_size, data_buffer)
        self.__logger.debug(f'transformed spike trains: {spike_trains}')
        # 2) spike_trains to rate in analyzer
        times, rate = self.__analyzer.spiketrains_to_rate(count, spike_trains)
        self.__logger.debug(f'analyzed rates, time: {times}, rate: {rate}')
        self.__logger.info(f'__DEBUG__ analyzed time: {times}')
        return times, rate
