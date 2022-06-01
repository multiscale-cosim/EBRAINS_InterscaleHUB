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
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_BUFFER_STATES

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

    def rate_to_spikes(self):
        '''converts rate to spike trains'''
        if self.__data_buffer_manager.get_at(index=-2) == DATA_BUFFER_STATES.HEADER:
            time_step = self.__data_buffer_manager.get_upto(index=2)
            data_buffer = self.__data_buffer_manager.get_from(starting_index=2)
        else:
            mpi_shared_data_buffer = self.__data_buffer_manager.mpi_shared_memory_buffer
            time_step = self.__data_buffer_manager.get_upto(index=2)
            data_buffer = self.__data_buffer_manager.get_from_range(
                start=2,
                end=int(mpi_shared_data_buffer[-2]))
        
        spike_trains = self.__transformer.rate_to_spikes(time_step, data_buffer)
        self.__logger.debug(f'spikes after conversion: {spike_trains}')
        return spike_trains

    def spikes_to_rate(self, count, size_at_index):
        '''
        Two step conversion from spikes/spike events to firing rates.
        '''
        # TODO refactor buffer indexing and buffer access inside analyzer and transformer
        buffer_size = self.__data_buffer_manager.get_at(index=size_at_index)
        data_buffer = self.__data_buffer_manager.mpi_shared_memory_buffer
        # 1) spike to spike_trains in transformer
        spike_trains = self.__transformer.spike_to_spiketrains(count, buffer_size, data_buffer)
        self.__logger.debug(f'transformed spike trains: {spike_trains}')
        # 2) spike_trains to rate in analyzer
        times, data = self.__analyzer.spiketrains_to_rate(count, spike_trains)
        self.__logger.debug(f'analyzed rates, time: {times}, data: {data}')
        return times, data
