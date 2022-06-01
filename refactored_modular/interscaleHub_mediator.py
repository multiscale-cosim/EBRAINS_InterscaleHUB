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
        # TODO: make correct calls to data_buffer_manager
        if self.__data_buffer_manager.get_at(index=-2) == DATA_BUFFER_STATES.HEAD:
            spikes = self.__transformer.rate_to_spikes(data_buffer[:2],
                                                       data_buffer[2:])
        else:
            spikes = self.__transformer.rate_to_spikes(data_buffer[:2],
                                                       data_buffer[2:int(data_buffer[-2])])
        # NOTE: former call to function, removed unused first argument 'count'
        # if int(data_buffer[-2]) == 0:
        #     spikes_times = self.__transformer.rate_to_spikes(0,
        #                                         data_buffer[:2],
        #                                         data_buffer[2:])
        # else:
        #     spikes_times = self.__transformer.rate_to_spikes(0,
        #                                         data_buffer[:2],
        #                                         data_buffer[2:int(data_buffer[-2])])

    def spike_to_rate(self, count, buffer_size, data_buffer):
        '''
        Two step conversion from spikes/spike events to firing rates.
        TODO: use databuffer manager
        '''
        # 1) spike to spiketrains in transfomer
        spiketrains = self.__transformer.spike_to_spiketrains(count, buffer_size, data_buffer)
        # 2) spiketrains to rate in analyzer
        times,data = self.__analyzer.spiketrains_to_rate(count, spiketrains)

''' NOTE: split spike_to_rate call into two?
    def __spike_to_spike_train(self, count, data_buffer):
        pass

    def __spike_train_to_rate(self, count, data_buffer):
        pass
'''
