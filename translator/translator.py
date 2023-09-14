# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------
from EBRAINS_InterscaleHUB.translator.elephant_delegator import ElephantDelegator
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_InterscaleHUB.common.interscalehub_enums import TRANSLATION_FUNCTION_ID
from EBRAINS_InterscaleHUB.common.interscalehub_utils import debug_log_message


class Translator:
    """
    Translates the data e.g. to change the scales.
    It wraps the functionality of the libraries such as ELEPHANT for
    transformation and analysis.

    NOTE This wrapper class exposes only the functionality that is supported by
    InterscaleHub.
    """
    def __init__(self, configurations_manager, log_settings, params, sci_params):
        """
        """
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="InterscaleTranslator",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        self.__params = params
        self.__sci_params = sci_params
        self.__elephant_delegator = ElephantDelegator(configurations_manager,
                                                      log_settings,
                                                      sci_params=sci_params)
        self.__logger.debug("Initialised")
    
    def translate(self,
                  translation_function_id,
                  translation_function,
                  count,
                  raw_data,
                  transformer_intra_comm,
                  transformers_root_rank,
                  *args):
        
        if translation_function_id == TRANSLATION_FUNCTION_ID.SPIKE_TO_RATES:
            return self._spikes_to_rates(count, raw_data, transformer_intra_comm,
                                        transformers_root_rank)
        
        if translation_function_id == TRANSLATION_FUNCTION_ID.RATE_TO_SPIKES:
            return self._rate_to_spikes(raw_data, transformer_intra_comm,
                                       transformers_root_rank)
        
        if translation_function_id == TRANSLATION_FUNCTION_ID.USER_LAND:
            # translation function is defined in dir
            # /userland/translation_funcitons/...
            return translation_function(raw_data,
                                        transformer_intra_comm,
                                        transformers_root_rank)
    
    def _spikes_to_rates(self, count, data, comm, root_transformer_rank):
        """
        i) Transforms the data from spikes to spiketrains, and then
        ii) converts spiketrains into rates.
        
        Parameters
        ----------
        count: int
            counter of the number of time of the transformation
            (identify the timing of the simulation)

        data_size : int
            size of the data to be read from the buffer for transformation

        data_buffer: MPI shared memory window
            buffer contains id of devices, id of neurons and spike times

        Returns
        ------
             times, rate: numpy array, float
                tuple of interval and the rate for the interval if data is
                transformed successfully
        """
        # 1) prepare spike events from raw data
        nb_neurons = self.__sci_params.nb_neurons
        first_neuron_id = self.__params['id_first_neurons'][0]
        spike_events = [[] for _ in range(nb_neurons)]
        
        # NOTE NEST sends 3 values for each spike event i.e.
        # (spike detector id, neuron id, spike time)
        # --> Assumption: len(data) is always a multiple of 3
        for index in range(int(len(data)/3)):
            neuron_id = int(data[index*3+1])
            spike_time = data[index*3+2]
            spike_events[neuron_id - first_neuron_id].append(spike_time)
        
        # 2) transform spikes to spike_trains
        spike_trains = self.__elephant_delegator.spike_events_to_spiketrains(
            count,
            spike_events,
            comm,
            root_transformer_rank)
        
        # 3) convert the spike_trains to rate
        # NOTE only root rank has the result (spike_trains)
        times = None
        rate = None
        if comm.Get_rank() == root_transformer_rank:
            times, rate = self.__elephant_delegator.spiketrains_to_rate(count, spike_trains)
        # wait until root rank is done with analysis
        debug_log_message(rank=0,  # hardcoded
                          logger=self.__logger,
                          msg="wait until root transformer converts the "
                          "spike_trains to rate")
        comm.Barrier()
        return times, rate

    def _rate_to_spikes(self, raw_data, transformer_intra_comm, transformers_root_rank):
        """Transforms the data from one format to another .
        
        Parameters
        ----------
        time_step: [int, int]
            time start and time stops of the current simulation step
        
        data_buffer: MPI shared memory window
            buffer contains the rate to be converted into spikes
        
        Returns
        ------
            returns the spike trains from rate
        """
        # NOTE the first two indexes are always the time steps
        time_step = raw_data[:2]
        rates = raw_data[2:]
        return self.__elephant_delegator.rate_to_spikes(time_step,
                                                        rates,
                                                        transformer_intra_comm,
                                                        transformers_root_rank)
