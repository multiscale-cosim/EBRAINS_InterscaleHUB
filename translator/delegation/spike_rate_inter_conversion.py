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
import numpy as np

from quantities import ms, Hz
from neo.core import AnalogSignal, SpikeTrain

from elephant.statistics import instantaneous_rate  # , mean_firing_rate
from elephant.kernels import RectangularKernel

# TODO:
# 'inhomogeneous_poisson_process' is deprecated; use 'NonStationaryPoissonProcess'.
# 'homogeneous_poisson_process' is deprecated; use 'StationaryPoissonProcess'.
from elephant.spike_train_generation import inhomogeneous_poisson_process  # , homogeneous_poisson_process

from EBRAINS_InterscaleHUB.common.interscalehub_utils import debug_log_message

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class SpikeRateConvertor:

    def __init__(self, configurations_manager, log_settings, sci_params=None):
        self.__logger = configurations_manager.load_log_configurations(
            name="Elephant -- spike_rate_conversion",
            log_configurations=log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)

        # time of synchronization between 2 run
        self.__time_synch = sci_params.time_syncronization
        # the resolution of the integrator
        self.__dt = sci_params.dt
        self.__nb_neurons = sci_params.nb_neurons
        self.__nb_synapse = sci_params.nb_brain_synapses

        debug_log_message(rank=0,
                          logger=self.__logger,
                          msg="Initialised")

    def spike_events_to_spiketrains(self, count, spike_events, comm, transformers_root_rank):
        """
        get the spike time from the buffer and order them by neurons
        """
        number_transformers = comm.Get_size()
        transformer_rank = comm.Get_rank()  # NOTE this is the group rank
        # split the spike_events as per number of transformers
        spike_events_per_transformer = np.array_split(range(len(spike_events)),
                                                      number_transformers)
        partial_spike_trains = []

        # compute SpikeTrains in parallel on all Transformers
        for i in spike_events_per_transformer[transformer_rank]:
            try:
                if len(spike_events[i]) > 1:
                    partial_spike_trains.append(SpikeTrain(np.concatenate(spike_events[i]) * ms,
                                                t_start=np.around(count * self.__time_synch, decimals=2),
                                                t_stop=np.around((count + 1) * self.__time_synch, decimals=2) + 0.0001))
                else:
                    partial_spike_trains.append(SpikeTrain(spike_events[i] * ms,
                                                t_start=np.around(count * self.__time_synch, decimals=2),
                                                t_stop=np.around((count + 1) * self.__time_synch, decimals=2) + 0.0001))
            except Exception as e:
                self.__logger.exception(e)
                raise
        
        # gather the results on root
        gathered_spike_trains = comm.gather(partial_spike_trains, root=transformers_root_rank)
        # flatten the nested lists on root
        if transformer_rank == transformers_root_rank:
            spike_trains = gathered_spike_trains[0]
            for rank in range(1, comm.Get_size()):
                spike_trains += gathered_spike_trains[rank]
            return spike_trains
        else:
            return None

    def spiketrains_to_rate(self, count, spiketrains):
        """
        implements the abstract method for the transformation of the
        spike trains to rate.
        """
        rates = instantaneous_rate(spiketrains,
                                   t_start=np.around(count * self.__time_synch, decimals=2) * ms,
                                   t_stop=np.around((count + 1) * self.__time_synch, decimals=2) * ms,
                                   sampling_period=(self.__dt - 0.000001) * ms, kernel=RectangularKernel(1.0 * ms))
        rate = np.mean(rates, axis=1) / 10  # the division by 10 ia an adaptation for the model of TVB
        times = np.array([count * self.__time_synch, (count + 1) * self.__time_synch], dtype='d')
        return times, rate

    def rate_to_spikes(self, time_step, rates, comm, transformers_root_rank):
        """
        implements the abstract method for the transformation of the
        rate to spikes.
        """
        # rate of poisson generator ( due property of poisson process)
        rate_of_poisson_generator = rates * self.__nb_synapse
        rate_of_poisson_generator += 1e-12
        rate_of_poisson_generator = np.abs(rate_of_poisson_generator)  # avoid rate equals to zeros
        signal = AnalogSignal(rate_of_poisson_generator * Hz, t_start=(time_step[0] + 0.1) * ms,
                              sampling_period=(time_step[1] - time_step[0]) / rate_of_poisson_generator.shape[-1] * ms)
        partial_spike_trains = []
        gathered_spike_trains = []
        number_transformers = comm.Get_size()
        transformer_rank = comm.Get_rank()  # NOTE this is the group rank
        neuron_chunks_per_transformer = np.array_split(range(self.__nb_neurons), number_transformers)
        # split the computation
        for _ in range(len(neuron_chunks_per_transformer[transformer_rank])):
            # TODO: 'inhomogeneous_poisson_process' is deprecated; use 'NonStationaryPoissonProcess'.
            partial_spike_trains.append(np.around(np.sort(inhomogeneous_poisson_process(signal, as_array=True)), decimals=1))

        # gather the results at root_transformer_rank
        gathered_spike_trains = comm.gather(partial_spike_trains, root=transformers_root_rank)
        # flatten the nested lists on root
        if transformer_rank == transformers_root_rank:
            spike_trains = gathered_spike_trains[0]
            for rank in range(1, comm.Get_size()):
                spike_trains += gathered_spike_trains[rank]
            return spike_trains
        else:
            return None
