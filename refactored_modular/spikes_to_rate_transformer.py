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
import numpy as np
from quantities import ms

from neo.core import SpikeTrain

from elephant.statistics import instantaneous_rate
from elephant.kernels import RectangularKernel

from EBRAINS_InterscaleHUB.refactored_modular.interscale_transformer_base import InterscaleTransformerBaseClass
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class SpikesToRate(InterscaleTransformerBaseClass):
       
    def __init__(self, param, configurations_manager, log_settings):
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="Transformer--Spikes_to_Rate",
                                        log_configurations=self._log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)

        # time of synchronization between 2 run
        self.__time_synch = param['time_synchronization']
        
        # the resolution of the integrator
        self.__dt = param['resolution']
        self.__nb_neurons = param['nb_neurons'][0]
        
        # id of transformer is hardcoded to 0
        self.__first_id = param['id_first_neurons'][0]
        self.__logger.info("Initialised")

    def transform(self, count, size_buffer, buffer_of_spikes):
        """
        implements the abstract method for the transformation of the
        spike trains to rate.

        Parameters
        ----------
        count : int
            counter of the number of time of the transformation (identify the
            timing of the simulation)

        size_buffer: int
            size of the data to be read from the buffer for transformation

        buffer_of_spikes: Any  # e.g. 1D numpy array
            buffer which contains spikes (id of devices, id of neurons and
            spike times)

        Returns
        ------
            times, rate: numpy array, float
                tuple of interval and the rate for the interval if data is
                transformed successfully
        """
        spikes_neurons = self.__reshape_buffer_from_nest(count, size_buffer, buffer_of_spikes)
        rates = instantaneous_rate(spikes_neurons,
                                   t_start=np.around(count * self.__time_synch, decimals=2) * ms,
                                   t_stop=np.around((count + 1) * self.__time_synch, decimals=2) * ms,
                                   sampling_period=(self.__dt - 0.000001) * ms, kernel=RectangularKernel(1.0 * ms))
        rate = np.mean(rates, axis=1) / 10  # the division by 10 ia an adaptation for the model of TVB
        times = np.array([count * self.__time_synch, (count + 1) * self.__time_synch], dtype='d')
        return times, rate

    def __reshape_buffer_from_nest(self, count, size_buffer, buffer):
        """
        get the spike time from the buffer and order them by neurons

        Parameters
        ----------
        count : int
            counter of the number of time of the transformation (identify the
            timing of the simulation)

        size_buffer: int
            size of the data to be read from the buffer for transformation

        buffer: Any  # e.g. 1D numpy array
            buffer which contains spikes (id of devices, id of neurons and
            spike times)

        Returns
        ------
            spikes_neurons: list
                spike train of neurons
        """
        spikes_neurons = [[] for i in range(self.__nb_neurons)]
        # get all the time of the spike and add them in a histogram
        for index_data in range(int(np.rint(size_buffer / 3))):
            id_neurons = int(buffer[index_data * 3 + 1])
            time_step = buffer[index_data * 3 + 2]
            spikes_neurons[id_neurons - self.__first_id].append(time_step)
        for i in range(self.__nb_neurons):
            if len(spikes_neurons[i]) > 1:
                spikes_neurons[i] = SpikeTrain(np.concatenate(spikes_neurons[i]) * ms,
                                               t_start=np.around(count * self.__time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.__time_synch, decimals=2) + 0.0001)
                                               
            
            else:
                spikes_neurons[i] = SpikeTrain(spikes_neurons[i] * ms,
                                               t_start=np.around(count * self.__time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.__time_synch, decimals=2) + 0.0001)
        return spikes_neurons