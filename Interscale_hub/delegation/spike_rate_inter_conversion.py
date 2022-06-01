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
from neo.core import AnalogSignal,SpikeTrain

from elephant.statistics import instantaneous_rate, mean_firing_rate
from elephant.kernels import RectangularKernel
from elephant.spike_train_generation import inhomogeneous_poisson_process, homogeneous_poisson_process

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class SpikeRateConvertor:
    
    def __init__(self, param, configurations_manager, log_settings):
        self.__logger = configurations_manager.load_log_configurations(
                                        name="Elephant -- spike_rate_conversion",
                                        log_configurations=log_settings,
                                        target_directory=DefaultDirectories.SIMULATION_RESULTS)  
        
        # NOTE:params spike to rate
        # time of synchronization between 2 run
        self.__time_synch = param['time_synchronization']
        
        # the resolution of the integrator
        self.__dt = param['resolution']
        self.__nb_neurons = param['nb_neurons'][0]
        
        # id of transformer is hardcoded to 0
        self.__first_id = param['id_first_neurons'][0]
        
        # NOTE:params rate to spike
        # number of spike generators
        self.__nb_spike_generator = param['nb_neurons']
        # self.__path = param['path'] + "/transformation/"
        
        # # variable for saving values
        # self.__save_spike = bool(param['save_spikes'])
        # if self.__save_spike:
        #     self.__save_spike_buf = None
        # self.__save_rate = bool(param['save_rate'])
        # if self.__save_rate:
        #     self.__save_rate_buf = None
        
        # number of synapsis
        self.__nb_synapse = int(param["nb_brain_synapses"])
        
        self.__logger.info("Initialised")
    
    def spike_to_spiketrains(self, count, data_size, data_buffer):
        """
        get the spike time from the buffer and order them by neurons

        Parameters
        ----------
        count : int
            counter of the number of time of the transformation (identify the
            timing of the simulation)

        data_size: int
            size of the data to be read from the buffer for transformation

        data_buffer: Any  # e.g. 1D numpy array
            buffer which contains spikes (id of devices, id of neurons and
            spike times)

        Returns
        ------
            spiketrains: list
                spike train of neurons
        """
        spiketrains = [[] for i in range(self.__nb_neurons)]
        # get all the time of the spike and add them in a histogram
        for index_data in range(int(np.rint(data_size / 3))):
            id_neurons = int(data_buffer[index_data * 3 + 1])
            time_step = data_buffer[index_data * 3 + 2]
            spiketrains[id_neurons - self.__first_id].append(time_step)
        for i in range(self.__nb_neurons):
            if len(spiketrains[i]) > 1:
                spiketrains[i] = SpikeTrain(np.concatenate(spiketrains[i]) * ms,
                                               t_start=np.around(count * self.__time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.__time_synch, decimals=2) + 0.0001)
                                               
            
            else:
                spiketrains[i] = SpikeTrain(spiketrains[i] * ms,
                                               t_start=np.around(count * self.__time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.__time_synch, decimals=2) + 0.0001)
        return spiketrains
    
    
    def spiketrains_to_rate(self, count, spiketrains):
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
        rates = instantaneous_rate(spiketrains,
                                   t_start=np.around(count * self.__time_synch, decimals=2) * ms,
                                   t_stop=np.around((count + 1) * self.__time_synch, decimals=2) * ms,
                                   sampling_period=(self.__dt - 0.000001) * ms, kernel=RectangularKernel(1.0 * ms))
        rate = np.mean(rates, axis=1) / 10  # the division by 10 ia an adaptation for the model of TVB
        times = np.array([count * self.__time_synch, (count + 1) * self.__time_synch], dtype='d')
        return times, rate

    
    def rate_to_spikes(self, time_step, data_buffer):
        """
        implements the abstract method for the transformation of the
        rate to spikes.

        Parameters
        ----------
        count : int
            counter of the number of time of the transformation (identify the
            timing of the simulation)

        time_step: int
           time interval for the signal

        rate: Any  # e.g. 1D numpy array
            Spikes rate

        Returns
        ------
            spikes_train: list
                returns spikes train if data is transformed successfully
        """
        rate = data_buffer # NOTE: match argument names
        
        rate *= self.__nb_synapse  # rate of poisson generator ( due property of poisson process)
        rate += 1e-12
        rate = np.abs(rate)  # avoid rate equals to zeros
        signal = AnalogSignal(rate * Hz, t_start=(time_step[0] + 0.1) * ms,
                              sampling_period=(time_step[1] - time_step[0]) / rate.shape[-1] * ms)
        self.__logger.debug(f"rate: {rate}, signal: {signal}, time_step: {time_step}")
        spikes_train = []
        # generate individual spike trains
        for i in range(self.__nb_spike_generator[0]):
            spikes_train.append(np.around(np.sort(inhomogeneous_poisson_process(signal, as_array=True)), decimals=1))
        return spikes_train

        
'''
#NOTE: NEW VERSIONS of spike-rate converions.
# TODO confirm: params (np_synapse, ...) are not needed anymore.
# TODO check: data_buffer objects are already in correct shape (respectively)!?
    
    def spikes_to_rate_new(self, time_step, data_buffer, windows=0.0):
        # signature matched to old call, function itself is copy-paste of new
        t_start = time_step[0]
        t_stop = time_step[1]
        spikes = data_buffer
        
        # NOTE: COPY PASTE FROM HERE
        if windows == 0.0:
            #case without variation of rate
            if len(spikes[0].shape) ==0:
                # with only one rate
                result = [mean_firing_rate(spiketrain=spikes,t_start=t_start,t_stop=t_stop).rescale(Hz)]
            else:
                # with multiple rate
                result = []
                for spike in spikes:
                    result.append(mean_firing_rate(spiketrain=spike,t_start=t_start,t_stop=t_stop).rescale(Hz))
            return np.array(result)
        else:
            # case with variation of rate
            rate = []
            for time in np.arange(t_start,t_stop,windows):
                t_start_window = time*t_start.units
                t_stop_window = t_start_window+windows
                if len(spikes[0].shape) == 0:
                    # for only one spike train
                    result = [mean_firing_rate(spiketrain=spikes, t_start=t_start_window, t_stop=t_stop_window).rescale(Hz)]
                else:
                    # for multiple spike train
                    result = []
                    for spike in spikes:
                        result.append(mean_firing_rate(spiketrain=spike, t_start=t_start_window, t_stop=t_stop_window).rescale(Hz))
                rate.append(result)
            return np.array(rate)
    
    def rate_to_spikes_new(self, time_step, data_buffer, variation=False):
        # signature matched to old call, function itself is copy-paste of new
        t_start = time_step[0]
        t_stop = time_step[1]
        rates = data_buffer
        
        # NOTE: COPY PASTE FROM HERE
        if variation:
            # the case where the variation of the rate is include
            # We generate the inhomogenous poisson
            if len(rates.shape) == 1:
                # the case where we have only one rate
                signal = AnalogSignal(rates, t_start=t_start, sampling_period=(t_stop-t_start)/rates.shape[-1])
                result = [inhomogeneous_poisson_process(signal,as_array=True)]
                return np.array(result)
            else :
                # the case where we have multiple rates
                result = []
                for rate in rates:
                    signal = AnalogSignal(rate, t_start=t_start, sampling_period=(t_stop - t_start) / rates.shape[-1])
                    result.append(inhomogeneous_poisson_process(signal,as_array=True))
                return np.array(result)
        else:
            # the case we have only the rate
            # We generate the homogenous poisson
            if len(rates.shape) ==0:
                # the case where we have only one rate
                result = np.array([homogeneous_poisson_process(rate=rates, t_start=t_start, t_stop=t_stop, as_array=True)])
            else:
                # the case where we have multiple rates
                result = []
                for rate in rates:
                    result.append(homogeneous_poisson_process(rate=rate, t_start=t_start, t_stop=t_stop, as_array=True))
            return np.array(result)
    '''
