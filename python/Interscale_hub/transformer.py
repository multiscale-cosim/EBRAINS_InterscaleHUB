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


#TODO: This is a copy-paste of all the transformation and science from the usecase
#   - It contains also some 'pivot' steps, e.g. raw buffer data spikes to spike-train
#   - encapsulate this into transformer and science parts and make it suitable as plug-in

import numpy as np
import copy

import logging
import sys
# science related imports
from Interscale_hub.science import rates_to_spikes
from elephant.spike_train_generation import inhomogeneous_poisson_process
from quantities import ms,Hz
from neo.core import SpikeTrain, AnalogSignal
from elephant.statistics import instantaneous_rate
from elephant.kernels import RectangularKernel

############
# Transformation and Science for NEST-TVB direction
############
# TODO: proper transformation and science
def slidding_window(data,width):
    """
    use for mean field
    :param data: instantaneous firing rate
    :param width: windows or times average of the mean field
    :return: state variable of the mean field
    """
    res = np.zeros((data.shape[0]-width,width))
    res [:,:] = np.squeeze(data[np.array([[ i+j for i in range(width) ] for j in range(data.shape[0]-width)])])
    return res.mean(axis=1)

class store_data:
    def __init__(self,param):
        """
        initialisation
        :param path : path for the logger files
        :param param : parameters for the object
        """
        self.synch=param['time_synchronization']                # time of synchronization between 2 run
        self.dt=param['resolution']              # the resolution of the integrator
        self.shape = (int(self.synch/self.dt),1) # the shape of the buffer/histogram
        self.hist = np.zeros(self.shape)         # the initialisation of the histogram
        
    def add_spikes(self,count,datas):
        """
        adding spike in the histogram
        :param count: the number of synchronization times
        :param datas: the spike :(id,time)
        """
        for data in np.reshape(datas,(int(datas.shape[0]/3),3)):
            data[2]-=self.dt
            self.hist[int((data[2]-count*self.synch)/self.dt)]+=1
        #self.logger.info(int(datas.shape[0]/3))

    def return_data(self):
        """
        return the histogram and reinitialise the histogram
        :return: histogram
        """
        hist_copy = copy.copy(self.hist)
        self.hist = np.zeros(self.shape) # initialise histogram histogram of one region
        return hist_copy

class analyse_data:
    def __init__(self,param):
        """
        initialisation
        :param param : the parameters of analysis
        """

        self.width = int(param['width']/param['resolution']) # the window of the average in time
        self.synch = param['time_synchronization']     # synchronize time between simulator
        self.buffer = np.zeros((self.width,))                  #initialisation/ previous result for a good result
        self.coeff = 1 / ( param['nb_neurons'][0] * param['resolution'] ) # for the mean firing rate in in KHZ
        
    def analyse(self,count,hist):
        """
        analyse the histogram to generate state variable and the time
        :param count: the number of step of synchronization
        :param hist: the data
        :return:
        """
        hist_slide = np.concatenate((self.buffer,np.squeeze(hist,1)))
        data = slidding_window(hist_slide,self.width)
        self.buffer = np.squeeze(hist_slide[-self.width:])
        times = np.array([count*self.synch,(count+1)*self.synch], dtype='d')
        #self.logger.info(np.mean(data*self.coeff))
        return times,data*self.coeff


class spiketorate:

    def __init__(self,param):
        self.__logger = logging.getLogger("transformer--spiketorate")
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)
        self.__logger.setLevel(logging.DEBUG)
        self.id = 0
        self.time_synch = param['time_synchronization']  # time of synchronization between 2 run
        self.dt = param['resolution']  # the resolution of the integrator
        self.nb_neurons = param['nb_neurons'][0]
        # self.first_id = 0
        self.first_id = param['id_first_neurons'][0]  # id of transformer is hardcoded to 0


    def spike_to_rate(self, count, size_buffer, buffer_of_spikes):
        """
        function for the transformation of the spike trains to rate
        :param count: counter of the number of time of the transformation (identify the timing of the simulation)
        :param size_buffer: size of the data in the buffer
        :param buffer_of_spikes: buffer contains spikes
        :return: rate for the interval
        """
        spikes_neurons = self._reshape_buffer_from_nest(count, size_buffer, buffer_of_spikes)
        rates = instantaneous_rate(spikes_neurons,
                                   t_start=np.around(count * self.time_synch, decimals=2) * ms,
                                   t_stop=np.around((count + 1) * self.time_synch, decimals=2) * ms,
                                   sampling_period=(self.dt - 0.000001) * ms, kernel=RectangularKernel(1.0 * ms))
        rate = np.mean(rates, axis=1) / 10  # the division by 10 ia an adaptation for the model of TVB
        times = np.array([count * self.time_synch, (count + 1) * self.time_synch], dtype='d')
        return times, rate

    def _reshape_buffer_from_nest(self, count, size_buffer, buffer):
        """
        get the spike time from the buffer and order them by neurons
        :param count: counter of the number of time of the transformation (identify the timing of the simulation)
        :param size_buffer: size of the data in the buffer
        :param buffer: buffer contains id of devices, id of neurons and spike times
        :return:
        """
        spikes_neurons = [[] for i in range(self.nb_neurons)]
        # get all the time of the spike and add them in a histogram
        for index_data in range(int(np.rint(size_buffer / 3))):
            id_neurons = int(buffer[index_data * 3 + 1])
            time_step = buffer[index_data * 3 + 2]
            spikes_neurons[id_neurons - self.first_id].append(time_step)
        for i in range(self.nb_neurons):
            if len(spikes_neurons[i]) > 1:
                spikes_neurons[i] = SpikeTrain(np.concatenate(spikes_neurons[i]) * ms,
                                               t_start=np.around(count * self.time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.time_synch, decimals=2) + 0.0001)
                                               
            
            else:
                spikes_neurons[i] = SpikeTrain(spikes_neurons[i] * ms,
                                               t_start=np.around(count * self.time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.time_synch, decimals=2) + 0.0001)
        return spikes_neurons


############
# Transformation and Science for TVB-NEST direction
############
# TODO: proper transformation and science
def toy_rates_to_spikes(rates,t_start,t_stop):
    # Can be changed to the function we had with elephant, this is just a toy function
    '''
    transform rate in spike with random value for testing
    :param rates: rates from tvb
    :param t_start: time of starting simulation
    :param t_stop: time of ending simulation
    :return: times of spikes
    '''
    times = t_start + np.random.rand(rates.shape[-1]) * (t_stop-t_start)
    times = np.around(np.sort(np.array(times)), decimals=1)
    return times

class generate_data:
    def __init__(self,param):
        """
        generate spike train for each neurons
        :param path : path for the logger files
        :param nb_spike_generator: number of spike generator/neurons in each regions
        """
        self.__logger = logging.getLogger("transformer--generate_data")
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)
        self.__logger.setLevel(logging.DEBUG)

        # self.percentage_shared = param['percentage_shared']  # percentage of shared rate between neurons
        # self.nb_spike_generator = param['nb_spike_generator']         # number of spike generator
        self.nb_spike_generator = param['nb_neurons']         # number of spike generator
        # self.nb_synapse = param['nb_brain_synapses']               # number of synapses by neurons
        # self.function_translation = param['function_select'] # choose the function for the translation
        # np.random.seed(param['seed'])
        
        # id_transformer = 0  # TODO check if it is correct
        # self.id = id_transformer  # TODO check if it is needed
        
        # self.nb_spike_generator = nb_spike_generator  # number of spike generator
        self.path = param['path'] + "/transformation/"
        # variable for saving values:
        self.save_spike = bool(param['save_spikes'])
        if self.save_spike:
            self.save_spike_buf = None
        self.save_rate = bool(param['save_rate'])
        if self.save_rate:
            self.save_rate_buf = None
        # self.logger.info('TRS : end init transformation')
        self.nb_synapse = int(param["nb_brain_synapses"])
        
    def generate_spike(self,count,time_step,rate):
        #if time_step[0] == -1e5:
        #    self.get_time_rate_exit = True
        #    self.logger.info("MPI Internal : rate(get) : times"+str(self.sender_rank))
        #    return times, None
        rate *= self.nb_synapse  # rate of poisson generator ( due property of poisson process)
        rate += 1e-12
        rate = np.abs(rate)  # avoid rate equals to zeros
        signal = AnalogSignal(rate * Hz, t_start=(time_step[0] + 0.1) * ms,
                              sampling_period=(time_step[1] - time_step[0]) / rate.shape[-1] * ms)
        spike_generate = []
        # print("rate:",rate,"\nsignal:",signal,"\ntime step:",time_step)
        for i in range(self.nb_spike_generator[0]):
            # generate individual spike trains
            spike_generate.append(np.around(np.sort(inhomogeneous_poisson_process(signal, as_array=True)), decimals=1))
        return spike_generate
       
       
        # """
        # generate spike
        # This function are based on the paper : Kuhn, Alexandre, Ad Aertsen, and Stefan Rotter. “Higher-Order Statistics of Input Ensembles and the Response of Simple Model Neurons.” Neural Computation 15, no. 1 (January 2003): 67–101. https://doi.org/10.1162/089976603321043702.
        # DOI: 10.1162/089976603321043702
        # function 1 : Single Interaction Process Model
        # function 2 : Multiple Interaction Process Model
        # :param count: the number of step of synchronization between simulators
        # :param time_step: the time of synchronization
        # :param rate: the input rate of the mean field
        # :return:
        # """
        # if self.function_translation == 1:
        #     # Single Interaction Process Model
        #     # Compute the rate to spike trains
        #     rate *= self.nb_synapse # rate of poisson generator ( due property of poisson process)
        #     rate += 1e-12 # avoid rate equals to zeros
        #     spike_shared = \
        #         rates_to_spikes(rate * self.percentage_shared * Hz,
        #                         time_step[0] * ms, time_step[1] * ms, variation=True)[0]
        #     spike_generate = rates_to_spikes(np.repeat([rate],self.nb_spike_generator,axis=0) * (1 - self.percentage_shared) * Hz, time_step[0] * ms, time_step[1] * ms,
        #                             variation=True)
        #     for i in range(self.nb_spike_generator):
        #         spike_generate[i] = np.around(np.sort(np.concatenate((spike_generate, spike_shared))), decimals=1)
        #     #self.logger.info('rate :'+str(rate)+' spikes :'+str(np.concatenate(spike_generate).shape))
        #     return spike_generate
        # elif self.function_translation == 2:
        #     # Multiple Interaction Process Model
        #     rate *= self.nb_synapse / self.percentage_shared # rate of poisson generator ( due property of poisson process)
        #     rate += 1e-12  # avoid rate equals to zeros
        #     spike_shared = np.round(rates_to_spikes(rate * Hz, time_step[0] * ms, time_step[1] * ms, variation=True)[0],1)
        #     select = np.random.binomial(n=1,p=self.percentage_shared,size=(self.nb_spike_generator,spike_shared.shape[0]))
        #     result = []
        #     for i in np.repeat([spike_shared],self.nb_spike_generator,axis=0)*select :
        #         result.append(i[np.where(i!=0)])
        #     #self.logger.info('rate :'+str(rate)+' spikes :'+str(spike_shared))
        #     return result
    
