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
#TODO: It contains also some 'pivot' steps, e.g. raw buffer data spikes to spike-train
#TODO: encapsulate this into transformer and science parts and make it suitable as plug-in
import numpy as np
import copy
# science related imports
# here elephant is imported and used
from placeholders.science_rate_spike import rates_to_spikes
from quantities import ms,Hz

# TODO: proper NEST-TVB direction transformation and science
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
        self.synch=param['synch']                # time of synchronization between 2 run
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
        self.synch = param['synch']                          # synchronize time between simulator
        self.buffer = np.zeros((self.width,))                  #initialisation/ previous result for a good result
        self.coeff = 1 / ( param['nb_neurons'] * param['resolution'] ) # for the mean firing rate in in KHZ
        
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



# TODO: proper TVB-NEST direction transformation and science
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
        self.percentage_shared = param['percentage_shared']  # percentage of shared rate between neurons
        self.nb_spike_generator = param['nb_spike_generator']         # number of spike generator
        self.nb_synapse = param['nb_synapses']               # number of synapses by neurons
        self.function_translation = param['function_select'] # choose the function for the translation
        np.random.seed(param['seed'])
        
    def generate_spike(self,count,time_step,rate):
        """
        generate spike
        This function are based on the paper : Kuhn, Alexandre, Ad Aertsen, and Stefan Rotter. “Higher-Order Statistics of Input Ensembles and the Response of Simple Model Neurons.” Neural Computation 15, no. 1 (January 2003): 67–101. https://doi.org/10.1162/089976603321043702.
        DOI: 10.1162/089976603321043702
        function 1 : Single Interaction Process Model
        function 2 : Multiple Interaction Process Model
        :param count: the number of step of synchronization between simulators
        :param time_step: the time of synchronization
        :param rate: the input rate of the mean field
        :return:
        """
        if self.function_translation == 1:
            # Single Interaction Process Model
            # Compute the rate to spike trains
            rate *= self.nb_synapse # rate of poisson generator ( due property of poisson process)
            rate += 1e-12 # avoid rate equals to zeros
            spike_shared = \
                rates_to_spikes(rate * self.percentage_shared * Hz,
                                time_step[0] * ms, time_step[1] * ms, variation=True)[0]
            spike_generate = rates_to_spikes(np.repeat([rate],self.nb_spike_generator,axis=0) * (1 - self.percentage_shared) * Hz, time_step[0] * ms, time_step[1] * ms,
                                    variation=True)
            for i in range(self.nb_spike_generator):
                spike_generate[i] = np.around(np.sort(np.concatenate((spike_generate, spike_shared))), decimals=1)
            #self.logger.info('rate :'+str(rate)+' spikes :'+str(np.concatenate(spike_generate).shape))
            return spike_generate
        elif self.function_translation == 2:
            # Multiple Interaction Process Model
            rate *= self.nb_synapse / self.percentage_shared # rate of poisson generator ( due property of poisson process)
            rate += 1e-12  # avoid rate equals to zeros
            spike_shared = np.round(rates_to_spikes(rate * Hz, time_step[0] * ms, time_step[1] * ms, variation=True)[0],1)
            select = np.random.binomial(n=1,p=self.percentage_shared,size=(self.nb_spike_generator,spike_shared.shape[0]))
            result = []
            for i in np.repeat([spike_shared],self.nb_spike_generator,axis=0)*select :
                result.append(i[np.where(i!=0)])
            #self.logger.info('rate :'+str(rate)+' spikes :'+str(spike_shared))
            return result
    
