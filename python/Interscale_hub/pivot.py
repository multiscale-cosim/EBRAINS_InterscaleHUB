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
# 
from mpi4py import MPI
import time
import numpy as np
import logging

#nest to tvb
from Interscale_hub.transformer import store_data, analyse_data
#tvb to nest
from Interscale_hub.transformer import generate_data


# NestTvbPivot and TvbNestPivot classes:
# TODO: proper abstraction -> extract the usecase details from the general implementation
# -> Init, start, stop are pretty much the same every time
# -> incoming (receive) and outgoing (send) loops (M:N mapping)
# -> the analyse (method) should be 
#   a) pivot, as raw data to cosim data 
#   b) transform (might be trivial) and 
#   c) analysis (might be trivial)

# TODO: rework on the receive and send loops (both, general coding style and usecase specifics)

class NestTvbPivot:
    def __init__(self, param, comm_receiver, comm_sender, databuffer):
        '''
        '''
        
        # TODO: logger placeholder for testing
        import sys
        self.__logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)
        self.__logger.setLevel(logging.DEBUG)
        
        # Parameter for transformation and analysis
        self.__param = param
        # INTERcommunicator
        self.__comm_receiver = comm_receiver
        self.__comm_sender = comm_sender
        # How many Nest ranks are sending, how many Tvb ranks are receiving
        self.__num_sending = self.__comm_receiver.Get_remote_size()
        self.__num_receiving = self.__comm_sender.Get_remote_size()
        self.__databuffer = databuffer
    
    
    def start(self, intracomm):
        '''
        Start the pivot operation.
        M:N mapping of MPI ranks, receive data, further process data.
        
        MVP: receive on rank 0, do the rest on rank 1.
        '''
        if intracomm.Get_rank() == 0: # Receiver from input sim, rank 0
            self._receive()
        else: #  Science/analyse and sender to TVB, rank 1-x
            self._send()


    def stop(self):
        '''
        TODO: proper execution of stop command
        '''
        self.__stop = True


    def _receive(self):
        '''
        Receive data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        # The last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        self.__databuffer[-1] = 1 # set buffer to 'ready to receive from nest'
        self.__databuffer[-2] = 0 # marks the 'head' of the buffer
        # It seems the 'check' variable is used to receive tags from NEST, i.e. ready for send...
        # change this in the future, also mentioned in the FatEndPoint solution from Wouter.
        check = np.empty(1,dtype='b')
        shape = np.empty(1, dtype='i')    
        count = 0
        status_ = MPI.Status()
        while True:
            head_ = 0 # head of the buffer, reset after each iteration            
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # IF all ranks send always the same tag in one iteration (simulation step)
            # then this works. But it should be handled differently!!!!
            for i in range(self.__num_sending):
                # new: We do not care which source sends first, give MPI the freedom to send in whichever order.
                self.__comm_receiver.Recv([check, 1, MPI.CXX_BOOL], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
            if status_.Get_tag() == 0:
                # wait until ready to receive new data (i.e. the sender has cleared the buffer)
                while self.__databuffer[-1] != 1: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                for source in range(self.__num_sending):
                    # send 'ready' to the nest rank
                    self.__comm_receiver.Send([np.array(True,dtype='b'),MPI.BOOL],dest=source,tag=0)
                    # receive package size info
                    self.__comm_receiver.Recv([shape, 1, MPI.INT], source=source, tag=0, status=status_)
                    # NEW: receive directly into the buffer
                    self.__comm_receiver.Recv([self.__databuffer[head_:], MPI.DOUBLE], source=source, tag=0, status=status_)
                    head_ += shape[0] # move head 
                # Mark as 'ready to do analysis'
                self.__databuffer[-1] = 0
                # important: head_ is first buffer index WITHOUT data.
                self.__databuffer[-2] = head_
            elif status_.Get_tag() == 1:
                count += 1
            elif status_.Get_tag() == 2:
                # NOTE: simulation ended
                break
            else:
                raise Exception("bad mpi tag"+str(status_.Get_tag()))
    
    
    def _send(self):
        '''
        Send data to TVB (multiple MPI ranks possible).
        Replaces the former 'send' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        count=0 # simulation/iteration step
        status_ = MPI.Status()
        while True:
            # TODO: this communication has the 'rank 0' problem described in the beginning
            accept = False
            #logger.info("Nest to TVB : wait to send " )
            while not accept:
                req = self.__comm_sender.irecv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
                accept = req.wait(status_)
            #logger.info(" Nest to TVB : send data status : " +str(status_.Get_tag()))
            if status_.Get_tag() == 0:
                # wait until the receiver has cleared the buffer, i.e. filled with new data
                while self.__databuffer[-1] != 0: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                # TODO: All science/analysis here. Move to a proper place.
                times,data = self._transform(count)
                
                # Mark as 'ready to receive next simulation step'
                self.__databuffer[-1] = 1
                
                ### OLD Code
                #logger.info("Nest to TVB : send data :"+str(np.sum(data)) )
                # time of sim step
                self.__comm_sender.Send([times, MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                # send the size of the rate
                size = np.array(int(data.shape[0]),dtype='i')
                self.__comm_sender.Send([size,MPI.INT], dest=status_.Get_source(), tag=0)
                # send the rates
                self.__comm_sender.Send([data,MPI.DOUBLE], dest=status_.Get_source(), tag=0)
                ### OLD Code end
            elif status_.Get_tag() == 1:
                # NOTE: simulation ended
                break
            else:
                raise Exception("bad mpi tag"+str(status_.Get_tag()))
            count+=1
        #logger.info('NEST_to_TVB: End of send function')

    
    def _transform(self, count):
        '''
        This step contains some pivoting, transformation and analysis.
        TODO: encapsulate
        :param count: Simulation iteration/step
        :return times, data: simulation times and the calculated rates
        '''
        #store: Python object, create the histogram 
        #analyse: Python object, calculate rates
        store = store_data(self.__param)
        analyse = analyse_data(self.__param)
        
        # TODO: Step 1 and 2 can be merged into one step. Buffer is no longer filled rank by rank.
        # Make this parallel with the INTRA communicator (should be embarrassingly parallel).
        # Step 1) take all data from buffer and create histogram
        # second to last index in databuffer denotes how much data there is
        store.add_spikes(count, self.__databuffer[:int(self.__databuffer[-2])])
        # Step 2) take the resulting histogram
        data_to_analyse = store.return_data()
        # Step 3) Analyse this data, i.e. calculate rates?
        times,data = analyse.analyse(count, data_to_analyse)
        
        return times, data
    


class TvbNestPivot: 
    def __init__(self, param, comm_receiver, comm_sender, databuffer):
        '''
        '''
        
        # TODO: logger placeholder for testing
        self.__logger = logging.getLogger(__name__)
        self.__logger.info("Initialise...")
        
        # Parameter for transformation and analysis
        self.__param = param
        # INTERcommunicator
        self.__comm_receiver = comm_receiver
        self.__comm_sender = comm_sender
        # How many Nest ranks are sending, how many Tvb ranks are receiving
        self.__num_sending = self.__comm_receiver.Get_remote_size()
        self.__num_receiving = self.__comm_sender.Get_remote_size()
        self.__databuffer = databuffer


    def start(self, intracomm):
        '''
        Start the pivot operation.
        M:N mapping of MPI ranks, receive data, further process data.
        
        MVP: receive on rank 0, do the rest on rank 1.
        '''
        if intracomm.Get_rank() == 0: # Receiver from input sim, rank 0
            self._receive()
        else: #  Science/analyse and sender to TVB, rank 1-x
            self._send()


    def stop(self):
        '''
        TODO: proper execution of stop command
        '''
        self.__stop = True


    def _receive(self):
        '''
        Receive data on rank 0. Put it into the shared mem buffer.
        Replaces the former 'receive' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        # The last two buffer entries are used for shared information
        # --> they replace the status_data variable from previous version
        # --> find more elegant solution?
        self.__databuffer[-1] = 1 # set buffer to 'ready to receive from tvb'
        self.__databuffer[-2] = 0 # marks the 'head' of the buffer
        # init placeholder for incoming data
        size = np.empty(1, dtype='i') # size of the rate-array
        status_ = MPI.Status()
        while True:
            # NOTE: Check communication protocol between simulators and transformers!
            requests=[]
            for rank in range(self.__num_sending):
                requests.append(self.__comm_receiver.isend(True,dest=rank,tag=0))
            MPI.Request.Waitall(requests)
            # NOTE: works for now, needs rework if multiple ranks are used on TVB side
            # we receive from "ANY_SOURCE", but only check the status_ of the last receive...
            # get the starting and ending time of the simulation step
            # NEW: receive directly into the buffer
            self.__comm_receiver.Recv([self.__databuffer[0:], MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status_)
            if status_.Get_tag() == 0:
                # wait until ready to receive new data (i.e. the sender has cleared the buffer)
                while self.__databuffer[-1] != 1: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass
                # Get the size of the data
                self.__comm_receiver.Recv([size, 1, MPI.INT], source=status_.Get_source(), tag=0, status=status_)
                # NEW: receive directly into the buffer
                # First two entries are the times, see above
                self.__comm_receiver.Recv([self.__databuffer[2:], MPI.DOUBLE], source=status_.Get_source(), tag=0, status=status_)
                # Mark as 'ready to do analysis'
                self.__databuffer[-1] = 0
                self.__databuffer[-2] = size # info about size of data array
                # logger.info(" TVB to Nest: update buffer")
            elif status_.Get_tag() == 1:
                # logger.info('TVB: simulation ended, waiting for stop command')
                # NOTE: simulation ended
                break
            else:
                raise Exception("bad mpi tag"+str(status_.Get_tag()))
        
        # logger.info('TVB_to_NEST: End of receive function')


    def _send(self):
        '''
        Send data to NEST (multiple MPI ranks possible).
        Replaces the former 'send' function.
        NOTE: First refactored version -> not pretty, not final. 
        '''
        status_ = MPI.Status()
        # NOTE: hardcoded...
        check = np.empty(1,dtype='b')
        size_list = np.empty(1, dtype='i')
        id_first_spike_detector = self.__param['id_first_spike_detector']
        while True:
            # TODO: This is still not correct. We only check for the Tag of the last rank.
            # IF all ranks send always the same tag in one iteration (simulation step)
            # then this works. But it should be handled differently!!!!
            for rank in range(self.__num_sending):
                self.__comm_sender.Recv([check, 1, MPI.CXX_BOOL], source=rank, tag=MPI.ANY_TAG, status=status_)
            if status_.Get_tag() == 0:
                # wait until the receiver has cleared the buffer, i.e. filled with new data
                while self.__databuffer[-1] != 0: # TODO: use MPI, remove the sleep
                    time.sleep(0.001)
                    pass

                # TODO: All science/generate here. Move to a proper place.
                spikes_times = self._transform()
                
                # Mark as 'ready to receive next simulation step'
                self.__databuffer[-1] = 1
                
                ### OLD code, kept the communication and science as it is for now
                # NOTE: Receive from status_.Get_source() and rank
                # Send to status_.Get_source() and rank
                # why?
                # a second status_ object is used, should not be named the same
                for rank in range(self.__num_sending):
                    # NOTE: hardcoded 10 in simulation mocks
                    self.__comm_sender.Recv([size_list, 1, MPI.INT], source=rank, tag=0, status=status_)
                    if size_list[0] != 0:
                        list_id = np.empty(size_list, dtype='i')
                        # NOTE: hardcoded np.arange(0,10,1) in simulation mocks
                        self.__comm_sender.Recv([list_id, size_list, MPI.INT], source=status_.Get_source(), tag=0, status=status_)
                        # Select the good spike train and send it
                        # TODO: create lists, append to lists, nested loops
                        # this is slow and will be a bottleneck when we scale up.
                        data = []
                        shape = []
                        for i in list_id:
                            shape += [spikes_times[i-id_first_spike_detector].shape[0]]
                            data += [spikes_times[i-id_first_spike_detector]]
                        send_shape = np.array(np.concatenate(([np.sum(shape)],shape)), dtype='i')
                        # firstly send the size of the spikes train
                        self.__comm_sender.Send([send_shape, MPI.INT], dest=status_.Get_source(), tag=list_id[0])
                        # secondly send the spikes train
                        data = np.concatenate(data).astype('d')
                        self.__comm_sender.Send([data, MPI.DOUBLE], dest=rank, tag=list_id[0])
                ### OLD code end
            elif  status_.Get_tag() == 1:
                # NOTE: one sim step? inconsistent with receiving side
                # logger.info(" TVB to Nest end sending") 
                continue
            elif status_.Get_tag() == 2:
                # logger.info(" TVB to Nest end simulation ")
                # NOTE: simulation ended
                break
            else:
                raise Exception("bad mpi tag : "+str(status_.Get_tag()))
        
        # logger.info('TVB_to_NEST: End of send function')


    def _transform(self):
        '''
        This step contains some pivoting, transformation and analysis.
        TODO: encapsulate
        '''
        generator = generate_data(self.__param)
        # NOTE: count is a hardcoded '0'. Why?
        # time_step are the first two doubles in the buffer
        # rate is a double array, which size is stored in the second to last index
        spikes_times = generator.generate_spike(0,
                                                self.__databuffer[:2],
                                                self.__databuffer[2:int(self.__databuffer[-2])])
        
        return spikes_times
