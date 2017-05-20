import struct
import numpy as np
import os.path
from ctypes import *
import time

###-------------------------------- Binary loading ---------------------------------------------------------------------------
class LoaderMfsc:
    file_index = 0
    file_number = 0
    context = 0
    verbose = True
    file_base = ""
    currentSampleNumber = -1
    currentBatchNo = -1
    formated_outputs = None
    formated_inputs2 = None
    

    '''binary data format:
    (total number of data)(number of phrases)(number of feature per frame)(number of states)    <<-- Header
    |-------------------------------|----------------------|----------------------|-------------|------------------------|
    | (number of frames from wav-1) | f1_1 f1_2 ... f1_123 | f2_1 f2_2 ... f2_123 | ... f_n1_123| (o_1) (0_2) ... (o_n1) |
    |-------------------------------|----------------------|----------------------|-------------|------------------------|
    |                               |                      |                      |             |                        |
    |-------------------------------|----------------------|----------------------|-------------|------------------------|
    | (number of frames from wav-k) | f1_1 f1_2 ... f1_123 | f2_1 f2_2 ... f2_123 |... f_nk_123 | (o_1) (0_2) ... (o_nk) |
    |-------------------------------|----------------------|----------------------|-------------|------------------------|
    NOTE:
    (number of frames from wav-1) + (number of frames from wav-2) + ... + (number of frames from wav-k) = (total number of data);
    k = (number of phrases);
    f*_* <- 4-byte float
    (*)  <- 4-byte integer
    '''
    
    #file_base is path + file name without index, index start from 0
    def __init__(self, file_base, context):
        self.file_base = file_base
        self.context = context
        file_full_path = ("%s_%d") % (file_base, self.file_number)
        while os.path.exists(file_full_path):
            print "detected:   " + file_full_path
            self.file_number += 1
            file_full_path = ("%s_%d") % (file_base, self.file_number)
            
    def next(self):
        # last is set to True if we start new cycle
        last = False
        self.currentBatchNo = self.file_index
        if self.file_number == 1:
            last = True
            if self.file_index != 100:
                fileName = ("%s_0") % (self.file_base)
                self.formated_inputs2, self.formated_outputs, featur_per_layer, out_vec_len, samples = trim(fileName, self.context, self.verbose)
                self.file_index = 100
                print "one batch mode - loaded"
        else: 
            if self.formated_inputs2 != None:
                del self.formated_inputs2
            if self.formated_outputs != None:
                del self.formated_outputs
            fileName = ("%s_%d") % (self.file_base, self.file_index)
            self.formated_inputs2, self.formated_outputs, featur_per_layer, out_vec_len, samples = trim(fileName, self.context, self.verbose)
            self.file_index += 1
            if self.file_number == self.file_index:
                last = True
                self.file_index = 0
                
        self.verbose = False
        self.currentSampleNumber = samples
        return self.formated_inputs2, self.formated_outputs, featur_per_layer, out_vec_len, samples, last
        
    def printBatchInfo(self):
        if self.file_number == 1:
            print "one batch mode, \tsample number: %d" % (self.currentSampleNumber)
        else:
            print "batch: %d / %d, \tsample number: %d" % (self.currentBatchNo+1, self.file_number, self.currentSampleNumber)
        

#read 'sample' binary values of given 4-byte 'type' and store to 'array'
def read_from_file(f, array, samples, type):
    pointer = 0
    batch_size = 10000    #byte size
    while pointer < samples:
        batch  = np.minimum(batch_size, (samples - pointer) * 4)     #byte size  
        new_elem_no = batch / 4
        pattern = type * new_elem_no
        buffor = f.read(batch)
        array[pointer : pointer + new_elem_no] = struct.unpack(pattern, buffor)
        pointer += new_elem_no

#read one wav-data set
def read_one(f, feature_per_frame):
    frame_no = struct.unpack('I', f.read(4))[0]
    features = np.empty(frame_no*feature_per_frame)
    outputs = np.empty(frame_no)
    read_from_file(f, features, frame_no*feature_per_frame, 'f')
    read_from_file(f, outputs, frame_no, 'I')
    return features, outputs

#read data form file
#inputs is python array of C-style arrays, C-arrays are different length
#inputs[i] contains data from i-th wav, inputs[i] length is equal to number of frames in i-th wav
def read_data_bin_ctx(fileName, verbose):
    
    with open(fileName,"rb") as f:
        #read header
        total_samples = struct.unpack('I', f.read(4))[0]
        phrase_no = struct.unpack('I', f.read(4))[0]
        feature_per_frame = struct.unpack('I', f.read(4))[0]
        out_vec_len = struct.unpack('I', f.read(4))[0]
        
        if verbose:
            print ""
            print "--------------------------------------------"
            print "Total number of samples: %d" % (total_samples)
            print "Phrase number          : %d" % (phrase_no)
            print "Feature per frame      : %d" % (feature_per_frame)
            print "Output vector length   : %d" % (out_vec_len)
            print "--------------------------------------------"
            print ""
        
        inputs = [None] * phrase_no
        outputs = [None] * phrase_no
        phrase_i = 0

        for i in range(0, phrase_no):
            inputs[i], outputs[i] = read_one(f, feature_per_frame)
            if phrase_i == 300:
                print "loaded %d phrases from %d" % (i, phrase_no)
                phrase_i = 0
            phrase_i += 1
            
    return inputs, outputs, total_samples, phrase_no, out_vec_len, feature_per_frame

#reformat data by adding context, create one long 2D array (samples x features)
#release used c-style arrays to save memory
def create_context(inputs, outputs, context, formated_inputs, formated_outputs, offset, feature_per_frame):
    input_len = inputs.size / feature_per_frame
    data_in_layer = inputs.size / 3
    feature_per_layer = feature_per_frame / 3

    j = (context * 2 + 1) * feature_per_layer 
    #'i' is input number, from beginning of oryginal data
    for i in range(input_len-context * 2):    

        internal_offset = feature_per_layer * i
        formated_inputs[offset][0] = np.copy(inputs[internal_offset : internal_offset + j])
        internal_offset += data_in_layer
        formated_inputs[offset][1] = np.copy(inputs[internal_offset : internal_offset + j])
        internal_offset += data_in_layer
        formated_inputs[offset][2] = np.copy(inputs[internal_offset : internal_offset + j])
            
        formated_outputs[offset] = outputs[i]
        offset += 1
    #relese c-array to save memory
    del inputs
    del outputs
    return offset
        
    
#mian load function
def load_and_set_context(fileName, context, verbose):
    inputs, outputs, total_samples, phrase_no, out_vec_len, feature_per_frame  = read_data_bin_ctx(fileName, verbose)
    formated_inputs = np.zeros((total_samples - 2*phrase_no, 3, int(feature_per_frame *(2*context+1)) / 3), dtype='float32')
    formated_outputs = np.zeros(total_samples - 2*phrase_no, dtype='uint8')
    offset = 0
    
    if verbose:
        print "data loaded, creating context..."
    phrase_counter = 0
    
    millis = int(round(time.time() * 1000))
    for i in range(0, phrase_no):
        offset = create_context(inputs[i], outputs[i], context, formated_inputs, formated_outputs, offset, feature_per_frame)
        if phrase_counter == 300:
            print "formed %d phrases from %d" % (i, phrase_no)
            phrase_counter = 0
        phrase_counter += 1
       
    if verbose:
        print "time: %d [ms]" % (int(round(time.time() * 1000)) - millis)
    
    
    return formated_inputs, formated_outputs, feature_per_frame, out_vec_len, total_samples - 2*phrase_no
def trim(fileName, context, verbose):
    formated_inputs, formated_outputs, feature_per_frame, out_vec_len, samples = load_and_set_context(fileName, context, verbose)
    return formated_inputs, formated_outputs, 41, out_vec_len, samples