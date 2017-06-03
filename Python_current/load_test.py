from scipy.ndimage.filters import convolve
import struct
import numpy as np
import time
from load_mfsc import load_and_set_context, trim
#from theano_fun import compile_cuda, get_net_params#
from load_mfsc_batch import LoaderMfsc

import os
import sys

context = 1
filter_num = 64
neuron_num = 2400

dataDir = "D:\\Alek\\sieci\\16\\"
in_frame_num = 0
out_vec_len = 0
feature_per_frame = 0
   

train_inputs_2 = None
train_outputs_2 = None
samples_2 = 0

in_frame_num = context * 2 + 1
train_set_file = 'train_b3_mfsc'
trainLoader = LoaderMfsc(dataDir + train_set_file, context)
#train_inputs, train_outputs, feature_per_frame, out_vec_len, samples = trim(dataDir + train_set_file, context)
train_inputs, train_outputs, feature_per_frame, out_vec_len, samples, lastBatch = trainLoader.next()
sample_num = len(train_inputs)
print "train read"

###-------------------------------- Final lunch nad training ----------------------------------------------

print "context:            %d" % (in_frame_num)
print "feature per frame:  %d" % (feature_per_frame)
print "output vector szie: %d" % (out_vec_len)
print ""
print("Start training...")
best_result = 0.0
begin = 0
batch = 256
start_learning_rate = 5e-3
num_epochs =160
best_state_acc = 0.0
worsening_epoche = 0
best_minbatch_error = 1000.0

for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0.0
    start_time = time.time()
    last_time = int(round(time.time() * 1000))

    while(True):
        begin = 0
        trainLoader.printBatchInfo()       
        
        train_inputs = None
        train_outputs = None
        if lastBatch:
            train_inputs, train_outputs, feature_per_frame, out_vec_len, samples, lastBatch = trainLoader.next()
            sample_num = len(train_inputs)
            break
        train_inputs, train_outputs, feature_per_frame, out_vec_len, samples, lastBatch = trainLoader.next()
        sample_num = len(train_inputs)



