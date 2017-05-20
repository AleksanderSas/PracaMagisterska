from scipy.ndimage.filters import convolve
import struct
import numpy as np
import time
from load_mfsc import load_and_set_context, trim
#from theano_fun import compile_cuda, get_net_params#
from load_mfsc_batch import LoaderMfsc

import os
import sys

# Fix a bug in printing SVG
if sys.platform == 'win32':
    print "Monkey-patching pydot"
    import pydot

    def force_find_graphviz(graphviz_root):
        binpath = os.path.join(graphviz_root, 'bin')
        programs = 'dot twopi neato circo fdp sfdp'
        def helper():
            for prg in programs.split():
                if os.path.exists(os.path.join(binpath, prg)):
                    yield ((prg, os.path.join(binpath, prg)))
                elif os.path.exists(os.path.join(binpath, prg+'.exe')):
                    yield ((prg, os.path.join(binpath, prg+'.exe')))
        progdict = dict(helper())
        return lambda: progdict

    pydot.find_graphviz = force_find_graphviz('c:/Program Files (x86)/Graphviz2.34/')

print "Theano fun 1"
	
import theano
print "Theano fun 2"
import theano.tensor as T
print "Theano fun 3"
import lasagne
print "Theano fun 4"
from lasagne.regularization import regularize_layer_params_weighted, l2, l1

###-------------------------------- Functions -------------------------------------
print "Theano fun 5"

def build_cnn(input_var=None, filter_num = 0, neuron_num = 0, in_frame_num = 0, feature_per_frame = 0, out_vec_len = 0):

    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 3,in_frame_num, feature_per_frame),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network,
             #lasagne.layers.dropout(network, p=.2),
            num_filters=filter_num, filter_size=(in_frame_num,6),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())  #DIM 1, feature_per_frame

    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 2))   #DIM 14

    '''network = lasagne.layers.Conv2DLayer(network,
            #lasagne.layers.dropout(network, p=.2),
            num_filters=800, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.leaky_rectify) #DIM 10'''
    
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))  #DIM 5

    #network =  lasagne.layers.DenseLayer(
    layer2reg = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=neuron_num,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
   
    '''network =  lasagne.layers.DenseLayer(network,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify)'''

    network = lasagne.layers.DenseLayer(
            layer2reg,
            num_units=out_vec_len,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network, layer2reg 
    
def compile_cuda(filter_num, neuron_num, in_frame_num, feature_per_frame, out_vec_len):
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    lrate_var = theano.tensor.scalar('lrate',dtype='float32')
    momentum_var = theano.tensor.scalar('momentum',dtype='float32')

    #create network
    network, layer2reg = build_cnn(input_var, filter_num, neuron_num, in_frame_num, feature_per_frame, out_vec_len)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    #regularization decrises accuracy
    penalty = lasagne.regularization.regularize_layer_params(layer2reg, l1) * 1.0e-5
    loss = loss + penalty

    # SGB with momentm and changing learning rate
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate = lrate_var, momentum = momentum_var)
    #updates = lasagne.updates.adadelta(loss, params)
            


    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a training function
    train_fn = theano.function([input_var, target_var, lrate_var, momentum_var], loss, updates=updates)
    #train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_result = T.argmax(test_prediction, axis=1)
                    
    # Compile a second function computing the validation loss and accuracy:
    predict_fn = theano.function([input_var], [test_result])
    return train_fn, predict_fn, test_prediction, network   
    
def get_net_params(_network):
    return lasagne.layers.get_all_param_values(_network)

###-------------------------------- Functions -------------------------------------


context = 5
filter_num = 64
neuron_num = 2400

dataDir = "D:\\Alek\\sieci\\10\\"
in_frame_num = 0
out_vec_len = 0
feature_per_frame = 0
        
def test(inputs_all, outputs_all):
    test_number = 0
    test_ok = 0
    test_ok_p = 0
    err = 0
    acc = 0
    err_p = 0
    acc_p = 0
    batches = 0
    batch = 196
    begin = 0
    sample_num = len(inputs_all)

    while begin < sample_num:
        batch_size = np.minimum(batch, sample_num - begin) 
        inputs = inputs_all[begin:begin+batch].reshape(batch_size, 3, in_frame_num, feature_per_frame)
        outputs = outputs_all[begin:begin+batch]
        begin += batch

        result = predict_fn(inputs)
        for x in zip(result[0], outputs.flatten()):
            test_number += 1
            if x[0] == x[1]:
                test_ok += 1
            x1 = int(x[0]) / 3
            x2 = int(x[1]) / 3
            if x1 == x2:
                test_ok_p += 1  
	state_acc = 1.0 * test_ok / test_number		
    phone_acc = 1.0 * test_ok_p / test_number	
	
    print "\tstate:  {}->{}".format(test_ok, state_acc)
    print "\tphone:  {}->{}".format(test_ok_p, phone_acc)
    return state_acc, phone_acc
	
###-------------------------------- Load train and validation data -------------------------------------

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
valid_inputs, valid_outputs, feature_per_frame, out_vec_len, samples = trim(dataDir + "valid_b3_mfsc_0", context)
print "valid read"
print "------------------------------------------------"

###-------------------------------- BUILD Theano functions ----------------------------------------------

total_batch = 0
    
train_fn, predict_fn, test_prediction, network = compile_cuda(filter_num, neuron_num, in_frame_num, feature_per_frame, out_vec_len)
print "compiled"
print "------------------------------------------------"


###-------------------------------- Final lunch nad training ----------------------------------------------

print "context:            %d" % (in_frame_num)
print "feature per frame:  %d" % (feature_per_frame)
print "output vector szie: %d" % (out_vec_len)
print ""
print("Start training...")
best_result = 0.0
begin = 0
batch = 256
start_learning_rate = 9e-3
num_epochs = 1500
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
        while begin < sample_num:     
            batch_size = np.minimum(batch, sample_num - begin) 
            inputs = train_inputs[begin:begin+batch_size].reshape(batch_size, 3, in_frame_num, feature_per_frame)
            outputs = train_outputs[begin:begin+batch_size]
            begin += batch
            
            K = 160000
            lrate = start_learning_rate;
            train_err += train_fn(inputs, outputs.flatten(), lrate, 0.9)
            total_batch += 1
            if total_batch % 3000 == 0:
                print "minibatch err %f \t\tTIME: %d ms" % (1.0*train_err / total_batch, int(round(time.time() * 1000)) - last_time)
                last_time = int(round(time.time() * 1000))
            '''if best_minbatch_error < 1.0*train_err / total_batch:
                start_learning_rate *=0.987
            else:
                best_minbatch_error = 1.0*train_err / total_batch'''
        trainLoader.printBatchInfo()       
        
        train_inputs = None
        train_outputs = None
        if lastBatch:
            train_inputs, train_outputs, feature_per_frame, out_vec_len, samples, lastBatch = trainLoader.next()
            sample_num = len(train_inputs)
            break
        train_inputs, train_outputs, feature_per_frame, out_vec_len, samples, lastBatch = trainLoader.next()
        sample_num = len(train_inputs)
    
    # And a full pass over the validation data:
    print "EPOCHE %d" % (epoch)
    print "VALID SET ACCURACY"
    state_acc, phone_acc = test(valid_inputs, valid_outputs)
    #print "TRAIN SET ACCURACY"
    #test(train_inputs, train_outputs)
    
    phone_acc = 0.0
    
    if state_acc > best_state_acc:
        worsening_epoche = 0
        best_state_acc = state_acc 
        #net_2_C_exporter('D:\\NN_state_recognizer\\sieci\\net_forC_%d_%d_mfsc.net2' % (filter_num, neuron_num, get_net_params() )
        with open("D:\\Alek\\sieci\\11\\net2_tmp_%d_%d_r_e-7_ctx_%d_mfsc.net2" % (filter_num, neuron_num, context), "wb") as f:
            '''
            kontekst
            liczba filtrow,
            liczba neuronow w warstwie feed forward
            '''
            f.write("%d\n" % (context))
            f.write("%d\n" % (filter_num))
            f.write("%d\n" % (neuron_num))
            np.save(f, get_net_params(network))
    else:
        worsening_epoche += 1
        start_learning_rate *=0.94
    if worsening_epoche == 12:
        print "TRAIN COMPLETE"
        break
print "best result: %f" % best_state_acc



