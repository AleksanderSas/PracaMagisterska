from __future__ import print_function
from scipy.ndimage.filters import convolve
import os
import sys


# Fix a bug in printing SVG
if sys.platform == 'win32':
    print("Monkey-patching pydot")
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
    
import theano
import theano.tensor.signal.downsample
import numpy as np
import theano.tensor as T
import lasagne
import time
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from MFSC_reader import load_mfsc_data_with_context
import struct
import glob

###-------------------------------- Functions -------------------------------------

#file_handler = open("C:\\Users\\Alek\\Desktop\\test.txt", 'r')
#dataDir = "B:\\NN_data\\THEANO_DATA\\tmp\\"
in_frame_num = 0
out_vec_len = 0
feature_per_frame = 0
    
def build_cnn(input_var, in_frame_num, feature_per_frame, out_vec_len, filter_num, neuron_num):

    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 3,in_frame_num, feature_per_frame),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network,
            num_filters=filter_num, filter_size=(in_frame_num,6),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    layer2reg =  lasagne.layers.DenseLayer(network,
            num_units=neuron_num,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            layer2reg,
            num_units=out_vec_len,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network, layer2reg

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
def test_acc(inputs_all, outputs_all):
    err = 0
    acc = 0
    batches = 0
    batch = 25
    begin = 0
    sample_num = len(inputs_all)
             
    while begin < sample_num:
        batch_size = np.minimum(batch, sample_num - begin) 
        inputs = inputs_all[begin:begin+batch].reshape(batch_size, 3, in_frame_num, feature_per_frame)
        outputs = outputs_all[begin:begin+batch]
        begin += batch
        
        err, acc = val_fn(inputs, outputs.flatten())
        err += err
        acc += acc
        batches += 1
    return err, acc, batches
    
        
def test2(inputs_all, outputs_all):
    test_number = 0
    test_ok = 0
    test_ok_p = 0


    err = 0
    acc = 0
    err_p = 0
    acc_p = 0
    batches = 0
    batch = 25
    begin = 2000
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
    print("\tstate:  {}->{}".format(test_ok, 1.0 * test_ok / test_number))
    print("\tphone:  {}->{}".format(test_ok_p, 1.0 * test_ok_p / test_number))
    
def read_data_only_inputs(fileName):
    
    with open(fileName,"r") as f:
        #reda header
        values = f.readline().split()
        samples = int(values[0])
        feature_per_frame = int(values[1])
        in_frame_num = int (values[2])
        output_vec_len = int (values[3])
        
        #read inputs
        input_data = [None] * samples
        i_iuput = 0
        for line in f:
            #read outputs
            if i_iuput == samples:
                break
                
            input_data[i_iuput] = map(float, line.split())
            i_iuput += 1

            
        inputs = np.array(input_data, dtype='float32').reshape(samples,in_frame_num,feature_per_frame)
    return inputs, in_frame_num, feature_per_frame, output_vec_len
def swap32(x):
    return (((x << 24) & 0xFF000000) |
            ((x <<  8) & 0x00FF0000) |
            ((x >>  8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF))
            
def read_state_apriori_ppb(file_name, output_len):
    ppb = [None] * output_len
    with open(file_name,"r") as f:
        for i in range(0, output_len):
            ppb[i] = 1.0 / float(f.readline())
    return np.float32(np.diag(ppb))

def main(argv):
    ###-------------------------------- Get files to proccess ----------------------------------------------
        
    # just read header to get layer dimensions
    #files = glob.glob("B:\\NN_data\\THEANO_DATA\\tmp\\*.tmp")
    context = 5
    in_frame_num = context * 2 + 1
    feature_per_frame = 41
    output_vec_len = 214

    files = glob.glob(argv[0] + "\\*.mfsc")
    #inputs, in_frame_num, feature_per_frame, output_vec_len = read_data_only_inputs(files[0])
    print("\feature_per_frame:  {}".format(feature_per_frame))
    print("\output_vec_len:  {}".format(output_vec_len))
    
    ###-------------------------------- Read apriori state ppb ---------------------------------------------
    apriori_ppb_input = read_state_apriori_ppb(argv[2], output_vec_len)
    #print(apriori_ppb)
    #exit(0)
    ###-------------------------------- BUILD Theano functions ----------------------------------------------

    network = None
    layer2reg = None
    total_batch = 0

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    apriori_ppb = T.fmatrix('apriori')
    target_var = T.ivector('targets')
    lrate_var = theano.tensor.scalar('lrate',dtype='float32')
    momentum_var = theano.tensor.scalar('momentum',dtype='float32')

    #create network
    #network, layer2reg = build_cnn(input_var, in_frame_num, feature_per_frame, output_vec_len)
    with open(argv[1], "rb") as f2:
        context = int(f2.readline())
        in_frame_num = context * 2 + 1
        filter_num = int(f2.readline())
        neuron_num = int(f2.readline())
        network, layer2reg = build_cnn(input_var, in_frame_num, feature_per_frame, output_vec_len, filter_num, neuron_num)
        lasagne.layers.set_all_param_values(network, np.load(f2))


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # SGB with momentm and changing learning rate
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate = lrate_var, momentum = momentum_var)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a training function
    train_fn = theano.function([input_var, target_var, lrate_var, momentum_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    test_prediction_1 = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.dot(test_prediction_1, apriori_ppb)
    test_result = T.argmax(test_prediction_1, axis=1)
                    
    # Compile a second function computing the validation loss and accuracy:
    predict_fn = theano.function([input_var], [test_result])
    predict_fn_j = theano.function([input_var, apriori_ppb], [test_prediction])
    print("theano functions compiled")


    ###-------------------------------- Final lunch ----------------------------------------------
    
    file_counter = 0
    print("")

    for one_file in files:       
        inputs_j = load_mfsc_data_with_context(one_file, context)
        dot_index = one_file.find('.mfsc')
        output_file_name = one_file[0:dot_index]+".bin"
        print(inputs_j[0][0])
        sample_num = len(inputs_j)
        base_file_name = output_file_name[output_file_name.rfind('\\')+1:]
        file_counter += 1
        if file_counter % 5 == 0:
            print(base_file_name)
        else:
            print(base_file_name + "   ", end="")
        
        with open(output_file_name, 'wb') as f:
            f.write(struct.pack('I',swap32(len(inputs_j))))
            f.write(struct.pack('I',swap32(100000)))
            f.write(struct.pack('H',output_vec_len*4)[::-1])
            f.write(struct.pack('H',9)[::-1])
                
            begin = 0
            batch = 8
            while begin < sample_num:
                batch_size = np.minimum(batch, sample_num - begin) 
                result = predict_fn_j(inputs_j[begin:begin+batch_size].reshape(batch_size,3,in_frame_num,feature_per_frame), apriori_ppb_input)
                
                res = np.log(result)[0]
                for aaa in res:
                    for ppb in aaa:
                        f.write(ppb)
                        #f.write(struct.pack('f',ppb)[::-1])
                begin += batch_size
    print("")
'''
    ARGUMENTS:
        1) Directory containing thn.tmp files 
            (files with input vectors, one file per utterance)
        2) file storing neural network
        
    Result:
        Julius binary files are stored in the same directory as
        thn.tmp files, name is the same but extension is 'bin' instead of 'thn.tmp'
'''                
if __name__ == "__main__":
   main(sys.argv[1:])

