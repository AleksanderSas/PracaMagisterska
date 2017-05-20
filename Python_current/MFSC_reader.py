from struct import unpack, pack
import numpy

def load_mfsc(file_name):
    samples = 0
    params = None
    file = open(file_name, "rb")
    header = file.read(12)
    samples, sampPeriod, sampSize, parmKind = \
        unpack(">IIHH", header)
    params = numpy.fromfile(file, 'f')
    #print params[0:41]
    file.close()
    params = params.byteswap()

    return params.reshape((samples,123)), samples

def normalizeZeroMeanUnitStdDev(x,pos,means,stdDevs):
    y = x-means[pos]
    y = y/stdDevs[pos]
    return y
    
def normalizeData(tableMFSC):

    means = tableMFSC.mean(axis=0)
    stdDevs = tableMFSC.std(axis=0)

    dim_x = tableMFSC.shape[0]
    dim_y = tableMFSC.shape[1]

    for i in range(0, dim_x):
        for j in range (0, dim_y):
            tableMFSC[i][j] = normalizeZeroMeanUnitStdDev(tableMFSC[i][j], j, means, stdDevs)        
    
def create_mfsc_context(inputs, context, formated_inputs, data_index, feature_per_frame):
    f_in_layer = 41 * (context * 2 + 1)
    for layer in range(3):
        for ctx in range(context * 2 + 1):
            for f in range(41):
                formated_inputs[data_index][layer * f_in_layer + ctx * 41 + f] \
                = inputs[data_index * feature_per_frame + ctx * 123 + layer * 41 + f]

def load_mfsc_data_with_context(file_name, context):
    features, samples = load_mfsc(file_name)
    normalizeData(features)
    data_size = features.shape[0]
    flatten_fetures =features.flatten()
    features_with_ctx = numpy.zeros((data_size - 2 * context, (context * 2 + 1)*123), dtype='float32')
    for i in range(0,data_size - 2 * context):
        create_mfsc_context(flatten_fetures, context, features_with_ctx, i, 123)
    del features
    '''print features_with_ctx[0][0:41]
    print features_with_ctx[0][41:82]
    print features_with_ctx[0][82:123]'''
    return features_with_ctx
    
'''features = load_data_with_context('B:\\NN_State_Recognizer_jurek_data\\train\\10014.mfc', 4)
print features.shape
print features[0][39: 39*2]
print features[1][0:39]'''

