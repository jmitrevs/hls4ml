
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.keras.qkeras import get_quantizer_from_config



@keras_handler('FiLM')
def parse_film_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'FiLM')

    layer = parse_default_keras_layer(keras_layer, input_names)

    output_shape = input_shapes[0][:]
    

    if len(input_shapes[0]) < len(input_shapes[1]):
        output_shape = input_shapes[1][:]
    

    
    return layer, output_shape


@keras_handler('Mask_track')
def parse_mask_track_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'Mask_track')

    layer = parse_default_keras_layer(keras_layer, input_names)

    output_shape = input_shapes[0][:]
    output_shape[-1] = 1
    
    return layer, output_shape


@keras_handler('Sum1D')
def parse_sum1d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'Sum1D')

    layer = parse_default_keras_layer(keras_layer, input_names)
    print('\n' * 10)
    print("Sum1D shape: ", input_shapes[:])
    print('\n' * 10)
    
    # input shape: [[None, 76, 128]]

    output_shape = [input_shapes[0][0], input_shapes[0][-1]]
   
    
    print('\n' * 10)
    print("Sum1D out shape: ", output_shape)
    print('\n' * 10)
    
    return layer, output_shape


@keras_handler('Slice_tensor1D')
def parse_slice_tensor1d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'Slice_tensor1D')

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    layer['start'] = keras_layer['config']['start']
    layer['end'] = keras_layer['config']['end']
 

    output_shape = input_shapes[0][:]
    output_shape[-1] = keras_layer['config']['end'] - keras_layer['config']['start']

    
    return layer, output_shape


@keras_handler('TimeDistributed')
def parse_timedistributed_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'TimeDistributed')
    
    layer = parse_default_keras_layer(keras_layer, input_names)
    
    
    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]
    
    layers_config = keras_layer['config']['layer']['config']['layers']
    
    # qdensebatchnorm1
    qdensebatchnorm1_config = layers_config[1]
    layer['n_hid'] = qdensebatchnorm1_config['config']['units']
    layer['epsilon1'] = qdensebatchnorm1_config['config']['epsilon']
    layer['weight_quantizer1'] = get_quantizer_from_config(qdensebatchnorm1_config,'kernel' )
    layer['bias_quantizer1'] = get_quantizer_from_config(qdensebatchnorm1_config,'bias' )
    
    # qdensebatchnorm2
    qdensebatchnorm2_config = layers_config[3]
    layer['n_out'] = qdensebatchnorm2_config['config']['units']
    layer['epsilon2'] = qdensebatchnorm2_config['config']['epsilon']
    layer['weight_quantizer2'] = get_quantizer_from_config(qdensebatchnorm2_config,'kernel' )
    layer['bias_quantizer2'] = get_quantizer_from_config(qdensebatchnorm2_config,'bias' )
    
    output_shape = input_shapes[0][:]
    output_shape[-1] = layer['n_out']
    

    return layer, output_shape