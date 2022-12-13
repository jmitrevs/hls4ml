
from hls4ml.model.layers import  TimeDistributed, FiLM,  Sum1D, Slice_tensor1D, Mask_track
from hls4ml.backends.backend import get_backend

from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# Deepcalo layers templates

film_config_template = """struct config{index} : nnet::film_config {{
    static const unsigned height = {height};
    static const unsigned width = {width};
    static const unsigned n_chan = {n_chan};
    static const unsigned n_inp1 = {n_inp1};
    static const unsigned n_inp2 = {n_inp2};
    typedef {accum_t.name} accum_t;
}};\n"""

mask_track_config_template = """struct config{index} : nnet::mask_track_config {{	 
    static const unsigned n_in = {n_in};
    static const unsigned height = {height};
    static const unsigned width = {width};	
}};\n"""

sum1d_config_template = """struct config{index} : nnet::sum1d_config {{	 
    static const unsigned n_in = {n_in};
    static const unsigned height = {height};
    static const unsigned width = {width};
    typedef {accum_t.name} accum_t;	
}};\n"""

slice_tensor1d_config_template = """struct config{index} : nnet::slice_tensor1d_config {{	 
    static const unsigned n_in = {n_in};
    static const unsigned start = {start};
    static const unsigned end = {end};	
}};\n"""

film_function_template = 'nnet::film_ss<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
mask_track_function_template = 'nnet::mask_track_ss<{input_t}, {output_t}, {config}>({input}, {output});'
sum1d_function_template = 'nnet::sum1d_ss<{input_t}, {output_t}, {config}>({input}, {output});'
slice_tensor1d_function_template = 'nnet::slice_tensor1d_ss<{input_t}, {output_t}, {config}>({input}, {output});'

deepcalo_include_list = ['nnet_utils/nnet_deepcalo.h', 'nnet_utils/nnet_deepcalo_stream.h']


# FiLM 
class FiLMConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(FiLM)
        self.template = film_config_template
        
    def format(self, node):
        inp1 = node.get_input_variable(node.inputs[0]) 
        inp2 = node.get_input_variable(node.inputs[1]) 

        if len(inp1.shape) < len(inp2.shape): 
           inp1,inp2 =  inp2,inp1   #inp1 is the image, inp2 is the scalar
           
        params = self._default_config_params(node)
     
        params['height'] = inp1.shape[0]
        params['width'] = inp1.shape[1]
        params['n_chan'] = inp1.shape[-1]
        params['n_inp1'] = inp1.size_cpp()
        params['n_inp2'] = inp2.size_cpp()
  
        return self.template.format(**params)

        
class FiLMFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(FiLM, include_header=deepcalo_include_list)
        self.template = film_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['input1_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input2_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['output_t'] = node.get_output_variable().type.name
        params['input1'] = node.get_input_variable(node.inputs[0]).name
        params['input2'] = node.get_input_variable(node.inputs[1]).name
        params['output'] = node.get_output_variable().name

        return self.template.format(**params)

# Mask_track
class MaskTrackConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Mask_track)
        self.template = mask_track_config_template
        
    def format(self, node):           
        params = self._default_config_params(node)
     
        inp = node.get_input_variable()
        shape = inp.shape
        params['n_in'] = inp.size_cpp()
        params['height'] = shape[0]            
        params['width'] =  shape[1]
  
        return self.template.format(**params)

        
class MaskTrackFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Mask_track, include_header=deepcalo_include_list)
        self.template = mask_track_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


#Sum1D
class Sum1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Sum1D)
        self.template = sum1d_config_template
        
    def format(self, node):           
        params = self._default_config_params(node)
     
        inp = node.get_input_variable()
        shape = inp.shape
        params['n_in'] = inp.size_cpp()
        params['height'] = shape[0]
        if len(shape) == 1:          
            params['width'] =  1
        else:
            params['width'] =  shape[1]
  
        return self.template.format(**params)

        
class Sum1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Sum1D, include_header=deepcalo_include_list)
        self.template = sum1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


        
#Slice_tensor1D
class SliceTensor1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Slice_tensor1D)
        self.template = slice_tensor1d_config_template
        
    def format(self, node):           
        params = self._default_config_params(node)
     
        params['n_in'] = node.get_input_variable().size_cpp()
        params['start'] = node.get_attr('start')
        params['end'] =  node.get_attr('end')
  
        return self.template.format(**params)

        
class SliceTensor1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Slice_tensor1D, include_header=deepcalo_include_list)
        self.template = slice_tensor1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


# TimeDistributed templates

timedistributed_function_template = 'nnet::timedistributed_ss<{input_t}, {output_t}, {config}>({input}, {output}, {d_weight1}, {d_bias1}, {d_weight2}, {d_bias2});'

# dense  
timedistributed_mult1_config_template = """struct config{index}_mult1 : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

timedistributed_mult2_config_template = """struct config{index}_mult2 : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# relu
timedistributed_relu1_config_template = """struct config{index}_relu1 : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};    
}};\n"""

timedistributed_relu2_config_template = """struct config{index}_relu2 : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""


timedistributed_config_template = """struct config{index} : nnet::timedistributed_config {{    
    static const unsigned n_in = {n_in};
    static const unsigned n_hid = {n_hid};
    static const unsigned n_out = {n_out};
    static const unsigned n_timesteps = {n_timesteps};  
    
    // dense config
    typedef {config_mult_t1} dense1_config;
    typedef {config_mult_t2} dense2_config;
    

    
    //  relu config
    typedef {config_relu_t1} relu1_config;
    typedef {config_relu_t2} relu2_config    ;
    
    
}};"""



class TimeDistributedConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(TimeDistributed)
        self.template = timedistributed_config_template
        self.mult1_template = timedistributed_mult1_config_template
        self.mult2_template = timedistributed_mult2_config_template
        self.relu1_template = timedistributed_relu1_config_template
        self.relu2_template = timedistributed_relu2_config_template
    
    def format(self, node):
        # main config
        params = self._default_config_params(node)
        params['n_in'] = node.get_attr('n_in')
        params['n_hid'] = node.get_attr('n_hid')
        params['n_out'] = node.get_attr('n_out')        
        params['n_timesteps'] = node.get_attr('n_timesteps')       
        params['config_mult_t1'] = 'config{}_mult1'.format(node.index)
        params['config_mult_t2'] = 'config{}_mult2'.format(node.index)        
        params['config_relu_t1'] = 'config{}_relu1'.format(node.index)
        params['config_relu_t2'] = 'config{}_relu2'.format(node.index)
        
        timedistributed_config = self.template.format(**params)
        
        # dense1 config
        mult1_params = self._default_config_params(node)
        
        mult1_params['n_in'] = params['n_in']
        mult1_params['n_out'] = params['n_hid']
        mult1_params['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('d_weight1').type.precision)
        mult1_params['strategy'] = 'resource'
        mult1_params['accum_t'] = node.get_attr('accum_t')
        mult1_params['bias_t'] = node.get_weights('d_bias1').type
        mult1_params['weight_t'] = node.get_weights('d_weight1').type
        mult1_config = self.mult1_template.format(**mult1_params)
        
        # dense2 config
        mult2_params = self._default_config_params(node)
        
        mult2_params['n_in'] = params['n_hid']
        mult2_params['n_out'] = params['n_out']
        mult2_params['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('d_weight2').type.precision)
        mult2_params['strategy'] = 'resource'
        mult2_params['accum_t'] = node.get_attr('accum1_t')
        mult2_params['bias_t'] = node.get_weights('d_bias2').type
        mult2_params['weight_t'] = node.get_weights('d_weight2').type
        mult2_config = self.mult2_template.format(**mult2_params)
        
        #  relu1 config
        relu1_params = self._default_config_params(node)
        relu1_params['n_in'] = params['n_hid']
        relu1_config = self.relu1_template.format(**relu1_params)
        
        #  relu2 config
        relu2_params = self._default_config_params(node)
        relu2_params['n_in'] = params['n_out']
        relu2_config = self.relu2_template.format(**relu2_params)

        return  mult1_config + '\n' + mult2_config + '\n' + relu1_config + '\n' + relu2_config + '\n' + timedistributed_config
        
class TimeDistributedFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(TimeDistributed, include_header=deepcalo_include_list)
        self.template = timedistributed_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['d_weight1'] = node.get_weights('d_weight1').name
        params['d_weight2'] = node.get_weights('d_weight2').name
        params['d_bias1'] = node.get_weights('d_bias1').name
        params['d_bias2'] = node.get_weights('d_bias2').name
        
        return self.template.format(**params)