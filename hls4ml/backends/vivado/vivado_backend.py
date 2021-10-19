import numpy as np
import math
import os
import sys
from bisect import bisect_left
from queue import Queue
from collections.abc import Iterable

from hls4ml.model.hls_types import FixedPrecisionType, NamedType, IntegerPrecisionType
from hls4ml.model.hls_layers import Layer, Dense, BatchNormalization, Conv1D, Conv2D, Conv2DBatchnorm, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Activation, ParametrizedActivation, PReLU, Softmax, Pooling1D, Pooling2D, GlobalPooling1D, GlobalPooling2D, ZeroPadding1D, ZeroPadding2D, Merge, Concatenate, Dot, Resize, Transpose, GarNet, GarNetStack
from hls4ml.model.hls_attributes import Attribute
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.flow import register_flow
from hls4ml.backends import FPGABackend
from hls4ml.report import parse_vivado_report

#TODO Switch GarNet to the new API
garnet_common_config_template = """
    static const unsigned n_vertices = {n_vertices};
    static const unsigned n_vertices_width = {n_vertices_width};
    static const unsigned n_in_features = {n_in_features};
    static const unsigned distance_width = {distance_width};
    static const unsigned output_collapse = {collapse_type};
    static const bool mean_by_nvert = {mean_by_nvert};

    typedef {norm_t} norm_t;
    typedef ap_fixed<{distance_width}, {distance_nint}, AP_TRN, AP_SAT> distance_t;
    typedef {edge_weight_t} edge_weight_t;
    typedef {edge_weight_aggr_t} edge_weight_aggr_t;
    typedef {aggr_t} aggr_t;
    typedef {output_t} output_t;

    static const unsigned reuse_factor = {reuse};
    static const unsigned log2_reuse_factor = {log2_reuse};
"""

garnet_config_template = """struct config{index} : nnet::garnet_config {{"""
garnet_config_template += garnet_common_config_template
garnet_config_template += """
    static const unsigned n_propagate = {n_propagate};
    static const unsigned n_aggregators = {n_aggregators};
    static const unsigned n_out_features = {n_out_features};

    typedef {input_transform_weights_t} input_transform_weights_t;
    typedef {input_transform_biases_t} input_transform_biases_t;
    typedef {aggregator_distance_weights_t} aggregator_distance_weights_t;
    typedef {aggregator_distance_biases_t} aggregator_distance_biases_t;
    typedef {output_transform_weights_t} output_transform_weights_t;
    typedef {output_transform_biases_t} output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[{input_transform_weights_size}];
    static const input_transform_biases_t (&input_transform_biases)[{input_transform_biases_size}];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[{aggregator_distance_weights_size}];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[{aggregator_distance_biases_size}];
    static const output_transform_weights_t (&output_transform_weights)[{output_transform_weights_size}];
    static const output_transform_biases_t (&output_transform_biases)[{output_transform_biases_size}];

    typedef config{index} base_t;
}};

const config{index}::input_transform_weights_t (&config{index}::input_transform_weights)[{input_transform_weights_size}] = {input_transform_weights};
const config{index}::input_transform_biases_t (&config{index}::input_transform_biases)[{input_transform_biases_size}] = {input_transform_biases};
const config{index}::aggregator_distance_weights_t (&config{index}::aggregator_distance_weights)[{aggregator_distance_weights_size}] = {aggregator_distance_weights};
const config{index}::aggregator_distance_biases_t (&config{index}::aggregator_distance_biases)[{aggregator_distance_biases_size}] = {aggregator_distance_biases};
const config{index}::output_transform_weights_t (&config{index}::output_transform_weights)[{output_transform_weights_size}] = {output_transform_weights};
const config{index}::output_transform_biases_t (&config{index}::output_transform_biases)[{output_transform_biases_size}] = {output_transform_biases};
"""

garnet_stack_base_config_template = """struct config{index}_base : nnet::garnet_config {{"""
garnet_stack_base_config_template += garnet_common_config_template
garnet_stack_base_config_template += """
    static const bool is_stack = true;

    typedef config{index}_base base_t;
}};

struct config{index} : config{index}_base {{
    static const unsigned n_sublayers = {n_sublayers};

    template<int L>
    struct sublayer_t : config{index}_base {{}};
}};

{sublayer_configs}
"""

garnet_stack_sublayer_config_template = """template<>
struct config{index}::sublayer_t<{il}> : config{index}_base {{
    static const unsigned n_in_features = {n_in_features};
    static const unsigned n_propagate = {n_propagate};
    static const unsigned n_aggregators = {n_aggregators};
    static const unsigned n_out_features = {n_out_features};

    typedef {input_transform_weights_t} input_transform_weights_t;
    typedef {input_transform_biases_t} input_transform_biases_t;
    typedef {aggregator_distance_weights_t} aggregator_distance_weights_t;
    typedef {aggregator_distance_biases_t} aggregator_distance_biases_t;
    typedef {output_transform_biases_t} output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[{input_transform_weights_size}];
    static const input_transform_biases_t (&input_transform_biases)[{input_transform_biases_size}];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[{aggregator_distance_weights_size}];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[{aggregator_distance_biases_size}];
    static const output_transform_biases_t (&output_transform_biases)[{output_transform_biases_size}];

    typedef config{index}::sublayer_t<{next}> next_layer_t;
}};

const config{index}::sublayer_t<{il}>::input_transform_weights_t (&config{index}::sublayer_t<{il}>::input_transform_weights)[{input_transform_weights_size}] = {input_transform_weights};
const config{index}::sublayer_t<{il}>::input_transform_biases_t (&config{index}::sublayer_t<{il}>::input_transform_biases)[{input_transform_biases_size}] = {input_transform_biases};
const config{index}::sublayer_t<{il}>::aggregator_distance_weights_t (&config{index}::sublayer_t<{il}>::aggregator_distance_weights)[{aggregator_distance_weights_size}] = {aggregator_distance_weights};
const config{index}::sublayer_t<{il}>::aggregator_distance_biases_t (&config{index}::sublayer_t<{il}>::aggregator_distance_biases)[{aggregator_distance_biases_size}] = {aggregator_distance_biases};
const config{index}::sublayer_t<{il}>::output_transform_biases_t (&config{index}::sublayer_t<{il}>::output_transform_biases)[{output_transform_biases_size}] = {output_transform_biases};
"""

garnet_stack_config_template = (garnet_stack_base_config_template, garnet_stack_sublayer_config_template)

garnet_function_template = 'nnet::garnet{impl}<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'
garnet_stack_function_template = 'nnet::garnet_stack<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'

garnet_include_list = ['nnet_utils/nnet_garnet.h']

class VivadoBackend(FPGABackend):
    def __init__(self):
        super(VivadoBackend, self).__init__('Vivado')
        self._register_flows()

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            'vivado:reshape_stream',
            'vivado:clone_output',
            'vivado:insert_zero_padding_before_conv1d',
            'vivado:insert_zero_padding_before_conv2d',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'vivado:merge_batch_norm_quantized_tanh',
            'vivado:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'vivado:optimize_pointwise_conv',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        vivado_types = [
            'vivado:transform_types',
            'vivado:generate_conv_streaming_instructions',
            'vivado:apply_resource_strategy',
        ]
        vivado_types_flow = register_flow('specific_types', vivado_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', templates, requires=[init_flow], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass for opt_pass in all_passes if opt_pass not in initializers + streaming_passes + quantization_passes + optimization_passes + vivado_types + templates
        ]

        if len(extras) > 0:
            extras_flow = register_flow('extras', extras, requires=[init_flow], backend=self.name)
        else:
            extras_flow = None

        ip_flow_requirements = ['optimize', init_flow, streaming_flow, quantization_flow, optimization_flow, vivado_types_flow, extras_flow, template_flow]
        ip_flow_requirements = list(filter(None, ip_flow_requirements))

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow
    
    def create_initial_config(self, part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel', external_weights=False):
        config = {}

        config['Part'] = part if part is not None else 'xcku115-flvb2104-2-i'
        config['ClockPeriod'] = clock_period
        config['IOType'] = io_type
        config['HLSConfig'] = {}
        # external weights not yet supported

        return config

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        if 'linux' in sys.platform:
            found = os.system('command -v vivado_hls > /dev/null')
            if found != 0:
                raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')
        
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        os.system('vivado_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={vsynth}"'
            .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth))
        os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)
        compression = layer.model.config.get_compression(layer)
        if layer.model.config.is_resource_strategy(layer):
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer)
            if compression:
                layer.set_attr('strategy', 'compressed')
                index_t = layer.get_weights('weight').type.index_precision
            else:
                layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')
        layer.set_attr('index_t', NamedType('layer{}_index'.format(layer.index), index_t))

    #TODO consolidate these functions into a single `init_conv`
    @layer_optimizer(Conv1D)
    def init_conv1d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2: # This can happen if we assign weights of Dense layer to 1x1 Conv1D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0,1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv1D)
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2: # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0,1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv2D)
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(DepthwiseConv2D)
    def init_depconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Activation)
    def init_activation(self, layer):
        if 'table_t' not in layer.attributes:
            layer.set_attr('table_t', NamedType(name=layer.name + '_table_t', precision=FixedPrecisionType(width=18, integer=8)))
        if 'table_size' not in layer.attributes:
            layer.set_attr('table_size', 1024)

    @layer_optimizer(Softmax)
    def init_softmax(self, layer):
        if 'exp_table_t' not in layer.attributes:
            layer.set_attr('exp_table_t', layer.get_attr('table_t'))
        if 'inv_table_t' not in layer.attributes:
            layer.set_attr('inv_table_t', layer.get_attr('table_t'))
        if layer.model.config.is_resource_strategy(layer):
            # 'resource' strategy = 'latency' for Softmax
            layer.set_attr('implementation', 'latency')
        else:
            layer.set_attr('implementation', layer.model.config.get_strategy(layer).lower())
