#ifndef NNET_SEPARABLE_CONV2D_STREAM_H_
#define NNET_SEPARABLE_CONV2D_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include "nnet_sepconv_stream.h"
#include "nnet_conv2d_stream.h"

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_encoded_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_chan])
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == CONFIG_T::filt_width);

    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    const int win_depth = CONFIG_T::filt_height * CONFIG_T::out_width;
    for (unsigned i_out = 0; i_out < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; i_out++) {
        #pragma HLS STREAM variable=data_window[i_out] depth=win_depth
    }

    #pragma HLS ARRAY_PARTITION variable=CONFIG_T::pixels complete

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack
    unsigned outputs_ready = 0;

    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> pixel_idx[data_T::size / CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=pixel_idx complete

    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            #pragma HLS LOOP_FLATTEN
            if (CONFIG_T::strategy == nnet::latency && data_T::size / CONFIG_T::n_chan == 1) {
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            }
            compute_scaled_indices_2d<data_T, CONFIG_T>(i_ih, i_iw, pixel_idx);
            compute_depthwise_output_encoded<data_T, res_T, CONFIG_T>(data.read(), data_window, res, res_pack, outputs_ready, weights, biases, pixel_idx);
        }
    }
}

// Line Buffer Implementation (Phil's)
template<class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_buffer_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_chan])
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[CONFIG_T::filt_height - 1][CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            if (CONFIG_T::strategy == nnet::latency) {
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            }
            if (CONFIG_T::filt_height > 1) {
                compute_depthwise_output_buffer_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res, weights, biases);
            } else {
                compute_depthwise_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
            }
        }
    }
}

//Single stream for Depthwise Conv2d
template<class data_T, class res_T, typename CONFIG_T>
void depthwise_ss_product(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    #pragma HLS INLINE

    typename CONFIG_T::accum_t mult[CONFIG_T::n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    #pragma HLS ARRAY_PARTITION variable=mult complete

    
	
	const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);
	
    CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::limit(multiplier_limit);

    // Do the matrix-multiply
    Product: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS UNROLL
        mult[ii] = CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[ii], weights[ii]);
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
	
	const int kernel_size = CONFIG_T::n_in / CONFIG_T::n_out;
    Accum1: for(int ii = 0; ii < kernel_size; ii++) {
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            int index = ii * CONFIG_T::n_out + jj;
            acc[jj] += mult[index];
        }
    }

    
	for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        
        res_T tmp = acc[ires];
        res[ires] =  tmp;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_cl_ss(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]){
    
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    std::cout  << "USE DEPTHWISE CONV2D SS"<< std::endl;
    
    // use HLS ap_shift_reg as linebuffer
    static ap_shift_reg<data_T, (CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right)> linebuffer[(CONFIG_T::filt_height)-1][CONFIG_T::n_chan];
    // #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=2
    
    // use array as linebuffer
    //static data_T linebuffer[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][(CONFIG_T::filt_height)-1][CONFIG_T::n_chan];
	//#pragma HLS ARRAY_PARTITION variable=linebuffer complete
    
    data_T tmpdata[CONFIG_T::n_chan];
    // #pragma HLS ARRAY_RESHAPE variable=tmpdata complete

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    // #pragma HLS ARRAY_RESHAPE variable=layer_in complete

    res_T layer_out[CONFIG_T::n_filt];
    // #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0
    
	res_T out_data;
	
	
	// Thresholds
    const static int lShiftX = CONFIG_T::filt_width - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    // Counters
    static int pX = 0; // Pixel X
    static int pY = 0; // Pixel Y

    static int sX = 0; // Stride X
    static int sY = 0; // Stride Y
	
    Conv2dWHLoop:
    for (unsigned i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width; i++) {
        Conv2dChanLoop:
        for(int i1 = 0; i1 < CONFIG_T::n_chan; i1++) { 
        // #pragma HLS UNROLL
            tmpdata[i1] = data.read();
        }
        
        // Perform LineBuffer
		nnet::cnnshift_apshiftreg<data_T,res_T,  CONFIG_T>(tmpdata, linebuffer, layer_in);
        //nnet::cnnshift_arr<data_T,res_T,  CONFIG_T>(tmpdata, linebuffer, layer_in);
        
		
		// Check to see if we have a full kernel
		if ( (sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
			
			// Dense multiply
			#pragma HLS INLINE region
			
				depthwise_ss_product<data_T,res_T, typename CONFIG_T::mult_config>(layer_in, layer_out, weights, biases);
			

			// Write output to stream when output ready
            Con2dCastLoop:
			for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
				// #pragma HLS UNROLL
				out_data = layer_out[i_ic];
				res.write(out_data);
			}		
		}

		// Counter Housekeeping
		if (pX + 1 == CONFIG_T::in_width)  // Includes padding, end of line (padded)
		{
			pX = 0; 
			sX = 0;
			if (pY + 1 == CONFIG_T::in_height) {  // Reached bottom of image
				pY = 0; 
				sY = 0;
			} else {
				pY = pY + 1;
				// Update stride (threshold) ? subtract stride : increment stride
				sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1; 
			}
		} else {
			pX = pX + 1;
			// Update stride (threshold) ? subtract stride : increment stride
			sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1; 
		}
	}
}
	

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            if (CONFIG_T::strategy == nnet::latency && data_T::size / CONFIG_T::n_chan == 1) {
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            }
            if (i_ih % CONFIG_T::stride_height == 0 && i_iw % CONFIG_T::stride_width == 0) {
                pointwise_mult_buffer<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
            } else {
                data.read();
            }
        }
    }
}

// Single Stream for PointWise Conv2d
template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_cl_ss(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);
	

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    
    static data_T layer_in[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=layer_in complete
    
    res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=layer_out complete
    
   	ReadInputHeight: 
	for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        	ReadInputWidth: 
		for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {			
			if (CONFIG_T::strategy == nnet::latency) {
				  #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
			}
			// full kernel
			if (i_ih % CONFIG_T::stride_height == 0 && i_iw % CONFIG_T::stride_width == 0) {				
				ReadInputChan: 
				for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
					layer_in[i_ic] = data.read();				
				}
				// Dense
				#pragma HLS INLINE region					
				if (CONFIG_T::strategy == nnet::latency) {
					dense_latency<data_T,res_T, typename CONFIG_T::mult_config>(layer_in, layer_out, weights, biases);
				} 
				else {
					dense_large<data_T,res_T, typename CONFIG_T::mult_config>(layer_in, layer_out, weights, biases);
				}
				// Write output to stream when output ready
				Pointwise2dCastLoop:
				for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
					// #pragma HLS UNROLL
					res_T out_data = layer_out[i_ic];
					res.write(out_data);
				}		
						
			}
			// skip data
			else {
				SkipInputChan: 
				for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
					data.read();				
				}		  
			}
        	}
	}
}

template<class data_T, class res_T, typename CONFIG_T>
void separable_conv_2d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::depthwise_config::weight_t depthwise_weights[CONFIG_T::depthwise_config::filt_height * CONFIG_T::depthwise_config::filt_width * CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::weight_t pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
    typename CONFIG_T::depthwise_config::bias_t   depthwise_biases[CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::bias_t   pointwise_biases[CONFIG_T::pointwise_config::n_filt]
) {
    #pragma HLS DATAFLOW

    hls::stream<data_T> depthwise_res;
    unsigned res_depth = CONFIG_T::depthwise_config::out_height * CONFIG_T::depthwise_config::out_width;
    #pragma HLS STREAM variable=depthwise_res depth=res_depth

    switch(CONFIG_T::depthwise_config::implementation){
        case conv_implementation::linebuffer:
            depthwise_conv_2d_buffer_cl<data_T, data_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights, depthwise_biases);
            break;
        case conv_implementation::encoded:
            depthwise_conv_2d_encoded_cl<data_T, data_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights, depthwise_biases);
            break;
    } 

    pointwise_conv_2d_cl<data_T, res_T, typename CONFIG_T::pointwise_config>(depthwise_res, res, pointwise_weights, pointwise_biases);
}

// Single stream for Seperable Conv2d
template<class data_T, class res_T, typename CONFIG_T>
void separable_conv_2d_cl_ss(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::depthwise_config::weight_t depthwise_weights[CONFIG_T::depthwise_config::filt_height * CONFIG_T::depthwise_config::filt_width * CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::weight_t pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
    typename CONFIG_T::depthwise_config::bias_t   depthwise_biases[CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::bias_t   pointwise_biases[CONFIG_T::pointwise_config::n_filt]
) {
    #pragma HLS DATAFLOW
	
    // output stream for depthwise conv2d
    hls::stream<data_T> depthwise_res;
    #pragma HLS STREAM variable=depthwise_res depth=1

    
    depthwise_conv_2d_cl_ss<data_T, data_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights, depthwise_biases);
    pointwise_conv_2d_cl_ss<data_T, res_T, typename CONFIG_T::pointwise_config>(depthwise_res, res, pointwise_weights, pointwise_biases);
}
    
}
#endif
