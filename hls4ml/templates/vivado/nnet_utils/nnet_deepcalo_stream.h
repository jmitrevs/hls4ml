#ifndef NNET_DEEPCALO_STREAM_H_
#define NNET_DEEPCALO_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include "nnet_batchnorm_stream.h"
#include "nnet_activation_stream.h"
#include "nnet_dense.h"
#include "nnet_batchnorm.h"
#include "nnet_activation.h"

#include <cmath>
namespace nnet {


template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void film(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
	
	input2_T scalar_gamma;
	res_T out_data;
	
	input2_T in_data2 = data2.read();	
	for (int i = 0; i < CONFIG_T::n_inp1/input1_T::size; i++) {
		
		input1_T in_data1 = data1.read();
		
		FilmPack: for (int k = 0; k < input1_T::size; k++) {
			
			// out = img*(gamma+1)+beta
			out_data[k] = in_data1[k]*(in_data2[k]+1) + in_data2[input1_T::size+k];
		}
		
		res.write(out_data);
		
	}			
					
}


template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void film_ss(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
	
	// input1 is the image, input2 is the scalar
	// n_inp2 is the double of n_chan, first half of scalar is considered as gamma, last half is beta
	input1_T img_data;
	
	input2_T scalar[CONFIG_T::n_inp2];
	#pragma HLS ARRAY_PARTITION variable=scalar complete
	
	res_T out_data;
	typename CONFIG_T::accum_t acc;
	
	ScalarLoop:
	for (unsigned i = 0; i < CONFIG_T::n_inp2; i++) {
		#pragma HLS PIPELINE
		scalar[i] = data2.read();	
	}
		
	OutLoop:
	for (unsigned i = 0; i < CONFIG_T::height * CONFIG_T::width; i++) {
		#pragma HLS loop_flatten
		#pragma HLS PIPELINE
		
		for (unsigned j = 0; j < CONFIG_T::n_chan; j++) {
			
			img_data = data1.read();
		
			//out = img*(gamma+1)+beta
			acc = img_data*(scalar[j]+1) + scalar[CONFIG_T::n_chan+j];
			
			out_data = (res_T)acc;			
			res.write(out_data);
		}		
	}	
	
}


template<class data_T, class res_T, typename CONFIG_T>
void mask_track(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {

    	MaskLoop1: 
		for (unsigned i = 0; i < CONFIG_T::n_in/data_T::size; i++) {
        	#pragma HLS PIPELINE
			
        	data_T in_data = data.read();   
			
        	res_T out_data;
			#pragma HLS DATA_PACK variable=out_data

        	MaskLoop2: 
			for (unsigned k = 0; k < data_T::size; k++){
				#pragma HLS UNROLL
               	
				if(in_data[k]==0){
					out_data[0] = 0;
				}else{
					out_data[0] = 1;
				}
            }
            res.write(out_data);
    	}
}

template<class data_T, class res_T, typename CONFIG_T>
void mask_track_ss(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        
        res_T out_data;
        MaskLoop1: for (unsigned i = 0; i < CONFIG_T::n_in/6; i++) {
            #pragma HLS PIPELINE II=1
            MaskLoop2: for (unsigned k = 0; k < 6; k++){
                data_T in_data = data.read();
                if(in_data==0){
                    out_data = 0;
                }else{
                    out_data = 1;
                }
            }
            res.write(out_data);
        }
}

template<class data_T, class res_T, typename CONFIG_T>
void sum1d(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {

	// sum up in dimension1(height)
	// example: input (None, 66, 128), output (None, 128)
	
	res_T out_data;
	#pragma HLS DATA_PACK variable=out_data

	for (unsigned i = 0; i < res_T::size; i++)	out_data[i] = 0;

    	for (unsigned i = 0; i < CONFIG_T::n_in/data_T::size; i++) {
        	
        	data_T in_data = data.read();

        	for (unsigned k = 0; k < data_T::size; k++){
            		if (res_T::size == 1) 
            			out_data[0] = out_data[0] + in_data[k];
            		else                  
            			out_data[k] = out_data[k] + in_data[k];
       	 }
	}
    	res.write(out_data);
}

template<class data_T, class res_T, typename CONFIG_T>
void sum1d_ss(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {

	data_T in_data;
		
	typename CONFIG_T::accum_t acc[CONFIG_T::width];
	#pragma HLS ARRAY_PARTITION variable=acc complete
	
	res_T out_data;
	
	InitLoop:
	for (unsigned i = 0; i < CONFIG_T::width; i++)	
		#pragma HLS PIPELINE
		acc[i] = 0;
	
	AccLoop:
    for (unsigned i = 0; i < CONFIG_T::height; i++) {
		#pragma HLS loop_flatten
		#pragma HLS PIPELINE
		for (unsigned k = 0; k < CONFIG_T::width; k++){
			in_data = data.read();
			acc[k] += in_data;
		}
	}
	
	CastLoop:
	for (unsigned i = 0; i < CONFIG_T::width; i++){	
		#pragma HLS PIPELINE
		out_data = (res_T) acc[i];
    	res.write(out_data);
	}
}


template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        
	data_T in_data = data.read();
	res_T out_data;
	#pragma HLS DATA_PACK variable=out_data

	for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
		if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
			out_data[i-CONFIG_T::start] = in_data[i];
		}
    }
	res.write(out_data);
}


template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d_ss(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        

	for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS PIPELINE II=1
		data_T in_data = data.read();
		if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
			res_T out_data = in_data;
			res.write(out_data);
		}
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void dense_mult_ss(
	hls::stream<data_T> &data,
	hls::stream<res_T>  &res,
	typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out]
	) {
	
	
  

	typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
      #pragma HLS ARRAY_PARTITION variable=acc complete
	
	InitAccum:
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS PIPELINE
        acc[iacc] = (typename CONFIG_T::accum_t) 0;
    }

	for(unsigned i_in = 0; i_in < CONFIG_T::n_in; i_in++) {
		#pragma HLS PIPELINE 
		data_T tmpdata = data.read();
	
		for (int i_out = 0; i_out < CONFIG_T::n_out; i_out++) {
	    
			unsigned w_index  =  i_in + (CONFIG_T::n_in*i_out); 
	    acc[i_out] += tmpdata * weights[w_index];
		}
	}
	
	 ResWrite: 
	 for(unsigned i_out = 0; i_out < CONFIG_T::n_out; i_out++) {         
          #pragma HLS PIPELINE 
          
          res_T tmpres = (res_T) acc[i_out];
          
          res.write(tmpres);
      }
}


template<class data_T, class res_T, typename CONFIG_T>
void timedistributed_ss(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res,
		typename CONFIG_T::dense1_config::weight_t     d_weight1[CONFIG_T::n_in][CONFIG_T::n_hid],		
		typename CONFIG_T::dense1_config::bias_t       d_bias1[CONFIG_T::n_hid],		
		typename CONFIG_T::dense2_config::weight_t     d_weight2[CONFIG_T::n_hid][CONFIG_T::n_out],
		typename CONFIG_T::dense2_config::bias_t       d_bias2[CONFIG_T::n_out]	

	  ) {
    
	for (unsigned i = 0; i < CONFIG_T::n_timesteps; i++) {
		
		#pragma HLS DATAFLOW 
		
		// dense1
		hls::stream<data_T> dense1_res;
		#pragma HLS STREAM variable=dense1_res depth=1
		nnet::dense_ss<data_T, data_T, typename CONFIG_T::dense1_config>(data, dense1_res, d_weight1, d_bias1);
		
		// relu 1
		hls::stream<data_T> relu1_res;
		#pragma HLS STREAM variable=relu1_res depth=1
		nnet::relu_ss<data_T, data_T, typename CONFIG_T::relu1_config>(dense1_res, relu1_res);
		
		// dense2
		hls::stream<data_T> dense2_res;
		#pragma HLS STREAM variable=dense2_res depth=1
		nnet::dense_ss<data_T, data_T, typename CONFIG_T::dense2_config>(relu1_res, dense2_res, d_weight2, d_bias2);

		// relu 2		
		nnet::relu_ss<data_T, res_T, typename CONFIG_T::relu2_config>(dense2_res, res);
	}
	
}








}
#endif
