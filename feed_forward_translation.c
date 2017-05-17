/**
 * This module serves to tile and compute the feed forward
 * results for a convolutional neural network.
 *
 * The eventual goal of this is to offload the low level
 * computations to the FPGA.
 *
 * Once we can do this reliably, we want to expose a higher
 * level API such that the performance gains of the feed-
 * forward being done in the FPGA can be evaluated.
 */

#include "feed_forward_translation.h"

void feed_forward(struct LayerParameters * params)
{
	// Check if the parameters specify a feed-forward network.
	if ((params->layer_type & FF_LAYER_TYPE_CHECK) == FF_FC_LAYER)
	{
		feed_forward_fc(params->fc_structure);
	}
	else if ((params->layer_type & FF_LAYER_TYPE_CHECK) == FF_CONV_LAYER)
	{
		feed_forward_conv(params->conv_structure);
	}
}

void feed_forward_fc(struct LayerFC * layer_spec)
{
	/*
	 * The first loop moves across the output neurons.
	 */
    uint32_t ot;
	for (ot = 0; ot < layer_spec->output_size; ot += hw.output_buffer_size)
	{
		// Cover edge case of partially overlapping the buffer.
		// It's possible we have less data than can fill the output buffer.
		uint32_t o_buff_size = hw.output_buffer_size;
		uint32_t required_o_buff_space = layer_spec->output_size - ot;

		hw.tile_height_reg = MIN(o_buff_size, required_o_buff_space);

		// Place the biases for the current outputs in the output buffers.
        uint32_t i;
		for (i = 0; i < hw.tile_height_reg; i++)
		{
			hw.output_buffer[i] = layer_spec->biases[ot + i]; // Eventually this will be placed in the FPGA buffer.
		}

		/*
		 * The second loop moves across the inputs neurons.
		 */
        uint32_t it;
		for (it = 0; it < layer_spec->input_size; it += hw.input_buffer_size)
		{
			// Again, cover edge cases for tiling.
			uint32_t i_buff_size = hw.input_buffer_size;
			uint32_t required_i_buff_space = layer_spec->input_size - it;

			hw.tile_width_reg = MIN(i_buff_size, required_i_buff_space);

			// Set the apply_activation_flag if we are at the last set of inputs for these particular outputs.
			if (it + hw.input_buffer_size > layer_spec->input_size)
			{
				hw.apply_activation_flag = HW_MODEL_APPLY_ACTIVATION;
			}
			else
			{
				hw.apply_activation_flag = HW_MODEL_NO_ACTIVATION;
			}

			// Place the inputs in the input buffer.
            uint32_t j;
			for (j = 0; j < hw.tile_width_reg; j++)
			{
				hw.input_buffer[j] = layer_spec->inputs[it + j];
			}

			// weights_temp_origin serves as a starting point in the matrix of weights from which
			// we can reference the data to be into buffers relatively.
			uint32_t weights_temp_origin = ot * layer_spec->input_size + it;

			// Iterate over the weights and place them in the buffer in row major order.
            uint32_t k;
			for (k = 0; k < hw.tile_height_reg; k++)
			{
                uint32_t m;
				for (m = 0; m < hw.tile_width_reg; m++)
				{
					hw.weight_buffer[k * hw.tile_width_reg + m] = layer_spec->weights[weights_temp_origin
					        + k * layer_spec->input_size + m];
				}
			}

			/*
			 * Now we have the input, output and weights buffer for the hardware accelerator completely filled.
			 */
			begin_hardware_acceleration();
		}

		/*
		 * Read the output buffers from hardware accelerator and put it in memory.
		 */
		uint32_t out_buffer_index;
		for (out_buffer_index = 0; out_buffer_index < hw.tile_height_reg; out_buffer_index++)
		{
			layer_spec->outputs[ot + out_buffer_index] = hw.output_buffer[out_buffer_index];
		}
	}
}

void feed_forward_conv(struct LayerConv * layer_spec)
{
	// TODO
}

void set_hardware_model(uint32_t input_buffer_size, uint32_t weight_buffer_size, uint32_t output_buffer_size,
        uint32_t m_o, uint32_t m_i)
{
	hw.input_buffer_size = input_buffer_size;
	hw.weight_buffer_size = weight_buffer_size;
	hw.output_buffer_size = output_buffer_size;
	hw.m_o = m_o;
	hw.m_i = m_i;
}

void initialize_hardware_model()
{
	hw.input_buffer = (float *) malloc(sizeof(float) * hw.input_buffer_size);
	hw.output_buffer = (float *) malloc(sizeof(float) * hw.output_buffer_size);
	hw.weight_buffer = (float *) malloc(sizeof(float) * hw.weight_buffer_size);
}

void teardown_hardware_model()
{
	free(hw.input_buffer);
	free(hw.output_buffer);
	free(hw.weight_buffer);
}
