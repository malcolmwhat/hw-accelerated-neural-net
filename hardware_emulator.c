/*
 * hardware_emulator.cpp
 *
 *  Created on: May 15, 2017
 *      Author: Malcolm Watt
 */

#include "hardware_emulator.h"

hardware_model hw = {};
control_signal ctrl_sig = {};

void begin_hardware_acceleration()
{
	// Loop over the outputs for this particular tile.
    uint32_t ob;
	for (ob = 0; ob < hw.tile_height_reg; ob += hw.m_o)
	{
		// Cover the edge case of having not enough outputs to fill the m_o.
		ctrl_sig.m_o = MIN(hw.m_o, hw.tile_height_reg - ob);
		ctrl_sig.o_0 = ob;

		// Loop over the inputs for this particular tile.
        uint32_t ib;
		for (ib = 0; ib < hw.tile_width_reg; ib += hw.m_i)
		{
			// Handle the edge case where m_i and m_o are not factors of t_i and t_o.
			ctrl_sig.m_i = MIN(hw.m_i, hw.tile_width_reg - ib);

			// Handle the end of line application of the activation function.
			if (ib + hw.m_i > hw.tile_width_reg && hw.apply_activation_flag == HW_MODEL_APPLY_ACTIVATION)
			{
				ctrl_sig.activation = HW_MODEL_APPLY_ACTIVATION;
			}
			else
			{
				ctrl_sig.activation = HW_MODEL_NO_ACTIVATION;
			}

			ctrl_sig.i_0 = ib;

			// TODO: It looks like the input offsets are not correct. Fix that.
			compute_step();

		}
	}
}

void compute_step()
{
    uint32_t o;
	for (o = 0; o < ctrl_sig.m_o; o++)
	{
        uint32_t i;
		for (i = 0; i < ctrl_sig.m_i; i++)
		{
			// TODO: Debug this.
			hw.output_buffer[o + ctrl_sig.o_0] = hw.input_buffer[i + ctrl_sig.i_0]
			        * hw.weight_buffer[(ctrl_sig.o_0 + o) * hw.tile_width_reg + ctrl_sig.i_0 + i];
			if (i + 1 == ctrl_sig.m_i && ctrl_sig.activation == HW_MODEL_APPLY_ACTIVATION)
			{
				apply_activation(&(hw.output_buffer[o + ctrl_sig.o_0]));
			}
		}
	}
}

void apply_activation(float * output_register)
{
	/*
	 * Assume ReLU for now.
	 */
	if (*(output_register) < 0.0)
	{
		*(output_register) = 0.0;
	}
}
