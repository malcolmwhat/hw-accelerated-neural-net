/*
 * hardware_emulator.cpp
 *
 *  Created on: May 15, 2017
 *      Author: Malcolm Watt
 */

#include "hardware_emulator.h"

hardware_model hw = {};
fc_control_signal fc_ctrl_sig = {};

void fc_begin_hardware_acceleration()
{
	// Loop over the outputs for this particular tile.
    uint32_t ob;
	for (ob = 0; ob < hw.tile_height_reg; ob += hw.m_o)
	{
		// Cover the edge case of having not enough outputs to fill the m_o.
		fc_ctrl_sig.m_o = MIN(hw.m_o, hw.tile_height_reg - ob);
		fc_ctrl_sig.o_0 = ob;

		// Loop over the inputs for this particular tile.
        uint32_t ib;
		for (ib = 0; ib < hw.tile_width_reg; ib += hw.m_i)
		{
			// Handle the edge case where m_i and m_o are not factors of t_i and t_o.
			fc_ctrl_sig.m_i = MIN(hw.m_i, hw.tile_width_reg - ib);

			// Handle the end of line application of the activation function.
			if (ib + hw.m_i >= hw.tile_width_reg && hw.apply_activation_flag == HW_MODEL_APPLY_ACTIVATION)
			{
				fc_ctrl_sig.activation = HW_MODEL_APPLY_ACTIVATION;
			}
			else
			{
				fc_ctrl_sig.activation = HW_MODEL_NO_ACTIVATION;
			}

			fc_ctrl_sig.i_0 = ib;

			fc_compute_step();

		}
	}
}

void fc_compute_step()
{
    uint32_t o;
	for (o = 0; o < fc_ctrl_sig.m_o; o++)
	{
        uint32_t i;
		for (i = 0; i < fc_ctrl_sig.m_i; i++)
		{
			hw.output_buffer[o + fc_ctrl_sig.o_0] += hw.input_buffer[i + fc_ctrl_sig.i_0]
			        * hw.weight_buffer[(fc_ctrl_sig.o_0 + o) * hw.tile_width_reg + fc_ctrl_sig.i_0 + i];
			if (i + 1 == fc_ctrl_sig.m_i && fc_ctrl_sig.activation == HW_MODEL_APPLY_ACTIVATION)
			{
				apply_activation(&(hw.output_buffer[o + fc_ctrl_sig.o_0]));
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

void conv_begin_hardware_acceleration(uint32_t kernel_size_y, uint32_t kernel_size_x) {
	// TODO: Tiling at the hardware level.
    /*
     * Loop over the tiles in the buffer.
     */
    uint32_t oy, ox, ozb, ky, kx, kzb, oz, kz;
    for (oy = 0; oy < hw.conv_t_ofm_y; oy++)
    {
        for (ox = 0; ox < hw.conv_t_ofm_x; ox++)
        {
            for (ozb = 0; ozb < hw.conv_t_ofm_z; ozb += hw.m_o)
            {
                // TODO: Cover edge cases.

                for (ky = 0; ky < kernel_size_y; ky++)
                {
                    for (kx = 0; kx < kernel_size_x; kx++)
                    {
                        for (kzb = 0; kzb < ; kzb += hw.m_i)
                        {

                        }
                    }
                }
            }
        }
    }
}
