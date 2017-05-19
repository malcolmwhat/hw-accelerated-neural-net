/*
 * hardware_model.h
 *
 *  Created on: May 15, 2017
 *      Author: Malcolm Watt
 */

#ifndef HARDWARE_EMULATOR_H_
#define HARDWARE_EMULATOR_H_

#include <stdint.h>
#include "utilities.h"

#define HW_MODEL_APPLY_ACTIVATION 0x01
#define HW_MODEL_NO_ACTIVATION 0x00

/*
 * Model for the FPGA.
 */
typedef struct hardware_model
{
	// The register addresses for the start of the various buffers.
	float * input_buffer;
	float * weight_buffer;
	float * output_buffer;

	// All sizes are in the number of input elements we can place.
	uint32_t fc_input_buffer_size;
	uint32_t fc_weight_buffer_size;
	uint32_t fc_output_buffer_size;

	// All sizes are in the number of input elements we can place.
	uint32_t conv_t_ofm_z;
	uint32_t conv_t_ofm_y;
	uint32_t conv_t_ofm_x;

	// Registers to set the tile heights.
	uint32_t tile_height_reg;
	uint32_t tile_width_reg;

	// Notifies the hardware accelerator whether the end of the tile is the end
	// of the output and therefore whether we should apply the activation function.
	uint8_t apply_activation_flag;

	// Number of multipliers available for use on board.
	uint32_t m; // Number of multipliers on chip.
	uint32_t m_o; // Number of dot product operators on chip.
	uint32_t m_i; // Number of multipliers per dot product operator.
} hardware_model;

/*
 * Model for the control_signals of the compute unit.
 */
typedef struct fc_control_signal
{
	// Relative address of the input and output for this compute cycle.
	uint32_t i_0;
	uint32_t o_0;

	uint32_t m_i;
	uint32_t m_o;
	uint8_t activation;
} fc_control_signal;

/*
 * Model for the control signal for the conv compute unit.
 */
typedef struct conv_control_signal
{
	// Don't worry about this yet...
} conv_control_signal;

extern hardware_model hw;
extern fc_control_signal fc_ctrl_sig;

/**
 * This function emulates the behaviour of the on chip controller.
 */
void fc_begin_hardware_acceleration();

/**
 * This function emulates the on chip controller for conv layers.
 */
void conv_begin_hardware_acceleration();

/**
 * This function emulates the behaviour of the on chip compute unit.
 */
void fc_compute_step();

/**
 * Applies the current activation function.
 */
void apply_activation(float * output_register);

#endif /* HARDWARE_EMULATOR_H_ */
