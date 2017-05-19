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

#ifndef FEED_FORWARD_TRANSLATION_H_INCLUDED
#define FEED_FORWARD_TRANSLATION_H_INCLUDED

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "layer.h"
#include "hardware_emulator.h"
#include "utilities.h"

#define FF_LAYER_TYPE_CHECK 0x01
#define FF_FC_LAYER 0x00
#define FF_CONV_LAYER 0x01

typedef struct conv_indices
{
    uint32_t ozt;
    uint32_t oyt;
    uint32_t oxt;
    uint32_t t_ofm_z;
    uint32_t t_ofm_y;
    uint32_t t_ofm_x;
} conv_indices;

/**
 * This function is the interface with the outside world.
 *
 * It takes in a set of parameters defining the layer we
 * are operating on and it dispatches this to the relevant
 * processing elements.
 */
void feed_forward(struct LayerParameters * params);

/**
 * This function feeds forward if we have a fully connected layer.
 */
void feed_forward_fc(struct LayerFC * layer_spec);

/**
 * This function handles feed forward for convolutional layers.
 */
void feed_forward_conv(struct LayerConv * layer_spec);

/**
 * Move range of kernels into weight buffer.
 */
void write_kernels_to_buffer(uint32_t start, uint32_t end, struct LayerConv * layer_spec);

/**
 * Move range of input feature maps into input buffer.
 */
void write_ifm_to_buffer(uint32_t oxt, uint32_t oyt, uint32_t t_ofm_y, uint32_t t_ofm_x, struct LayerConv * layer_spec);

/**
 * Move biases to the output buffer for conv layers.
 */
void write_conv_bias_to_buffer(struct LayerConv *layer_spec, struct conv_indices *indices);

/**
 * Allow the hardware parameters to be set externally.
 */
void set_hardware_model(uint32_t input_buffer_size, uint32_t weight_buffer_size, uint32_t output_buffer_size,
        uint32_t mo, uint32_t mi);

/**
 * Initialize the buffers as arrays.
 */
void initialize_hardware_model();

/**
 * Free the memory required for the array pointers.
 */
void teardown_hardware_model();

#endif // FEED_FORWARD_TRANSLATION_H_INCLUDED
