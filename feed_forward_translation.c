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

void feed_forward(struct LayerParameters *params) {
    // Check if the parameters specify a feed-forward network.
    if ((params->layer_type & FF_LAYER_TYPE_CHECK) == FF_FC_LAYER) {
        feed_forward_fc(params->fc_structure);
    } else if ((params->layer_type & FF_LAYER_TYPE_CHECK) == FF_CONV_LAYER) {
        feed_forward_conv(params->conv_structure);
    }
}

void feed_forward_fc(struct LayerFC *layer_spec) {
    /*
     * The first loop moves across the output neurons.
     */
    uint32_t ot;
    for (ot = 0; ot < layer_spec->output_size; ot += hw.fc_output_buffer_size) {
        // Cover edge case of partially overlapping the buffer.
        // It's possible we have less data than can fill the output buffer.
        uint32_t o_buff_size = hw.fc_output_buffer_size;
        uint32_t required_o_buff_space = layer_spec->output_size - ot;

        hw.tile_height_reg = MIN(o_buff_size, required_o_buff_space);

        // Place the biases for the current outputs in the output buffers.
        uint32_t i;
        for (i = 0; i < hw.tile_height_reg; i++) {
            hw.output_buffer[i] = layer_spec->biases[ot + i]; // Eventually this will be placed in the FPGA buffer.
        }

        /*
         * The second loop moves across the inputs neurons.
         */
        uint32_t it;
        for (it = 0; it < layer_spec->input_size; it += hw.fc_input_buffer_size) {
            // Again, cover edge cases for tiling.
            uint32_t i_buff_size = hw.fc_input_buffer_size;
            uint32_t required_i_buff_space = layer_spec->input_size - it;

            hw.tile_width_reg = MIN(i_buff_size, required_i_buff_space);

            // Set the apply_activation_flag if we are at the last set of inputs for these particular outputs.
            if (it + hw.fc_input_buffer_size > layer_spec->input_size) {
                hw.apply_activation_flag = HW_MODEL_APPLY_ACTIVATION;
            } else {
                hw.apply_activation_flag = HW_MODEL_NO_ACTIVATION;
            }

            // Place the inputs in the input buffer.
            uint32_t j;
            for (j = 0; j < hw.tile_width_reg; j++) {
                hw.input_buffer[j] = layer_spec->inputs[it + j];
            }

            // weights_temp_origin serves as a starting point in the matrix of weights from which
            // we can reference the data to be into buffers relatively.
            uint32_t weights_temp_origin = ot * layer_spec->input_size + it;

            // Iterate over the weights and place them in the buffer in row major order.
            uint32_t k;
            for (k = 0; k < hw.tile_height_reg; k++) {
                uint32_t m;
                for (m = 0; m < hw.tile_width_reg; m++) {
                    hw.weight_buffer[k * hw.tile_width_reg + m] = layer_spec->weights[weights_temp_origin
                                                                                      + k * layer_spec->input_size + m];
                }
            }

            /*
             * Now we have the input, output and weights buffer for the hardware accelerator completely filled.
             */
            fc_begin_hardware_acceleration();
        }

        /*
         * Read the output buffers from hardware accelerator and put it in memory.
         */
        uint32_t out_buffer_index;
        for (out_buffer_index = 0; out_buffer_index < hw.tile_height_reg; out_buffer_index++) {
            layer_spec->outputs[ot + out_buffer_index] = hw.output_buffer[out_buffer_index];
        }
    }
}

void feed_forward_conv(struct LayerConv *layer_spec) {
    // OFM -> Output feature map
    uint32_t ofm_size_z = layer_spec->ofm_dims[2];
    uint32_t ofm_size_y = layer_spec->ofm_dims[0];
    uint32_t ofm_size_x = layer_spec->ofm_dims[1];

    // First loop is over the output feature maps.
    uint32_t ozt, oyt, oxt;

    conv_indices indices = (conv_indices) {.oxt=0, .oyt=0, .ozt=0, .t_ofm_x=hw.conv_t_ofm_x, .t_ofm_y=hw.conv_t_ofm_y,
            .t_ofm_z=hw.conv_t_ofm_z};

    for (ozt = 0; ozt < ofm_size_z; ozt += hw.conv_t_ofm_z) {
        indices.ozt = ozt;
        // Move the corresponding kernels into the weights buffer.
        write_kernels_to_buffer(ozt, ozt + hw.conv_t_ofm_z, layer_spec);

        // Tiling the input feature maps
        for (oyt = 0; oyt < ofm_size_y; oyt++) {
            indices.oyt = oyt;
            for (oxt = 0; oxt < ofm_size_x; oxt++) {
                indices.oxt = oxt;
                // Move the input feature maps into the input buffer.
                write_ifm_to_buffer(oxt, oyt, hw.conv_t_ofm_y, hw.conv_t_ofm_x, layer_spec);

                // Move the bias to the output buffer.
                write_conv_bias_to_buffer(layer_spec, &indices);


                // TODO: Read output buffer into main memory when done.
            }
        }
    }
}

void set_hardware_model(uint32_t input_buffer_size, uint32_t weight_buffer_size, uint32_t output_buffer_size,
                        uint32_t m_o, uint32_t m_i) {
    hw.fc_input_buffer_size = input_buffer_size;
    hw.fc_weight_buffer_size = weight_buffer_size;
    hw.fc_output_buffer_size = output_buffer_size;
    hw.m_o = m_o;
    hw.m_i = m_i;
}

void initialize_hardware_model() {
    hw.input_buffer = (float *) malloc(sizeof(float) * hw.fc_input_buffer_size);
    hw.output_buffer = (float *) malloc(sizeof(float) * hw.fc_output_buffer_size);
    hw.weight_buffer = (float *) malloc(sizeof(float) * hw.fc_weight_buffer_size);
}

void teardown_hardware_model() {
    free(hw.input_buffer);
    free(hw.output_buffer);
    free(hw.weight_buffer);
}

void write_kernels_to_buffer(uint32_t start, uint32_t end, struct LayerConv *layer_spec) {
    uint32_t i, j, k, l;
    uint32_t kernel_rows = layer_spec->kernel_dims[0];
    uint32_t kernel_cols = layer_spec->kernel_dims[1];
    uint32_t number_of_input_feature_maps = layer_spec->ifm_dims[2];
    for (i = 0; i < end - start; i++) {
        for (j = 0; j < kernel_rows; j++) {
            for (k = 0; k < kernel_cols; k++) {
                for (l = 0; l < number_of_input_feature_maps; l++) {
                    hw.weight_buffer[l + k * number_of_input_feature_maps + j * kernel_cols + i * kernel_rows]
                            = layer_spec->kernel[l + k * number_of_input_feature_maps +
                                                 j * kernel_cols + (i + start) * kernel_rows];
                }
            }
        }
    }
}

void write_ifm_to_buffer(uint32_t oxt, uint32_t oyt, uint32_t t_ofm_y, uint32_t t_ofm_x, struct LayerConv *layer_spec) {
    uint32_t i, j, k;
    uint32_t stride = layer_spec->stride;
    uint32_t kernel_rows = layer_spec->kernel_dims[0];
    uint32_t kernel_cols = layer_spec->kernel_dims[1];
    uint32_t ifm_depth = layer_spec->ifm_dims[2];
    uint32_t ifm_cols = layer_spec->ifm_dims[1];
    float *inputs = layer_spec->ifm;

    uint32_t i_upper_lim = stride * (t_ofm_y - 1) + kernel_rows;
    uint32_t j_upper_lim = stride * (t_ofm_x - 1) + kernel_cols;

    for (i = 0; i < i_upper_lim; i++) {
        for (j = 0; j < j_upper_lim; j++) {
            for (k = 0; k < ifm_depth; k++) {
                hw.input_buffer[k + j * ifm_depth + i * j_upper_lim] =
                        inputs[k + (j + oxt) * ifm_depth + (i + oyt) * ifm_cols];
            }
        }
    }
}

void write_conv_bias_to_buffer(struct LayerConv *layer_spec, struct conv_indices *indices) {
    uint32_t i, j, k;

    uint32_t bias_depth = layer_spec->ofm_dims[2];
    uint32_t bias_cols = layer_spec->ofm_dims[0];

    for (i = 0; i < indices->t_ofm_y; i++) {
        for (j = 0; j < indices->t_ofm_x; j++) {
            for (k = 0; k < indices->t_ofm_z; k++) {
                hw.output_buffer[k + j * indices->t_ofm_z + i * indices->t_ofm_x] =
                        layer_spec->biases[(k + indices->ozt) + (j + indices->oxt) * bias_depth +
                                           (i + indices->oyt) * bias_cols];
            }
        }
    }
}
