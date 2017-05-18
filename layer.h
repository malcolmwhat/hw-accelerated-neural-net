/*
 * Layer.h
 *
 *  Created on: May 15, 2017
 *      Author: mw
 */

#ifndef LAYER_H_
#define LAYER_H_

/*
 * Structure for a convolutional layer.
 */
typedef struct LayerConv
{
	uint32_t * ofm_dims; // Output feature map dimensions.
	uint32_t * ifm_dims; // Input feature map dimensions.
	uint32_t * kernel_dims; // Number of filters is inferred.
	uint8_t * stride;

	int32_t * ifm;
	int32_t * ofm;
	int32_t * kernel;
	int32_t * biases;
} LayerConv;

/*
 * Define the structure for a fully connected network.
 */
typedef struct LayerFC
{
	uint32_t input_size;
	uint32_t output_size; // Size of the weight matrix is inferred.

	float * inputs;
	float * outputs;
	float * weights;
	float * biases;
} LayerFC;

/*
 * Layer parameters which are sent from the calling program.
 */
typedef struct LayerParameters
{
	// Booleans defining whether this is a FC or CONV layer.
	uint8_t layer_type;

	LayerFC *fc_structure;
	LayerConv * conv_structure;
} LayerParameters;

#endif /* LAYER_H_ */
