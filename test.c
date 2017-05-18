//
// Created by mw on 17/05/17.
//

#include "test.h"

/**
 * Tests the relu activation function.
 */
void test_relu()
{
    uint32_t local_success_count = success_count;
    uint32_t local_error_count = error_count;

    float x = 10.0;
    float y;
    y = x;
    apply_activation(&y);
    (y == x) ? success_count++ : error_count++;

    y = -10.0;
    apply_activation(&y);
    (y == 0.0) ? success_count++: error_count++;

    puts("Relu Activation Function Test:");
    printf("%d Successes\n", success_count - local_success_count);
    printf("%d Failures\n", error_count - local_error_count);
    puts(delimiter);
}

/**
 * Test the feed forward functionality from the highest level.
 */
void test_feed_forward()
{
    uint32_t local_success_count = success_count;
    uint32_t local_error_count = error_count;

    // Set the hardware model to something small.
    set_hardware_model(3, 6, 2, 1, 2);

    // Initialize the hardware model.
    initialize_hardware_model();

    int input_size = 5;
    int output_size = 5;

    float inputs[5] = {0.1, 0.2, 0.1, 0.5, 0.9};

    float biases[5] = {0.5, 0.3, -0.1, -0.4, -0.1};

    float weights[25] = {0.0, 0.1, 0.1, 0.5, -0.3,
                         0.9, -0.1, -0.1, -0.6, 0.8,
                         0.0, 0.1, 0.1, 0.0, 0.1,
                         0.9, -0.1, -0.1, -0.6, -0.2,
                         0.9, -0.1, -0.1, -0.6, 0.8};

    float * outputs;
    outputs = (float *) malloc(output_size);

    // Generate layer specification.
    LayerFC fcLayer = (LayerFC) {.input_size = input_size, .output_size = output_size,
    .inputs = inputs, .biases = biases, .weights = weights, .outputs = outputs};

    LayerParameters params = {.layer_type = FF_FC_LAYER, .fc_structure = &fcLayer};

    feed_forward(&params);

    // Calculated separately.
    float expected[] = {0.51, 0.78, 0.02, 0.0, 0.38};
    int i;
    for (i = 0; i < output_size; i++){
        // Since there are precision issues with the numbers being used, there is a very slight deviation.
        (expected[i] - params.fc_structure->outputs[i] < 0.00001) ? success_count++ : error_count++;
    }

    // Teardown the hardware model.
    teardown_hardware_model();

    free(outputs);

    puts("Feed Forward Function Test:");
    printf("%d Successes\n", success_count - local_success_count);
    printf("%d Failures\n", error_count - local_error_count);
    puts(delimiter);
}

/**
 * Basic testing interface which writes to console.
 */
void run_tests()
{
    success_count = 0;
    error_count = 0;

    /*
     * Basically just modify what we call here.
     */
    test_relu();
    test_feed_forward();

    puts("Overall test results:");
    printf("%d Successes\n", success_count);
    printf("%d Failures\n", error_count);
    puts(delimiter);
}

int main(void)
{
    run_tests();
    return 0;
}