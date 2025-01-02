#include <stdio.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

int main(){
    srand(time(NULL));
    NN_Matrix input = {
        .columns = 2,
        .rows = 4,
        .stride = 3,
        .data = td
    };
    NN_Matrix output = {
        .columns = 1,
        .rows = 4,
        .stride = 3,
        .data = td + 2
    };
    size_t arch[] = {2, 3, 1};
    NN neural_network = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(neural_network, 0, 1);
    for (size_t i = 0; i < 50000; i++)
    {
        //nn_back_propagation(neural_network, gradient, input, output);
        nn_finite_difference(neural_network, gradient, 0.1, input, output);
        nn_learn(neural_network, gradient, 30);
        printf("%f\n", nn_cost(neural_network, input, output));
    }
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MAT_AT(NN_INPUT(neural_network), 0, 0) = i; 
            MAT_AT(NN_INPUT(neural_network), 0, 1) = j; 
            nn_forward(neural_network);
            printf("%zu & %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(neural_network), 0, 0));
        }
    }
    printf("\n");
    NN_PRINT(neural_network);
    return 0;
}