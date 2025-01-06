#define NN_IMPLEMENTATION
#include "nn.h"
#include <stdlib.h>
#include <stdbool.h>

#define BITS 6
#define ITERATIONS 5000
#define TRAIN

size_t arch[] = {2 * BITS, 4 * BITS, BITS + 1};


void writer(char* file_path, NN neural_network){
    FILE* file = fopen(file_path, "wb");
    if(file == NULL) printf("File was not written\n");
    nn_save(neural_network, arch, ARRAY_LEN(arch), file);
    fclose(file);
}

NN get_nn(char* file_path){
    NN neural_network;
    FILE* out_file = fopen(file_path, "rb");
    if (out_file){
        neural_network = nn_load(out_file);
        fclose(out_file);
    }
    else{
        neural_network = nn_alloc(arch, ARRAY_LEN(arch));
        nn_rand(neural_network, -1, 1);
    }
    return neural_network;
}

void generate_input_output(NN_Matrix input, NN_Matrix output){
    size_t n = 1<<BITS;
    for (size_t i = 0; i < input.rows; i++)
    {
        int x = i/n;
        int y = i%n;
        int z = x + y;
        for (size_t j = 0; j < BITS; j++)
        {
            MAT_AT(input, i, j) = (x >> j) & 1;
            MAT_AT(input, i, j + BITS) = (y >> j) & 1;
            MAT_AT(output, i, j) = (z >> j) & 1;
        }
        MAT_AT(output, i, BITS) = (z >> BITS) & 1;        
    }
}

void trainer(NN neural_network, NN gradient, NN_Matrix in, NN_Matrix out, size_t count){
    char file_path[32];
    snprintf(file_path, sizeof(file_path), "../adder/adders/%d_bit_adder.nn", BITS);
    for (size_t i = 0; i < count; i++)
    {
        nn_back_propagation(neural_network, gradient, in, out);
        nn_learn(neural_network, gradient, 1);
        writer(file_path, neural_network);
        printf("%f\n", nn_cost(neural_network, in, out));
    }
}

bool checker(NN neural_network, NN_Matrix input){
    bool correct = true;
    for (size_t in = 0; in < input.rows; in++){
        matrix_copy(NN_INPUT(neural_network), matrix_get_row(input, in));
        nn_forward(neural_network);
        int a = 0;
        int b = 0;
        int c = 0;
        for (int j = BITS - 1; j >= 0; j--)
        {
            a = (a << 1) | (int)MAT_AT(NN_INPUT(neural_network), 0, j);
            b = (b << 1) | (int)MAT_AT(NN_INPUT(neural_network), 0, j + BITS);
        }
        for (int j = BITS; j >= 0; j--)
        {
            c = (c << 1) | (int)(MAT_AT(NN_OUTPUT(neural_network), 0, j) + 0.5);
        }
        if(a + b != c){
            correct = false;
            printf("%d + %d = %d\n", a, b, c);
        }
    }
    return correct;
}

int main(){
    size_t n = 1<<BITS;
    size_t rows = n*n;
    NN_Matrix input =  matrix_alloc(rows, 2 * BITS);
    NN_Matrix output = matrix_alloc(rows, BITS + 1);
    generate_input_output(input, output);

    char file_path[32];
    snprintf(file_path, sizeof(file_path), "../adder/adders/%d_bit_adder.nn", BITS);
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    NN neural_network = get_nn(file_path);

#ifdef TRAIN
    trainer(neural_network, gradient, input, output, ITERATIONS);
#endif
    if(!checker(neural_network, input)){
        printf("Rerun the program, there are mistakes");
    }

    return 0;
}
