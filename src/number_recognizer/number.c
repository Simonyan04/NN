#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define NN_IMPLEMENTATION
#include "nn.h"
#include <stdio.h>

#define COUNT_PER_NUMBER 1000
#define START 0

size_t arch[] = {28*28, 64, 32, 10};

#define FILE_PATH "../src/number_recognizer/number_recognition.nn"

int num_from_out(NN nn){
    NN_Matrix out = NN_OUTPUT(nn);
    int res;
    for (size_t i = 0; i < out.columns; i++)
    {
        res = MAT_AT(out, 0, i) + 0.5;
        if (res) return i;
    }
    return -1;
}

void writer(char* file_path, NN neural_network){
    FILE* file = fopen(file_path, "wb");
    if(file == NULL){
        printf("File was not written\n");
        return;
    }
    nn_save(neural_network, arch, ARRAY_LEN(arch), file);
    fclose(file);
}


void input_output(NN_Matrix input, NN_Matrix output, size_t start){
    char file_path[64];
    int width, height, channels;
    matrix_set(output, 0);
    for (size_t i = 0; i < 10; i++)
    {
        for (size_t j = 0; j < COUNT_PER_NUMBER; j++)
        {
            snprintf(file_path, sizeof(file_path), "../src/number_recognizer/dataset/%zu/%zu/%zu.png", i, i, j + start);
            unsigned char *img = stbi_load(file_path, &width, &height, &channels, 0);
            if (img == NULL) {
                printf("Failed to load image %s\n", file_path);
                continue;
            }
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int alpha_index = (k * width + l) * channels + 3;
                    float a = (float)img[alpha_index] / 255;
                    MAT_AT(input, i * COUNT_PER_NUMBER + j,  k * width + l) = a;
                    MAT_AT(output, i * COUNT_PER_NUMBER + j, i) = 1;
                }
            }
            stbi_image_free(img);
        }
    }
}

void trainer(NN neural_network, NN gradient, NN_Matrix in, NN_Matrix out, float rate, float desired_cost){
    float cost = nn_cost(neural_network, in, out);
    while(cost > desired_cost)
    {
        nn_back_propagation(neural_network, gradient, in, out);
        nn_learn(neural_network, gradient, rate);
        writer(FILE_PATH, neural_network);
        cost = nn_cost(neural_network, in, out);
        printf("%f\n", cost);
    }
}


float checker(NN nn){
    char file_path[64];
    int width, height, channels;
    int counter = 0;
    for (size_t i = 0; i < 10; i++)
    {
        for (size_t j = 9500; j < 10773; j++)
        {
            snprintf(file_path, sizeof(file_path), "../src/number_recognizer/dataset/%zu/%zu/%zu.png", i, i, j);
            unsigned char *img = stbi_load(file_path, &width, &height, &channels, 0);
            if (img == NULL) {
                printf("Failed to load image\n");
                continue;
            }
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    int alpha_index = (k * width + l) * channels + 3;
                    float a = (float)img[alpha_index] / 255;
                    MAT_AT(NN_INPUT(nn), 0,  k * width + l) = a;
                }
            }
            nn_forward(nn);
            int res = num_from_out(nn);
            //printf("%d = %zu\n", res, i);
            if(res == i) counter++;
            stbi_image_free(img);
        }
    }
    return (float)counter / (float) 12730;
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


int main() {
    NN_Matrix input = matrix_alloc(COUNT_PER_NUMBER * 10, 28 * 28);
    NN_Matrix output = matrix_alloc(COUNT_PER_NUMBER * 10, 10);

    NN nn = get_nn(FILE_PATH);
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    size_t batch_size = 10;
    
    for (size_t j = 0; j < batch_size; j++)
    {
        for (size_t i = START; i < 9500; i += 500){
            input_output(input, output, i);
            trainer(nn, gradient, input, output, 1, 0.04);
        }
    }

    printf("\n%f\n", checker(nn));

    return 0;
}
