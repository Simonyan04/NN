#ifndef NN_H_
#define NN_H_

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define ARRAY_LEN(da) sizeof(da)/sizeof(da[0])

#define MAT_AT(m, i, j) m.data[(i) * (m).stride + (j)]
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)

#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_INPUT(nn) (nn).activations[0]
#define NN_OUTPUT(nn) (nn).activations[(nn).count]


float rand_float();
float sigmoidf(float x);

typedef struct
{
    size_t rows;
    size_t columns;
    size_t stride;
    float* data;
}NN_Matrix;


NN_Matrix matrix_alloc(size_t rows, size_t columns);
void matrix_multiplication(NN_Matrix destination, NN_Matrix a, NN_Matrix b);
void matrix_sum(NN_Matrix destination, NN_Matrix a);
NN_Matrix matrix_get_row(NN_Matrix matrix, size_t row);
void matrix_print(NN_Matrix matrix, const char* name, size_t padding);
void matrix_rand(NN_Matrix matrix, float low, float high);
void matrix_copy(NN_Matrix destination, NN_Matrix source);
void matrix_sigmoid(NN_Matrix mat);
void matrix_set(NN_Matrix mat, float num);
void matrix_save(NN_Matrix mat, FILE* file);
NN_Matrix matrix_load(FILE* file);

typedef struct
{
    size_t count;
    NN_Matrix* activations;
    NN_Matrix* weights;
    NN_Matrix* biases;
}NN;

NN nn_alloc(size_t* architecture, size_t size);
void nn_rand(NN nn, float low, float high);
void nn_zero(NN nn);
void nn_print(NN nn, const char* name);
void nn_forward(NN nn);
void nn_finite_difference(NN nn, NN gradient, float eps, NN_Matrix input, NN_Matrix output);
void nn_back_propagation(NN nn, NN gradient, NN_Matrix input, NN_Matrix output);
float nn_cost(NN nn, NN_Matrix input, NN_Matrix output);
void nn_learn(NN nn, NN gradient, float rate);
void nn_save(NN nn, size_t* arch, size_t arch_count, FILE* file);

#endif 
#ifdef NN_IMPLEMENTATION

float rand_float(){
    return (float)rand() / (float)RAND_MAX; 
}

float sigmoidf(float x){
    return 1.f / (1.f + expf(-x));
}

NN_Matrix matrix_alloc(size_t rows, size_t columns){
    NN_Matrix result;
    result.rows = rows;
    result.columns = columns;
    result.stride = columns;
    result.data = (float*)malloc(sizeof(*(result.data)) * rows * columns);
    assert(result.data != NULL);
    return result;
}

NN_Matrix matrix_get_row(NN_Matrix matrix, size_t row){
    return (NN_Matrix){
        .rows = 1,
        .columns = matrix.columns,
        .stride = matrix.stride,
        .data = &MAT_AT(matrix, row, 0)
    };
}

void matrix_print(NN_Matrix mat, const char* name, size_t padding){
    printf("%*s", (int)padding, "");
    printf("%*s%s [\n", (int)padding, "", name);
    for(size_t rows = 0; rows < mat.rows; ++rows){
        printf("%*s    ", (int)padding, "");
        for(size_t columns = 0; columns < mat.columns; ++columns){
            printf("%f ", MAT_AT(mat, rows, columns));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void matrix_rand(NN_Matrix matrix, float low, float high){
    for (size_t i = 0; i < matrix.rows; i++)
    {
        for (size_t j = 0; j < matrix.columns; j++)
        {
            MAT_AT(matrix, i, j) = rand_float()*(high - low) + low;
        }
    }
}

void matrix_copy(NN_Matrix destination, NN_Matrix source){
    assert(destination.rows == source.rows);
    assert(destination.columns == source.columns);
    for (size_t i = 0; i < source.rows; i++)
    {
        for (size_t j = 0; j < source.columns; j++)
        {
            MAT_AT(destination, i, j) = MAT_AT(source, i, j);
        }
    }
    
}

void matrix_sum(NN_Matrix destination, NN_Matrix a){
    assert(destination.rows == a.rows);
    assert(destination.columns == a.columns);
    for (size_t i = 0; i < destination.rows; i++)
    {
        for (size_t j = 0; j < destination.columns; j++)
        {
            MAT_AT(destination, i, j) += MAT_AT(a, i, j);
        }
    }
}

void matrix_multiplication(NN_Matrix destination, NN_Matrix a, NN_Matrix b){
    assert(a.columns == b.rows);
    assert(destination.rows == a.rows);
    assert(destination.columns == b.columns);
    for (size_t i = 0; i < destination.rows; i++)
    {
        for (size_t j = 0; j < destination.columns; j++)
        {
            MAT_AT(destination, i, j) = 0;
            for (size_t k = 0; k < a.columns; k++)
            {
                MAT_AT(destination, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}


void matrix_sigmoid(NN_Matrix mat){
    for (size_t i = 0; i < mat.rows; i++)
    {
        for (size_t j = 0; j < mat.columns; j++)
        {
            MAT_AT(mat, i, j) = sigmoidf(MAT_AT(mat, i, j));
        }
    }
}

void matrix_set(NN_Matrix mat, float num){
    for (size_t i = 0; i < mat.rows; i++)
    {
        for (size_t j = 0; j < mat.columns; j++)
        {
            MAT_AT(mat, i, j) = num;
        }
    }
}

void matrix_save(NN_Matrix mat, FILE* file){
    const char* key = "nn.h.mat";
    fwrite(key, strlen(key), 1, file);
    fwrite(&mat.rows, sizeof(mat.rows), 1, file);
    fwrite(&mat.columns, sizeof(mat.columns), 1, file);
    for (size_t i = 0; i < mat.rows; i++)
    {
        size_t n = fwrite(&MAT_AT(mat, i, 0), sizeof(*mat.data), mat.columns, file);
        while (n < mat.columns && !ferror(file))
        {
            size_t k = fwrite(&MAT_AT(mat, i, n), sizeof(*mat.data), mat.columns - n, file);
            n += k;
        }
    }
}

NN_Matrix matrix_load(FILE* file){
    uint64_t key;
    fread(&key, sizeof(key), 1, file);
    assert(key == 0x74616d2e682e6e6e);
    size_t rows, columns;
    fread(&rows, sizeof(rows), 1, file);
    fread(&columns, sizeof(columns), 1, file);
    NN_Matrix mat = matrix_alloc(rows, columns);
    size_t n = fread(mat.data, sizeof(*mat.data), rows * columns, file);
    while (n < rows * columns && !ferror(file))
    {
        size_t k = fread(mat.data + n, sizeof(*mat.data), rows * columns - n, file);
        n += k;
    }
    return mat;
}

NN nn_alloc(size_t* architecture, size_t count){
    assert(count > 0);
    NN nn;
    nn.count = count - 1;
    nn.weights = (NN_Matrix*)malloc(sizeof(NN_Matrix) * nn.count);
    assert(nn.weights != NULL);
    nn.biases = (NN_Matrix*)malloc(sizeof(NN_Matrix) * nn.count);
    assert(nn.biases != NULL);
    nn.activations = (NN_Matrix*)malloc(sizeof(NN_Matrix) * count);
    assert(nn.activations != NULL);
    nn.activations[0] = matrix_alloc(1, architecture[0]);
    for (size_t i = 1; i < count; i++)
    {
        nn.weights[i-1]   = matrix_alloc(nn.activations[i-1].columns, architecture[i]);
        nn.biases[i-1]    = matrix_alloc(1, architecture[i]);
        nn.activations[i] = matrix_alloc(1, architecture[i]);
    }
    return nn;   
}

void nn_print(NN nn, const char* name){
    char buf[64];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "weights %zu =", i+1);
        matrix_print(nn.weights[i], buf, 4);
        snprintf(buf, sizeof(buf), "biases %zu =", i+1);
        matrix_print(nn.biases[i], buf, 4);   
    }
    
    printf("]\n");
}

void nn_rand(NN nn, float low, float high){
    for (size_t i = 0; i < nn.count; i++)
    {
        matrix_rand(nn.weights[i], low, high);
        matrix_rand(nn.biases[i], low, high);
    }
}

void nn_zero(NN nn){
    for (size_t i = 0; i < nn.count; i++)
    {
        matrix_set(nn.activations[i], 0);
        matrix_set(nn.biases[i], 0);
        matrix_set(nn.weights[i], 0);
    }
    matrix_set(nn.activations[nn.count], 0);
}

void nn_forward(NN nn){
    for (size_t i = 0; i < nn.count; i++)
    {
        matrix_multiplication(nn.activations[i+1], nn.activations[i], nn.weights[i]);
        matrix_sum(nn.activations[i+1], nn.biases[i]);
        matrix_sigmoid(nn.activations[i+1]);
    }   
}


float nn_cost(NN nn, NN_Matrix input, NN_Matrix output){
    assert(input.rows == output.rows);
    assert(NN_OUTPUT(nn).columns == output.columns);
    size_t n = input.rows;
    float c = 0;

    for (size_t i = 0; i < n; i++)
    {
        NN_Matrix input_layer = matrix_get_row(input, i);
        NN_Matrix expected_output_layer = matrix_get_row(output, i);
        matrix_copy(NN_INPUT(nn), input_layer);
        nn_forward(nn);
        for (size_t j = 0; j < output.columns; j++)
        {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(expected_output_layer, 0, j);
            c += d*d;
        }
    }
    return c/n;
}

void nn_finite_difference(NN nn, NN gradient, float eps, NN_Matrix input, NN_Matrix output){
    float cost_before = nn_cost(nn, input, output);
    for (size_t k = 0; k < nn.count; k++)
    {   for (size_t i = 0; i < nn.biases[k].rows; i++)
        {
            for (size_t j = 0; j < nn.biases[k].columns; j++)
            {
                MAT_AT(nn.biases[k], i, j) += eps;
                MAT_AT(gradient.biases[k], i, j) = (nn_cost(nn, input, output) - cost_before)/eps;
                MAT_AT(nn.biases[k], i, j) -= eps;
            }
        }
        for (size_t i = 0; i < nn.weights[k].rows; i++)
        {
            for (size_t j = 0; j < nn.weights[k].columns; j++)
            {
                MAT_AT(nn.weights[k], i, j) += eps;
                MAT_AT(gradient.weights[k], i, j) = (nn_cost(nn, input, output) - cost_before)/eps;
                MAT_AT(nn.weights[k], i, j) -= eps;
            }
        }   
    }
}

void nn_back_propagation(NN nn, NN gradient, NN_Matrix input, NN_Matrix output){
    assert(input.rows == output.rows);
    assert(output.columns == NN_OUTPUT(nn).columns);
    
    nn_zero(gradient);

    size_t n = input.rows;
    for (size_t i = 0; i < n; i++)
    {
        matrix_copy(NN_INPUT(nn), matrix_get_row(input, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; j++)
        {
            matrix_set(gradient.activations[j], 0);
        }
        

        for (size_t j = 0; j < output.columns; j++)
        {
            MAT_AT(NN_OUTPUT(gradient), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) -  MAT_AT(output, i, j);
        }
        for (size_t l = nn.count; l > 0; l--)
        {
            for (size_t j = 0; j < nn.activations[l].columns; j++)
            {
                float a = MAT_AT(nn.activations[l], 0, j);
                float da = MAT_AT(gradient.activations[l], 0, j);
                MAT_AT(gradient.biases[l-1], 0, j) += 2*da*a*(1-a);
                for (size_t k = 0; k < nn.activations[l-1].columns; k++)
                {
                    float pa = MAT_AT(nn.activations[l-1], 0, k);
                    float w = MAT_AT(nn.weights[l-1], k, j);
                    MAT_AT(gradient.weights[l-1], k, j) += 2*da*a*(1-a)*pa;
                    MAT_AT(gradient.activations[l-1], 0, k) += 2*da*a*(1-a)*w;
                }
            }
        }
    }
    for (size_t i = 0; i < gradient.count; i++){
        for (size_t j = 0; j < gradient.weights[i].rows; j++){
            for (size_t k = 0; k < gradient.weights[i].columns; k++){
                MAT_AT(gradient.weights[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < gradient.biases[i].rows; j++){
            for (size_t k = 0; k < gradient.biases[i].columns; k++){
                MAT_AT(gradient.biases[i], j, k) /= n;
            }
        }
    }
    
}

void nn_learn(NN nn, NN gradient, float rate){
    for (size_t k = 0; k < nn.count; k++)
    {   for (size_t i = 0; i < nn.biases[k].rows; i++)
        {
            for (size_t j = 0; j < nn.biases[k].columns; j++)
            {
                MAT_AT(nn.biases[k], i, j) -= rate * MAT_AT(gradient.biases[k], i, j);
            }
        }
        for (size_t i = 0; i < nn.weights[k].rows; i++)
        {
            for (size_t j = 0; j < nn.weights[k].columns; j++)
            {
                MAT_AT(nn.weights[k], i, j) -= rate * MAT_AT(gradient.weights[k], i, j);
            }
        }   
    }   
}

void nn_save(NN nn, size_t* arch, size_t arch_count, FILE* file){
    const char* key = "nn.h.nn ";
    fwrite(key, strlen(key), 1, file);
    fwrite(&arch_count, sizeof(arch_count), 1, file);
    fwrite(arch, sizeof(*arch), arch_count, file);
    for (size_t i = 0; i < nn.count; i++)
    {
        matrix_save(nn.weights[i], file);
        matrix_save(nn.biases[i], file);
    }   
}

NN nn_load(FILE* file){
    uint64_t key;
    size_t arch_count;
    fread(&key, sizeof(key), 1, file);
    assert(key == 0x206E6E2E682E6E6E);
    fread(&arch_count, sizeof(arch_count), 1, file);
    size_t* arch = (size_t*)malloc(arch_count * sizeof(size_t));
    for (size_t i = 0; i < arch_count; i++)
    {
        fread(arch + i, sizeof(*arch), 1, file);
    }
    NN nn = nn_alloc(arch, arch_count);
    for (size_t i = 0; i < nn.count; i++)
    {
        nn.weights[i] = matrix_load(file);
        nn.biases[i] = matrix_load(file);
    }
    return nn;
}

#endif
