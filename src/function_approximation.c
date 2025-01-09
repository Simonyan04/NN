#define NN_IMPLEMENTATION
#include "nn.h"
#include <math.h>

#include <raylib.h>

#define WIDTH 800
#define HEIGHT 600

#define Y_SCREEN(y) (HEIGHT/2 - (y*2 - 1)*HEIGHT/2)
#define NORMALIZED(x, a, b) (((x) - (a))/((b) - (a)))

typedef struct {
    float x;
    float y;
}Point;

float function(float x) {
    return sinf(5*x);
}

Point min_max(const Point* points, size_t n) {
    Point res;
    res.x = points[0].y;
    res.y = points[0].y;


    for (int i = 1; i < n; i++) {
        res.x = res.x < points[i].y ? res.x : points[i].y;
        res.y = res.y > points[i].y ? res.y : points[i].y;
    }
    return res;
}

int normal_function_points(float (*function)(float), float a, float b, Point* points, size_t n) {
    float step = (b - a) / (float) n;
    for (int i = 0; i < n; i++) {
        points[i].x = (float)i/n;
        points[i].y = function(a + i*step);
    }
    Point normalizer = min_max(points, n);
    for (int i = 0; i < n; i++) {
        points[i].y = NORMALIZED(points[i].y, normalizer.x, normalizer.y);
    }
    return n;
}

int interpolated_points(NN nn, Point* points, size_t n) {
    for (int i = 0; i < n; i++) {
        points[i].x = (float)i/n;
        MAT_AT(NN_INPUT(nn), 0, 0) = points[i].x;
        nn_forward(nn);
        points[i].y = MAT_AT(NN_OUTPUT(nn), 0, 0);
    }
    return n;
}


void input_output(NN_Matrix* in, NN_Matrix* out, const Point* points, size_t n){
    float* new_in = malloc(sizeof(float) * n);
    float* new_out = malloc(sizeof(float) * n);

    for (size_t i = 0; i < n; i++)
    {
        new_in[i] = points[i].x;
        new_out[i] = points[i].y;
    }

    in->rows = n;
    in->columns = 1;
    in->stride = 1;
    in->data = new_in;

    out->rows = n;
    out->columns = 1;
    out->stride = 1;
    out->data = new_out;
}

void back_and_learn(NN nn, NN gradient, NN_Matrix in, NN_Matrix out, size_t count){
    for (size_t i = 0; i < count; i++){
        nn_back_propagation(nn, gradient, in, out);
        nn_learn(nn, gradient, 1);
    }
    printf("%f\n", nn_cost(nn, in, out));
}

int main(){
    size_t arch[] = {1, 12, 1};


    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, -1, 1);

    MAT_AT(nn.weights[0], 0, 0) = 1;
    MAT_AT(nn.biases[0], 0, 0) = 0;

    NN_Matrix in = matrix_alloc(0,0);
    NN_Matrix out = matrix_alloc(0,0);


    float a = -3; float b = 3;
    int n = 40;

    Point points[n];
    normal_function_points(function, a, b, points, n);

    input_output(&in, &out, points, n);


    InitWindow(WIDTH, HEIGHT, "graph");

    SetTargetFPS(300);

    while (!WindowShouldClose()) {
        Point real_points[WIDTH];
        Point nn_points[WIDTH];
        BeginDrawing();
        ClearBackground(BLACK);

        back_and_learn(nn, gradient, in, out, 250);

        interpolated_points(nn, nn_points, WIDTH);
        normal_function_points(function, a, b, real_points, WIDTH);


        for (size_t i = 0; i < WIDTH-1; i++) {
            DrawPixel(nn_points[i].x * WIDTH, Y_SCREEN(nn_points[i].y) , RED);
            DrawPixel(real_points[i].x * WIDTH, Y_SCREEN(real_points[i].y), BLUE);
        }

        EndDrawing();
    }

    NN_PRINT(nn);
}