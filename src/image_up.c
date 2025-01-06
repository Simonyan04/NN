#define NN_IMPLEMENTATION
#include "../include/nn.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#include <raylib.h>

#include <pthread.h>

#define WIDTH 1350
#define HEIGHT 450

#define PATH "../src/number_recognizer/dataset/0/0/2.png"


typedef struct{
    int width;
    int height;
    int channels;
    unsigned char* img;
}Img;


void input_output(NN_Matrix in, NN_Matrix out, Img image){
    for (size_t i = 0; i < image.height; i++){
        for (size_t j = 0; j < image.width; j++){
            int alpha_index = (i * image.width + j) * image.channels + 3;
            float a = (float)image.img[alpha_index] / 255;
            MAT_AT(in, i * image.width + j, 0) = (float)j / (image.width - 1);
            MAT_AT(in, i * image.width + j, 1) = (float)i / (image.height - 1);
            MAT_AT(out, i * image.width + j, 0) = a;
        }
    }
}

float nn_interpolate(NN nn, float x, float y){
    MAT_AT(NN_INPUT(nn), 0, 0) = x;
    MAT_AT(NN_INPUT(nn), 0, 1) = y;
    nn_forward(nn);
    return MAT_AT(NN_OUTPUT(nn), 0, 0);
}

void back_and_learn(NN nn, NN gradient, NN_Matrix in, NN_Matrix out, size_t count){
    for (size_t i = 0; i < count; i++){
        nn_back_propagation(nn, gradient, in, out);
        nn_learn(nn, gradient, 1);
    }
    printf("%f\n", nn_cost(nn, in, out));
}

void draw_nn(NN nn, int dx, int dy, int width, int height){
    Color high_color = (Color){0, 255, 0, 255};
    Color low_color  = (Color){255, 0, 255, 255};

    int layer_padding = width / (nn.count + 1);

    for (size_t i = 0; i < NN_INPUT(nn).columns; i++)
    {
        DrawCircle(dx + (layer_padding / 2), i * (height / NN_INPUT(nn).columns) + dy + (height / NN_INPUT(nn).columns) / 2, 10, GRAY);
    }

    for (size_t i = 0; i < nn.count; i++){
        NN_Matrix current_weights = nn.weights[i];
        for (size_t j = 0; j < current_weights.rows; j++)
        {
            int current_neuron_padding = HEIGHT / current_weights.rows;
            int next_neuron_padding = HEIGHT / nn.activations[i + 1].columns;
            for (size_t k = 0; k < current_weights.columns; k++){
                int value = sigmoidf(MAT_AT(current_weights, 0, j)) * 255.f;
                high_color.a = value;
                Color final_color = ColorAlphaBlend(low_color, high_color, WHITE);
                DrawLine(i * layer_padding + layer_padding/2 + dx, j * current_neuron_padding + current_neuron_padding / 2 + dy, (i+1) * layer_padding + layer_padding/2 + dx, next_neuron_padding * k + dy + next_neuron_padding / 2 , final_color);
            }
        }
        NN_Matrix current_biases = nn.biases[i];
        for (size_t j = 0; j < current_biases.columns; j++){
            int value = sigmoidf(MAT_AT(current_biases, 0, j)) * 255.f;
            high_color.a = value;
            Color final_color = ColorAlphaBlend(low_color, high_color, WHITE);
            int current_neuron_padding = height / current_biases.columns; 
            DrawCircle((i+1) * layer_padding + dx + layer_padding/2, current_neuron_padding * j + dy + current_neuron_padding / 2, 10, final_color);
        }
    }
}

void render_upscaled_image(size_t x, size_t y, size_t new_width ,size_t new_height, NN nn, Image image, Texture2D texture){
    for (size_t i = 0; i < new_height; i++){
            for (size_t j = 0; j < new_width; j++){
                int a = 255 * nn_interpolate(nn, (float)j / (new_width - 1), (float)i / (new_height - 1));
                Color color = (Color){a, a, a, 255}; 
                ImageDrawPixel(&image, j, i, color);
            }
        }
    UpdateTexture(texture, image.data);
    DrawTexture(texture, x, y, WHITE);
}

void render_image(Image image, Texture2D texture, Img img){
    for (size_t i = 0; i < image.height; i++){
        for (size_t j = 0; j < image.width; j++){
            int a =  img.img[(i * image.width + j) * img.channels + 3];
            Color color = (Color){a, a, a, 255}; 
            ImageDrawPixel(&image, j, i, color);
        }
    }
    UpdateTexture(texture, image.data);
}

int main(){
    int image_width, image_height, image_channels;
    unsigned char *img = stbi_load(PATH, &image_width, &image_height, &image_channels, 0);
    if (img == NULL) {
        printf("Failed to load image %s\n", PATH);
        return 1;
    }

    Img file_image = {
        .width = image_height,
        .height = image_height,
        .channels = image_channels,
        .img = img
    };
    
    NN_Matrix in = matrix_alloc(image_width * image_height, 2);
    NN_Matrix out = matrix_alloc(image_width * image_height, 1);

    size_t arch[] = {2, 7, 4, 1};
    
    input_output(in, out, file_image);

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, -1, 1);

    InitWindow(WIDTH, HEIGHT, "picture");

    SetTargetFPS(300);

    int new_width = WIDTH / 3;
    int new_height = HEIGHT;

    Image ray_new_image = GenImageColor(new_width, new_height, BLACK);
    Texture2D new_texture = LoadTextureFromImage(ray_new_image);

    Image ray_old_image = GenImageColor(image_width, image_height, BLACK);
    Texture2D old_texture = LoadTextureFromImage(ray_old_image);

    render_image(ray_old_image, old_texture, file_image);

    while (!WindowShouldClose())
    {
        BeginDrawing();

        ClearBackground(BLACK);
        
        back_and_learn(nn, gradient, in, out, 100);

        render_upscaled_image((WIDTH * 2) / 3, 0,  new_width, new_height, nn, ray_new_image, new_texture);
        
        DrawTextureEx(old_texture, (Vector2){0, 0}, 0, WIDTH / (3 * 28), WHITE);

        draw_nn(nn, WIDTH / 3, 0, new_width, new_height);

        EndDrawing();
    }
    
    return 0;
}