#include <raylib.h>
#define NN_IMPLEMENTATION
#include "../nn.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"


#define WIDTH 1000
#define HEIGHT 400

#define SCALAR 15

typedef struct{
    int width;
    int height;
    int channels;
    unsigned char* img;
}Img;


int num_from_out(NN nn){
    NN_Matrix out = NN_OUTPUT(nn);
    int res;
    for (size_t i = 0; i < out.columns; i++)
    {
        res = MAT_AT(out, 0, i) + 0.6;
        printf("res is %f\n", MAT_AT(out, 0, i));
        if (res) return i;
    }
    return -1;
}

void draw_nn(NN nn, int dx, int dy, int width, int height){
    Color high_color = CLITERAL(Color){0, 255, 0, 255};
    Color low_color  = CLITERAL(Color){255, 0, 255, 255};

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

float checker(NN nn){
    char file_path[32];
    int width, height, channels;
    int counter = 0;
    for (size_t i = 1; i < 2; i++)
    {
        for (size_t j = 10772; j < 10773; j++)
        {
            snprintf(file_path, sizeof(file_path), "dataset/%zu/%zu/%zu.png", i, i, j);
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
            //nn_forward(nn);
            //int res = num_from_out(nn);

            stbi_image_free(img);
        }
    }
    return (float)counter / (float) 2730;
}

void render_image(Image image, Texture2D texture, Img img){
    for (size_t i = 0; i < image.height; i++){
        for (size_t j = 0; j < image.width; j++){
            int a =  img.img[(i * image.width + j) * img.channels + 3];
            Color color = CLITERAL(Color){a, a, a, 255}; 
            ImageDrawPixel(&image, j, i, color);
        }
    }
    UpdateTexture(texture, image.data);
    DrawTextureEx(texture, CLITERAL(Vector2){0, 0}, 0, SCALAR, WHITE);
}


int main(void){
    InitWindow(WIDTH, HEIGHT, "neural network");
    const char* file_path = "/Users/armansimonyan/CLionProjects/NN/number_recognition.nn";
    NN nn;
    FILE* out_file = fopen(file_path, "rb");
    if (out_file){
        nn = nn_load(out_file);
        fclose(out_file);
    }
    NN_PRINT(nn);
    

    SetTargetFPS(60);
    

    Image bad_image = GenImageColor(28, 28, BLACK);
    Texture2D bad_texture = LoadTextureFromImage(bad_image);

    while (!WindowShouldClose()){
        char file_path[60];
        int width, height, channels;
        int counter = 0;
        for (size_t i = 0; i < 10; i++)
        {
            for (size_t j = 0; j < 20; j++)
            {
                snprintf(file_path, sizeof(file_path), "/Users/armansimonyan/CLionProjects/NN/dataset/%zu/%zu/%zu.png", i, i, j);
                printf("Loading image: %s\n", file_path);
                unsigned char *img = stbi_load(file_path, &width, &height, &channels, 0);
                if (img == NULL) {
                    printf("Failed to load image\n");
                    continue;
                }

                Img new_img = {
                    .width = width,
                    .height = height,
                    .channels = channels,
                    .img = img
                };

                for (int k = 0; k < height; k++) {
                    for (int l = 0; l < width; l++) {
                        int alpha_index = (k * width + l) * channels + 3;
                        float a = (float)img[alpha_index] / 255;
                        MAT_AT(NN_INPUT(nn), 0,  k * width + l) = a;
                    }
                }
                nn_forward(nn);
                bool space_pressed = false;
                while (!(space_pressed || WindowShouldClose()))
                {
                    BeginDrawing();
                    ClearBackground(BLACK);
                    render_image(bad_image, bad_texture, new_img);
                    space_pressed = IsKeyPressed(KEY_SPACE);

                    char res[2];
                    res[0] = num_from_out(nn) + '0';
                    res[1] = '\0';
                    printf("%d\n", num_from_out(nn));
                    DrawText(res, WIDTH / 2, HEIGHT /2, 40, RED);
                    EndDrawing();
                }
                stbi_image_free(img);
            }
        }
    }

    CloseWindow();

    return 0;
}
