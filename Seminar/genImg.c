#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define WIDTH 30720
#define HEIGHT 17280
#define CHANNELS 1
#define GRAYSCALE_VALUE 128

int main() {
    unsigned char *image1 = malloc(WIDTH * HEIGHT * CHANNELS);
    unsigned char *image2 = malloc(WIDTH * HEIGHT * CHANNELS);
    
    if (image1 == NULL|| image2 == NULL) {
        printf("Unable to allocate memory for image\n");
        return 1;
    }

    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; i++) {
        image1[i] = GRAYSCALE_VALUE;
    }

    int grayscale_value = 0;
    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; i++) {
        image2[i] = grayscale_value;
        grayscale_value = (grayscale_value + 1) % 256;
    }

    if (!stbi_write_jpg("oneColor.jpg", WIDTH, HEIGHT, CHANNELS, image1, 100)) {
        printf("Failed to write image\n");
        free(image1);
        return 1;
    }
    if (!stbi_write_jpg("perfectColor.jpg", WIDTH, HEIGHT, CHANNELS, image2, 100)) {
        printf("Failed to write image\n");
        free(image2);
        return 1;
    }
    printf("Happy\n");
    free(image1);
    free(image2);

    return 0;
}