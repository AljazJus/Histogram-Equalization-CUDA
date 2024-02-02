#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1


unsigned long findMin(unsigned long* cdf){
    
    unsigned long min = 0;
    // grem skozi CDF dokler ne najdem prvi nenicelni element ali pridem do konca
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
    }
    
    return min;
}

unsigned char Scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize){
    
    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = round(scale * (float)(GRAYLEVELS-1));
    
    return (int)scale;
}

void CalculateHistogram(unsigned char* image, int width, int height, unsigned long* histogram){
    
    //Clear histogram:
    for (int i=0; i<GRAYLEVELS; i++) {
        histogram[i] = 0;
    }
    
    //Calculate histogram
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            histogram[image[i*width + j]]++;
        }
    }
}

/*
https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
*/
void CalculateCDF(unsigned long* histogram, unsigned long* cdf){
    
    // clear cdf:
    for (int i=0; i<GRAYLEVELS; i++) {
        cdf[i] = 0;
    }
    
    // calculate cdf from histogram
    cdf[0] = histogram[0];
    for (int i=1; i<GRAYLEVELS; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
}


void Equalize(unsigned char * image_in, unsigned char * image_out, int width, int height, unsigned long* cdf){
     
    unsigned long imageSize = width * height;
    
    unsigned long cdfmin = findMin(cdf);
    
    //Equalize: namig: blok niti naj si CDF naloži v skupni pomnilnik
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            image_out[(i*width + j)] = Scale(cdf[image_in[i*width + j]], cdfmin, imageSize);
        }
    }
}

int main(int argc, char const *argv[]){

    if (argc < 2)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);
    char szImage_in_name_with_path[256]; // Adjust size as needed
    strcpy(szImage_in_name_with_path, "imgIn/");
    strcat(szImage_in_name_with_path, szImage_in_name);
    

    // Read image from file
    int width, height, cpp;
    // read only DESIRED_NCHANNELS channels from the input image:
    unsigned char *imageIn = stbi_load(szImage_in_name_with_path, &width, &height, &cpp, DESIRED_NCHANNELS);
    if(imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    // printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);
    //Allocate memory for raw output image data, histogram, and CDF 
	unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned long));
    unsigned long *histogram= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));
    unsigned long *CDF= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));

    // 1. izracunaj histogram:
    clock_t start = clock();
    CalculateHistogram(imageIn, width, height, histogram);
    clock_t end = clock();
    double histTime = ((double) (end - start)) / CLOCKS_PER_SEC;


    // 2. izracunaj CDF:
    start = clock();
    CalculateCDF(histogram, CDF);
    end = clock();
    double CDFTime = ((double) (end - start)) / CLOCKS_PER_SEC;

    // 3. izenači hisotgram:
    start = clock();
    Equalize(imageIn, imageOut, width, height, CDF);
    end = clock();
    double eqTime = ((double) (end - start)) / CLOCKS_PER_SEC;

    // stbi_write_jpg("oneColor-CPU.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);


    // printf("Beam       Hit_GPU       CDF_GPU\n");
    // for(int i = 0; i < GRAYLEVELS; i++){
    //     printf("%3d      %6d    %6d   \n", i, histogram[i],CDF[i]);
    // }

    printf("Hist: %f ms CDF: %f ms equalize: %f ms\n", histTime*1000,CDFTime*1000,eqTime*1000);

    //Free memory
	free(imageIn);
    free(imageOut);
    free(histogram);
    free(CDF);


    return 0;
}
//srun --reservation=psistemi --nodelist=wn162 ./hist

//srun --partition=gpu --gres=gpu:1 --reservation=psistemi --mem-per-cpu=8GB --time=00:00:50 --ntasks=5  Seminar city-neq.jpg a >>meritve/city-t.txt