#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define GRAY_LEVELS 256
#define DESIRED_NCHANNELS 1

#define BSX 32
#define BSY 32


__global__ void histogram_kernel(unsigned char* img, int width, int height, unsigned int* hist)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int lx = threadIdx.x; 
    int ly = threadIdx.y;

    __shared__ unsigned int hist_private[256];

    int tID=ly * blockDim.x + lx;
    
    if(tID<GRAY_LEVELS)
        hist_private[tID] = 0;
    
    __syncthreads();

    if (x < width && y < height)
    {
        atomicAdd(&hist_private[img[y * width + x]], 1);
    }
    __syncthreads();

    // kopiranje v skupni pomnilnik
    if(tID<GRAY_LEVELS)
        atomicAdd(&hist[tID], hist_private[tID]);
}

__global__ void CDF_kernel( unsigned int* hist, unsigned int* cdf){

    __shared__ unsigned int tmp[GRAY_LEVELS*2];
    int tid = threadIdx.x;

    int pout=0, pin=1;

    tmp[tid] = hist[tid];
    __syncthreads();
    
    for(int offset = 1; offset < GRAY_LEVELS; offset <<= 1){
        
        pout = 1 - pout;
        pin = 1 - pout;
        if(tid >= offset){
            tmp[pout*GRAY_LEVELS + tid] = tmp[pin*GRAY_LEVELS + tid]+ tmp[pin*GRAY_LEVELS + tid - offset];
        }else{
            tmp[pout*GRAY_LEVELS + tid] = tmp[pin*GRAY_LEVELS + tid];
        }

        __syncthreads();
    }

    cdf[tid] = tmp[pout*GRAY_LEVELS + tid];
}

__global__ void CDF_kernal_optimized(unsigned int* hist, unsigned int* cdf){
    __shared__ unsigned int tmp[GRAY_LEVELS+1];
    int tid = threadIdx.x;

    tmp[tid] = hist[tid];

    int offset = 1;
    __syncthreads();
    
    for(size_t d =GRAY_LEVELS/2 ;d > 0; d >>= 1){
    
        if(tid < d){
            tmp[offset*(2*tid+2)-1] += tmp[offset*(2*tid+1)-1];
        }
        offset <<=1;
        __syncthreads();
    }

    if(tid == 0){
        tmp[GRAY_LEVELS] = tmp[GRAY_LEVELS-1];
        tmp[GRAY_LEVELS-1] = 0;
    }
    __syncthreads();

    offset= GRAY_LEVELS/2;
    for(int d = 1; d <= GRAY_LEVELS/2; d <<= 1){

        if(tid < d){
            unsigned int t = tmp[offset*(2*tid+1)-1];
            tmp[offset*(2*tid+1)-1] = tmp[offset*(2*tid+2)-1];
            tmp[offset*(2*tid+2)-1] += t;
            
        }
        offset >>=1;
        __syncthreads();
        
    }
    cdf[tid] = tmp[tid+1];

}

__device__ unsigned int scale(unsigned int cdf, unsigned int cdf_min, unsigned int image_size)
{
    float scale = (float)(cdf - cdf_min) / (float)(image_size - cdf_min);
    scale = round(scale* float(GRAY_LEVELS - 1));
    return (unsigned int)scale;
}

// CUDA kernel to find the minimum non-zero value in an array
__global__ void equalize_kernel(unsigned char *img_in, unsigned char *img_out, int width, int height,int image_size, unsigned int *min, unsigned int *cdf) {

    int min_cdf=*min;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        img_out[y * width + x] = scale(cdf[img_in[y * width + x]], min_cdf, image_size);
    }
}

__global__ void FindMin(unsigned int *hist, unsigned int *min) {
    __shared__ unsigned int minShared;
    if (threadIdx.x == 0) minShared = GRAY_LEVELS - 1;
    __syncthreads();

    if (hist[threadIdx.x]) atomicMin(&minShared, threadIdx.x);
    __syncthreads();

    if (threadIdx.x == 0) *min = hist[minShared];
}

// coda na gostitelju
int main(int argc, char const *argv[])
{   

    if (argc < 3)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }
    char szImage_in_name[255];
    char szImage_out_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);
    char szImage_in_name_with_path[256]; // Adjust size as needed
    strcpy(szImage_in_name_with_path, "imgIn/");
    strcat(szImage_in_name_with_path, szImage_in_name);
    char szImage_out_name_with_path[256]; // Adjust size as needed
    strcpy(szImage_out_name_with_path, "imgOut/");
    strcat(szImage_out_name_with_path, szImage_out_name);
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Read image from file
    
    // read only DESIRED_NCHANNELS channels from the input image:
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name_with_path, &width, &height, &cpp, DESIRED_NCHANNELS);

    if(h_imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    // printf("Loaded image name: %s\n", szImage_in_name);
    // printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);
    cpp = DESIRED_NCHANNELS;
    // CPU memory allocation

    // rezerviraj prostor v pomnilniku za izhodno sliko:
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);
    unsigned int *h_histogram= (unsigned int *)malloc(GRAY_LEVELS * sizeof(unsigned int));
    unsigned int *h_N_histogram= (unsigned int *)malloc(GRAY_LEVELS * sizeof(unsigned int));
    
    unsigned int *h_N_CDF_op= (unsigned int *)malloc(GRAY_LEVELS * sizeof(unsigned int));
    unsigned int *h_CDF= (unsigned int *)malloc(GRAY_LEVELS * sizeof(unsigned int));
    unsigned int *h_CDF_op= (unsigned int *)malloc(GRAY_LEVELS * sizeof(unsigned int));
    unsigned int *h_CDF_min= (unsigned int *)malloc(sizeof(unsigned int));

    // GPU memory allocation

    // rezerviraj prostor v pomnilniku na grafični kartici
    unsigned char *d_imageIn, *d_imageOut;
    cudaMalloc((void **)&d_imageIn, datasize);
    cudaMalloc((void **)&d_imageOut, datasize);

    unsigned int *d_histogram, *d_CDF,*d_min,*d_N_histogram,*d_N_CDF,*d_CDF_OP;
    cudaMalloc((void **)&d_histogram, GRAY_LEVELS * sizeof(unsigned int));
    cudaMalloc((void **)&d_N_histogram, GRAY_LEVELS * sizeof(unsigned int));
    
    cudaMalloc((void **)&d_N_CDF, GRAY_LEVELS * sizeof(unsigned int));
    cudaMalloc((void **)&d_CDF_OP, GRAY_LEVELS * sizeof(unsigned int));
    cudaMalloc((void **)&d_CDF, GRAY_LEVELS * sizeof(unsigned int));
    cudaMalloc((void **)&d_min, sizeof(unsigned int));


    // kopiraj vhodno sliko v pomnilnik na grafični kartici
    cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice);

    //prepear for kernel launch  
    dim3 blockSize(BSX, BSY);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Record start event on the stream
    cudaEventRecord(start);

    // izracunaj histogram
    histogram_kernel<<<gridSize, blockSize>>>(d_imageIn, width, height, d_histogram);
    
    // Record stop event on the stream
    cudaEventRecord(stop);
    // Wait until the stop event completes
    cudaEventSynchronize(stop); 
    
    // Copy the histogram from the device to the host
    cudaMemcpy(h_histogram, d_histogram, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Calculate the elapsed time between two events
    float hist_mls = 0;
    cudaEventElapsedTime(&hist_mls, start, stop);

    // izracunaj CDF
    cudaEventRecord(start);
    CDF_kernel<<<1, GRAY_LEVELS>>>(d_histogram, d_CDF);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float CDF_mls = 0;
    cudaEventElapsedTime(&CDF_mls, start, stop);
    cudaMemcpy(h_CDF, d_CDF, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // izracunaj CDF optimized
    cudaEventRecord(start);
    CDF_kernal_optimized<<<1, GRAY_LEVELS>>>(d_histogram, d_CDF_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float CDF_mls_op = 0;
    cudaEventElapsedTime(&CDF_mls_op, start, stop);
    cudaMemcpy(h_CDF_op, d_CDF_OP, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // equalize the image
    cudaEventRecord(start);
    FindMin<<<1, GRAY_LEVELS>>>(d_histogram, d_min);
    equalize_kernel<<<gridSize, blockSize>>>(d_imageIn,d_imageIn, width, height,width*height,d_min, d_CDF);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float equalize_mls = 0;
    cudaEventElapsedTime(&equalize_mls, start, stop);

    // New histogram for equalized image
    // histogram_kernel<<<gridSize, blockSize>>>(d_imageIn, width, height, d_N_histogram);
    // cudaMemcpy(h_N_histogram, d_N_histogram, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // CDF_kernal_optimized<<<1, GRAY_LEVELS>>>(d_N_histogram, d_N_CDF);
    // cudaMemcpy(h_N_CDF_op, d_N_CDF, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  

    // cudaMemcpy(h_imageOut, d_imageIn, datasize, cudaMemcpyDeviceToHost);
    // stbi_write_jpg(szImage_out_name_with_path, width, height, cpp, h_imageOut, 100);
    // // Write image to file
    // printf("%5s %10s %10s %10s %10s %10s\n","Beam","Hist_GPU","CDF_GPU","O_CDF_GPU","New_Hist","New_CDF");
    // for(int i = 0; i < GRAY_LEVELS; i++){
    //     printf("%5d %10d %10d %10d %10d %10d\n", i, h_histogram[i],h_CDF[i], h_CDF_op[i],h_N_histogram[i],h_N_CDF_op[i]);
    // }

    // Get the current device ID
    int device;
    cudaGetDevice(&device);
    // Get the properties of the device
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    // Print the name of the device
    // printf("Running on GPU: %s\n", props.name);
    printf("Hist: %f ms CDF: %f ms CDF op: %f ms equalize: %f ms\n", hist_mls,CDF_mls,CDF_mls_op,equalize_mls);
    
    // free GPU memory
    cudaFree(d_imageIn);
    cudaFree(d_imageOut);
    cudaFree(d_histogram);
    cudaFree(d_CDF);
    cudaFree(d_min);
    cudaFree(d_N_histogram);
    cudaFree(d_N_CDF);
    cudaFree(d_CDF_OP);


    // free CPU memory
    free(h_CDF_op);
    free(h_N_CDF_op);
    free(h_CDF_min);
    free(h_imageIn);
    free(h_imageOut);
    free(h_histogram);
    free(h_CDF);
    free(h_N_histogram);

    return 0;
}
// nvcc -o Seminar Seminar.cu -lm

//srun --partition=gpu --gres=gpu:1 --reservation=psistemi --mem-per-cpu=8GB --time=00:00:30 --ntasks=1  Seminar perfectColor-s.jpg perfectColor-s-eq-GPU.jpg >meritve/perfectColor-s.txt 

//srun --partition=gpu --gres=gpu:1 --reservation=psistemi --mem-per-cpu=8GB --time=00:00:30 --ntasks=5  Seminar perfectColor-s.jpg perfectColor-s-eq-GPU.jpg >>meritve/perfectColor-s.txt 

//srun --partition=gpu --gres=gpu:1 --reservation=psistemi --mem-per-cpu=8GB --time=00:00:50 --ntasks=5  Seminar city-neq.jpg a >>meritve/city-t.txt