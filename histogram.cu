#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define MAX_BINS 4096
//Comparing the multi and single threaded histograms
bool compare(unsigned int *one, unsigned int *two, int size)
{
    for(int p = 0; p<size; p++)
    {
        if (one[p] != two[p])
        {
            return false;
        }
    }
    return true;
}
//Printing the datas if the required parameter is set to 1
void Data(unsigned int *data, unsigned int dataSize)
{
    printf("Data generated : ");
    for (int a = 0; a < dataSize; a++)
    {
        printf("%d", data[a]);
        if (a == dataSize - 1)
        {
            printf("]\n");
        }
        if (a != dataSize - 1)
        {
            printf("-");
        }
    }
}

__global__
static void histogram(unsigned int *input, unsigned int *histo, unsigned int dataSize, unsigned int binSize)
{
    int th = blockIdx.x * blockDim.x + threadIdx.x;
//Using extern shared and setting its size when calling the kernel
    extern __shared__ int local_histogram[];
//init histo
    for (int y = threadIdx.x; y < binSize; y += blockDim.x)
    {
        local_histogram[y] = 0;
    }
    __syncthreads();
//Filling the results on the histogram
    for (int i = th; i < dataSize; i += blockDim.x * gridDim.x)
    {
        //Atomic add is used as 2 datas can have the same number, to not have issues if they add 1 at the same time
        atomicAdd(&local_histogram[input[i]], 1);
    }
    __syncthreads();
        //add / blocks
    for (int z = threadIdx.x; z < binSize; z += blockDim.x)
    {
        atomicAdd(&histo[z], local_histogram[z]);
    }

}
//Printing the results
void result(unsigned int *res, int threadNb, unsigned int Size )
{
    printf("Result for %d threads: [", threadNb);
    for (int i = 0; i < Size; i++)
    {
        printf("%d", res[i]);
        if (i != Size - 1)
        {
            printf("|");
        }
    }
}

//Cleaning the histogram by putting 0's in it
__global__ static void cleanHisto(unsigned int *histo, unsigned int binSize)
{
    for (int i = threadIdx.x; i < binSize; i += blockDim.x)
    {
        histo[i] = 0;
    }
    __syncthreads();

}

void wrapper(unsigned int dataSize, unsigned int binSize, int display, int threadNb, int blockCount)
{
    unsigned int *histo = NULL;
    unsigned int *histo_single = NULL;
    unsigned int *d_histo = NULL;
    unsigned int *data = NULL;
    unsigned int *d_data = NULL;
    cudaEvent_t start;
    cudaEvent_t start_single;
    cudaEvent_t stop;
    cudaEvent_t stop_single;

    // Defining the structures
    data = (unsigned int *)malloc(dataSize * sizeof(unsigned int));
    histo = (unsigned int *)malloc(binSize * sizeof(unsigned int));
    histo_single = (unsigned int *)malloc(binSize * sizeof(unsigned int));

    // Generate data set on the host
    printf("Generation of data sets randomly.\n");
    srand(time(NULL));
    srand(time(NULL));
    for (int i = 0; i < dataSize; i++){
        data[i] = rand() % binSize;
    }
    printf("Done\n");

    // Print the input if it was asked while lauching the program
    if (display == 1)
    {
        Data(data, dataSize);
    }

    // Allocating memory
    checkCudaErrors(cudaMalloc((void **)&d_histo, sizeof(unsigned int) * binSize));
    checkCudaErrors(cudaMalloc((void **)&d_data, sizeof(unsigned int) * dataSize));

    // Copy the data to the device
    checkCudaErrors(cudaMemcpy(d_data, data, sizeof(unsigned int) * dataSize, cudaMemcpyHostToDevice));

    // Record the start event
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Launch the kernel
    histogram<<<blockCount, threadNb,sizeof(unsigned int) * binSize>>>(d_data, d_histo, dataSize, binSize);
    cudaDeviceSynchronize();

    // Fetch the result from device to host into histo
    printf("End of the kernel, fetching the results :\n");
    checkCudaErrors(cudaMemcpy(histo, d_histo, sizeof(unsigned int) * binSize, cudaMemcpyDeviceToHost));

    // Record the stop event and wait for the stop event to complete
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventCreate(&start_single));
    checkCudaErrors(cudaEventCreate(&stop_single));
    checkCudaErrors(cudaEventRecord(start_single, NULL));

    // Clean the first histogram as I re-use it afterwards
    cleanHisto<<<1, threadNb>>>(d_histo, binSize);
    cudaDeviceSynchronize();

    // Launch the kernel on a single thread
    histogram<<<1, 1,sizeof(unsigned int) * binSize>>>(d_data, d_histo, dataSize, binSize);
    cudaDeviceSynchronize();

    // Fetch the result of the last kernel onto the host
    checkCudaErrors(cudaMemcpy(histo_single, d_histo, sizeof(unsigned int) * binSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop_single, NULL));
    checkCudaErrors(cudaEventSynchronize(stop_single));
    float msecTotal = 0.0f;
    float msecTotal_single = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal_single, start_single, stop_single));
    double gigaFlops = (dataSize * 1.0e-9f) / (msecTotal / 1000.0f);
    double gigaFlops_single = (dataSize * 1.0e-9f) / (msecTotal_single / 1000.0f);

    // Print the histograms if the parameter
    if (display == 1)
    {
        result(histo, threadNb, binSize);
        result(histo_single, 1, binSize);
    }
    // Compare the results of the two histograms
    if (compare(histo, histo_single, binSize))
    {
        printf("All good ! Histograms match\n");
    }
    else
    {
        printf("Wrong ! Histograms don't match\n");
    }
    // Print performances
    printf("%d threads :\nCuda processing time = %.3fms, \n Perf = %.3f Gflops\n",threadNb, msecTotal, gigaFlops);
    printf("1 thread :\nCuda processing time = %.3fms, \n Perf = %.3f Gflops\n", msecTotal_single, gigaFlops_single);
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_histo));
    free(histo);
    free(histo_single);
    free(data);

}

int main(int argc, char **argv)
{
    int print = 0;
    unsigned int binSize = MAX_BINS;
    unsigned long long ds = 256;

    char *dataSize = NULL;
    cudaDeviceProp cudaprop;

    // retrieve device
    int dev = findCudaDevice(argc, (const char **)argv);
    cudaGetDeviceProperties(&cudaprop, dev);
    //Retrieving parameters
    if (checkCmdLineFlag(argc, (const char **)argv, "size"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "size", &dataSize);
        ds = atoll(dataSize);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "displayData"))
    {
        print = 1;
    }

    printf("Data Size is: %d \n", ds);
    //Max is 2^32 as asked
    if (ds >= 4294967296 || ds == 0) {
        printf("Error: Data size > 4,294,967,296");
        exit(EXIT_FAILURE);
    }
    //Defining the number of threads to follow the need (ds) with max value 256 (multiple of 32) and < 1024
    int nbThread = min((int)ds, 256);
    printf("nb thread: %d \n", nbThread);
    //Defining the number of blocks to follow the need (if ds = 500 only 2 blocks) with max value a multiple of 30
    int nbBlock =  min(((int)ds/256),18000);
    //if the data size is below 256 we still have to have atleast 1 block
    if (nbBlock == 0) nbBlock = 1;
    printf("nbblock: %d \n", nbBlock);
    wrapper(ds, binSize, print, nbThread, nbBlock);
    return EXIT_SUCCESS;
}
