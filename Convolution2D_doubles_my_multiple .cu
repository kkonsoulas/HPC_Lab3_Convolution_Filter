/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>



unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define ACCURACY 	0.000000005 

 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}


//////////////////////////////////////////////////////////////////////////////////
//              CUDA CODE 
//////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
                       int imageW, int imageH, int filterR) {

  int k;
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
        }     

        d_Dst[y * imageW + x] = sum;
      }
   
        
}


__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter,
    			   int imageW, int imageH, int filterR) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int k;
 
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
        }   
 
        d_Dst[y * imageW + x] = sum;
      }
   
}



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU,
    *d_Input,
    *d_Buffer,
    *d_Filter,
    *d_OutputGPU;


    int imageW;
    int imageH;
    unsigned int i;
    double accuracy = 0.0;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...

  
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));

    //Check the mallocs
    if(h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL)
        return -1;
  

    
    cudaMalloc((void**) &d_Filter,FILTER_LENGTH*sizeof(double));
    cudaMalloc((void**) &d_OutputGPU,imageW * imageH *sizeof(double));
    cudaMalloc((void**) &d_Buffer,imageW * imageH *sizeof(double));
    cudaMalloc((void**) &d_Input,imageW * imageH *sizeof(double));

    
    //!Check also CUDA malloc!
    if(d_Filter == NULL || d_Input == NULL || d_Buffer == NULL || d_OutputGPU == NULL)
        return -1;


    
  
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    }

    clock_t start_t,stop_t;
    double total_t;
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    //timestamp CPU
    start_t = clock(); 
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    // timestamp CPU
    stop_t = clock();
    printf("CPU computation finished\n");
    //print the total CPU time.
    total_t = (double)(stop_t - start_t) / CLOCKS_PER_SEC;
    


    //Initialisation of the blocks.
    dim3 grid_dim;
    dim3 block_dim;
    block_dim.x = imageW >= 32 ? 32 : imageW;
    block_dim.y = imageH >= 32 ? 32 : imageW;
    grid_dim.x = imageW >= 32 ? imageW/32 : 1;
    grid_dim.y = imageH >= 32 ? imageH/32 : 1;


    cudaEvent_t start,stop;
    printf("GPU computation...\n");
    //timestamp GPU time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(double), cudaMemcpyHostToDevice);


    convolutionRowGPU<<<grid_dim, block_dim>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    convolutionColumnGPU<<<grid_dim, block_dim>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius); 

    cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //timestamp GPU time
    printf("GPU computation finished\n");
    //print total GPU time.
    printf("Elapsed time in CPU: %lf (s)\n",total_t);
    printf("Elapsed time in GPU: %lf (s)\n",milliseconds/1000);


    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    for(int i=0; i < imageH * imageW; i++){
      if( (double) ABS(h_OutputCPU[i] - h_OutputGPU[i]) > accuracy){
        accuracy = (double) ABS((h_OutputCPU[i] - h_OutputGPU[i]));
        if(accuracy > ACCURACY){
          cudaDeviceReset();
          fprintf(stderr,"Unexpected Diviation: %lf\nMax Permited Diviation: %lf\n",accuracy,ACCURACY);
          return(-1);  
        }
      }
    }
    fprintf(stderr,"Comparing results with accuracy %.12lf\n",accuracy);


    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputGPU);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
