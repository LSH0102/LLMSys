#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum[1] = {0.0};
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum[0] += val.x + val.y + val.z + val.w;
  }

  float l_exp[1]={0.0};
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_exp[0] += val.x* val.x+ val.y *val.y+ val.z*val.z + val.w*val.w;
  }
  
  // Step 2
  blockReduce<ReduceType::kSum, 1>(l_sum);
  blockReduce<ReduceType::kSum, 1>(l_exp);
  __syncthreads();
  
  float mean=l_sum[0]/(hidden_size<<2);
  float std=sqrt(l_exp[0]/(hidden_size<<2)-mean*mean+LN_EPSILON);
  means[blockIdx.x]=mean;
  vars[blockIdx.x]=std;
  // Step 3
  
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale) ;
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  ln_res+=blockIdx.x * (hidden_size<<2);
   for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 scale = scale_f4[idx];
    float4 val=inp_f4[idx];
    float4 b=bias_f4[idx];
    ln_res[4*idx]= (val.x-mean)/std*scale.x+b.x;
    ln_res[4*idx+1]= (val.y-mean)/std*scale.y+b.y;
    ln_res[4*idx+2]= (val.z-mean)/std*scale.z+b.z;
    ln_res[4*idx+3]= (val.w-mean)/std*scale.w+b.w;
  }

  
  
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  
  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];


  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1                                                 //每个block只负责计算一个hidden_dim列的总和  
  int j=blockIdx.x;                                      //锁定第j列
  int i=0;
  int iters=rows/blockDim.y+1;   
  float u=0.0;
  float v=0.0;                                          //每个Idx.y需要遍历几行
  for(i=0;i<iters;i+=1)                  //遍历第row行第j列的全部元素 0\leq row \leq width-1, 用Idx.y加速计算
  {                                      //第idx.y个thread负责计算从第idx.y*iters到idx.y*(iters)+iters行的和
     int row_idx=threadIdx.y*iters+i;
     
     if(j+row_idx*width<rows*width && row_idx<rows)        
     {
        u+=out_grad[j+row_idx*width];  
        v+=out_grad[j+row_idx*width]*(inp[j+row_idx*width]-means[row_idx])/vars[row_idx];
     }
  }
  betta_buffer[threadIdx.y][threadIdx.x]=u;
  gamma_buffer[threadIdx.y][threadIdx.x]=v;
  __syncthreads();
  
  
  // Step 2
  
  // Step 3
  
  float b_val=betta_buffer[threadIdx.x][threadIdx.y];
  float g_val=gamma_buffer[threadIdx.x][threadIdx.y];
  for(i=g.size()/2;i>0;i/=2)
  {
    b_val+=g.shfl_down(b_val,i);
    g_val+=g.shfl_down(g_val,i);
  }

  // Step 4
  if(blockIdx.x<width && threadIdx.x==0)
  {
    betta_grad[blockIdx.x]=b_val;
    gamma_grad[blockIdx.x]=g_val; 
  }
  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  // Step 1

  float dxhat[4]={0.0}; 
  inp_grad+=blockIdx.x * (hidden_dim<<2);  
  const float4 *grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_dim;  
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);  
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {            //其实循环只有一节 因为blockDim.x=hidden_dim
    float4 gamma_val = gamma_f4[idx];
    float4 grad_val=grad_f4[idx];
    dxhat[0]=gamma_val.x*grad_val.x;
    dxhat[1]=gamma_val.y*grad_val.y;
    dxhat[2]=gamma_val.z*grad_val.z;
    dxhat[3]=gamma_val.w*grad_val.w;                                            //先更新一次inp_grad
    
    inp_grad[4*idx]=dxhat[0];
    inp_grad[4*idx+1]=dxhat[1];
    inp_grad[4*idx+2]=dxhat[2];
    inp_grad[4*idx+3]=dxhat[3];
  }


  // Step 2
   
   float xhat[4]={0.0};
   float mean=means[blockIdx.x];
   float std=vars[blockIdx.x];                         //vars本质上传入的就是std 所以不再需要开根号
   const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_dim;
   for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) 
   {            
    float4 inp_val=inp_f4[idx];
    xhat[0]=(inp_val.x-mean)/std;
    xhat[1]=(inp_val.y-mean)/std;
    xhat[2]=(inp_val.z-mean)/std;
    xhat[3]=(inp_val.w-mean)/std;
   }


  // Step 3
  float mul[4]={0.0};
  int i;
  for(i=0;i<4;i++)
  {
    mul[i]=xhat[i]*dxhat[i];
  }
  blockReduce<ReduceType::kSum, 4>(dxhat);
  blockReduce<ReduceType::kSum, 4>(mul);
  __syncthreads();
  // Step 4
  int m=hidden_dim<<2;                 //传入的hidden_dim是右移两次之后的结果 所以要移回去
  float t1=mul[0]+mul[1]+mul[2]+mul[3];
  float t2=dxhat[0]+dxhat[1]+dxhat[2]+dxhat[3];
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x)
  {
    inp_grad[4*idx]-=(t2+xhat[0]*t1)/m;
    inp_grad[4*idx+1]-=(t2+xhat[1]*t1)/m;
    inp_grad[4*idx+2]-=(t2+xhat[2]*t1)/m;
    inp_grad[4*idx+3]-=(t2+xhat[3]*t1)/m;
  }
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x)
  {
    inp_grad[4*idx]/=std;
    inp_grad[4*idx+1]/=std;
    inp_grad[4*idx+2]/=std;
    inp_grad[4*idx+3]/=std;
  }
  
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);
  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
