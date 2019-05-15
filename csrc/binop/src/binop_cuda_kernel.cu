#include <stdint.h>
#include <stdio.h>
#include "binop_cuda_kernel.h"

int GET_BLOCKS(int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void binary_gemm_kernel(uint32_t *A, uint32_t *B, float *C, int m, int nn, int k, int transb, int alpha, int beta, float *alphas)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    int n = 1 + (nn - 1) / ENCODE_BITS;
    int startLocation = BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol;

    float *Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    __shared__ uint32_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint32_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    int Cvalue = 0;

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int lim = 1 + ((n - 1) / BLOCK_SIZE);
    for (int i = 0; i < lim; ++i)
    {

        // Get sub-matrix Asub of A
        uint32_t *Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B
        uint32_t *Bsub = transb ? &B[BLOCK_SIZE * blockCol * n + BLOCK_SIZE * i] : &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        if ((BLOCK_SIZE * i + col) < n && r < m)
            As[row][col] = Asub[row * n + col];
        else
            As[row][col] = 0;
        if ((BLOCK_SIZE * i + row) < n && c < k)
            Bs[row][col] = transb ? Bsub[row + col * n] : Bsub[row * k + col];
        else
            Bs[row][col] = 0;

        __syncthreads();
#pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j)
            Cvalue += __popc(As[row][j] ^ Bs[j][col]);
        __syncthreads();
    }

    if (col + blockCol * BLOCK_SIZE < k && row + blockRow * BLOCK_SIZE < m)
    {
        Csub[row * k + col] = beta ? Csub[row * k + col] : 0;
        Csub[row * k + col] += alpha ? (1.0 * nn - (Cvalue << 1)) * alphas[(startLocation + row * k + col) / k] : 1.0 * nn - (Cvalue << 1);
    }
}

__global__ void im2col_kernel(int n, float *data_im, int height, int width,
                              int ksize_h, int ksize_w, int pad_h, int pad_w,
                              int stride_h, int stride_w, int dilation_h, int dilation_w,
                              int height_col, int width_col, float *data_col)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        int w_out = index % width_col;
        index /= width_col;
        int h_out = index % height_col;
        int channel_in = index / height_col;
        int channel_out = channel_in * ksize_h * ksize_w;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;
        data_col += (channel_out * height_col + h_out) * width_col + w_out;
        data_im += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize_h; ++i)
        {
            for (int j = 0; j < ksize_w; ++j)
            {
                int h = h_in + i * dilation_h;
                int w = w_in + j * dilation_w;
                *data_col = (h >= 0 && w >= 0 && h < height && w < width) ? data_im[i * dilation_h * width + j * dilation_w] : 0;
                data_col += height_col * width_col;
            }
        }
    }
}

__forceinline__ __device__ uint32_t encode_val(float *array, int n)
{
    uint32_t r = 0;
    for (int i = 0; i < ENCODE_BITS && i < n; i++)
    {
        r |= (array[i] > 0) << i;
    }
    return r;
}

__global__ void encode_rows_kernel(float *input, uint32_t *output, int m, int n, int l)
{ // l = 1+(n-1)/ENCODE_BITS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = n * (i / l) + ENCODE_BITS * (i % l);
    if (i < m * l)
        output[i] = encode_val(&input[p], n - ENCODE_BITS * (i % l));
}

__global__ void encode_cols_kernel(float *a, uint32_t *b, int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int i32 = i * ENCODE_BITS;
    if (j < n && i32 < m)
    {
        uint32_t r = 0;
        for (int k = 0; j + n * (i32 + k) < m * n && k < ENCODE_BITS; k++)
        {
            r |= (a[j + n * (i32 + k)] > 0) << k;
        }
        b[j + n * i] = r;
    }
}

void binary_gemm_cuda(uint32_t *A, uint32_t *B, float *C, int m, int n, int k, int transb, int alpha, int beta, float *alphas, cudaStream_t stream)
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(k / BLOCK_SIZE + 1, m / BLOCK_SIZE + 1);
    binary_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, transb, alpha, beta, alphas);
}

void im2col_cuda(int n, float *data_im, int height, int width,
                 int ksize_h, int ksize_w, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w,
                 int height_col, int width_col, float *data_col, cudaStream_t stream)
{
    im2col_kernel<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>>(
        n, data_im, height, width, ksize_h, ksize_w,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w,
        height_col, width_col, data_col);
}

void encode_rows_cuda(float *input, uint32_t *output, int m, int n, int l, cudaStream_t stream)
{
    encode_rows_kernel<<<GET_BLOCKS(m * l), CUDA_NUM_THREADS, 0, stream>>>(input, output, m, n, l);
}

void encode_cols_cuda(float *input, uint32_t *output, int n, int k, cudaStream_t stream)
{
    dim3 blockDim(ENCODE_BITS, ENCODE_BITS, 1);
    dim3 gridDim(k / ENCODE_BITS + 1, n / ENCODE_BITS + 1, 1);

    encode_cols_kernel<<<gridDim, blockDim, 0, stream>>>(input, output, n, k);
}

__global__ void concatenate_kernel(float *self, float *src,
                                   int self_sz0, int self_sz1, int self_sz2, int self_sz3,
                                   int src_sz0, int src_sz1, int src_sz2, int src_sz3, int group)
{
    int T = self_sz0;
    int data_size = src_sz2 * src_sz3;

    int self_batch_size = self_sz1 * data_size;
    int self_skipped_channels_size = group * src_sz1 * data_size;

    int src_batch_size = src_sz1 * data_size;
    int n_channels = src_sz1;

    int t, c, p_t1, p_t2, p_c, i;

    for (t = 0; t < T; ++t)
    {
        p_t1 = t * self_batch_size + self_skipped_channels_size;
        p_t2 = t * src_batch_size;

        for (c = 0; c < n_channels; ++c)
        {
            p_c = c * data_size;

            for (i = 0; i < data_size; ++i)
            {
                self[p_t1 + p_c + i] = src[p_t2 + p_c + i];
            }
        }
    }
}

void concatenate_cuda(float *self, float *src,
                      int self_sz0, int self_sz1, int self_sz2, int self_sz3,
                      int src_sz0, int src_sz1, int src_sz2, int src_sz3, int group, cudaStream_t stream)
{
    dim3 blockDim(ENCODE_BITS, 1, 1);

    concatenate_kernel<<<blockDim, blockDim, 0, stream>>>(self, src,
                                                          self_sz0, self_sz1, self_sz2, self_sz3,
                                                          src_sz0, src_sz1, src_sz2, src_sz3, group);
}