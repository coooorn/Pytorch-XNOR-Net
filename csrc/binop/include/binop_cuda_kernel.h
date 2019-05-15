#ifndef _BINCONV_CUDA_KERNEL
#define _BINCONV_CUDA_KERNEL

#define BLOCK_SIZE 16
#define BLOCK_DIM 16
#define CUDA_NUM_THREADS 1024
#define ENCODE_BITS 32

#ifdef __cplusplus
extern "C"
{
#endif

    void binary_gemm_cuda(uint32_t *A, uint32_t *B, float *C, int m, int nn, int k, int transb, int alpha, int beta, float *alphas, cudaStream_t stream);

    void im2col_cuda(int n, float *data_im, int height, int width,
                     int ksize_h, int ksize_w, int pad_h, int pad_w,
                     int stride_h, int stride_w, int dilation_h, int dilation_w,
                     int height_col, int width_col, float *data_col, cudaStream_t stream);

    void encode_rows_cuda(float *input, uint32_t *output, int m, int n, int l, cudaStream_t stream);
    void encode_cols_cuda(float *input, uint32_t *output, int n, int k, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
