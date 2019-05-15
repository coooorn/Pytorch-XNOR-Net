#include <THC/THC.h>
#include <stdint.h>
#include "binop_cuda_kernel.h"

extern THCState *state;

void binary_gemm(THCudaIntTensor *a, THCudaIntTensor *b, THCudaTensor *c, int m, int nn, int k, int transb, int alpha, int beta, THCudaTensor *alphas)
{
    if (c->nDimension != 2 || c->size[0] * c->size[1] < m * k)
    {
        THCudaTensor_resize2d(state, c, m, k);
    }
    uint32_t *A = (uint32_t *)THCudaIntTensor_data(state, a);
    uint32_t *B = (uint32_t *)THCudaIntTensor_data(state, b);
    float *C = THCudaTensor_data(state, c);
    float *D = alpha ? THCudaTensor_data(state, alphas) : NULL;
    cudaStream_t stream = THCState_getCurrentStream(state);

    binary_gemm_cuda(A, B, C, m, nn, k, transb, alpha, beta, D, stream);
}

void im2col(THCudaTensor *data_im, int channels,
            int height, int width,
            int ksize_h, int ksize_w, int pad_h,
            int pad_w, int stride_h, int stride_w,
            int dilation_h, int dilation_w, THCudaTensor *data_col)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;

    float *d_im = THCudaTensor_data(state, data_im);
    float *d_col = THCudaTensor_data(state, data_col);
    cudaStream_t stream = THCState_getCurrentStream(state);

    im2col_cuda(
        num_kernels, d_im, height, width, ksize_h, ksize_w,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w,
        height_col, width_col, d_col, stream);

    THCudaCheck(cudaGetLastError());
}

void encode_rows(THCudaTensor *input, THCudaIntTensor *output)
{
    //THCUNN_assertSameGPU(state, 2, input, output);

    int m = input->size[0];
    int n = input->size[1];
    int l = 1 + (n - 1) / ENCODE_BITS;

    THCudaIntTensor_resize2d(state, output, m, l);
    float *a = THCudaTensor_data(state, input);
    uint32_t *b = (uint32_t *)THCudaIntTensor_data(state, output);
    cudaStream_t stream = THCState_getCurrentStream(state);

    encode_rows_cuda(a, b, m, n, l, stream);
}

void encode_cols(THCudaTensor *input, THCudaIntTensor *output)
{
    //THCUNN_assertSameGPU(state, 2, input, output);

    int n = input->size[0];
    int k = input->size[1];
    int l = 1 + (n - 1) / ENCODE_BITS;

    THCudaIntTensor_resize2d(state, output, l, k);
    float *a = THCudaTensor_data(state, input);
    uint32_t *b = (uint32_t *)THCudaIntTensor_data(state, output);
    cudaStream_t stream = THCState_getCurrentStream(state);

    encode_cols_cuda(a, b, n, k, stream);
}

// Based on the torch SpatialConvolutionMM_updateOutput
void BinarySpatialConvolution_updateOutput(
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaIntTensor *weight,
    THCudaTensor *bias,
    THCudaTensor *alphas,
    int kH, int kW,
    int sH, int sW,
    int padH, int padW)
{

    //THCUNN_assertSameGPU(state, 5, input, output, weight, columns, columns_binary);

    // Params:
    int nInputPlane = input->size[1];
    int nOutputPlane = weight->size[0];

    input = THCudaTensor_newContiguous(state, input);
    int batch = 1;
    if (input->nDimension == 3)
    {
        // Force batch
        batch = 0;
        THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    }

    int64_t inputWidth = input->size[3];
    int64_t inputHeight = input->size[2];
    int64_t outputWidth = (inputWidth + 2 * padW - kW) / sW + 1;
    int64_t outputHeight = (inputHeight + 2 * padH - kH) / sH + 1;

    // Batch size + input planes
    int64_t batchSize = input->size[0];

    // Resize output
    THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

    // Resize temporary columns
    THCudaTensor *columns = THCudaTensor_newWithSize2d(state, nInputPlane * kW * kH, outputHeight * outputWidth);

    // Define a buffer of ones, for bias accumulation
    // Note: this buffer can be shared with other modules, it only ever gets increased,
    // and always contains ones.
    THCudaTensor *ones = THCudaTensor_newWithSize2d(state, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);

    THCudaIntTensor *columns_binary = THCudaIntTensor_newWithSize2d(state, weight->size[1], outputHeight * outputWidth);

    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt++)
    {
        // Matrix mulitply per output:
        THCudaTensor *input_n = THCudaTensor_newSelect(state, input, 0, elt);
        THCudaTensor *output_n = THCudaTensor_newSelect(state, output, 0, elt);

        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        int64_t m_ = nOutputPlane;
        int64_t n_ = outputHeight * outputWidth;
        int64_t k_ = 1;

        // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
        if (bias->nDimension)
        {
            THCudaBlas_Sgemm(
                state,
                't', 'n',
                n_, m_, k_,
                1,
                THCudaTensor_data(state, ones), k_,
                THCudaTensor_data(state, bias), k_,
                0,
                THCudaTensor_data(state, output_n), n_);
        }
        else
        {
            THCudaTensor_zero(state, output_n);
        }

        // Extract columns:
        im2col(input_n, nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, sH, sW, 1, 1, columns);

        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // row-major to column-major change
        int m = weight->size[0];
        //int n = weight->size[1];
        int k = columns->size[1];

        encode_cols(columns, columns_binary);
        binary_gemm(weight, columns_binary, output_n, m, nInputPlane * kW * kH, k, 0, 1, 1, alphas);

        THCudaTensor_free(state, input_n);
        THCudaTensor_free(state, output_n);
    }

    if (batch == 0)
    {
        THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
        THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    }

    // Free
    THCudaTensor_free(state, ones);
    THCudaTensor_free(state, input);
    THCudaTensor_free(state, columns);
    THCudaIntTensor_free(state, columns_binary);
}

void concat_group_gpu(THCudaTensor *self, THCudaTensor *src, int group)
{
    float *A = THCudaTensor_data(state, self);
    float *B = THCudaTensor_data(state, src);

    cudaStream_t stream = THCState_getCurrentStream(state);

    concatenate_cuda(A, B,
                     (int)self->size[0], (int)self->size[1], (int)self->size[2], (int)self->size[3],
                     (int)src->size[0], (int)src->size[1], (int)src->size[2], (int)src->size[3], group, stream);
}

void update_conv_output_gpu(
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaIntTensor *weight,
    THCudaTensor *bias,
    THCudaTensor *alphas,
    int kernel_height,
    int kernel_width,
    int stride_vertical,
    int stride_horizontal,
    int padding_rows,
    int pad_columns,
    int groups)
{
    if (groups == 1)
    {
        BinarySpatialConvolution_updateOutput(input, output, weight, bias, alphas,
                                              kernel_height, kernel_width, stride_vertical, stride_horizontal,
                                              padding_rows, pad_columns);
        return;
    }

    int g;
    uint64_t channels = input->size[1];
    uint64_t input_group_size = channels / groups;
    uint64_t weight_group_size = weight->size[0] / groups;

    uint64_t output_height = (input->size[2] + 2 * padding_rows - kernel_height) / stride_vertical + 1;
    uint64_t output_width = (input->size[3] + 2 * pad_columns - kernel_width) / stride_horizontal + 1;

    uint64_t T = input->size[0];

    THCudaTensor_resize4d(state, output, T, weight->size[0], output_height, output_width);

    for (g = 0; g < groups; ++g)
    {
        THCudaTensor *grouped_input = THCudaTensor_newNarrow(state, input, 1,
                                                             g * input_group_size, input_group_size);

        THCudaIntTensor *grouped_weight = THCudaIntTensor_newNarrow(state, weight, 0,
                                                                    g * weight_group_size, weight_group_size);

        THCudaTensor *grouped_output = THCudaTensor_new(state);

        BinarySpatialConvolution_updateOutput(grouped_input, grouped_output, grouped_weight, bias, alphas,
                                              kernel_height, kernel_width, stride_vertical, stride_horizontal,
                                              padding_rows, pad_columns);

        concat_group_gpu(output, grouped_output, g);

        THCudaTensor_free(state, grouped_input);
        THCudaIntTensor_free(state, grouped_weight);
        THCudaTensor_free(state, grouped_output);
    }
}