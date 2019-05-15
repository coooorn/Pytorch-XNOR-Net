#include <TH/TH.h>
#include <stdio.h>
#include <stdint.h>

#include "matmul.h"

inline uint32_t encode_val(float *array, int n)
{
    uint32_t sign, r = 0;
    for (int i = 0; i < ENCODE_BIT && i < n; i++)
    {
        sign = array[i] > 0;
        r |= (sign << i);
    }
    return r;
}

void encode_rows_cpu_kernel(float *columns, uint32_t *columns_binary, int m, int n)
{
    int i, l = 1 + (n - 1) / ENCODE_BIT;
    //#pragma omp parallel for
    for (i = 0; i < m * l; i++)
    {
        int p = n * (i / l) + ENCODE_BIT * (i % l);

        columns_binary[i] = encode_val(&columns[p], n - ENCODE_BIT * (i % l));
    }
}

void encode_cols_cpu_kernel(float *columns, uint32_t *columns_binary, int m, int n)
{
    int col_bin_m = 1 + (m - 1) / ENCODE_BIT;
    int i, j, k;
    //#pragma omp parallel for
    for (i = 0; i < col_bin_m; i++)
    {
        int i64 = i * ENCODE_BIT;
        for (j = 0; j < n && i64 < m; j++)
        {

            uint32_t sign, rvalue = 0;

            for (k = 0; j + n * (i64 + k) < m * n && k < ENCODE_BIT; k++)
            {
                sign = columns[j + n * (i64 + k)] > 0;
                rvalue |= (sign << k);
            }

            columns_binary[j + n * i] = rvalue;
        }
    }
}

void encode_rows_cpu(THFloatTensor *input, THIntTensor *output)
{
    int m = input->size[0];
    int n = input->size[1];
    int l = 1 + (n - 1) / ENCODE_BIT;

    THIntTensor_resize2d(output, m, l);
    float *a = THFloatTensor_data(input);
    uint32_t *b = (uint32_t *)THIntTensor_data(output);

    encode_rows_cpu_kernel(a, b, m, n);
}

void encode_cols_cpu(THFloatTensor *input, THIntTensor *output)
{
    int n = input->size[0];
    int k = input->size[1];
    int l = 1 + (n - 1) / ENCODE_BIT;

    THIntTensor_resize2d(output, l, k);
    float *a = THFloatTensor_data(input);
    uint32_t *b = (uint32_t *)THIntTensor_data(output);

    encode_cols_cpu_kernel(a, b, n, k);
}

void binary_gemm_cpu(THIntTensor *a, THIntTensor *b, THFloatTensor *c, int m, int nn,
                     int k, int transb, int beta, int alpha, THFloatTensor *alphas)
{
    if (c->nDimension != 2 || c->size[0] * c->size[1] < m * k)
    {
        THFloatTensor_resize2d(c, m, k);
    }
    uint32_t *A = (uint32_t *)THIntTensor_data(a);
    uint32_t *B = (uint32_t *)THIntTensor_data(b);
    float *C = THFloatTensor_data(c);
    float *D = THFloatTensor_data(alphas);
    int n = 1 + (nn - 1) / ENCODE_BIT, brow = transb ? 1 : k, bcol = transb ? n : 1;
    dgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, D);
}

void THNN_unfolded_copy(THFloatTensor *columns, THFloatTensor *input,
                        int kW, int kH, int dW, int dH, int padW, int padH,
                        int nInputPlane, int inputWidth, int inputHeight,
                        int outputWidth, int outputHeight)
{
    // This function assumes that
    // kH*kW does not overflow an int
    // nInputPlane*kH*kW does not overflow a int64_t
    // outputHeight*dH does not overflow a int64_t
    // outputWidth*dW does not overflow a int64_t

    int64_t k;
    float *input_data = THFloatTensor_data(input);
    float *columns_data = THFloatTensor_data(columns);

#pragma omp parallel for private(k)
    for (k = 0; k < (int64_t)nInputPlane * kH * kW; k++)
    {
        int64_t nip = k / (kH * kW);
        int64_t rest = k % (kH * kW);
        int64_t kh = rest / kW;
        int64_t kw = rest % kW;
        int x, y;
        int64_t ix, iy;
        float *dst = columns_data + nip * ((size_t)kH * kW * outputHeight * outputWidth) + kh * ((size_t)kW * outputHeight * outputWidth) + kw * ((size_t)outputHeight * outputWidth);
        float *src = input_data + nip * ((size_t)inputHeight * inputWidth);
        if (padW > 0 || padH > 0)
        {
            int64_t lpad, rpad;
            for (y = 0; y < outputHeight; y++)
            {
                iy = (int64_t)y * dH - padH + kh;
                if (iy < 0 || iy >= inputHeight)
                {
                    memset(dst + (size_t)y * outputWidth, 0, sizeof(float) * outputWidth);
                }
                else
                {
                    if (dW == 1)
                    {
                        ix = 0 - padW + kw;
                        lpad = fmaxf(0, padW - kw);
                        rpad = fmaxf(0, padW - (kW - kw - 1));
                        if (outputWidth - rpad - lpad <= 0)
                        {
                            memset(dst + (size_t)y * outputWidth, 0, sizeof(float) * outputWidth);
                        }
                        else
                        {
                            if (lpad > 0)
                                memset(dst + (size_t)y * outputWidth, 0, sizeof(float) * lpad);
                            memcpy(dst + (size_t)y * outputWidth + lpad, src + (size_t)iy * inputWidth + ix + lpad, sizeof(float) * (outputWidth - rpad - lpad));
                            if (rpad > 0)
                                memset(dst + (size_t)y * outputWidth + outputWidth - rpad, 0, sizeof(float) * rpad);
                        }
                    }
                    else
                    {
                        for (x = 0; x < outputWidth; x++)
                        {
                            ix = (int64_t)x * dW - padW + kw;
                            if (ix < 0 || ix >= inputWidth)
                                memset(dst + (size_t)y * outputWidth + x, 0, sizeof(float) * 1);
                            else
                                memcpy(dst + (size_t)y * outputWidth + x, src + (size_t)iy * inputWidth + ix, sizeof(float) * (1));
                        }
                    }
                }
            }
        }
        else
        {
            for (y = 0; y < outputHeight; y++)
            {
                iy = (int64_t)y * dH + kh;
                ix = 0 + kw;
                if (dW == 1)
                    memcpy(dst + (size_t)y * outputWidth, src + (size_t)iy * inputWidth + ix, sizeof(float) * outputWidth);
                else
                {
                    for (x = 0; x < outputWidth; x++)
                        memcpy(dst + (size_t)y * outputWidth + x, src + (size_t)iy * inputWidth + ix + (int64_t)x * dW, sizeof(float) * (1));
                }
            }
        }
    }
}

static void THNN_Bin_SpatialConvolutionMM_updateOutput_frame(
    THFloatTensor *output,
    THIntTensor *weight,
    THFloatTensor *bias,
    THFloatTensor *ones,
    THIntTensor *bin_col,
    THFloatTensor *alphas,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int64_t nInputPlane,
    int64_t inputWidth, int64_t inputHeight,
    int64_t nOutputPlane,
    int64_t outputWidth, int64_t outputHeight)
{
    THFloatTensor *output2d;

    output2d = THFloatTensor_newWithStorage2d(output->storage, output->storageOffset,
                                              nOutputPlane, -1, outputHeight * outputWidth, -1);
    THFloatTensor_zero(output2d);

    binary_gemm_cpu(weight, bin_col, output2d, nOutputPlane, kW * kH * nInputPlane,
                    outputHeight * outputWidth, 0, 1, 1, alphas);
    if (bias->nDimension)
    {
        THFloatTensor_addmm(output2d, 1, output2d, 1, bias, ones);
    }
    THFloatTensor_free(output2d);
}

void THNN_Bin_SpatialConvolutionMM_updateOutput(
    THFloatTensor *input,
    THFloatTensor *output,
    THIntTensor *weight,
    THFloatTensor *bias,
    THFloatTensor *alphas,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW)
{
    input = THFloatTensor_newContiguous(input);
    int ndim = input->nDimension;
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4)
    {
        dimf++;
        dimh++;
        dimw++;
    }

    int64_t nInputPlane = input->size[dimf];
    int64_t inputHeight = input->size[dimh];
    int64_t inputWidth = input->size[dimw];
    int64_t nOutputPlane = weight->size[0];
    int64_t outputHeight = (inputHeight + 2 * padH - kH) / dH + 1;
    int64_t outputWidth = (inputWidth + 2 * padW - kW) / dW + 1;

    if (bias->nDimension == 1)
    {
        THFloatTensor_resize2d(bias, bias->size[0], 1);
    }

    THFloatTensor *ones = THFloatTensor_newWithSize2d(1, outputHeight * outputWidth);
    THFloatTensor_fill(ones, 1);

    int64_t T = input->size[0];
    int64_t t;

    THFloatTensor_resize4d(output, T, nOutputPlane, outputHeight, outputWidth);

    THFloatTensor *columns = THFloatTensor_newWithSize3d(T, kW * kH * nInputPlane, outputHeight * outputWidth);
    THIntTensor *bin_col = THIntTensor_newWithSize3d(T, weight->size[0], outputHeight * outputWidth);

#pragma omp parallel for private(t)
    for (t = 0; t < T; t++)
    {
        THFloatTensor *input_t = THFloatTensor_newSelect(input, 0, t);
        THFloatTensor *columns_t = THFloatTensor_newSelect(columns, 0, t);
        THIntTensor *bin_col_t = THIntTensor_newSelect(bin_col, 0, t);

        THNN_unfolded_copy(
            columns_t, input_t, kW, kH, dW, dH, padW, padH,
            nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);
        encode_cols_cpu(columns_t, bin_col_t);

        THFloatTensor_free(input_t);
        THFloatTensor_free(columns_t);
        THIntTensor_free(bin_col_t);
    }

    for (t = 0; t < T; t++)
    {
        THFloatTensor *output_t = THFloatTensor_newSelect(output, 0, t);
        THIntTensor *bin_col_t = THIntTensor_newSelect(bin_col, 0, t);

        THNN_Bin_SpatialConvolutionMM_updateOutput_frame(
            output_t, weight, bias, ones, bin_col_t, alphas, kW, kH, dW, dH, padW, padH,
            nInputPlane, inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight);

        THFloatTensor_free(output_t);
        THIntTensor_free(bin_col_t);
    }
    THFloatTensor_free(input);
    THFloatTensor_free(ones);
    THIntTensor_free(bin_col);
    THFloatTensor_free(columns);
}

void concat_group_cpu(THFloatTensor *self, THFloatTensor *src, int group)
{
    uint64_t T = self->size[0];
    uint64_t data_size = src->size[2] * src->size[3];

    uint64_t self_batch_size = self->size[1] * data_size;
    uint16_t self_skipped_channels_size = group * src->size[1] * data_size;

    uint64_t src_batch_size = src->size[1] * data_size;
    uint64_t n_channels = src->size[1];

    float *A = self->storage->data;
    float *B = src->storage->data;

    for (uint64_t t = 0; t < T; ++t)
    {
        uint64_t p_t1 = t * self_batch_size + self_skipped_channels_size;
        uint64_t p_t2 = t * src_batch_size;

        for (uint64_t c = 0; c < n_channels; ++c)
        {
            uint64_t p_c = c * data_size;

            for (uint64_t i = 0; i < data_size; ++i)
            {
                A[p_t1 + p_c + i] = B[p_t2 + p_c + i];
            }
        }
    }
}

void update_conv_output_cpu(
    THFloatTensor *input,
    THFloatTensor *output,
    THIntTensor *weight,
    THFloatTensor *bias,
    THFloatTensor *alphas,
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
        THNN_Bin_SpatialConvolutionMM_updateOutput(input, output, weight, bias, alphas,
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

    THFloatTensor_resize4d(output, T, weight->size[0], output_height, output_width);

    for (g = 0; g < groups; ++g)
    {
        THFloatTensor *grouped_input = THFloatTensor_newNarrow(input, 1,
                                                               g * input_group_size, input_group_size);
        THIntTensor *grouped_weight = THIntTensor_newNarrow(weight, 0,
                                                            g * weight_group_size, weight_group_size);
        THFloatTensor *grouped_output = THFloatTensor_new();

        THNN_Bin_SpatialConvolutionMM_updateOutput(grouped_input, grouped_output, grouped_weight, bias, alphas,
                                                   kernel_height, kernel_width, stride_vertical, stride_horizontal,
                                                   padding_rows, pad_columns);

        concat_group_cpu(output, grouped_output, g);

        THFloatTensor_free(grouped_input);
        THIntTensor_free(grouped_weight);
        THFloatTensor_free(grouped_output);
    }
}
