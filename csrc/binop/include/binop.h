void encode_rows_cpu(THFloatTensor *input, THIntTensor *output);

void encode_cols_cpu(THFloatTensor *input, THIntTensor *output);

void binary_gemm_cpu(
    THIntTensor *a,
    THIntTensor *b,
    THFloatTensor *c,
    int m,
    int nn,
    int k,
    int transb,
    int beta,
    int alpha,
    THFloatTensor *alphas);

void update_conv_output_cpu(
    THFloatTensor *input,
    THFloatTensor *output,
    THIntTensor *weight,
    THFloatTensor *bias,
    THFloatTensor *columns,
    THFloatTensor *alphas,
    int kernel_height,
    int kernel_width,
    int stride_vertical,
    int stride_horizontal,
    int padding_rows,
    int pad_columns,
    int groups);