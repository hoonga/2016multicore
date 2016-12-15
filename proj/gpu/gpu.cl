#define ReLU(x) (((x)>0)?(x):0)
__kernel void small_convolution_layer(__global float * inputs, __global float * outputs, 
  __global float * filters, __global float * biases, int N, int D1, int D2,
  __local float * in)
{
// GLOBAL : D1*N*N
// LOCAL : N*N, in : N*N
    size_t d1 = get_group_id(0);
    size_t lid = get_local_id(0);
    int i, k, l;
    if (lid == 0)
    {
        // reading input
        int off = d1 * N * N;
        for (i = 0; i < N * N; i++)
        {
            in[i] = inputs[off + i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (i = 0; i < D2; i++)
    {
        float sum = 0;
        for (k = 0; k < 3; k++)
        {
            for (l = 0; l < 3; l++)
            {
                int x = lid / N + k - 1;
                int y = lid % N + l - 1;
                __local float fil;
                if (lid == 0)
                    fil = filters[(i * D1 + d1) * 9 + k * 3 + l];
                barrier(CLK_LOCAL_MEM_FENCE);
                if (x >= 0 && x < N && y >= 0 && y < N)
                    sum += in[x * N + y] * fil;
            }
        }
        __local float bias;
        if (lid == 0)
            bias = biases[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        outputs[N * N * i + lid] = ReLU(sum + bias);
    }
}
__kernel void big_convolution_layer(__global float * inputs, __global float * outputs,
  __global float * filters, __global float * biases, int N, int D1, int D2, 
  __local float * in)
{
    // GLOBAL : D1*N*N
    // LOCAL : N*16, in : N*16
    size_t d1 = get_group_id(0)/(N/16);
    size_t n = get_group_id(0)%(N/16);
    size_t lid = get_local_id(0);
    int j, k, l;
    if (lid == 0)
    {
         int off = d1 * N * N + N * 16 * n;
         for (j = 0; j < N*16; j++)
         {
             in[j] = inputs[off + j];
         }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j = 0; j < D2; j++)
    {
        float sum = 0;
        for (k = 0; k < 3; k++)
        {
            for (l = 0; l < 3; l++)
            {
                int x = lid / N + k - 1;
                int y = lid % N + l - 1;
                __local float fil;
                if (lid == 0)
                    fil = filters[(j * D1 + d1) * 9+ k * 3 + l];
                if (x >= 0 && x < 16 && y >= 0 && y < N)
                    sum += in[x * N + y] * fil;
            }
        }
        __local float bias;
        if (lid == 0)
            bias = biases[j];
        barrier(CLK_LOCAL_MEM_FENCE);
        outputs[j * N * N + n * N * 16 + lid] = ReLU(sum + bias);
    }
}
