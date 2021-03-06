#define ReLU(x) (((x)>0)?(x):0)
__kernel void convolution_layer(__global float * inputs, __global float * outputs, 
  __global float * filters, __global float * biases, int N, int D1, int D2,
  __local float * in, __local float * fil)
{
    size_t d2 = get_group_id(0);
    size_t X = get_group_id(1);
    size_t Y = get_group_id(2);
    size_t x = get_local_id(0);
    size_t y = get_local_id(1);
    int i, j, k;
    float sum = 0;
    for (i = 0; i < D1; i++)
    {
        if (x == 0 && y == 0)
        {
            for (j = 0; j < 16; j++)
            {
                for (k = 0; k < 16; k++)
                {
                    int w = X * 14 + k - 1;
                    int h = Y * 14 + j - 1;
                    if (w >= 0 && w < N && h >= 0 && h < N)
                        in[j*16 + k] = inputs[i*N*N + N*h + w];
                    else
                        in[j*16 + k] = 0;
                }
            }
            for (j = 0; j < 9; j++)
            {
                fil[j] = filters[(d2 * D1 + i) * 9 +j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (j = 0; j < 3; j++)
        {
            for (k = 0; k < 3; k++)
            {
                int w = x + k;
                int h = y + j;
                sum += in[h*14 + w]*fil[j*3 + k];
            }
        }
    }
    __local float bias;
    if (x == 0 && y == 0)
        bias = biases[d2];
    barrier(CLK_LOCAL_MEM_FENCE);
    outputs[N*N*d2 + X*14 + x + Y*14*14 + y*14] = ReLU(sum + bias);
}
