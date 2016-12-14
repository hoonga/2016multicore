void pooling2x2(__global float * input, __global float * output, int N)
{
    int i, j;
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            float m = 0 > input[i*4*N+j*2] ? 0 : input[i*4*N+j*2];
            m = m > input[i*4*4+j*2+1]? m : input[i*4*N+j*2+1];
            m = m > input[(i*2+1)*2*N+j*2] ? m : input[(i*2+1)*2*N+j*2];
            m = m > input[(i*2+1)*2*N+j*2+1] ? m : input[(i*2+1)*2*N+j*2+1];
            output[i*N+j] = m;
        }
    }
}
__kernel void pooling_layer(__global float * inputs, __global float * outputs, int N)
{
    size_t i = get_global_id(0);
    __global float * input = inputs + i * N * N * 4;
    __global float * output = outputs + i * N * N;
    pooling2x2(input, output, N);
}

void convolution3x3(__global float * input, __global float * output,
        __global float * filter, int N)
{
    int i, j, k, l;
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            float sum = 0;
            for(k = 0; k < 3; k++)
            {
                for(l = 0; l < 3; l++)
                {
                    int y = i + k - 1;
                    int x = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += input[y * N + x] * filter[k * 3 + l];
                }
            }
            output[i * N + j] += sum;
        }
    }
}

#define ReLU(x) (((x)>0)?(x):0)
__kernel void convolution_layer(__global float * inputs, __global float * outputs,
        __global float * filters, __global float * biases, int N, int D1)
{
    size_t j = get_global_id(0);
    int i;
    for(i = 0; i < D1; i++) {
        __global float * input = inputs + N * N * i;
        __global float * output = outputs + N * N * j;
        __global float * filter = filters + 3 * 3 * (j * D1 + i);
        convolution3x3(input, output, filter, N);
    }
    __global float * output = outputs + N * N * j;
    float bias = biases[j];
    for(i = 0; i < N * N; i++)
        output[i] = ReLU(output[i] + bias);
}
