__kernel void pooling_layer(__global float * inputs, __global float * outputs,
        int N)
{
    // local size == 16
    // global size == D!
    size_t i = get_global_id(0); // i of sequential code
    float * input = inputs + i * N * N * 4;
    float * output = outputs + i * N * N;
    pooling2x2(input, output, N);
}

void pooling2x2(float * input, float output, int N)
{
    int i, j;
    N = N/16;
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
