const char * kernel = \
"#define ReLU(x) (((x)>0)?(x):0)\n"
"__kernel void convolution_layer(__global float * inputs, __global float * outputs, "
"  __global float * filters, __global float * biases, int N, int D1, int D2,"
"  __local float * in, __local float * fil)"
"{"
"    size_t d2 = get_group_id(0);"
"    size_t X = get_group_id(1);"
"    size_t Y = get_group_id(2);"
"    size_t x = get_local_id(1);"
"    size_t y = get_local_id(2);"
"    int i, j, k;"
"    float sum = 0;"
"    for (i = 0; i < D1; i++)"
"    {"
"        for (j = 0; j < 3; j++)"
"        {"
"            for (k = 0; k < 3; k++)"
"            {"
"                int w = X * 14 + x + k - 1;"
"                int h = Y * 14 + y + j - 1;"
"                if (w >= 0 && w < N && h >= 0 && h < N)"
"                    sum += inputs[i*N*N + h*N + w]*filters[(D1*d2 + i)*9 + j*3 + k];"
"            }"
"        }"
"    }"
"    outputs[N*N*d2 + X*14 + x + (Y*14 + y)*N] = ReLU(sum + biases[d2]);"
"}";
