__kernel void conv3x3(__global float * input, __global float * output,
        __global float * filter, int N, __local float * inp, __local float * fil) {
    size_t x = get_local_id(0);
    size_t X = get_global_id(0);
    size_t G = X * N + x;
    inp[x] = input[G];
    fil[x] = filter[x];
    barrier(CLK_LOCAL_MEM_FENCE);
}
