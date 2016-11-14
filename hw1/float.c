#include <stdio.h>
#include <inttypes.h>

#ifdef X64
typedef double float_t;
typedef int64_t int_t;
#define FORMAT "%lf"
#define SIZE 64
#else
typedef float float_t;
typedef int32_t int_t;
#define FORMAT "%f"
#define SIZE 32
#endif

union f2i {
    float_t f;
    int_t i;
};

int main() {
    union f2i f2i;
    float_t f;
    char binary[SIZE+1];

    scanf(FORMAT, &f);
    
    f2i.f = f;
    for (int i = SIZE-1; i >= 0; i--) {
        binary[i] = (f2i.i & 1) + '0';
        f2i.i = f2i.i >> 1;
    }
    binary[SIZE] = '\0';

    printf("%s""\n", binary);
    printf(FORMAT"\n", f);
    return 0;
}
