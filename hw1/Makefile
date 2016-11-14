all: float double fma nofma

float: float.c
	gcc -std=c99 -o float float.c

double: float.c
	gcc -std=c99 -o double float.c -DX64

fma: fma.c
	gcc -o fma fma.c -mfma -O3 -lrt -DUSINGD
	objdump -d fma > fma.obj

nofma: fma.c
	gcc -o nofma fma.c -mno-fma -O3 -lrt -DUISNGD
	objdump -d nofma > nofma.obj

clean:
	rm float double fma nofma fma.obj nofma.obj
