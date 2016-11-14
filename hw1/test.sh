# !/bin/bash

len=(1000 5000 10000)
iter=(1000 5000)
filename="log.txt"

for l in "${len[@]}"; do
    for i in "${iter[@]}"; do
        echo "len $l iter $i" >> $filename
        gcc -o fma fma.c "-DLEN=$l" "-DITER=$i" "-mfma" "-O3"
        gcc -o nofma fma.c "-DLEN=$l" "-DITER=$i" "-mno-fma" "-O3"
	printf "fma\t" >> $filename
        for j in {0..100}; do
            ./fma >> $filename
            printf "\t" >> $filename
        done
        printf "\nnofma\t" >> $filename
        for j in {0..100}; do
            ./nofma >> $filename
            printf "\t" >> $filename
        done
	printf "\n" >> $filename
     done
done
