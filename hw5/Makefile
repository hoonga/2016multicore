TARGET=GPU1 GPU2 GPU4 CPU CPUGPU1 CPUGPU4
OBJS=timers.o
CC=gcc
CFLAGS=-g -O2 -Wall -std=c99
LDFLAGS=-I"/usr/include/CL" -lOpenCL

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(CFLAGS) -DFLAG=$@ mat_mul.c $(OBJS) -o $@ $(LDFLAGS) 

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add ./$(TARGET) 8912 -v
