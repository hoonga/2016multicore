TARGET=GPU1 GPU2 GPU4 CPU CPUGPU1 CPUGPU4
OBJS=timers.o
CC=gcc
CFLAGS_=-g -O2 -Wall
LDFLAGS=-I"/usr/include/CL" -lOpenCL

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) -DFLAG=$@ mat_mul.c $(OBJS) -o $@ 

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add ./$(TARGET) 8912 -v
