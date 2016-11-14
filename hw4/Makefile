TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS= -g -O2 -Wall -I"/usr/include/CL"
LDFLAGS= -lOpenCL

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --device gpu ./$(TARGET) 2048 -v
