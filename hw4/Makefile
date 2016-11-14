TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS= -g -O2 -Wall -DTRANSMUL
LDFLAGS= -I"/usr/include/CL" -lOpenCL 

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS) 

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --device gpu ./$(TARGET) 8912 -v
