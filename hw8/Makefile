TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-mavx -g -O3 -Wall -DINTRINSIC
LDFLAGS=

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add ./$(TARGET) 4096 -v
