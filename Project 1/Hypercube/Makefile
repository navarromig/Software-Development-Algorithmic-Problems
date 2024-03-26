CC = g++
CFLAGS = -Wall -g3 -O3 -std=c++11 -I $(INC)
OUT = hypercube
UTILS_OBJS = ../Utils/utils.o ../Utils/knn.o ../Utils/h_function.o
OBJS = main.o hypercube.o
INC =  ../Utils/
INPUT_FILE = ../input.dat
QUERY_FILE = ../query.dat

.PHONY: test clean

test: $(OUT)
	./$(OUT) -d $(INPUT_FILE) -q $(QUERY_FILE) -o ./hypercube.out -M 6000 -N 1 -k 18 -probes 1

$(OUT): $(OBJS) $(UTILS_OBJS)
	$(CC) -o $(OUT) $(OBJS) $(UTILS_OBJS)

main.o: main.cc
	$(CC) $(CFLAGS) -c main.cc

hypercube.o: hypercube.cc hypercube.h
	$(CC) $(CFLAGS) -c hypercube.cc

../Utils/%.o:
	$(MAKE) -C ../Utils $*.o

clean:
	rm -f $(OBJS) $(OUT)