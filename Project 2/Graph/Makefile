CC = g++
CFLAGS = -Wall -g3 -O3 -std=c++11 -I $(INC)
OUT = graph_search
UTILS_OBJS = ../Utils/utils.o ../Utils/knn.o ../Utils/h_function.o
EXTRA_OBJS = ../LSH/lsh.o
OBJS = main.o  gnns.o mrng.o
INC =  ../Utils/
INPUT_FILE = ../input.dat
QUERY_FILE = ../query.dat

.PHONY: test clean

test_GNNS: $(OUT)
	./$(OUT) -d $(INPUT_FILE) -q $(QUERY_FILE) -o ./gnns.out -k 50 -R 1 -E 30 -m 1

test_MRNG: $(OUT)
	./$(OUT) -d $(INPUT_FILE) -q $(QUERY_FILE) -o ./mrng.out -l 40 -m 2

$(OUT): $(OBJS) $(UTILS_OBJS) $(EXTRA_OBJS)
	$(CC) $(CFLAGS) -o $(OUT) $(OBJS) $(UTILS_OBJS) $(EXTRA_OBJS)

main.o: main.cc
	$(CC) $(CFLAGS) -c main.cc

mrng.o: mrng.cc mrng.h
	$(CC) $(CFLAGS) -c mrng.cc

gnns.o: gnns.cc gnns.h
	$(CC) $(CFLAGS) -c gnns.cc

../Utils/%.o:
	$(MAKE) -C ../Utils $*.o

../LSH/%.o:
	$(MAKE) -C ../LSH $*.o


clean:
	rm -f $(OBJS) $(OUT)
	$(MAKE) -C ../Utils clean