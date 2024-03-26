CC = g++
CFLAGS = -Wall -g3 -O3 -std=c++11 -I $(INC)
OUT = cluster
UTILS_OBJS = ../Utils/utils.o ../Utils/knn.o ../Utils/h_function.o
OBJS = main.o kmeans.o
EXTRA_OBJS = ../LSH/lsh.o ../Hypercube/hypercube.o
INC =  ../Utils/
INPUT_FILE = ../input.dat

.PHONY: test clean Utils/%.o LSH/%.o Hypercube/%.o reduced_test

test: $(OUT)
	./$(OUT) -i $(INPUT_FILE) -c cluster.conf -o ./cluster.out -m Classic

reduced_test-%: $(OUT)
	./$(OUT) -i $(INPUT_FILE) -ri ../reduced_input_$*.dat -c cluster.conf -o ./reduced_cluster_$*.out -m Classic

reduced_test: reduced_test-10 reduced_test-50 reduced_test-75 reduced_test-100 reduced_test-150

test-%: $(OUT)
	./$(OUT) -i $(INPUT_FILE) -c cluster.conf -o ./cluster-$*.out -complete -m $*

$(OUT): $(UTILS_OBJS) $(OBJS) $(EXTRA_OBJS)
	$(CC) $(CFLAGS) -o $(OUT) $(OBJS) $(UTILS_OBJS) $(EXTRA_OBJS) 

main.o: main.cc
	$(CC) $(CFLAGS) -c main.cc

kmeans.o: kmeans.cc kmeans.h
	$(CC) $(CFLAGS) -c kmeans.cc


../Utils/%.o:
	$(MAKE) -C ../Utils $*.o

../LSH/%.o:
	$(MAKE) -C ../LSH $*.o

../Hypercube/%.o:
	$(MAKE) -C ../Hypercube $*.o

clean:
	rm -f $(OBJS) $(OUT)