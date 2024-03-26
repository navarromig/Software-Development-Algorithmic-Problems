CC = g++
CFLAGS = -Wall -g3 -O3 -std=c++11

%.o: %.cc %.h
	$(CC) $(CFLAGS) -c $<
 
.PHONY: clean
clean:
	rm -f *.o