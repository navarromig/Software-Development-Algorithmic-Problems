.PHONY: clean all

all:
	+$(MAKE) -C Cluster cluster
	+$(MAKE) -C LSH lsh
	+$(MAKE) -C Hypercube hypercube

test:
	+$(MAKE) -C Cluster test-Classic
	+$(MAKE) -C Cluster test-LSH
	+$(MAKE) -C Cluster test-Hypercube
	+$(MAKE) -C LSH test
	+$(MAKE) -C Hypercube test

clean:
	+$(MAKE) -C Cluster clean
	+$(MAKE) -C LSH clean
	+$(MAKE) -C Hypercube clean
	+$(MAKE) -C Utils clean