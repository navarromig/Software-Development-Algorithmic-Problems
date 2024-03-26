.PHONY: clean all

all:
	+$(MAKE) -C Cluster cluster
	+$(MAKE) -C LSH lsh
	+$(MAKE) -C Hypercube hypercube
	+$(MAKE) -C Graph graph_search


clean:
	+$(MAKE) -C Cluster clean
	+$(MAKE) -C LSH clean
	+$(MAKE) -C Hypercube clean
	+$(MAKE) -C Utils clean
	+$(MAKE) -C Graph clean