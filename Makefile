all: clean multiTrain multiPred2 multiPred3 sparsify

#.PHONY: multiTrain

multiTrain:
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp
multiPred:
	g++ -fopenmp -std=c++11 -O3 -o multiPred multiPred.cpp
multiPred2:
	g++ -fopenmp -std=c++11 -O3 -o multiPred2 multiPred2.cpp
multiPred3:
	g++ -fopenmp -std=c++11 -O3 -o multiPred3 multiPred3.cpp
sparsify:
	g++ -fopenmp -std=c++11 -O3 -o sparsify sparsify.cpp
clean:
	rm -f multiTrain multiPred multiPred2 multiPred3 sparsify
