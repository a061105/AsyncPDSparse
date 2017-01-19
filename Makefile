all: clean multiTrain multiPred

#.PHONY: multiTrain

multiTrain:
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp
multiPred:
	g++ -fopenmp -std=c++11 -O3 -o multiPred multiPred.cpp
clean:
	rm -f multiTrain multiPred
