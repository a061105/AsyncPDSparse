all: clean multiTrain

#.PHONY: multiTrain

multiTrain:
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp
clean:
	rm -f multiTrain
