all:
	mpic++.mpich -O3 -std=c++11 -fopenmp -o MPI_PDSparse MPI_PDSparse.cpp
sector:
	mpiexec.mpich -n 4 -f hostfile4 ./MPI_PDSparse -l 0.1 ../../data/sector.train
Dmoz:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.1 ../../data/ODP/Dmoz.train
