FLAG = -O3 -std=c++11 -fopenmp

all:
	mpic++.mpich ${FLAG} -o MPI_PDSparse MPI_PDSparse.cpp
sector:
	mpiexec.mpich -n 4 -f hostfile4 ./MPI_PDSparse -l 0.1 ../../data/sector.train
Dmoz:
	mpiexec.mpich -n 200 -f hostfile200 ./MPI_PDSparse -l 0.1 ../../data/ODP/Dmoz.train
LSHTC1:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.1 ../../data/LSHTC/LSHTC1/LSHTC1.ri.train
wiki325:
	mpiexec.mpich -n 200 -f hostfile200 ./MPI_PDSparse -l 0.1 ../../data/VarmasData/WikiLSHTC/wikiLSHTC_train model.wiki
test:
	mpic++.mpich ${FLAG} -o test test_send_recv.cpp
	mpiexec.mpich -n 4 -f hostfile4 ./test
	rm -f test
