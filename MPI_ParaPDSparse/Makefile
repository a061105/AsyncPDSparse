FLAG = -O3 -std=c++11 -fopenmp
predict = ../multiPred

all:
	mpic++.mpich ${FLAG} -o MPI_PDSparse MPI_PDSparse.cpp
sector: 
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.1 ../../data/sector.train
	${predict} ../../data/sector.test  model
aloi:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.01 ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.train
	${predict} ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.test model
LSHTC1:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.01 ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.train
	${predict} ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.test model
Dmoz:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.01 ../../ExtremeMulticlass/examples/Dmoz/Dmoz.train
	${predict} ../../ExtremeMulticlass/examples/Dmoz/Dmoz.test model
wiki325:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.05 ../../data/VarmasData/WikiLSHTC/wikiLSHTC_train model.wiki.2
	${predict} ../../data/VarmasData/WikiLSHTC/wikiLSHTC_test  model.wiki.2
test:
	mpic++.mpich ${FLAG} -o test test_send_recv.cpp
	mpiexec.mpich -n 4 -f hostfile4 ./test
	rm -f test
