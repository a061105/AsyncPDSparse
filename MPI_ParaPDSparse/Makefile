FLAG = -O3 -std=c++11 -fopenmp
predict = ../multiPred3

all:
	mpic++.mpich ${FLAG} -o MPI_PDSparse MPI_PDSparse.cpp
sector: 
	mpiexec.mpich -n 50 -f hostfile50 ./MPI_PDSparse -l 0.1 ../../data/sector.train
	${predict} ../../data/sector.test  model
aloi:
	mpiexec.mpich -n 50 -f hostfile50 ./MPI_PDSparse -l 0.01 ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.train
	${predict} ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.test model
LSHTC1:
	mpiexec.mpich -n 100_2 -f hostfile100 ./MPI_PDSparse -l 0.01 ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.train
	${predict} ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.test model
Dmoz:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -l 0.01 ../../ExtremeMulticlass/examples/Dmoz/Dmoz.train
	${predict} ../../ExtremeMulticlass/examples/Dmoz/Dmoz.test model
wiki325:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -c 1 -l 0.005 ../../data/VarmasData/WikiLSHTC/wikiLSHTC.tfidf.norm.train model.wiki.c1.l005
	${predict} ../../data/VarmasData/WikiLSHTC/wikiLSHTC.tfidf.norm.test  model.wiki.c1.l005
wiki_small:
	mpiexec.mpich -n 50 -f hostfile50 ./MPI_PDSparse -c 0.1 -l 0.001 ../../data/VarmasData/WikiLSHTC/wikiLSHTC_train.170k model
	${predict} ../../data/VarmasData/WikiLSHTC/wikiLSHTC_val  model
wiki31:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -c 1 -l 1e-2 ../../data/VarmasData/Wiki10/wiki10.tfidf.norm.train  model
	${predict} ../../data/VarmasData/Wiki10/wiki10.tfidf.norm.test model
bibtex:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -c 0.01 -l 0.0 ../../data/VarmasData/Bibtex/Bibtex.train.fs model
	${predict} ../../data/VarmasData/Bibtex/Bibtex.test.fs model
delicious:
	mpiexec.mpich -n 50 -f hostfile50 ./MPI_PDSparse -c 0.01 -l 0.01 ../../data/VarmasData/Delicious/Delicious.train.fs model
	${predict} ../../data/VarmasData/Delicious/Delicious.test.fs model
large_delicious:
	mpiexec.mpich -n 50 -f hostfile50 ./MPI_PDSparse -c 0.01 -l 0.01 ../../data/VarmasData/DeliciousLarge/deliciousLarge_train2.txt.fs model
	${predict} ../../data/VarmasData/DeliciousLarge/deliciousLarge_test2.txt.fs model
mediamill:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -c 0.1 -l 0.0 ../../data/VarmasData/Mediamill/Mediamill.train.fs model
	${predict} ../../data/VarmasData/Mediamill/Mediamill.test.fs model
eurlex:
	mpiexec.mpich -n 50 -f hostfile50 ./MPI_PDSparse -c 1 -l 0.1 ../../data/VarmasData/Eurlex/Eurlex.train.fs model
	${predict} ../../data/VarmasData/Eurlex/Eurlex.test.fs model
amazoncat:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -c 1 -l 0.01 ../../data/VarmasData/AmazonCat/amazonCat.tfidf.norm.train model
	${predict} ../../data/VarmasData/AmazonCat/amazonCat.tfidf.norm.test model
amazon670K:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -c 1 -l 0.1 ../../data/VarmasData/Amazon/amazon.tfidf.norm.train model
	${predict} ../../data/VarmasData/Amazon/amazon.tfidf.norm.test model
amazon3M:
	mpiexec.mpich -n 100 -f hostfile100 ./MPI_PDSparse -c 1 -l 0.1 ../../data/VarmasData/Amazon-3M/amazon-3M.tfidf.norm.train.fs model.amazon3M
	${predict} ../../data/VarmasData/Amazon-3M/amazon-3M.tfidf.norm.test.fs model.amazon3M 20
test:
	mpic++.mpich ${FLAG} -o test test_send_recv.cpp
	mpiexec.mpich -n 4 -f hostfile4 ./test
	rm -f test
