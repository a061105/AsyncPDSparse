FLAG = -O3 -std=c++11 -fopenmp
predict = ../multiPred3
nhosts = 50
nthreads = 1

all:
	mpic++.mpich ${FLAG} -o MPI_PDSparse MPI_PDSparse.cpp
sector: 
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -l 0.1 -n $(nthreads) -b 20 ../../data/sector.train
	${predict} ../../data/sector.test  model 20
aloi:
	mpiexec.mpich  -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -l 0.01 -n $(nthreads) -b 50 ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.train model
	${predict} ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.test model 20
LSHTC1:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -l 0.01 -n $(nthreads) -b 300 ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.train
	${predict} ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.test model 20
Dmoz:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -l 0.01 -n $(nthreads) -b 50  ../../ExtremeMulticlass/examples/Dmoz/Dmoz.train
	${predict} ../../ExtremeMulticlass/examples/Dmoz/Dmoz.test model 20
wiki325:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1 -n $(nthreads) -b 100 ../../data/VarmasData/WikiLSHTC/wikiLSHTC.tfidf.norm.train model.wiki.c1.l1
	${predict} ../../data/VarmasData/WikiLSHTC/wikiLSHTC.tfidf.norm.test  model.wiki.c1.l1 20
wiki_small:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.1 -l 0.001 -n $(nthreads)  ../../data/VarmasData/WikiLSHTC/wikiLSHTC_train.170k model
	${predict} ../../data/VarmasData/WikiLSHTC/wikiLSHTC_val  model 20
wiki31:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 1e-2 -n $(nthreads)  ../../data/VarmasData/Wiki10/wiki10.tfidf.norm.train  model
	${predict} ../../data/VarmasData/Wiki10/wiki10.tfidf.norm.test model 20
bibtex:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.01 -l 0.0 -n $(nthreads)  ../../data/VarmasData/Bibtex/Bibtex.train.fs model
	${predict} ../../data/VarmasData/Bibtex/Bibtex.test.fs model 20
delicious:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.01 -l 0.01 -n $(nthreads)  ../../data/VarmasData/Delicious/Delicious.train.fs model
	${predict} ../../data/VarmasData/Delicious/Delicious.test.fs model 20
large_delicious:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.01 -l 0.01  -n $(nthreads) ../../data/VarmasData/DeliciousLarge/deliciousLarge_train2.txt.fs model
	${predict} ../../data/VarmasData/DeliciousLarge/deliciousLarge_test2.txt.fs model 20
mediamill:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.1 -l 0.0 -n $(nthreads)  ../../data/VarmasData/Mediamill/Mediamill.train.fs model
	${predict} ../../data/VarmasData/Mediamill/Mediamill.test.fs model 20
eurlex:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1  -n $(nthreads) ../../data/VarmasData/Eurlex/Eurlex.train.fs model
	${predict} ../../data/VarmasData/Eurlex/Eurlex.test.fs model 20
amazoncat:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.01 -n $(nthreads)  ../../data/VarmasData/AmazonCat/amazonCat.tfidf.norm.train model
	${predict} ../../data/VarmasData/AmazonCat/amazonCat.tfidf.norm.test model 20
amazon670K:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1  -n $(nthreads) ../../data/VarmasData/Amazon/amazon.tfidf.norm.train model
	${predict} ../../data/VarmasData/Amazon/amazon.tfidf.norm.test model 20
amazon3M:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1  -n $(nthreads) -b 10000 ../../data/VarmasData/Amazon-3M/amazon-3M.tfidf.norm.train.fs model.amazon3M
	${predict} ../../data/VarmasData/Amazon-3M/amazon-3M.tfidf.norm.test.fs model.amazon3M 20
test:
	mpic++.mpich ${FLAG} -o test test_send_recv.cpp
	mpiexec.mpich -n 4 -f hostfile4 ./test
	rm -f test
