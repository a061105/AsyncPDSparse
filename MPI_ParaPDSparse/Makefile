FLAG = -w -O3 -std=c++11 -fopenmp
predict = ../multiPred3
nhosts = 11
nthreads = 10

all:
	mpic++.mpich ${FLAG} -o MPI_PDSparse MPI_PDSparse.cpp
sector: 
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts)-uvanimor ./MPI_PDSparse -l 0.1 -n $(nthreads) -b 10 ../../data/sector.train
	${predict} ../../data/sector.test  model 20
aloi:
	mpiexec.mpich  -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -l 0.01 -n $(nthreads) -b 5 ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.train model
	${predict} ../../ExtremeMulticlass/examples/aloi.bin/aloi.bin.test model_dir.model 20
LSHTC1:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.01 -n $(nthreads) -b 5 ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.train
	${predict} ../../ExtremeMulticlass/examples/LSHTC1/LSHTC1.test model_dir.model 20
Dmoz:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.01 -n $(nthreads) -b 3  ../../ExtremeMulticlass/examples/Dmoz/Dmoz.train
	${predict} ../../ExtremeMulticlass/examples/Dmoz/Dmoz.test model_dir.model 20
wiki325:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1 -n $(nthreads) -b 5 ../../data/VarmasData/WikiLSHTC/wikiLSHTC.tfidf.scale.train model
	${predict} ../../data/VarmasData/WikiLSHTC/wikiLSHTC.tfidf.scale.test.10k model_dir.model 20
wiki500:
	#mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.001 -n $(nthreads) -b 1 ../../data/VarmasData/WikiLSHTC500k/Wiki500k.tfidf.norm.train.fm  model.wiki500
	${predict} ../../data/VarmasData/WikiLSHTC500k/Wiki500k.tfidf.norm.test.10k  model_dir.model.wiki500  20
wiki_small:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.1 -l 0.001 -n $(nthreads)  ../../data/VarmasData/WikiLSHTC/wikiLSHTC_train.170k model
	${predict} ../../data/VarmasData/WikiLSHTC/wikiLSHTC_val  model 20
wiki31:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 1e-2 -n $(nthreads)  ../../data/VarmasData/Wiki10/wiki10.tfidf.norm.train  model
	${predict} ../../data/VarmasData/Wiki10/wiki10.tfidf.norm.test model 20
bibtex:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.01 -l 0.01 -n $(nthreads) -b 1 ../../data/VarmasData/Bibtex/Bibtex.train.fs model
	${predict} ../../data/VarmasData/Bibtex/Bibtex.test.fs model_dir.model 20
delicious:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 0.01 -l 0.01 -n $(nthreads)  ../../data/VarmasData/Delicious/Delicious.train.fs model
	${predict} ../../data/VarmasData/Delicious/Delicious.test.fs model 20
large_delicious:
	#mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.01  -n $(nthreads) -b 5 ../../data/VarmasData/DeliciousLarge/deliciousLarge.tfidf.norm.train model
	${predict} ../../data/VarmasData/DeliciousLarge/deliciousLarge.tfidf.norm.test model_dir.model 20
mediamill:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 1 -n $(nthreads) -b 1 ../../data/VarmasData/Mediamill/Mediamill.train.fs model
	${predict} ../../data/VarmasData/Mediamill/Mediamill.test.fs model_dir.model 20
eurlex:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1 -n $(nthreads) -b 1 ../../data/VarmasData/Eurlex/Eurlex.train.fs  model
	${predict} ../../data/VarmasData/Eurlex/Eurlex.test.fs model_dir.model 20
rcv1:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 1 -n $(nthreads) -b 1 ../../data/VarmasData/RCV1-x/rcv1x_train.txt.fs  model
	${predict} ../../data/VarmasData/RCV1-x/rcv1x_test.txt.fs  model_dir.model 20
amazoncat:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1 -n $(nthreads) -b 2 ../../data/VarmasData/AmazonCat/amazonCat.tfidf.scale.train  model.amazoncat
	${predict} ../../data/VarmasData/AmazonCat/amazonCat.tfidf.scale.test model_dir.model.amazoncat 20
amazon670K:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.01  -n $(nthreads) -b 50 ../../data/VarmasData/Amazon/amazon.tfidf.norm.train model.amazon
	${predict} ../../data/VarmasData/Amazon/amazon.tfidf.norm.test.10k model_dir.model.amazon 20
amazon3M:
	mpiexec.mpich -n $(nhosts) -f hostfile$(nhosts) ./MPI_PDSparse -c 1 -l 0.1  -n $(nthreads) -b 10000 ../../data/VarmasData/Amazon-3M/amazon-3M.tfidf.norm.train.fs model.amazon3M
	${predict} ../../data/VarmasData/Amazon-3M/amazon-3M.tfidf.norm.test.fs model.amazon3M 20
test:
	mpic++.mpich ${FLAG} -o test test.cpp
	#mpiexec.mpich -n 4 -f hostfile4 ./test
	./test
	rm -f test
