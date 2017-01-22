#include <iostream>
#include <list>
#include <vector>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

int K = 100;
int ROOT = 0;

int TAG=0;
int main(int argc, char** argv){
	
	MPI::Init(argc, argv);
	int mpi_rank = MPI::COMM_WORLD.Get_rank();
	int mpi_num_proc = MPI::COMM_WORLD.Get_size();
	
	vector<int> labels;
	if( mpi_rank ==  0 ){
		for(int i=0;i<K;i++)
			labels.push_back(i);
	}
	
	int batch_size = 5;
	if( mpi_rank == ROOT ){
		
		int r=0;
		int* send_buf = new int[batch_size];
		int req_rank;
		while( r < labels.size() ){
			for(int i=0;i<batch_size;i++)
				send_buf[i] = -1;
			for(int i=0;i<batch_size && r<labels.size();i++,r++)
				send_buf[i] = labels[r];
			MPI::COMM_WORLD.Recv( &req_rank, 1, MPI::INT, MPI::ANY_SOURCE, TAG);
			MPI::COMM_WORLD.Send( send_buf, batch_size, MPI::INT, req_rank, TAG);
		}
		
		for(int i=0;i<batch_size;i++)
			send_buf[i] = -1;
		for(int i=1;i<mpi_num_proc;i++){
			MPI::COMM_WORLD.Recv( &req_rank, 1, MPI::INT, MPI::ANY_SOURCE, TAG);
			MPI::COMM_WORLD.Send( send_buf, batch_size, MPI::INT, req_rank, TAG);
		}	
		
	}else{
		int* recv_buf = new int[batch_size];
		//MPI::Status stat;
		while(1){
			
			MPI::COMM_WORLD.Sendrecv(&mpi_rank,1,MPI::INT, ROOT, TAG, 
					recv_buf, batch_size, MPI::INT, ROOT, TAG);
			int i;
			for(i=0;i<batch_size && recv_buf[i]!=-1;i++)
				cerr << "rank=" << mpi_rank << ", solve=" << recv_buf[i] << endl;

			if( i==0 )
				break;
		}

	}
	cerr << mpi_rank << " done" << endl;
	MPI::COMM_WORLD.Barrier();
	
	MPI::Finalize();
}
