#include "../util.h"
#include "../multi.h"
#include "../ParallelPDSparse.h"
#include <mpi.h>
#include <omp.h>

void exit_with_help(){
	cerr << "Usage: mpiexec -n [nodes] -f [hostfile] ./multiTrain (options) [train_data] (model)" << endl;	
	cerr << "options:" << endl;
	cerr << "-l lambda: L1 regularization weight (default 0.1)" << endl;
	cerr << "-n num_thread: number of parallel threads per node (default 1)" << endl;
	cerr << "-c cost: cost of each sample (default 1.0)" << endl;
	cerr << "-b batch_size: number of classes assigned by a node (with n threads) at a time (default 100)" << endl;
	//cerr << "-r speed_up_rate: sample 1/r fraction of non-zero features to estimate gradient (default r = ceil(min( 5DK/(Clog(K)nnz(X)), nnz(X)/(5N) )) )" << endl;
	//cerr << "-q split_up_rate: divide all classes into q disjoint subsets (default 1)" << endl;
	cerr << "-m max_iter: maximum number of iterations allowed if -h not used (default 30)" << endl;
	//cerr << "-u uniform_sampling: use uniform sampling instead of importance sampling (default not)" << endl;
	cerr << "-g max_select: maximum number of dual variables selected during search (default: -1 (i.e. dynamically adjusted during iterations) )" << endl;
	//cerr << "-p post_train_iter: #iter of post-training without L1R (default auto)" << endl;
	cerr << "-h <file>: using accuracy on heldout file '<file>' to terminate iterations" << endl;
	cerr << "-k shrink_ratio: shrink update by this ratio in solver 2 (default 1.0)." << endl;
	//cerr << "-e early_terminate: how many iterations of non-increasing heldout accuracy required to earyly stop (default 3)" << endl;
	//cerr << "-d : dump model file when better heldout accuracy is achieved, model files will have name (model).<iter>" << endl;
	exit(0);
}

int batch_size = 10;

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 'l': param->lambda = atof(argv[i]);
								//param->lambda = 0.0;
				  break;
			case 'c': param->C = atof(argv[i]);
				  break;
			case 'n': param->num_threads = atoi(argv[i]);
				  break;
			case 'b': batch_size = atoi(argv[i]);
				  break;
			//case 'r': param->speed_up_rate = atoi(argv[i]);
			//	  break;
			//case 'q': param->split_up_rate = atoi(argv[i]);
			//	  break;
			case 'm': param->max_iter = atoi(argv[i]);
				  break;
			//case 'u': param->using_importance_sampling = false; --i;
			//	  break;
			case 'g': param->max_select = atoi(argv[i]);
				  break;
			//case 'p': param->post_solve_iter = atoi(argv[i]);
			//	  break;
			//case 'e': param->early_terminate = atoi(argv[i]);
			//	  break;
			case 'h': param->heldoutFname = argv[i];
				  break;
			case 'k': param->step_size_shrink = atof(argv[i]);
				  break;
			//case 'd': param->dump_model = true; --i;
			//	  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}
	
	if(i>=argc)
		exit_with_help();
	
	param->trainFname = argv[i];
	i++;

	if( i<argc )
		param->modelFname = argv[i];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

void label_freq_sort(vector<Labels>& labels, int N, int K, vector<pair<int,int> >& label_freq){
	
	//compute label frequency
	Float* freq = new Float[K];
	for(int k=0;k<K;k++)
		freq[k] = 0.0;
	for(int i=0;i<N;i++)
		for(Labels::iterator it=labels[i].begin(); it!=labels[i].end(); it++)
			freq[*it]+=1.0;
	
	//sort label index according to frequency
	vector<int> label_index;
	label_index.resize(K);
	for(int k=0;k<K;k++)
		label_index[k] = k;
	sort(label_index.begin(), label_index.end(), ScoreComp(freq));
	//random_shuffle(label_index.begin(), label_index.end());
	
	//put into SparseVec
	label_freq.resize(K);
	for(int i=0;i<K;i++){
		int k = label_index[i];
		label_freq[i].first = k;
		label_freq[i].second = (int)freq[k];
	}

	delete[] freq;
}

const int ROOT = 0;//cannto change to other number
const int TAG = 0;

int main(int argc, char** argv){
	
	MPI::Init_thread(argc,argv,MPI_THREAD_MULTIPLE);
	int mpi_rank = MPI::COMM_WORLD.Get_rank();
	int mpi_num_proc = MPI::COMM_WORLD.Get_size();
	int num_worker_node = mpi_num_proc-1;
	int worker_node_id = mpi_rank-1;//assuming rank=0 is the ROOT
	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);

	Problem* train = new Problem();
	readData( param->trainFname, train, true);
	param->train = train;
	
	int D = train->D;
	int K = train->K;
	int N = train->data.size();
	if( mpi_rank==0 ){
		cerr << "N=" << N << endl;
		cerr << "d=" << (Float)nnz(train->data)/N << endl;
		cerr << "D=" << D << endl; 
		cerr << "K=" << K << endl;
		ThreadModelWriter root_modelwriter(num_worker_node, param->num_threads, param);
		root_modelwriter.writeMeta();
	}

	Float overall_train_time = 0.0;
	if( mpi_rank == ROOT ){
		/////coordinator code

		overall_train_time = -omp_get_wtime();
		
		vector<pair<int,int> > label_freq;
		label_freq_sort( train->labels, N, K, label_freq );
		
		//int r=309790; //debug from 293600
		int r=0; //debug from 293600
		int* send_buf = new int[batch_size];
		int req_rank;
		while( r < label_freq.size() ){
			for(int i=0;i<batch_size;i++)
				send_buf[i] = -1;

			/*int freq_sum = 0;
			for(int i=0; freq_sum<batch_size && r<label_freq.size();i++,r++){
					send_buf[i] = label_freq[r].first;
					freq_sum += label_freq[r].second;
			}*/
			for(int i=0;i<batch_size && r< label_freq.size(); i++,r++)
					send_buf[i] = label_freq[r].first;
			
			MPI::COMM_WORLD.Recv( &req_rank, 1, MPI::INT, MPI::ANY_SOURCE, TAG);
			MPI::COMM_WORLD.Send( send_buf, batch_size, MPI::INT, req_rank, TAG);
			
			cerr << "#labels_solved=" << r << endl;
		}
		
		for(int i=0;i<batch_size;i++)
			send_buf[i] = -1;

		int num_worker = (mpi_num_proc-1)*(param->num_threads);
		for(int i=0;i<num_worker;i++){
			MPI::COMM_WORLD.Recv( &req_rank, 1, MPI::INT, MPI::ANY_SOURCE, TAG);
			MPI::COMM_WORLD.Send( send_buf, batch_size, MPI::INT, req_rank, TAG);
		}

		overall_train_time += omp_get_wtime();
	}else{ 
		
		////worker code
		int nthreads= param->num_threads;
		omp_set_num_threads(nthreads);
		
		ParallelPDSparse* solver = new ParallelPDSparse(param);
		
		////for multi thread
		vector<int>* labels_assigned_arr = new vector<int>[nthreads];
		vector<SparseVec>* wk_arr_arr = new vector<SparseVec>[nthreads];
		int** recv_buf_arr = new int*[nthreads];
		for(int i=0;i<nthreads;i++)
						recv_buf_arr[i] = new int[batch_size];
		vector<int>* label_batch_arr = new vector<int>[nthreads];
		
		int* offset_arr = new int[nthreads];
		for(int i=0;i<nthreads;i++)
						offset_arr[i] = 0;

		vector<ThreadModelWriter*> modelwriter_vec;
		modelwriter_vec.resize(nthreads);
		for(int i=0;i<nthreads;i++)
						modelwriter_vec[i] = new ThreadModelWriter(num_worker_node, nthreads, worker_node_id, i, param);
		/////////////////////
		
		#pragma omp parallel
		while(1){
			
			int t = omp_get_thread_num();
			
			vector<int>& labels_assigned = labels_assigned_arr[t];
			vector<SparseVec>& wk_arr = wk_arr_arr[t];
			int* recv_buf = recv_buf_arr[t];
			vector<int>& label_batch = label_batch_arr[t];
			ThreadModelWriter* modelwriter = modelwriter_vec[t];
			
			MPI::COMM_WORLD.Sendrecv(&mpi_rank,1,MPI::INT, ROOT, TAG,
					recv_buf, batch_size, MPI::INT, ROOT, TAG);
			
			//cerr << "r=" << mpi_rank << ", t=" << t << ", get new labels" << endl;
			int i;
			label_batch.clear();
			for(i=0;i<batch_size && recv_buf[i]!=-1;i++){
				int k = recv_buf[i];
				label_batch.push_back(k);
				labels_assigned.push_back(k);
				wk_arr.push_back(SparseVec());
			}
			
			if( i==0 ){ //not receiving any new label==> terminate
					
					modelwriter->close();
					delete modelwriter;
					
					cerr << "rank=" << mpi_rank << ", thread=" << t << ", done." << endl;
					break;
			}

			
			for(int r=0;r<label_batch.size();r++){
				int k = label_batch[r];
				vector<int>& pos_sample = solver->pos_samples[k];
				//solve class k
				//cerr << "r=" << mpi_rank << ", t=" << t << ", solve..." << endl;
				solver->solve_one_class(k, pos_sample, wk_arr[offset_arr[t]+r]);
				//cerr << "r=" << mpi_rank << ", t=" << t << ", solved" << endl;
				//write model
				modelwriter->writeVec(k, wk_arr[offset_arr[t]+r]);
				//cerr << "r=" << mpi_rank << ", t=" << t <<", model written" << endl;
			}

			offset_arr[t] += label_batch.size();
		}
	}
	MPI::COMM_WORLD.Barrier();

	
	if( mpi_rank==0 ){
		
		cerr << "overall train time=" << overall_train_time << endl;
		//cerr << "writing model..." << endl;
	}
	
	MPI::Finalize();
}
