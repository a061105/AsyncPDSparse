#include "../util.h"
#include "../multi.h"
#include "../ParallelPDSparse.h"
#include <mpi.h>

void exit_with_help(){
	cerr << "Usage: mpiexec -n [nodes] -f [hostfile] ./multiTrain (options) [train_data] (model)" << endl;	
	cerr << "options:" << endl;
	cerr << "-l lambda: L1 regularization weight (default 0.1)" << endl;
	cerr << "-c cost: cost of each sample (default 1.0)" << endl;
	cerr << "-t tau: degree of asynchronization (default 10)" << endl;
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

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 'l': param->lambda = atof(argv[i]);
				  break;
			case 'c': param->C = atof(argv[i]);
				  break;
			case 't': param->tau = atoi(argv[i]);
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

void compute_label_partition(vector<Labels>& labels, int N, int K, int mpi_rank, int mpi_num_proc,
			vector<int>& label_partition){
	
	//compute label frequency
	int* label_freq = new int[K];
	for(int k=0;k<K;k++)
		label_freq[k] = 0;
	for(int i=0;i<N;i++)
		for(Labels::iterator it=labels[i].begin(); it!=labels[i].end(); it++)
			label_freq[*it]++;
	
	int freq_sum = 0;
	for(int k=0;k<K;k++)
		freq_sum += label_freq[k];
	
	//create a random permutation of label index
	vector<int> label_index;
	label_index.resize(K);
	for(int k=0;k<K;k++)
		label_index[k] = k;
	//random_shuffle(label_index.begin(), label_index.end());

	//compute label partition
	int freq_per_proc = freq_sum/mpi_num_proc;
	int cumul = 0;
	int rank = 0;
	int r;
	for(r=0; r<label_index.size();r++){

		int k = label_index[r];
		cumul += label_freq[k];
		
		if( rank == mpi_rank )
			label_partition.push_back(k);

		if( cumul > freq_per_proc ){
			cumul = 0;
			rank++;
		}
	}

	delete[] label_freq;
}

int main(int argc, char** argv){
	
	MPI::Init(argc,argv);
	int mpi_rank = MPI::COMM_WORLD.Get_rank();
	int mpi_num_proc = MPI::COMM_WORLD.Get_size();
	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	
	Problem* train = new Problem();
	readData( param->trainFname, train);
	param->train = train;

	int D = train->D;
	int K = train->K;
	int N = train->data.size();
	if( mpi_rank==0 ){
		cerr << "N=" << N << endl;
		cerr << "d=" << (Float)nnz(train->data)/N << endl;
		cerr << "D=" << D << endl; 
		cerr << "K=" << K << endl;
	}
	
	vector<int> my_label_partition;
	compute_label_partition(train->labels, N, K, mpi_rank, mpi_num_proc, my_label_partition);
	cerr << "proc=" << mpi_rank << "/" << mpi_num_proc << ", |part|=" << my_label_partition.size() << endl;
	
	/*ParallelPDSparse* solver = new ParallelPDSparse(param);
	cerr << "#" ;
	for(int r=0;r<my_label_partition.size();r++){
		int k = my_label_partition[r];
		vector<int>& pos_sample = solver->pos_samples[k];
		
		//cerr << "class " << k <<" " <<"(|pos|="<<pos_sample.size()<<"):";
		for(vector<int>::iterator it=pos_sample.begin(); 
				it!=pos_sample.end(); it++)
			solver->y[*it] = 1.0;
		
		SparseVec w_k;
		solver->solve_one_class(pos_sample, w_k);
		
		for(vector<int>::iterator it=pos_sample.begin(); 
				it!=pos_sample.end(); it++)
			solver->y[*it] = -1.0;
	}
	cerr << "$";
	*/
	//StaticModel* model = new StaticModel(train);
	//model->writeModel(param->modelFname);
	
	MPI::Finalize();
}
