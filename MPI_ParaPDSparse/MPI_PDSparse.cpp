#include "../util.h"
#include "../multi.h"
#include "../ParallelPDSparse.h"
#include <mpi.h>

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

int batch_size = 100;

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
	double* freq = new double[K];
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
	//sort(label_index.begin(), label_index.end(), ScoreComp(freq));
	random_shuffle(label_index.begin(), label_index.end());
	
	//put into SparseVec
	label_freq.resize(K);
	for(int i=0;i<K;i++){
		int k = label_index[i];
		label_freq[i].first = k;
		label_freq[i].second = (int)freq[k];
	}

	delete[] freq;
}

const int ROOT = 0;
const int TAG = 0;

int main(int argc, char** argv){
	
	MPI::Init_thread(argc,argv,MPI_THREAD_FUNNELED);
	int mpi_rank = MPI::COMM_WORLD.Get_rank();
	int mpi_num_proc = MPI::COMM_WORLD.Get_size();
	
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
	}

	vector<int> labels_assigned;
	vector<SparseVec> wk_arr;
	double overall_train_time = 0.0;
	if( mpi_rank == ROOT ){
		/////coordinator code

		overall_train_time = -omp_get_wtime();
		
		vector<pair<int,int> > label_freq;
		label_freq_sort( train->labels, N, K, label_freq );
		
		int r=0;
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
		for(int i=1;i<mpi_num_proc;i++){
			MPI::COMM_WORLD.Recv( &req_rank, 1, MPI::INT, MPI::ANY_SOURCE, TAG);
			MPI::COMM_WORLD.Send( send_buf, batch_size, MPI::INT, req_rank, TAG);
		}

		overall_train_time += omp_get_wtime();
	}else{ 
		
		////worker code
		omp_set_num_threads(param->num_threads);

		ParallelPDSparse* solver = new ParallelPDSparse(param);
		int* recv_buf = new int[batch_size];
		vector<int> label_batch;
		int offset = 0;
		while(1){
			
			MPI::COMM_WORLD.Sendrecv(&mpi_rank,1,MPI::INT, ROOT, TAG,
					recv_buf, batch_size, MPI::INT, ROOT, TAG);
			int i;
			label_batch.clear();
			for(i=0;i<batch_size && recv_buf[i]!=-1;i++){
				int k = recv_buf[i];
				label_batch.push_back(k);
				labels_assigned.push_back(k);
				wk_arr.push_back(SparseVec());
			}

			if( i==0 ) //not receiving any new label==> terminate
				break;
			
			#pragma omp parallel for
			for(int r=0;r<label_batch.size();r++){
				int k = label_batch[r];
				vector<int>& pos_sample = solver->pos_samples[k];
				solver->solve_one_class(k, pos_sample, wk_arr[offset+r]);
			}
			offset += label_batch.size();
		}
		cerr << "rank=" << mpi_rank << " done." << endl;
	}
	
	
	//Gather wk, k=1...K, from processes
	MPI::COMM_WORLD.Barrier();
	if( mpi_rank==0 ){
		
		cerr << "overall train time=" << overall_train_time << endl;
		cerr << "writing model..." << endl;
	}
	
	int nnz_wkj = 0;
	for(int r=0;r<labels_assigned.size();r++)
		nnz_wkj += wk_arr[r].size();
	int* nnz_wkj_arr = NULL;
	if( mpi_rank==ROOT )
		nnz_wkj_arr = new int[mpi_num_proc];
	
	MPI::COMM_WORLD.Gather(&nnz_wkj,1,MPI::INT, nnz_wkj_arr,1, MPI::INT, ROOT);
	
	int* disp = NULL;
	int total_nnz = 0;
	if( mpi_rank==ROOT){
		disp = new int[mpi_num_proc];
		size_to_displacement(nnz_wkj_arr, mpi_num_proc, disp);
		total_nnz = disp[mpi_num_proc-1] + nnz_wkj_arr[mpi_num_proc-1];
	}
	
	int* k_arr = new int[nnz_wkj];
	int* j_arr = new int[nnz_wkj];
	Float* v_arr = new Float[nnz_wkj];
	int count=0;
	for(int r=0;r<labels_assigned.size();r++){
		int k = labels_assigned[r];
		for(SparseVec::iterator it=wk_arr[r].begin(); it!=wk_arr[r].end(); it++){
			int j = it->first;
			Float val = it->second;
			k_arr[count] = k;
			j_arr[count] = j;
			v_arr[count] = val;
			count++;
		}
	}
	
	int* k_merge = NULL;
	int* j_merge = NULL;
	Float* v_merge = NULL;
	if( mpi_rank == ROOT ){
		k_merge = new int[total_nnz];
		j_merge = new int[total_nnz];
		v_merge = new Float[total_nnz];
	}
	
	MPI::COMM_WORLD.Gatherv( k_arr, nnz_wkj, MPI::INT, k_merge, nnz_wkj_arr, disp, MPI::INT, ROOT);
	MPI::COMM_WORLD.Gatherv( j_arr, nnz_wkj, MPI::INT, j_merge, nnz_wkj_arr, disp, MPI::INT, ROOT);
	MPI::COMM_WORLD.Gatherv( v_arr, nnz_wkj, MPI::DOUBLE, v_merge, nnz_wkj_arr, disp, MPI::DOUBLE, ROOT);
	
	// Construct Model from k_arr, j_arr, v_arr
	if( mpi_rank==ROOT ){
		StaticModel* model = new StaticModel(train);
		SparseVec* w = model->w;
		for(int i=0;i<total_nnz;i++){
			int k = k_merge[i];
			int j = j_merge[i];
			Float v = v_merge[i];
			w[j].push_back(make_pair(k,v));
		}
		model->writeModel(param->modelFname);
	}

	
	MPI::Finalize();
}
