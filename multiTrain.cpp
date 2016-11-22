#include "util.h"
#include "multi.h"
#include "PDSparse.h"
#include "AsyncPDSparse.h"

double overall_time = 0.0;

void exit_with_help(){
	cerr << "Usage: ./multiTrain (options) [train_data] (model)" << endl;	
	cerr << "options:" << endl;
	cerr << "-s solver: (default 1)" << endl;
	cerr << "	1 -- PDSparse" << endl;
	cerr << "	2 -- Asynchronized PDSparse (degree tau)" << endl;
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
			
			case 's': param->solver = atoi(argv[i]);
				  break;
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


int main(int argc, char** argv){
	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	
	Problem* train = new Problem();
	readData( param->trainFname, train);
	param->train = train;
	
	overall_time -= omp_get_wtime();

	if (param->heldoutFname != NULL){
		Problem* heldout = new Problem();
		readData( param->heldoutFname, heldout);
		cerr << "heldout N=" << heldout->data.size() << endl;
		param->heldoutEval = new HeldoutEval(heldout);
	}
	int D = train->D;
	int K = train->K;
	int N = train->data.size();
	cerr << "N=" << N << endl;
	cerr << "d=" << (Float)nnz(train->data)/N << endl;
	cerr << "D=" << D << endl; 
	cerr << "K=" << K << endl;
	
	if( param->solver == 1 ){

		PDSparse* solver = new PDSparse(param);
		solver->solve();

	}else if( param->solver == 2 ){

		AsyncPDSparse* solver = new AsyncPDSparse(param);
		solver->solve();
	}
	//model->writeModel(param->modelFname);
	
	overall_time += omp_get_wtime();
	cerr << "overall_time=" << overall_time << endl;
}
