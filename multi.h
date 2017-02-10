#ifndef MULTITRAIN
#define MULTITRAIN

#include "util.h"
#include "NewHash.h"

class Problem{
	public:
	static map<string,int> label_index_map;
	static vector<string> label_name_list;
	static int D;//dimension
	static int K;
	
	vector<SparseVec*> data;
	vector<Labels> labels;
	int N;//number of samples
};

map<string,int> Problem::label_index_map;
vector<string> Problem::label_name_list;
int Problem::D = -1;
int Problem::K = -1;

class HeldoutEval{
	public:
	HeldoutEval(Problem* _heldout){
		heldout = _heldout;
		N = heldout->data.size();
		D = heldout->D;
		K = heldout->K;
		prod = new Float[K];
		for(int k=0;k<K;k++)
			prod[k] = 0.0;
		max_indices = new int[K];
		for (int k = 0; k < K; k++)
			max_indices[k] = k;
		prod_is_nonzero = new bool[K];
		for (int k = 0; k < K; k++)
			prod_is_nonzero[k] = false;

	}

	~HeldoutEval(){
		delete[] max_indices;
		delete[] prod_is_nonzero;
		delete[] prod;
	}
	
	//compute heldout accuracy using hash
	double calcAcc( NewHash** w ){
		hit=0.0;
		for(int i=0;i<heldout->N;i++){
			memset(prod, 0.0, sizeof(Float)*K);
			
			SparseVec* xi = heldout->data.at(i);
			Labels* yi = &(heldout->labels.at(i));
			int top = 1;
			if (top == -1)
				top = yi->size();
			// compute <w_k, x_i> where w_k is stored in hashmap
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
				int j= it->first;
				Float xij = it->second;
				NewHash* wj = w[j];
				for (NewHash::iterator it=wj->begin();it!=wj->end();++it){
					int k = it->first;
					Float wjk = it->second;
					prod[k] += wjk * xij;
				}
			}
			
			//sort to get rank
			sort(max_indices, max_indices+K, ScoreComp(prod));
			for(int k=0;k<top;k++){
				bool flag = false;
				for (int j = 0; j < yi->size(); j++){
					if (yi->at(j) == max_indices[k] ){
						flag = true;
					}
				}
				if (flag)
					hit += 1.0/top;
			}
		}
		return hit/N;
	}
	
	private:
	int N,D,K;
	Problem* heldout;
	Float* prod;
	int* max_indices;
	Float hit;
	bool* prod_is_nonzero;
};
class Param{
	public:
	char* trainFname;
	char* modelFname;
	char* heldoutFname;
	Float lambda; //for L1-norm (default 1/N)
	Float C; //weight of loss
	int tau;//degree of asynchronization (for AsyncPDSparse)
	int num_threads;//number of threads per node
	int speed_up_rate; // speed up rate for sampling
	int split_up_rate; // split up [K] into a number of subsets	
	Problem* train;
	HeldoutEval* heldoutEval = NULL;
	//solver-specific param
	int solver;
	int max_iter;
	int max_select;
	bool using_importance_sampling;
	int post_solve_iter;
	int early_terminate;
	bool dump_model;
	
	/** For AsyncPDSparse
	 */
	Float step_size_shrink;
	
	Param(){
		solver = 1;
		lambda = 0.1;
		C = 1.0;
		tau = 10;
		max_iter = 30;
		max_select = -1;
		using_importance_sampling = true;
		post_solve_iter = INF;
		early_terminate = 3;
		heldoutFname == NULL;
		train = NULL;
		dump_model = false;
		step_size_shrink = 0.1;
	}

	~Param(){
		delete[] trainFname;
		delete[] modelFname;
		delete[] heldoutFname;
	}
};


//only used for prediction
class StaticModel{

	public:
	StaticModel(){
		label_name_list = new vector<string>();
		label_index_map = new map<string,int>();
	}
	StaticModel(Problem* prob){
		label_index_map = &(prob->label_index_map);
		label_name_list = &(prob->label_name_list);
		D = prob->D;
		K = prob->K;
		w = new SparseVec[D];
	}
	SparseVec* w;
	int D;
	int K;
	vector<string>* label_name_list;
	map<string,int>* label_index_map;
	
	void writeModel( char* fname){
		ofstream fout(fname);
		fout << "nr_class " << K << endl;
		fout << "label ";
		for(vector<string>::iterator it=label_name_list->begin();
				it!=label_name_list->end(); it++){
			fout << *it << " ";
		}
		fout << endl;
		fout << "nr_feature " << D << endl;
		for(int j=0;j<D;j++){
			
			SparseVec& wj = w[j];
			fout << wj.size() << " ";
			for(SparseVec::iterator it=wj.begin(); it!=wj.end(); it++){
				fout << it->first << ":" << it->second << " ";
			}
			fout << endl;

			if( j % (D/100) == 0 )
							cerr << "." ;
		}
		cerr << endl;
		fout.close();
	}
};


class ThreadModelWriter{
	
				public:
				ThreadModelWriter(int _nNode, int _nThread, int node_id, int thread_id, Param* _param){
						
						nNode = _nNode;
						nThread = _nThread;
						param = _param;
						
						model_dir_name = new char[FNAME_LEN];
						sprintf(model_dir_name, "model_dir.%s", param->modelFname);
						
						char tmp[FNAME_LEN];
						sprintf(tmp, "mkdir -p %s", model_dir_name);
						system(tmp);
						
						modelFname = new char[FNAME_LEN];
						sprintf(modelFname, "%s/model.%d", model_dir_name, node_id*nThread+thread_id);
						modelFout = new ofstream(modelFname);
				}
				
				ThreadModelWriter(int _nNode, int _nThread, Param* _param){
							
						nNode = _nNode;
						nThread = _nThread;
						param = _param;
						
						model_dir_name = new char[FNAME_LEN];
						sprintf(model_dir_name, "model_dir.%s", param->modelFname);
						char tmp[FNAME_LEN];
						sprintf(tmp, "mkdir -p %s", model_dir_name);
						system(tmp);
						
						modelFname = param->modelFname;
						modelFout = NULL;
				}
				
				void mergeModel(){
						
						int D = param->train->D;
						vector<SparseVec> W;
						W.resize(D);
						for(int i=0;i<nNode*nThread;i++){
										char tmp[FNAME_LEN];
										sprintf(tmp, "%s/model.%d", model_dir_name, i);
										ifstream fin(tmp);
										
										while( !fin.eof() ){
												int class_id;
												fin.read( (char*)&class_id, sizeof(int) );
												if( fin.eof() )
																break;
												SparseVec sv;
												fin >> sv;
												cerr << "sv size=" << sv.size() << endl;
												for(SparseVec::iterator it=sv.begin(); it!=sv.end(); it++)
																W[it->first].push_back(make_pair(class_id, it->second));
										}
						}
						cerr << "modelFname=" << modelFname << endl;
						ofstream fout(modelFname);
						fout << "nr_class " << param->train->K << endl;
						fout << "label ";
						vector<string>& label_name_list = param->train->label_name_list;
						for(vector<string>::iterator it=label_name_list.begin();
														it!=label_name_list.end(); it++){
										fout << *it << " ";
						}
						fout << endl;
						fout << "nr_feature " << param->train->D << endl;
						
						for(int j=0;j<D;j++){

								fout << W[j].size() << " ";
								for(SparseVec::iterator it=W[j].begin(); it!=W[j].end(); it++)
												fout << it->first << ":" << it->second << " ";
								fout << endl;
						}
						fout.close();
						
						char tmp[FNAME_LEN];
						sprintf(tmp, "rm -rf model_dir.%s", param->modelFname);
						system(tmp);
				}

				void writeVec(int class_id, SparseVec& sv){
							
							(*modelFout).write( (char*) &class_id, sizeof(int) );
							(*modelFout) << sv;
				}
				
				void close(){
							
							(*modelFout).close();
				}
				
				~ThreadModelWriter(){
						delete[] model_dir_name;
						delete[] modelFname;
						if( modelFout != NULL )
										delete modelFout;
				}

				private:
				int nNode;
				int nThread;
				char* model_dir_name;
				char* modelFname;
				ofstream* modelFout;
				Param* param;
};


void readData(char* fname, Problem* prob, bool add_bias)
{
	map<string,int>* label_index_map = &(prob->label_index_map);
	vector<string>* label_name_list = &(prob->label_name_list);
	
	ifstream fin(fname);
	char* line = new char[LINE_LEN];
	int d = -1;
	int line_count = 1;
	while( !fin.eof() ){
		
		fin.getline(line, LINE_LEN);
		string line_str(line);
		
		if( line_str.length() < 2 && fin.eof() )
			break;
		size_t found = line_str.find("  ");
		while (found != string::npos){
			line_str = line_str.replace(found, 2, " ");
			found = line_str.find("  ");
		}
		found = line_str.find(", ");
		while (found != string::npos){
			line_str = line_str.replace(found, 2, ",");
			found = line_str.find(", ");
		}
		vector<string> tokens = split(line_str, " ");
		//get label index
		Labels lab_indices;
		lab_indices.clear();
		map<string,int>::iterator it;
		int st = 0;
		while (st < tokens.size() && tokens[st].find(":") == string::npos){
			// truncate , out
			if (tokens[st].size() == 0){
				st++;
				continue;
			}
			vector<string> subtokens = split(tokens[st], ",");
			for (vector<string>::iterator it_str = subtokens.begin(); it_str != subtokens.end(); it_str++){
				string str = *it_str;
				if (str == "" || str == " ")
					continue;
				if( (it=label_index_map->find(str)) == label_index_map->end() ){
					lab_indices.push_back(label_index_map->size());
					label_index_map->insert(make_pair(str, lab_indices.back()));
				}else{
					lab_indices.push_back(it->second);
				}
			}
			st++;
		}
		
		SparseVec* ins = new SparseVec();
		//adding Bias
		if( add_bias )
						ins->push_back(make_pair(0,1.0));
		/////////////
		for(int i=st;i<tokens.size();i++){
			vector<string> kv = split(tokens[i],":");
			int ind = atoi(kv[0].c_str());
			Float val = atof(kv[1].c_str());
			ins->push_back(make_pair(ind,val));
			if( ind > d )
				d = ind;
			if( ind < 1 ){
				cerr << "minimum feature index should be 1 (" << line_count << " line)" << endl;
				exit(0);
			}
		}
		
		prob->data.push_back(ins);
		prob->labels.push_back(lab_indices);
		
		line_count++;
	}
	fin.close();
	
	/* Adding Bias
	*/
	if (prob->D < d+1){
		prob->D = d+1;
	}

	prob->N = prob->data.size();
	prob->K = label_index_map->size();
	label_name_list->resize(prob->K);
	for(map<string,int>::iterator it=label_index_map->begin();
			it!=label_index_map->end();
			it++)
		(*label_name_list)[it->second] = it->first;
	
	//random rehash labels
	/*HashFunc* hashfun = new HashFunc(prob->K);
	for(map<string,int>::iterator it=label_index_map->begin(); 
		it!=label_index_map->end(); it++){
		it->second = hashfun->get(it->second);
		(*label_name_list)[it->second] = it->first;
	}
	for(int i=0;i<prob->labels.size();i++){
		for(int j=0;j<prob->labels[i].size();j++)
			prob->labels[i][j] = hashfun->get(prob->labels[i][j]);
	}
	delete hashfun;*/

	delete[] line;
}

StaticModel* readModel(char* file){

				StaticModel* model = new StaticModel();

				ifstream fin(file);
				char* tmp = new char[LINE_LEN];
				fin >> tmp >> (model->K);
				
				fin >> tmp;
				string name;
				for(int k=0;k<model->K;k++){
								fin >> name;
								model->label_name_list->push_back(name);
								model->label_index_map->insert(make_pair(name,k));
				}

				fin >> tmp >> (model->D);
				model->w = new SparseVec[model->D];

				vector<string> ind_val;
				int nnz_j;
				for(int j=0;j<model->D;j++){
								fin >> nnz_j;
								model->w[j].resize(nnz_j);
								for(int r=0;r<nnz_j;r++){
												fin >> tmp;
												ind_val = split(tmp,":");
												int k = atoi(ind_val[0].c_str());
												Float val = atof(ind_val[1].c_str());
												model->w[j][r].first = k;
												model->w[j][r].second = val;
								}
				}

				delete[] tmp;
				return model;
}

#endif
