#include "multi.h"
#include <omp.h>
#include <cassert>

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

int main(int argc, char** argv){

				if( argc < 1+2 ){
								cerr << "multiPred [testfile] [model]" << endl;
								cerr << "\tcompute top-1 accuracy" << endl;
								exit(0);
				}

				char* testFile = argv[1];
				char* modelFile = argv[2];
				char* outFname;

				StaticModel* model = readModel(modelFile);

				Problem* prob = new Problem();
				readData( testFile, prob, true );
				
				cerr << "Ntest=" << prob->N << endl;

				double start = omp_get_wtime();
				//compute accuracy
				vector<SparseVec*>* data = &(prob->data);
				vector<Labels>* labels = &(prob->labels);
				
				Float hit=0.0;
				Float* prod = new Float[model->K];
				memset(prod, 0.0, sizeof(Float)*model->K);
				
				vector<int> touched_index;
				for(int i=0;i<prob->N;i++){
								
								if( i % (prob->N/100) == 0 )
												cerr << ".";
								
								SparseVec* xi = data->at(i);
								Labels* yi = &(labels->at(i));
								//compute scores
								for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

												int j= it->first;
												Float xij = it->second;
												if( j >= model->D )
																continue;
												SparseVec* wj = &(model->w[j]);
												for(SparseVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
																int k = it2->first;
																if( prod[k] == 0.0 ){
																				touched_index.push_back(k);
																}
																prod[k] += it2->second*xij;
												}
								}
								
								Float max_val = -1e300;
								int max_k = 0;
								for(vector<int>::iterator it=touched_index.begin(); it!=touched_index.end(); it++){
												if( prod[*it] > max_val ){
																max_val = prod[*it];
																max_k = *it;
												}
								}

								//compute top-1 precision
								bool flag = false;
								for (int j = 0; j < yi->size(); j++){
												if( yi->at(j) >= prob->label_name_list.size() )
																continue;
												if (prob->label_name_list[yi->at(j)] == model->label_name_list->at(max_k)){
																flag = true;
												}
								}
								if (flag)
												hit += 1.0;

								//clear earse prod values touched index
								for(vector<int>::iterator it=touched_index.begin(); it!=touched_index.end(); it++)
												prod[*it] = 0.0;
								touched_index.clear();
				}
				cerr << endl;
				double end = omp_get_wtime();
				cerr << "Top " << 1 << " Acc=" << ((Float)hit/prob->N) << endl;
				cerr << "pred time=" << (end-start) << " s" << endl;
}
