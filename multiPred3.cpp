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

				if( argc < 1+3 ){
								cerr << "multiPred3 [testfile] [model] [num_thread]" << endl;
								cerr << "\tcompute top-1,2,3,4,5 accuracy" << endl;
								exit(0);
				}

				char* testFile = argv[1];
				char* modelFile = argv[2];
				int num_thread = atoi(argv[3]);
				omp_set_num_threads(num_thread);
				
				char* outFname;
				StaticModel* model = readModel(modelFile);
				
				Problem* prob = new Problem();
				readData( testFile, prob, true );
				
				cerr << "Ntest=" << prob->N << endl;
				
				double start = omp_get_wtime();
				//compute accuracy
				vector<SparseVec*>* data = &(prob->data);
				vector<Labels>* labels = &(prob->labels);
				
				Float** top_k_hit = new Float*[num_thread];
				for(int i=0;i<num_thread;i++){
								top_k_hit[i] = new Float[5];
								memset(top_k_hit[i], 0.0, sizeof(Float)*5);
				}
				Float** prod = new Float*[num_thread];
				for(int i=0;i<num_thread;i++){
								prod[i] = new Float[model->K];
								for(int k=0;k<model->K;k++)
												prod[i][k] = -1e300;
				}
				
				vector<int>* touched_index = new vector<int>[num_thread];
				
				#pragma omp parallel for
				for(int i=0;i<prob->N;i++){
								
								int thd_num = omp_get_thread_num();
								
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
																if( prod[thd_num][k] == -1e300 ){
																				touched_index[thd_num].push_back(k);
																				prod[thd_num][k] = 0.0;
																}
																prod[thd_num][k] += it2->second*xij;
												}
								}
								//find k1, k2, k3, k4, k5
							  nth_element(touched_index[thd_num].begin(), touched_index[thd_num].begin()+5, touched_index[thd_num].end(), ScoreComp(prod[thd_num]));
								int size = min( 5, (int)touched_index[thd_num].size() );
								sort(touched_index[thd_num].begin(), touched_index[thd_num].begin()+size, ScoreComp(prod[thd_num]));
								//compute top-1,2,3,4,5 precision
								for(int t=0;t<5&&t<touched_index[thd_num].size();t++){
												
												int k = touched_index[thd_num][t];
												bool flag = false;
												for (int j = 0; j < yi->size(); j++){
																if( yi->at(j) >= prob->label_name_list.size() )
																				continue;
																if (prob->label_name_list[yi->at(j)] == model->label_name_list->at(k)){
																				flag = true;
																}
												}
												if (flag){
																for(int j=4;j>=t;j--)
																				top_k_hit[thd_num][j] += 1.0;
												}
								}
								
								//clear earse prod values touched index
								for(vector<int>::iterator it=touched_index[thd_num].begin(); it!=touched_index[thd_num].end(); it++)
												prod[thd_num][*it] = -1e300;
								touched_index[thd_num].clear();
				}
				cerr << endl;
				double end = omp_get_wtime();
				
				Float* top_k_hit_sum = new Float[5];
				for(int j=0;j<5;j++)
								top_k_hit_sum[j] = 0.0;
				for(int i=0;i<num_thread;i++){
						for(int j=0;j<5;j++)
										top_k_hit_sum[j] += top_k_hit[i][j];
				}

				for(int t=0;t<5;t++){
								cerr << "Top " << t+1 << " Acc=" << ((Float)top_k_hit_sum[t]/(prob->N*(t+1))) << endl;
				}
				cerr << "pred time=" << (end-start) << " s" << endl;
}
