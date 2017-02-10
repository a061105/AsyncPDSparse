#include "multi.h"
#include <omp.h>
#include <cassert>


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

				//compute accuracy
				vector<SparseVec*>* data = &(prob->data);
				vector<Labels>* labels = &(prob->labels);
				
				Float hit=0.0;
				Float* prod = new Float[model->K];
				memset(prod, 0.0, sizeof(Float)*model->K);
				
				double start = omp_get_wtime();
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
