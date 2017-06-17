#include "multi.h"
#include <omp.h>
#include <cassert>


int main(int argc, char** argv){

				if( argc < 1+2 ){
								cerr << "multiPred [testfile] [model] (-p S <output_file>) (k)" << endl;
								cerr << "\t-p S <output_file>: print top S <label>:<prediction score> pairs to <output_file>, one line for each instance. (default S=0 and no file is generated)" << endl;
								cerr << "\tcompute top k accuracy, default k=1" << endl;
								exit(0);
				}

				char* testFile = argv[1];
				char* modelFile = argv[2];
				char* outFname;
				int S = 0, offset = 0;
				if (argc > 5 && strcmp(argv[3], "-p") == 0){
								S = atoi(argv[4]);
								outFname = argv[5];
								offset = 3;
				}
				int T = 1;
				if (argc > 3 + offset){
								T = atoi(argv[3 + offset]);
				}
				bool is_binary = !isFile(modelFile);
				StaticModel* model = readModel(modelFile, is_binary);

				if (T > model->K || S > model->K){
								cerr << "k or S is larger than domain size" << endl;
								exit(0);
				}
				Problem* prob = new Problem();
				readData( testFile, prob, true );

				cerr << "Ntest=" << prob->N << endl;

				//compute accuracy
				vector<SparseVec*>* data = &(prob->data);
				vector<Labels>* labels = &(prob->labels);
				Float hit=0.0;
				Float margin_hit = 0.0;
				Float* prod = new Float[model->K];
				int* max_indices = new int[model->K+1];
				for(int k = 0; k < model->K+1; k++){
								max_indices[k] = -1;
				}
				ofstream fout;
				if (S != 0){
								cerr << "Printing Top " << S << " <label>:<prediction score> pairs to " << outFname << ", one line per instance" << endl;
								fout.open(outFname);
				}
				
				double start = omp_get_wtime();
				for(int i=0;i<prob->N;i++){

								if( i % (prob->N/100) == 0 )
												cerr << ".";
								
								memset(prod, 0.0, sizeof(Float)*model->K);

								SparseVec* xi = data->at(i);
								Labels* yi = &(labels->at(i));
								int Ti = T;
								if (Ti <= 0)
												Ti = yi->size();
								int top = max(Ti, S);
								for(int ind = 0; ind < model->K; ind++){
												max_indices[ind] = ind;
								}
								for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

												int j= it->first;
												Float xij = it->second;
												if( j >= model->D )
																continue;
												SparseVec* wj = &(model->w[j]);
												for(SparseVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
																int k = it2->first;
																prod[k] += it2->second*xij;
												}
								}
								nth_element(max_indices, max_indices+top, max_indices+model->K, ScoreComp(prod));
								sort(max_indices, max_indices+top, ScoreComp(prod));
								for(int k=0;k<Ti;k++){
												bool flag = false;
												for (int j = 0; j < yi->size(); j++){
																if (prob->label_name_list[yi->at(j)] == model->label_name_list->at(max_indices[k])){
																				flag = true;
																}
												}
												if (flag)
																hit += 1.0/Ti;
								}
								if (S != 0){
												for (int k = 0; k < S; k++){
																if (k != 0){
																				fout << " ";
																}
																fout << model->label_name_list->at(max_indices[k]) << ":" << prod[max_indices[k]];
												}
												fout << endl;
								}
				}
				if (S != 0){
								fout.close();
				}
				cerr << endl;
				double end = omp_get_wtime();
				cerr << "Top " << T << " Acc=" << ((Float)hit/prob->N) << endl;
				cerr << "pred time=" << (end-start) << " s" << endl;
}
