#include "util.h"
#include "multi.h"
#include "NewHash.h"
#include <iomanip>
#include <cassert>

extern double overall_time;

class PDSparse{
	
	public:
	PDSparse(Param* param){

		train = param->train;
		heldoutEval = param->heldoutEval;	
		early_terminate = param->early_terminate;
		data = &(train->data);
		labels = &(train->labels);
		lambda = param->lambda;
		C = param->C;
		N = train->N;
		D = train->D;
		K = train->K;

		max_iter = param->max_iter;
		
		//setting up for sampling oracle
		cerr << "lambda=" << lambda << ", C=" << C << ", max_iter=" << max_iter << endl;
		
		//initialize alpha and v ( s.t. v = X^Talpha )
		HashFunc* hashfunc = new HashFunc(K);
		v = new NewHash*[D];
		w = new NewHash*[D];
		for(int j=0;j<D;j++){
			v[j] = new NewHash(hashfunc);
			w[j] = new NewHash(hashfunc);
		}
		//initialize active set out of [K] for each sample i
		alpha = new vector<pair<int, Float>>[N];
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
				//positive labels are always active
				alpha[i].push_back(make_pair(*it, 0.0));
			}
		}
		
		//number of variables added to active set each iteration.
		max_select = param->max_select;
		if (max_select == -1){
			int avg_label = 0;
			for (int i = 0; i < N; i++){
				avg_label += labels->at(i).size();
			}
			avg_label /= N;
			if (avg_label < 1)
				avg_label = 1;
			max_select = avg_label;
		}
		
		//global cache
		prod_cache = new Float[K];
		prod_is_tocheck = new bool[K];
		memset(prod_cache, 0.0, sizeof(Float)*K);
		memset(prod_is_tocheck, false, sizeof(bool)*K);
	}
	
	~PDSparse(){
		for(int j=0;j<D;j++)
			delete v[j];
		delete[] v;
		for(int j=0;j<D;j++)
			delete[] w[j];
		delete[] w;
		
		delete[] prod_cache;
		delete[] prod_is_tocheck;
		delete[] alpha;
	}

	StaticModel* solve(){
		
		//initialize Q_diag (Q=X*X') for the diagonal Hessian of each i-th subproblem
		Q_diag = new Float[N];
		for(int i=0;i<N;i++){
			SparseVec* ins = data->at(i);
			Float sq_sum = 0.0;
			for(SparseVec::iterator it=ins->begin(); it!=ins->end(); it++)
				sq_sum += it->second*it->second;
			Q_diag[i] = sq_sum;
		}
		//indexes for permutation of [N]
		int* index = new int[N];
		for(int i=0;i<N;i++)
			index[i] = i;
		
		//main loop
		int terminate_countdown = 0;
		double search_time=0.0, subsolve_time=0.0, maintain_time=0.0;
		double last_search_time = 0.0, last_subsolve_time = 0.0, last_maintain_time = 0.0;
		Float* alpha_i_new = new Float[K];
		iter = 0;
		while( iter < max_iter ){
			
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){
				
				int i = index[r];
				SparseVec* x_i = data->at(i);
				Labels* yi = &(labels->at(i));
				
				//search active variable
				search_time -= omp_get_wtime();
				search_active_i( i, alpha[i] );	
				search_time += omp_get_wtime();

				//solve subproblem
				if( alpha[i].size() < 2 )
					continue;
				
				subsolve_time -= omp_get_wtime();
				subSolve(i, alpha[i], alpha_i_new);
				subsolve_time += omp_get_wtime();
					
				//maintain v =  X^T\alpha;  w = prox_{l1}(v);
				maintain_time -= omp_get_wtime();
				Float* delta_alpha_i = new Float[alpha[i].size()];
				int ind = 0;
				for(vector<pair<int, Float>>::iterator it = alpha[i].begin(); it != alpha[i].end(); it++){
                                        delta_alpha_i[ind++] = alpha_i_new[it->first] - it->second;
				}
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
					int J = it->first; 
					Float xij = it->second;
					
					NewHash* vj = v[J];
					NewHash* wj = w[J];
					ind = 0;
					for (vector<pair<int, Float>>::iterator it2 = alpha[i].begin(); it2 != alpha[i].end(); it2++){
						int k = it2->first;
						Float delta_alpha = delta_alpha_i[ind++];
						if( fabs(delta_alpha) < EPS )
							continue;
						//update v
						pair<int,Float>* iv_pair = vj->get_addr(k);
						Float vjk_old = iv_pair->second;
						Float vjk_new = vjk_old + xij*delta_alpha;
						vj->set( k, iv_pair, vjk_new );
						//update w
						if( fabs(vjk_old)<lambda && fabs(vjk_new)<lambda )
							continue;
						Float wjk = prox_l1(vjk_new, lambda);
						wj->set( k, wjk );
					}
				}
				delete[] delta_alpha_i;
				
				//update alpha
				bool has_zero=0;
				for(vector<pair<int, Float>>::iterator it=alpha[i].begin(); it!=alpha[i].end(); it++){
					int k = it->first;
					it->second = alpha_i_new[k];
					has_zero |= (fabs(it->second)<EPS);
				}
				//shrink alpha
				if( has_zero ){
					vector<pair<int, Float>> tmp_vec;
					tmp_vec.reserve(alpha[i].size());
					for(vector<pair<int, Float>>::iterator it=alpha[i].begin(); it!=alpha[i].end(); it++){
						int k = it->first;
						Float alpha_ik = it->second;
						if( fabs(alpha_ik) > EPS || find(yi->begin(), yi->end(), k)!=yi->end() ){
							tmp_vec.push_back(make_pair(k, it->second));
						}
					}
					alpha[i] = tmp_vec;
				}
				maintain_time += omp_get_wtime();
			}

			cerr << "i=" << iter << "\t" ;
			nnz_a_i = 0.0;
			for(int i=0;i<N;i++){
				nnz_a_i += alpha[i].size();	
			}
			nnz_a_i /= N;
			cerr << "nnz_a_i="<< (nnz_a_i) << "\t";
			nnz_w_j = 0.0;
			for(int j=0;j<D;j++){
				nnz_w_j += w[j]->size(); //util_w[j][S];
			}
			nnz_w_j /= D;
			cerr << "nnz_w_j=" << (nnz_w_j) << "\t";
			cerr << "search=" << search_time-last_search_time << "\t";
			cerr << "subsolve=" << subsolve_time-last_subsolve_time << "\t";
			cerr << "maintain=" << maintain_time-last_maintain_time << "\t";
			if (search_time - last_search_time > (subsolve_time-last_subsolve_time + maintain_time - last_maintain_time)*2){
				max_select = min( max_select*2, 100 );
			}
			last_search_time = search_time;
			last_maintain_time = maintain_time;
			last_subsolve_time = subsolve_time;

			overall_time += omp_get_wtime();
			cerr << "dual_obj=" << dual_obj() << "\t";
			if( heldoutEval != NULL){
				Float heldout_test_acc = heldoutEval->calcAcc(w);
				cerr << "heldout Acc=" << heldout_test_acc << " ";
			}
			overall_time -= omp_get_wtime();
			
			cerr << endl;
			iter++;
		}
		cerr << endl;

	
		//computing heldout accuracy 	
		cerr << "train time=" << overall_time + omp_get_wtime() << endl;
		cerr << "search time=" << search_time << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] Q_diag;
		delete[] index;

		return new StaticModel();
	}
	
	//compute 1/2 \|w\|_2^2 + \sum_{i,k: k \not \in y_i} alpha_{i, k}
	Float dual_obj(){
		
		Float dual_obj = 0.0;
		for (int J = 0; J < D; J++){
			NewHash* wj = w[J];
			
			for (NewHash::iterator it=wj->begin(); it!=wj->end(); ++it){
				Float wjk = it->second;
				dual_obj += wjk*wjk;
			}
		}
		dual_obj /= 2.0;
		for (int i = 0; i < N; i++){
			vector<pair<int, Float>>& act_index = alpha[i];
			Labels* yi = &(labels->at(i));
			for (vector<pair<int, Float>>::iterator it = act_index.begin(); it != act_index.end(); it++){
				int k = it->first;
				Float alpha_ik = it->second;
				if (find(yi->begin(), yi->end(), k) == yi->end()){
					dual_obj += alpha_ik;
				}
			}
		}
		return dual_obj;
	}

	void subSolve(int I, vector<pair<int, Float> >& alpha, Float* alpha_i_new){	
			
		Labels* yi = &(labels->at(I));
		int m = yi->size(), n = alpha.size() - m;
		Float* b = new Float[n];
		Float* c = new Float[m];
		int* act_index_b = new int[n];
		int* act_index_c = new int[m];
		
		SparseVec* x_i = data->at(I);
		Float A = Q_diag[I];
		int i = 0, j = 0;
		for(vector<pair<int, Float>>::iterator it = alpha.begin(); it != alpha.end(); it++){
			int k = it->first;
			Float alpha_ik = it->second;
                        if( find(yi->begin(), yi->end(), k) == yi->end() ){
                                b[i] = 1.0 - A*alpha_ik;
                                act_index_b[i++] = k;
                        }else{
                                c[j] = A*alpha_ik;
                                act_index_c[j++] = k;
                        }
		}

		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int j = it->first;
			Float xij = it->second;
                        NewHash* wj = w[j];
			for(int i = 0; i < n; i++){
                                Float wjk = wj->get(act_index_b[i]);
                                b[i] += wjk*xij;
			}
			for(int j = 0; j < m; j++){
                                Float wjk = wj->get(act_index_c[j]);
                                c[j] -= wjk*xij;
			}
		}
		for (int i = 0; i < n; i++){
			b[i] /= A;
		}
		for (int j = 0; j < m; j++){
			c[j] /= A;
		}
			
		Float* x = new Float[n];
		Float* y = new Float[m];
		solve_bi_simplex(n, m, b, c, C, x, y);
		for(int i = 0; i < n; i++){
			int k = act_index_b[i];
			alpha_i_new[k] = -x[i];
		}
		for(int j = 0; j < m; j++){
                        int k = act_index_c[j];
			alpha_i_new[k] = y[j];
                }

		delete[] x; delete[] y;
		delete[] b; delete[] c;
		delete[] act_index_b; delete[] act_index_c;
	}
	
	
	void search_active_i(int i, vector<pair<int, Float>>& alpha){	
		
                Labels* yi = &(labels->at(i));
                SparseVec* xi = data->at(i);
		
		//prod_cache should be all zero
		////encorce not selected: {k | alpha_ik!=0 or y_ik=1}
		for(vector<pair<int, Float>>::iterator it = alpha.begin(); it != alpha.end(); it++){
			prod_cache[it->first] = -INFI;
		}
		for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
			prod_cache[*it] = -INFI;
		}
		////prepare to select "max_select" indices out of {K]
		int* max_indices = new int[max_select+1];
		for(int ind = 0; ind <= max_select; ind++){
			max_indices[ind] = -1;
		}
		
                //Main Loop: Compute <xi,wk> for k=1...K
                vector<int> check_indices;
		for (SparseVec::iterator it = xi->begin(); it < xi->end(); it++){
			
			Float xij = it->second;
			int j = it->first;
			NewHash* wj = w[j];
			
                        for(NewHash::iterator it2 = wj->begin(); it2!=wj->end(); ++it2 ){
				int k = it2->first;
				Float wjk = it2->second;
				
                                prod_cache[k] += wjk * xij;
				
				if (!prod_is_tocheck[k]){
					check_indices.push_back(k);
					prod_is_tocheck[k] = true;
				}
			}
                }
		//Find "max_select" k of largest products
		for (vector<int>::iterator it = check_indices.begin(); it != check_indices.end(); it++){
			int k = *it;
			if( prod_cache[k] > -1.0 )
				update_max_indices(max_indices, prod_cache, k, max_select);
		}
		//add random k \in [K] if not enough
		for (int j = 0; j < max_select; j++){
			if (max_indices[j] != -1 && prod_cache[max_indices[j]] > 0.0) 
				continue;
			for (int r = 0; r < K; r++){
				int k = rand()%K;
				if (prod_cache[k] == 0){
					if (update_max_indices(max_indices, prod_cache, k, max_select)){
						break;
					}
				}
			}
		}
		//put selected k into alpha
		for(int ind = 0; ind < max_select; ind++){
			int k = max_indices[ind];
			if ( k != -1 && prod_cache[k] > -1.0 )
				alpha.push_back(make_pair(max_indices[ind], 0.0));
		}
		
		//reset prod_cache to all zero
		for (vector<int>::iterator it = check_indices.begin(); it != check_indices.end(); it++){
			prod_cache[*it] = 0.0;
			prod_is_tocheck[*it] = false;
		}
		for(vector<pair<int, Float>>::iterator it = alpha.begin(); it != alpha.end(); it++){
			prod_cache[it->first] = 0.0;
		}
		for (Labels::iterator it = yi->begin(); it != yi->end(); it++){
			prod_cache[*it] = 0.0;
		}
		
		delete[] max_indices;
	}

	
	private:
	
	Problem* train;
	HeldoutEval* heldoutEval;
	Float lambda;
	Float C;
	vector<SparseVec*>* data;
	vector<Labels>* labels;
	int D; 
	int N;
	int K;
	Float* Q_diag;
	
	int max_iter;
	vector<int>* k_index;
		
	//for active set search
	int max_select;
	Float* prod_cache;
	bool* prod_is_tocheck;
	
	//heldout options
	int early_terminate;
		
	public:
	
	//useful statistics
	Float nnz_w_j = 1.0;
	Float nnz_a_i = 1.0;
	Float d = 1.0;

	//(index, val) representation of alpha
	vector<pair<int, Float> >* alpha;

	//iterations used so far	
	int iter;
	
	NewHash** w;
	NewHash** v;
};
