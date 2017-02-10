#include "util.h"
#include "multi.h"
#include "NewHash.h"
#include <iomanip>
#include <cassert>
#include <list>
#include <random>

extern Float overall_time;

class ParallelPDSparse{

	public:
		ParallelPDSparse(Param* param){

			train = param->train;
			heldoutEval = param->heldoutEval;	
			lambda = param->lambda;
			C = param->C;
			N = train->N;
			D = train->D;
			K = train->K;
			num_threads = param->num_threads;
			data = (train->data);
			labels = (train->labels);
			
			index_transpose(labels, N, K, pos_samples);
			transpose(data, N, D, data_inv);
			
			//sort data_inv[j] for j=1...D by its value
			for(int j=0;j<D;j++){
							//random_shuffle(data_inv[j].begin(), data_inv[j].end());
							sort(data_inv[j].begin(), data_inv[j].end(),ValueComp());
			}
			//build sampling table for data_inv[j], j=1...D. (Assume nonnegative features)
			/*data_inv_dist.resize(D);
			for(int j=0;j<D;j++)
							data_inv_dist[j] = new discrete_distribution<int>(data_inv[j].begin(), data_inv[j].end());
			*/
			alpha_arr = new Float*[num_threads];
			for(int t=0;t<num_threads;t++){
							alpha_arr[t] = new Float[N];
							for(int i=0;i<N;i++)
											alpha_arr[t][i] = 0.0;
			}
			
			y_arr = new Float*[num_threads];
			for(int t=0;t<num_threads;t++){
							y_arr[t] = new Float[N];
							for(int i=0;i<N;i++)
											y_arr[t][i] = -1.0;
			}

			w_arr = new Float*[num_threads];
			v_arr = new Float*[num_threads];
			for(int t=0;t<num_threads;t++){
							v_arr[t] = new Float[D];
							w_arr[t] = new Float[D];
							for(int j=0;j<D;j++){
											v_arr[t][j] = 0.0;
											w_arr[t][j] = 0.0;
							}
			}
			
			w_nz_arr = new NewHash*[num_threads];
			for(int t=0;t<num_threads;t++){
							w_nz_arr[t] = new NewHash(new HashFunc(D));
			}
			
			prod_arr = new Float*[num_threads];
			for(int t=0;t<num_threads;t++){
							prod_arr[t] = new Float[N];
							for(int i=0;i<N;i++)
											prod_arr[t][i] = -1e300;
			}
			
			XXt_diag = new Float[N];
			for(int i=0;i<N;i++){
				XXt_diag[i] = 0.0;
				for(SparseVec::iterator it=data[i]->begin(); it!=data[i]->end(); it++)
					XXt_diag[i] += it->second*it->second;
				//for L2-loss
				XXt_diag[i] += 1.0/(2.0*C);
				/////////////////
				
			}
		}

		~ParallelPDSparse(){
			
			for(int t=0;t<num_threads;t++){
							delete[] v_arr[t];
							delete[] w_arr[t];
							delete w_nz_arr[t];
							delete[] alpha_arr[t];
							delete[] prod_arr[t];
			}
			delete[] v_arr;
			delete[] w_arr;
			delete w_nz_arr;
			delete[] alpha_arr;
			delete[] prod_arr;
		}
		
		StaticModel* solve(){

			vector<int> class_index;
			for(int i=0;i<K;i++)
				class_index.push_back(i);

			random_shuffle(class_index.begin(), class_index.end());

			StaticModel* model = new StaticModel(train);
			for(int r=0;r<class_index.size();r++){

				int k = class_index[r];

				cerr << "r=" << r <<" " <<"(|pos|="<<pos_samples[k].size()<<"):";

				SparseVec w_k;
				solve_one_class(k, pos_samples[k], w_k);
				
				for(SparseVec::iterator it=w_k.begin(); it!=w_k.end(); it++)
					model->w[it->first].push_back( make_pair(k,it->second) );
			}
			cerr << endl;

			return model;
		}

		void solve_one_class(int k, vector<int>& pos_samples, SparseVec& w_k){
					
						int t = omp_get_thread_num();
						
						for(vector<int>::iterator it=pos_samples.begin(); it!=pos_samples.end(); it++)
										y_arr[t][*it] = 1.0;

						//if( 10*pos_samples.size() < N ){
						active_set_CD( k, pos_samples, 
													prod_arr[t], alpha_arr[t], y_arr[t], w_arr[t], w_nz_arr[t], v_arr[t],
														w_k );
						//}else{
						//				randomized_CD(k, pos_samples, 
						//											alpha_arr[t], y_arr[t], w_arr[t], w_nz_arr[t], v_arr[t],
						//											w_k );
						//}
						for(vector<int>::iterator it=pos_samples.begin(); it!=pos_samples.end(); it++)
										y_arr[t][*it] = -1.0;
		}

		void randomized_CD(int k, vector<int>& pos_samples, 
										Float* alpha, Float* y, Float* w, NewHash* w_nz, Float* v,
										SparseVec& w_k){

			vector<int> act_index;
			act_index.resize(N);
			for(int i=0;i<N;i++)
				act_index[i] = i;

			Float tol = 0.1;
			int max_iter = 100;
			int iter=0;
			while(iter<max_iter){

				Float residual = 0.0; //inf-norm of prox-grad

				random_shuffle(act_index.begin(), act_index.end());
				for(int r=0;r<act_index.size();r++){
					int i = act_index[r];
					Float yi = y[i];
					SparseVec* xi = data[i];
					//compute grad
					//Float gi = yi*inner_prod(w, xi) - 1.0;
					Float gi = yi*inner_prod(w, xi) - 1.0 + alpha[i]/(2.0*C);
					//compute update
					//Float alpha_i_new = min(max( alpha[i]-gi/XXt_diag[i], 0.0),C);
					Float alpha_i_new = max( alpha[i]-gi/XXt_diag[i], 0.0);

					//maintain v, w
					Float yi_diff = yi*(alpha_i_new-alpha[i]);
					Float diff_abs = fabs(yi_diff);
					if( diff_abs > 1e-8 ){

						residual = max(residual, diff_abs*XXt_diag[i]);

						alpha[i] = alpha_i_new;

						//other feature
						for(SparseVec::iterator it=xi->begin(); 
								it!=xi->end(); it++){

							int j = it->first;
							Float val = it->second;

							v[j] += yi_diff*val;

							//Float wj_new = prox_l1( v[j], lambda );
							if( j!=0 ) w[j] = prox_l1( v[j], lambda );
							else       w[j] = v[j];
						}
					}
				}

				if( residual < tol )
					break;

				iter++;
			}

			//clean v,w and copy to w_k
			w_k.clear();
			for(int j=0;j<D;j++){
				v[j] = 0.0;
				if( fabs(w[j]) > 1e-2 ) //Dismec Heuristic
					w_k.push_back(make_pair(j,w[j]));
				w[j] = 0.0;
			}

			//clean alpha
			int nSV=0;
			for(vector<int>::iterator it=act_index.begin(); it!=act_index.end(); it++){
				int i = *it;
				if( alpha[i] > 0.0 )
					nSV++;
				alpha[i] = 0.0;
			}

			cerr << "#pos=" << pos_samples.size() << ", #ran_iter=" << iter 
				<< ", w_nnz=" << w_k.size() << ", a_nnz=" << nSV
				<< endl;
		}
	
		void active_set_CD(int k, vector<int>& pos_samples, 
											Float* prod, Float* alpha, Float* y, Float* w, NewHash* w_nz, Float* v,
										SparseVec& w_k){

			//assert: alpha, w, v are all 0 (maintained by previous solver)

			//initialize act_index
			vector<int> act_index;
			act_index.reserve(RESERVE_SIZE);
			for(vector<int>::iterator it=pos_samples.begin(); it!=pos_samples.end(); it++)
				act_index.push_back(*it);
			for(int r=0;r<min((int)(10*pos_samples.size()),(int)N/10);r++){
				int i=rand()%N;
				while( y[i] > 0.0 )
					i=rand()%N;
				act_index.push_back(i);
			}

			//main loop
			double sub_time=0.0, prod_time=0.0, select_time=0.0;
			vector<int> v_change_ind; 
			vector<int> prod_change_ind;
			v_change_ind.reserve(RESERVE_SIZE);
			prod_change_ind.reserve(RESERVE_SIZE);
			Float tol = 0.1;
			int max_iter = 15;
			int max_inner = 1;
			double sample_speedup_rate = 10.0;
			int minimal_sample_size = 5;
			//int minimal_sample_size = 1000000000;
			int max_select = 100;
			SparseVec wk_samples;

			int iter=0;
			while(iter<max_iter){
				
				//Float residual = 0.0; //inf-norm of prox-grad
				Float residual = 1e300;
				sub_time -= omp_get_wtime();
				//subsolve
				for(int inner=0;inner<max_inner;inner++){

					//Float inner_residual=0.0;
					random_shuffle(act_index.begin(), act_index.end());
					for(int r=0;r<act_index.size();r++){
						int i = act_index[r];
						Float yi = y[i];
						SparseVec* xi = data[i];
						//compute grad
						//Float gi = yi*(w[0]+inner_prod(w, xi)) - 1.0;
						Float gi = yi*(inner_prod(w, xi)) - 1.0 + alpha[i]/(2.0*C);
						

						//compute update
						//Float alpha_i_new = min(max( alpha[i]-gi/XXt_diag[i], 0.0),C);
						Float alpha_i_new = max( alpha[i]-gi/XXt_diag[i], 0.0);
						
						//maintain v, w
						Float yi_diff = yi*(alpha_i_new-alpha[i]);
						Float diff_abs = fabs(yi_diff);
						if( diff_abs > 1e-6 ){

							//inner_residual = max(residual, diff_abs*XXt_diag[i]);
							//inner_residual = max(residual, diff_abs);
							
							alpha[i] = alpha_i_new;
							//other feature
							for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

								int j = it->first;
								Float val = it->second;
								if( v[j] == 0.0 )
									v_change_ind.push_back(j);

								v[j] += yi_diff*val;

								Float wj_new;
								if( j!=0 ){
												wj_new = prox_l1( v[j], lambda );
												if( wj_new!=0.0 || w[j] != 0.0 )
																w_nz->set( j, wj_new );
												w[j] = wj_new;
								}else{
												w[j] = v[j];
								}
							}
						}
					}
					//if( inner_residual < tol )
					//	break;
				}
				sub_time += omp_get_wtime();
				
				if( iter+1 == max_iter )
								break;

				//srhink active set
				vector<int> act_index_new;
				act_index_new.reserve(RESERVE_SIZE);
				for(int r=0;r<act_index.size();r++){
					int i = act_index[r];
					Float yi = y[i];
					if( alpha[i] > 0.0 ||  yi > 0.0 )
						act_index_new.push_back(i);
				}
				act_index = act_index_new;

				//search active negative samples
				
				////adjust select size
				max_select = min( max((int)act_index.size(),max_select), (int)(N/10) );
				
				/*for(int r=0;r<max_select;r++){
								int i=rand()%N;
								while( y[i] > 0.0 || alpha[i] > 0.0 )
												i=rand()%N;
								act_index.push_back(i);
				}*/
				
				
				
				////compute <w,x_i> for all i
				prod_time -= omp_get_wtime();

				int num_sample = (int)(w_nz->size()/sample_speedup_rate);
				if( w_nz->size() > minimal_sample_size*sample_speedup_rate ){
					importance_samples( *w_nz, num_sample, wk_samples);
				}else{
					hash_to_sv( *w_nz, wk_samples );
				}

				for(SparseVec::iterator it=wk_samples.begin();it!=wk_samples.end();++it){

					int j = it->first;
					if( j==0 || data_inv[j].size()==0 )
						continue;
					Float wj = it->second;
					
					int x_sample_size = max_select*10;
					if( data_inv[j].size() <= x_sample_size ){
									for(SparseVec::iterator it2=data_inv[j].begin(); 
																	it2!=data_inv[j].end(); it2++){
													int i = it2->first;
													Float xji = it2->second;
													if( prod[i] == -1e300 ){
																	prod_change_ind.push_back(i);
																	prod[i] = 0.0;
													}
													prod[i] += wj*xji;
									}
					}else{
									//if( wj > 0.0 ){
													//int ran = rand() % (data_inv[j].size()-x_sample_size+1);
													//double ratio = (double)data_inv[j].size()/x_sample_size;
													//for(int r=ran;r<ran+x_sample_size;r++){
													for(int r=0;r<x_sample_size;r++){
																	int i = data_inv[j][r].first;
																	//Float xji = data_inv[j][r].second*ratio;
																	Float xji = data_inv[j][r].second;
																	if( prod[i] == -1e300 ){
																					prod_change_ind.push_back(i);
																					prod[i] = 0.0;
																	}
																	prod[i] += wj*xji;
													}
									/*}else{
													for(int r=0;r<x_sample_size;r++){
																	int r2 = data_inv[j].size()-1-r;
																	int i = data_inv[j][r2].first;
																	//Float xji = data_inv[j][r].second*ratio;
																	Float xji = data_inv[j][r2].second;
																	if( prod[i] == -1e300 ){
																					prod_change_ind.push_back(i);
																					prod[i] = 0.0;
																	}
																	prod[i] += wj*xji;
													}

									}*/
					}
				}
				prod_time += omp_get_wtime();
				

				select_time -= omp_get_wtime();
				////find max
				if( max_select <= 1000 ){
					
					list<pair<int,Float> > cand;
					for(int k=0;k<max_select;k++)
						cand.push_front(make_pair(-1,-1e300));
					for(int r=0;r<prod_change_ind.size();r++){
						int i = prod_change_ind[r];
						if( alpha[i] > 0.0 || y[i] > 0.0 )
							continue;
						Float val = prod[i];
						//Float gi = -(val+w[0])-1.0;
						list<pair<int,Float> >::iterator it=cand.begin();
						//if( val <= it->second || gi >= 0.0 )
						if( val <= it->second )
							continue;
						it++;
						for(;it!=cand.end() && val>it->second; it++);
						cand.insert(it,make_pair(i,val));
						cand.pop_front();
					}

					for(list<pair<int,Float> >::iterator it=cand.begin();
							it!=cand.end(); it++){
						if(it->first!=-1){
							//Float gi = -(it->second+w[0])-1.0;
							//if( -gi > 0.0 ){
								act_index.push_back(it->first);
								//residual = max( residual, -gi );
							//}
						}
					}
				}else{
					nth_element(prod_change_ind.begin(), prod_change_ind.begin()+ max_select, prod_change_ind.end(), ScoreComp(prod));
					int count_added=0;
					for(int r=0;r<prod_change_ind.size();r++){
						int i = prod_change_ind[r];
						if( alpha[i] > 0.0 || y[i] > 0.0 )
							continue;
						
						//Float gi = -(prod[i]+w[0])-1.0;
						//if( -gi > 0.0 ){
							act_index.push_back(i);
							//residual = max( residual, -gi );
							count_added++;
							if( count_added >= max_select )
								break;
						//}
					}
				}
				select_time += omp_get_wtime();

				////clear prod
				for(vector<int>::iterator it=prod_change_ind.begin();
						it!=prod_change_ind.end(); it++)
					prod[*it] = -1e300;
				prod_change_ind.clear();
				

				//check exit or dump info
				//if( residual < tol )
				//	break;

				//if( iter%10==0 )
				//	cerr << ".";
				
				iter++;
			}

			cerr << setprecision(3) ;
			cerr << "k=" << k << ", #pos=" << pos_samples.size() << ", #act_iter=" << iter 
				<< ", w_nnz=" << w_nz->size() << ", a_nnz=" << act_index.size()
				<< ", prod_time=" << prod_time 
				<< ", select_time=" << select_time 
				<< ", sub_time=" << sub_time 
				<< endl;

			//clean v,w and copy to w_k
			w_k.clear();
			for(vector<int>::iterator it=v_change_ind.begin(); it!=v_change_ind.end(); it++){
				int j = *it;
				v[j] = 0.0;
				if( fabs(w[j]) > 1e-6 )//dismec heuristic
					w_k.push_back(make_pair(j,w[j]));
				w[j] = 0.0;
			}
			w_nz->clear();

			//clean alpha
			for(vector<int>::iterator it=act_index.begin(); it!=act_index.end(); it++){
				int i = *it;
				alpha[i] = 0.0;
			}
		}

	private:

		/*Float dual_objective(vector<int>& act_index){

			Float obj = 0.0;
			for(NewHash::iterator it=w_nz->begin(); it!=w_nz->end(); ++it){
				obj += it->second*it->second;
			}
			obj /= 2.0;

			for(vector<int>::iterator it=act_index.begin();it!=act_index.end();it++)
				obj -= alpha[*it];

			return obj;
		}*/

		Problem* train;
		HeldoutEval* heldoutEval;
		Float lambda;
		Float C;
		vector< SparseVec* > data;
		vector< SparseVec > data_inv;
		//vector< discrete_distribution<int>* > data_inv_dist;
		//default_random_engine generator;
		vector< Labels > labels;
		int D;
		int N;
		int K;
		int num_threads;
		Float* XXt_diag;
		
		//for active set search
		Float** prod_arr;

	public:

		vector<vector<int> > pos_samples;
		
		//dense representation of alpha
		Float** alpha_arr;
		Float** y_arr; //N*1
		//w = prox(v);
		Float** w_arr;
		NewHash** w_nz_arr;
		//v=X'alpha
		Float** v_arr;
};
