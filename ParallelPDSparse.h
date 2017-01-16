#include "util.h"
#include "multi.h"
#include "NewHash.h"
#include <iomanip>
#include <cassert>
#include <list>

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
		data = (train->data);
		labels = (train->labels);
		model = new StaticModel(param);
		
		
		index_transpose(labels, N, K, pos_samples);
		transpose(data, N, D, data_inv);
		
		alpha = new Float[N];
		for(int i=0;i<N;i++)
			alpha[i] = 0.0;
		
		y = new Float[N];
		for(int i=0;i<N;i++)
			y[i] = -1.0;
		
		v = new Float[D];
		w = new Float[D];
		for(int j=0;j<D;j++){
			v[j] = 0.0;
			w[j] = 0.0;
		}

		w_nz = new NewHash(new HashFunc(D));
		
		prod_arr = new Float[N];
		for(int i=0;i<N;i++)
			prod_arr[i] = 0.0;

		XXt_diag = new Float[N];
		for(int i=0;i<N;i++){
			XXt_diag[i] = 0.0;
			for(SparseVec::iterator it=data[i]->begin(); it!=data[i]->end(); it++)
				XXt_diag[i] += it->second*it->second;
		}
	}
	
	~ParallelPDSparse(){
		delete[] v;
		
		delete w;
		delete[] alpha;
		delete[] prod_arr;
	}
	
	StaticModel* solve(){
		
		for(int k=0;k<K;k++){
			//cerr << "class " << k <<" " <<"(|pos|="<<pos_samples[k].size()<<"):";
			
			for(vector<int>::iterator it=pos_samples[k].begin(); 
				it!=pos_samples[k].end(); it++)
				y[*it] = 1.0;
			
			SparseVec w_k;
			solve_one_class(pos_samples[k], w_k);
			
			for(vector<int>::iterator it=pos_samples[k].begin(); 
				it!=pos_samples[k].end(); it++)
				y[*it] = -1.0;
			
			for(SparseVec::iterator it=w_k.begin(); it!=w_k.end(); it++)
				model->w[it->first].push_back( make_pair(k,it->second) );

			if( k % (K/100) == 0 )
				cerr << ".";
		}
		cerr << endl;
		
		return model;
	}
	
	void solve_one_class(vector<int>& pos_samples, SparseVec& w_k){
		
		//assert: alpha, w, v are all 0 (maintained by previous solver)
		
		//initialize act_index
		vector<int> act_index;
		act_index.reserve(RESERVE_SIZE);
		for(vector<int>::iterator it=pos_samples.begin(); it!=pos_samples.end(); it++)
			act_index.push_back(*it);
		
		//main loop
		//double search_time, sub_time, overall_search_time=0.0, overall_sub_time=0.0;
		vector<int> v_change_ind; 
		vector<int> prod_change_ind;
		v_change_ind.reserve(RESERVE_SIZE);
		prod_change_ind.reserve(RESERVE_SIZE);
		Float tol = 0.1;
		int max_iter = 1000;
		int max_select = 5*pos_samples.size();
		int iter=0;
		while(iter<max_iter){
			
			Float residual = 0.0; //inf-norm of prox-grad

			//sub_time = -omp_get_wtime();
			//subsolve
			random_shuffle(act_index.begin(), act_index.end());
			for(int r=0;r<act_index.size();r++){
				int i = act_index[r];
				Float yi = y[i];
				SparseVec* xi = data[i];
				//compute grad
				Float gi = yi*inner_prod(w, xi) - 1.0;
				//compute update
				Float alpha_i_new = min(max( alpha[i]-gi/XXt_diag[i], 0.0),C);
				Float yi_diff = yi*(alpha_i_new-alpha[i]);
				//maintain v, w
				Float diff_abs = fabs(yi_diff);
				if( diff_abs > 1e-6 ){

					residual = max(residual, diff_abs);
					
					alpha[i] = alpha_i_new;
					
					for(SparseVec::iterator it=xi->begin(); 
						it!=xi->end(); it++){
						
						int j = it->first;
						Float val = it->second;
						if( v[j] == 0.0 )
							v_change_ind.push_back(j);
						
						v[j] += yi_diff*val;
						
						Float wj_new = prox_l1( v[j], lambda );
						if( wj_new != 0.0 || w[j] != 0.0 )
							w_nz->set( j, wj_new );
						w[j] = wj_new;
					}
				}
			}
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
			
			//sub_time += omp_get_wtime();
			
			//search_time = -omp_get_wtime();
			//search active negative samples
			////compute <w,x_i> for all i
			for(NewHash::iterator it=w_nz->begin();it!=w_nz->end();++it){
				int j = it->first;
				Float wj = it->second;
				for(SparseVec::iterator it2=data_inv[j].begin(); 
					it2!=data_inv[j].end(); it2++){
					int i = it2->first;
					Float xji = it2->second;
					if( prod_arr[i] == 0.0 )
						prod_change_ind.push_back(i);
					prod_arr[i] += wj*xji;
				}
			}
			
			////find max
			list<pair<int,Float> > cand;
			for(int k=0;k<max_select;k++)
				cand.push_front(make_pair(-1,-1e300));
			for(int r=0;r<prod_change_ind.size();r++){
				int i = prod_change_ind[r];
				if( alpha[i] > 0.0 || y[i] > 0.0 )
					continue;
				Float val = prod_arr[i];
				list<pair<int,Float> >::iterator it=cand.begin();
				if( val <= it->second  )
					continue;
				it++;
				for(;it!=cand.end() && val>it->second; it++);
				cand.insert(it,make_pair(i,val));
				cand.pop_front();
			}
			
			for(list<pair<int,Float> >::iterator it=cand.begin();it!=cand.end();it++)
				act_index.push_back(it->first);
			
			Float gi = -(cand.back().second)-1.0;
			residual = max( residual, -gi );

			////clear prod_arr
			for(vector<int>::iterator it=prod_change_ind.begin();
				it!=prod_change_ind.end(); it++)
				prod_arr[*it] = 0.0;
			prod_change_ind.clear();
			
			//search_time += omp_get_wtime();
			
			//adjust select size
			if( 2*max_select < act_index.size() )
				max_select = min( 2*max_select, 100 );
			
			//check exit or dump info
			if( residual < tol ){
				/*cerr << "w_nnz=" << w_nz->size() << ", a_nnz=" << act_index.size()
				     << ", search_time=" << overall_search_time 
				     << ", sub_time=" << overall_sub_time << endl;*/
				//cerr <<"*"<< endl;i
				break;
			}
			//if( iter%10==0 )
			//	cerr << ".";
			
			//overall_search_time += search_time;
			//overall_sub_time += sub_time;
			iter++;
		}
		
		//clean v,w and copy to w_k
		w_k.clear();
		for(vector<int>::iterator it=v_change_ind.begin(); it!=v_change_ind.end(); it++){
			int j = *it;
			v[j] = 0.0;
			if( fabs(w[j]) > 0.0 )
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
	
	Float dual_objective(vector<int>& act_index){
		
		Float obj = 0.0;
		for(NewHash::iterator it=w_nz->begin(); it!=w_nz->end(); ++it){
			obj += it->second*it->second;
		}
		obj /= 2.0;

		for(vector<int>::iterator it=act_index.begin();it!=act_index.end();it++)
			obj -= alpha[*it];
		
		return obj;
	}

	Problem* train;
	HeldoutEval* heldoutEval;
	Float lambda;
	Float C;
	vector< SparseVec* > data;
	vector< SparseVec > data_inv;
	vector< Labels > labels;
	vector<vector<int> > pos_samples;
	int D;
	int N;
	int K;
	Float* XXt_diag;
	
	//for active set search
	Float* prod_arr;
	
	public:
	
	//dense representation of alpha
	Float* alpha;
	Float* y; //N*1
	//w = prox(v);
	Float* w;
	NewHash* w_nz;
	//v=X'alpha
	Float* v;

	StaticModel* model;
};
