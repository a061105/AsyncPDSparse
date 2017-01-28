#ifndef NEWHASH
#define NEWHASH
#include "util.h"

const int INIT_CAP = 128;//Must be a multiple of 2
const Float UPPER_UTIL_RATE = 0.75;
const Float LOWER_UTIL_RATE = 0.5;

//Note this Hash Map does not preserve entries with value=0.0
//(it could be deleted when resizing)

class NewHash{

	public:
	
	NewHash(HashFunc* _hashfunc){
		
		hashfunc = _hashfunc;

		list_cap = INIT_CAP;
		list_size = 0;
		list = new pair<int, Float>[list_cap];
		for(int i=0;i<list_cap;i++){
			list[i].first = -1;
			list[i].second = 0.0;
		}
	}
	
	~NewHash(){
		delete[] list;
	}
	
	int size(){
		return list_size;
	}

	pair<int,Float>* get_addr(int index){
		
		int list_index;
		int size0 = list_cap-1;
		find_index( list, list_index, index, size0, hashfunc->hashindices );
		
		return &(list[ list_index ]);
	}

	Float get(int index){
		
		pair<int,Float>* p = get_addr(index);
		return (p->first==-1)?0.0:(p->second);
	}

	void set(int index, Float value){
		
		pair<int,Float>* p = get_addr(index);
		set( index, p, value );
	}
	
	void set(int index, pair<int,Float>* p, Float value){
		
		int prev_index = p->first;
		p->first = index;
		p->second = value;
		
		if( prev_index == -1 ){
			list_size+=1;
			pair<int,Float>* list2 = list;
			int list_cap2 = list_cap;
			int size0 = list_cap-1;
			if( list_size > list_cap * UPPER_UTIL_RATE ){
				resize(list2, list, list_cap, list_cap2, size0, list_size, hashfunc->hashindices);
			}
		}
	}
	
	void clear(){

		for(int i=0;i<list_cap;i++){
			list[i].first = -1;
			list[i].second = 0.0;
		}
		list_size=0;
	}


	class iterator{
		public:
		iterator(pair<int,Float>* _list, int _cap, int _pos){
			arr = _list;
			cap = _cap;
			pos = _pos;
			while( pos<cap && arr[pos].first==-1 )
				pos++;
		}
			
		iterator& operator++(){

			pos++;
			while( pos<cap && arr[pos].first==-1 )
				pos++;
			
			return *this;
		}
		
		bool operator==(const iterator& it2){
			return pos==it2.pos;
		}

		bool operator!=(const iterator& it2){
			return pos!=it2.pos;
		}

		pair<int,Float>* operator->() const {
			return &(arr[pos]);
		}
		
		private:
		pair<int,Float>* arr;
		int pos;
		int cap;
	};
	
	iterator begin(){
		iterator it(list, list_cap, 0);
		return it;
	}

	iterator end(){
		iterator it(list, list_cap, list_cap);
		return it;
	}
	
	//for debug
	void printList(){
		for(int i=0;i<list_cap;i++)
			cerr << "(" << list[i].first << "," << list[i].second << "),";
		cerr << endl;
	}
	
	private:
	
	HashFunc* hashfunc;
	pair<int, Float>* list;
	int list_cap;
	int list_size;
	
	inline void find_index(pair<int, Float>*& l, int& st, const int& index, int& size0, int*& hashindices){
	        st = hashindices[index] & size0;
	        if (l[st].first != index){
	                while (l[st].first != index && l[st].first != -1){
	                        st++;
	                        st &= size0;
	                }
	        }
	}
	

	inline void resize(pair<int, Float>*& l, pair<int, Float>*& L, int& size, int& new_size, int& new_size0, int& util, int*& hashindices){
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        if (new_size == size)
	                return;
	        pair<int, Float>* new_l = new pair<int, Float>[new_size];
	        new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, 0.0);
	        }
	 	int new_util=0;
	 	for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, Float> p = l[tt];
	                if (p.first != -1 && p.second != 0.0){
				find_index(new_l, index_l, p.first, new_size0, hashindices);
	                        new_l[index_l] = p;
				new_util++;
	                }
	        }
	        delete[] l;
	        l = new_l;
	        size = new_size;
	        L = new_l;
		util = new_util;
	}

	inline void trim(pair<int, Float>*& l, int& size, int& util, int*& hashindices){
		util = 0;
		for (int tt = 0; tt < size; tt++){
			pair<int, Float> p = l[tt];
			if (p.first != -1 && p.second != 0.0)
				util++;
		}
		int new_size = size;
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        while (new_size > INIT_CAP && util < LOWER_UTIL_RATE * new_size){
	                new_size /= 2;
	        }
	        pair<int, Float>* new_l = new pair<int, Float>[new_size];
	        int new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, 0.0);
	        }
	        for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, Float> p = l[tt];
	                if (p.first != -1 && p.second != 0.0){
	                        find_index(new_l, index_l, p.first, new_size0, hashindices);
	                        new_l[index_l] = p;
	                }
	        }
	        delete[] l;
	        l = new_l;
	        size = new_size;
	}

};
	
void importance_samples( NewHash& sv, int num_sample, SparseVec& samples ){
	
	samples.clear();
	if( sv.size() == 0 )
		return;
	
	double* cum = new double[sv.size()];
	NewHash::iterator it = sv.begin();
	if( it->first != 0 )//not bias
					cum[0] = fabs(it->second);
	else
					cum[0] = 0.0;
	++it;
	for(int i=1;it!=sv.end();++it,++i){
		if( it->first != 0 )//not bias
						cum[i] = cum[i-1] + fabs(it->second);
		else//bias
						cum[i] = cum[i-1];
		
	}
	double weight_sum = cum[sv.size()-1];
	
	vector<double> rands;
	for(int i=0;i<num_sample;i++)
		rands.push_back( weight_sum * ((double)rand()/RAND_MAX) );
	sort( rands.begin(), rands.end() );
	
	int i=0,j=0;
	it = sv.begin();
	while( i < rands.size() ){
		
		while( rands[i] > cum[j]  ){
	       		j++;
			++it;
		}
		
		double count = 0.0;
		
		while( i<rands.size() && rands[i] < cum[j] ){
			count += 1.0;
			i++;
		}
		
		if( it->second > 0.0 )
			samples.push_back(make_pair(it->first,count));
		else
			samples.push_back(make_pair(it->first,-count));
	}

	double tmp = (weight_sum/num_sample);
	for(SparseVec::iterator it2=samples.begin(); it2!=samples.end(); it2++)
		it2->second *= tmp;
	
	delete[] cum;
}

void hash_to_sv( NewHash& hash, SparseVec& sv ){
	
	sv.clear();
	for(NewHash::iterator it=hash.begin(); it!=hash.end(); ++it){
		if( it->second != 0.0 )
			sv.push_back(make_pair(it->first, it->second));
	}
}

#endif
