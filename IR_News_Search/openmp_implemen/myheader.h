#include<iostream>
#include<string>
#include<vector>
#include<cstring>
#include<unordered_set>
#include<stdlib.h>
#include <unordered_map> 
#include <unordered_set>
#include <utility>




class csr_matrix{

public:
	std::vector<int> row_ptrs;
	std::vector<int> col_ind;
	int rows = 0;
	int cols = 0;
	std::vector<int> term_freq;
};




class zone{

	public:
	const std::string zone_name;
	csr_matrix matrix;						
	std::unordered_map<std::string, int > vocabulary; 
	bool stem = true;
	bool remove_stop_words = true;
	int vocab_index = 0;
	int num_clusters = 20;
	std::vector<int> doc_freq;
	std::vector<std::vector<float>> centroids;
	std::vector<std::unordered_set<int>> clusters;
	int length_current = 0;

	zone(const std::string & name, bool do_stem, bool do_removal_stop): zone_name(name), stem(do_stem) , remove_stop_words(do_removal_stop){
		matrix.row_ptrs.push_back(0);
	}


	//for a zone, fill the csr matrix, and the vocabulary map.
	void fill_mat(std::vector<std::string> tokens_doc, int doc_ref,const std::unordered_set<std::string>& STOP_WORD_SET);
	
	void Print();





	void initialize_centroids();
	
	
	void assign_docs_to_clusters(std::vector<int> & cluster_for_docs);
	

	void recompute_centroids();
	
	void Print_cluster_info(const std::vector<std::string>& DATASET);
	
	void K_Means_Clustering(const std::vector<std::string>& DATASET,int K,bool print_info);
	
	void elbow_method(int start,int end,const std::vector<std::string>& DATASET);
	


	float cosine_similarity(int doc_index,std::vector<float> full_vec);
	
	float similarity_cluster_and_query(int i,const std::vector<float>& query_vector);
	




	std::vector<float> query_handler_cluster_based(const std::string& query);
	

	
	std::vector<float> query_transform_to_vector(const std::string& query);
	
	std::vector<float> query_handler(const std::string& query);
	

	

};




