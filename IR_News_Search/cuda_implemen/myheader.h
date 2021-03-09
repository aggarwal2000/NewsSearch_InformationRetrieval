#include<iostream>
#include<string>
#include<vector>
#include<cstring>
#include<unordered_set>
#include<stdlib.h>
#include <unordered_map> 
#include <unordered_set>
#include <utility>


#define THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

class csr_matrix{

public:
	std::vector<int> row_ptrs;
	std::vector<int> col_ind;
	int rows = 0;
	int cols = 0;
	std::vector<int> term_freq;
	


	int* row_ptrs_gpu;
	int* col_ind_gpu;
	int* term_freq_gpu;
	
};




class zone{

	public:

	int length_current = 0;//used to form csr matrix

	const std::string zone_name;

	csr_matrix matrix;						
	std::unordered_map<std::string, int > vocabulary; 
	bool stem = true;
	bool remove_stop_words = true;
	int vocab_index = 0;
	std::vector<int> doc_freq;
	std::vector<float> doc_norms;

	int num_clusters;
	std::vector<float> centroids;
	std::vector<int> cluster_ind_for_docs;
	std::vector<std::unordered_set<int>> clusters;
	std::vector<float> centroid_norms;


	int* doc_freq_gpu;
	float* doc_norms_gpu;

	int* cluster_ind_for_docs_gpu;
	float* centroids_gpu; //store one after the other
	float* centroid_norms_gpu;
	int* cardinality_clusters_gpu;



	



	zone(const std::string & name, bool do_stem, bool do_removal_stop): zone_name(name), stem(do_stem) , remove_stop_words(do_removal_stop){
		matrix.row_ptrs.push_back(0);
	}


	//for a zone, fill the csr matrix, and the vocabulary map
	void fill_mat(std::vector<std::string> tokens_doc, int doc_ref,const std::unordered_set<std::string>& STOP_WORD_SET);
	void Print_zonal_structure(); 

		

	void Basic_Initialization();
	void Compute_doc_norms();





	void Common_Allocation_for_clustering();
	void k_Allocation_for_clustering(int k);
	void k_Deallocation_for_clustering();
	void Common_Deallocation_for_clustering();

	void Elbow_method(const std::vector<std::string>& DATASET);

	void Clustering_docs(const std::vector<std::string>& DATASET);
	void Make_all_centroids_zero();
	void Initialize_centroids_with_docs();
	void Choose_seed_points();
	void Compute_centroid_norms();
	void Assign_docs_to_clusters();
	void Compute_cardinality_clusters();
	void Recompute_centroids();
	void Spherical_k_Means(int k, const std::vector<std::string>& DATASET,bool print_info);
    float Average_Dissimilarity();
    void Print_cluster_info(const std::vector<std::string>& DATASET);
    float Parallel_reduction(float* doc_in_cluster_dissim_gpu , int arr_length);

    std::vector<float> query_transform_to_vector(const std::string& query);

 	float Compute_Query_Norm(float* query_vector_gpu, int length);

 	int Find_closest_cluster_index(float* query_vector_gpu);
 	void Compute_query_zone_scores_cluster_based(float* zone_scores_gpu,float* query_vector_gpu, float query_norm, int closest_cluster);
 	float* query_handler_cluster_based(const std::string& query);

 	void Compute_query_zone_scores_exact(float* zone_scores_gpu,float* query_vector_gpu, float query_norm);
 	float* query_handler_exact(const std::string& query);


 	void Basic_Deallocation();
};




