#include<iostream>
#include<string>
#include<vector>
#include<cstring>
#include<unordered_set>
#include<fstream>
#include<stdlib.h>
#include <unordered_map> 
#include <unordered_set>
#include <utility>
#include <omp.h>
#include<time.h>
#include<cmath>


#include"myheader.h"

float zone::cosine_similarity(int doc_index,std::vector<float> full_vec)
	{
			int start_ind_col_arr = matrix.row_ptrs[doc_index];
			int end_ind_col_arr = matrix.row_ptrs[doc_index + 1];
			float dot_product = 0;
			float norm_sqr_doc_vec = 0;

			#pragma omp parallel for reduction(+ : dot_product) reduction( + : norm_sqr_doc_vec )
			for(int j = start_ind_col_arr; j < end_ind_col_arr;j++)
			{
				int word_ind = matrix.col_ind[j];
				float logtf_idf = 0;
				int tf = matrix.term_freq[j]; //logtf idf actually
				if(tf == 0)
					logtf_idf = 0; //never possible
				else
				{
					logtf_idf =  ( 1 + std::log10(double(tf)) )*std::log10(matrix.rows /doc_freq[word_ind]);	
				}

				dot_product += logtf_idf*full_vec[word_ind];
				norm_sqr_doc_vec += logtf_idf*logtf_idf;
				
			}

			if(norm_sqr_doc_vec == 0)
				return 0; //no term in document
			else{
				
				float norm_sqr_full_vec = 0;

				#pragma omp parallel for reduction(+ : norm_sqr_full_vec)
				for(int i = 0; i< full_vec.size();i++)
				{
					norm_sqr_full_vec += full_vec[i]*full_vec[i];
				}
				

				if(norm_sqr_full_vec == 0)
					return 0;
				float cosine = dot_product/(std::sqrt(norm_sqr_doc_vec) * std::sqrt(norm_sqr_full_vec));
				return cosine;
			}
	}

	float zone::similarity_cluster_and_query(int i,const std::vector<float>& query_vector)
	{	
		float dot_product = 0;
		float norm_sqr_centroid = 0;
		float norm_sqr_query = 0;

		#pragma omp parallel for reduction (+:dot_product) reduction( +:norm_sqr_centroid )  reduction(+:norm_sqr_query)
		for(int j=0;j< query_vector.size();j++)
		{
			dot_product += centroids[i][j]*query_vector[j];

			norm_sqr_centroid += centroids[i][j]*centroids[i][j];
			norm_sqr_query += query_vector[j]*query_vector[j];
		}

		if(norm_sqr_centroid == 0 || norm_sqr_query == 0)
			return 0;

		float cosine = dot_product/(std::sqrt(norm_sqr_centroid)*std::sqrt(norm_sqr_query));
		return cosine;
	}

	