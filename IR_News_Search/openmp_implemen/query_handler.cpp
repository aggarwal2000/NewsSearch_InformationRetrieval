#include<iostream>
#include<string>
#include<vector>
#include<cstring>
#include<unordered_set>
#include<fstream>
#include<stdlib.h>
#include<regex>
#include <unordered_map> 
#include <unordered_set>
#include"porter2_stemmer.h"
#include <utility>
#include <omp.h>
#include<time.h>
#include <iomanip>  


#include"myheader.h"

std::vector<float> zone::query_handler_cluster_based(const std::string& query)
	{	

		std::vector<float> query_vector = this->query_transform_to_vector(query);

		//find index of closest cluster
		float max_sim = 0;
		int closest_cluster_index = 0;

		// #pragma omp parallel for reduction(max: max_sim) reduction(max:closest_cluster_index) //NEW line ADDED //producing wrong result
		// for(int i = 0; i < clusters.size();i++)
		// {
		// 	float sim = similarity_cluster_and_query(i,query_vector);
		// 	//std::cout << " \nsim score:" << sim << "  cluster: " << i << std::endl;
		// 	if(sim > max_sim)
		// 	{
		// 		max_sim = sim;
		// 		closest_cluster_index = i;
		// 	}

		// }


		#pragma omp parallel for//NEW line ADDED
		for(int i = 0; i < clusters.size();i++)
		{
			float sim = similarity_cluster_and_query(i,query_vector);
			//std::cout << " \nsim score:" << sim << "  cluster: " << i << std::endl;

			#pragma omp critical
			if(sim > max_sim)
			{
				max_sim = sim;
				closest_cluster_index = i;
			}

		}
		//Got closest cluster index

		std::cout << "\n\n The closest cluster is: " << closest_cluster_index << std::endl;

		//Now inside this cluster have to compute similarity and return all (Note: for some-->  0 score -> don't mind)
	
		std::vector<float> scores(matrix.rows,0);

		
		
		#pragma omp parallel
		{
			#pragma omp single
			{

				for (auto itr =   clusters[closest_cluster_index].begin(); itr != clusters[closest_cluster_index].end(); ++itr) {
		    	
			    	  #pragma omp task
						{
							int doc_index = *itr;
							float sc = cosine_similarity(doc_index,query_vector);
							scores[doc_index] = sc;	
					    }	
					
				}
			}
		}
		

		// #pragma omp parallel for //Tried this way, but it does not work...., but does work for query_word.. in the function below
		// for(int doc_index : clusters[closest_cluster_index])
		// {	
		// 	float sc = cosine_similarity(doc_index,query_vector);
		// 	scores[doc_index] = sc;	
		// }
		return scores;


	}




	std::vector<float> zone::query_transform_to_vector(const std::string& query)
	{

		std::cout << "\n---------------------------------------------------------------------------------------\n\n" << " zone: " << zone_name << " ---- query received: " << query <<  std::endl;
		

		std::vector<float> query_vector(matrix.cols,0);

		std::regex re("[\\|\n\t,:;(){.} ]+|[-]");
        std::sregex_token_iterator first{query.begin(), query.end(), re, -1}, last;//the '-1' is what makes the regex split (-1 := what was not matched)
        std::vector<std::string> tokens_temp{first, last}; //might conatain few empty strings bcoz of regex splitting...
		std::vector<std::string> query_tokens;
		for(int i = 0;i<tokens_temp.size();i++)
		{
				if(tokens_temp[i] == std::string(""))
				   continue;
			    query_tokens.push_back(tokens_temp[i]);
		}


		std::cout << " Query words modified as: " << std::endl;

		#pragma omp parallel for //added NEW line...
		for(auto query_word : query_tokens)
		{	
			//std::cout << query_word << std::endl;
			std::transform(query_word.begin(), query_word.end(), query_word.begin(), ::tolower);

			if(stem == true)
			{
				Porter2Stemmer::trim(query_word);
        		Porter2Stemmer::stem(query_word);
			}
			
			std::string print_query_word = query_word + "  ";
			std::cout << print_query_word;

			//std::cout << query_word << "  "; //modified as sometimes 2 words printed together without space

        	if(query_word == "" ) 
              	continue;  

			//check if query word is in the vocab, if yes get the corr. index in vocab
			if(vocabulary.find(query_word) == vocabulary.end())
			{	
				//std::cout << "query word not in vocab" << std::endl;
				continue;
			}

			int query_word_index = vocabulary.at(query_word);
			//std::cout << "query word: " << query_word << "  index in vocab: " << query_word_index << std::endl;

			#pragma omp atomic
			query_vector[query_word_index] = query_vector[query_word_index] + 1;

		}
		std::cout << std::endl;
		


		//Now convert this raw tf to logtf-idf
		#pragma omp parallel for
		for(int i=0;i<query_vector.size();i++)
		{
			if(query_vector[i] != 0)
			{
				query_vector[i] = ( 1 + std::log10(double(query_vector[i])) )*( std::log10(matrix.rows / doc_freq[i]) );
				//std::cout << "wt: " << query_vector[i] << "word_index in vocab: " << i << std::endl;
			}
		}

		
		return query_vector;
	}




	std::vector<float> zone::query_handler(const std::string& query)
	{	
		

		std::vector<float> query_vector = this->query_transform_to_vector(query);

		std::vector<float> scores(matrix.rows,0);
		#pragma omp parallel for
		for(int doc_index =0; doc_index < matrix.rows ;doc_index ++)
		{
			float sc = cosine_similarity(doc_index,query_vector);
			scores[doc_index] = sc;	
		}

		
		return scores;


	}
