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
#include <utility>
#include <omp.h>
#include<chrono>
#include <iomanip>  
#include<cmath>	
#include"myheader.h"



	void zone::initialize_centroids()
	{
		//get K seed points
		//int array[num_clusters];
		int* array = (int*)malloc(num_clusters*sizeof(int));
		for(int i=0;i< num_clusters ;i++)
		{
				array[i] = i;
		}

		//initialize centroids
		#pragma omp parallel for //all local variables implicitly private
		for(int i=0;i< num_clusters;i++)
		{
			int doc_index = array[i];
			int start_ind_col_arr = matrix.row_ptrs[doc_index];
			int end_ind_col_arr = matrix.row_ptrs[doc_index + 1];

			#pragma omp parallel for 
			for(int j = start_ind_col_arr; j < end_ind_col_arr;j++)
			{
				int word_ind = matrix.col_ind[j];
				centroids[i][word_ind] = 0;
				int tf = matrix.term_freq[j]; //log tf idf actually
				if(tf == 0)
					centroids[i][word_ind] = 0;
				else
				{
					centroids[i][word_ind] =  ( 1 + std::log10(double(tf)) )*std::log10(matrix.rows /doc_freq[word_ind]);
					
				}
				
			}


		}

		free(array);

	}


	void zone::assign_docs_to_clusters(std::vector<int> & cluster_for_docs)
	{
		       #pragma omp parallel for	
					for(int doc_index = 0; doc_index < matrix.rows ; doc_index++)
					{	
						int cluster_index = 0;
						float max_sim = 0;

						// #pragma omp parallel for reduction(max:max_sim) reduction(max:cluster_index) //lNEW----line added//might produce a wrong result...
						// for(int i = 0; i < centroids.size();i++)
						// {		
						// 		float sim = this->cosine_similarity(doc_index,centroids[i]);
						// 		//std::cout << "sim: " << sim << std::endl;
						// 		if(sim > max_sim)
						// 			{
						// 				cluster_index = i;
						// 				max_sim = sim;
						// 			}
						// }


						#pragma omp parallel for //lNEW----line added
						for(int i = 0; i < centroids.size();i++)
						{		
								float sim = this->cosine_similarity(doc_index,centroids[i]);
								//std::cout << "sim: " << sim << std::endl;

								#pragma omp critical
								if(sim > max_sim)
									{
										cluster_index = i;
										max_sim = sim;
									}
						}

						//std::cout << "Cluster alloted to doc- " << doc_index << " is: " << cluster_index << std::endl;
						cluster_for_docs[doc_index] = cluster_index;
						//clusters[cluster_index].insert(doc_index); 
					
					}


	}


	void zone::recompute_centroids()
	{

		#pragma omp parallel for
					for(int i=0;i<clusters.size();i++)
					{	
						//make centroids[i] -> all zero
						std::fill(centroids[i].begin(), centroids[i].end(), 0);

						#pragma omp parallel
							#pragma omp single
									
									for(auto itr = clusters[i].begin(); itr != clusters[i].end(); itr++)	
									{	
										#pragma omp task
										{
												int doc_index = *itr;
												int start_ind_col_arr = matrix.row_ptrs[doc_index];
												int end_ind_col_arr = matrix.row_ptrs[doc_index + 1];


												#pragma omp parallel for
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

													#pragma omp atomic
													centroids[i][word_ind] = centroids[i][word_ind] + logtf_idf/clusters[i].size();	
												}	
										}
										
									}			
					}

	}



void zone::Print_cluster_info(const std::vector<std::string>& DATASET)
	{

			

 			std::ofstream myfile;
 			myfile.open("RESULTS/clustering_results_zone-" + zone_name + ".txt");

			myfile << " Clustering information for zone: " << zone_name << std::endl;

			myfile << " Total number of clusters - " << num_clusters << std::endl;

			for(int i = 0; i< clusters.size(); i++)
			{
				myfile << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
				myfile << " Cluster - " << i << std::endl ;
				std::vector<std::pair<float,int>> centroid_scores(matrix.cols);
				for(int l = 0; l < matrix.cols ; l ++)
				{
					centroid_scores[l].first = centroids[i][l];
					centroid_scores[l].second = l;
				}

				std::sort(centroid_scores.rbegin(),centroid_scores.rend());
				


				myfile << "Important terms are:" << std::endl;
				for(int j = 0 ; j < 5  && j < matrix.cols; j++)
				{	
					int term_ind = centroid_scores[j].second;
					std::string s;
					for(auto a : vocabulary)
					{
						if(a.second == term_ind)
						{	
							s = a.first;
							break;
						}
					}

					if(centroid_scores[j].first > 0)
						myfile << " term in vocab:  " << s << "  weight: " << centroid_scores[j].first << std::endl;
				}


				myfile << " doc names are- " << std::endl;
				for(int doc_id : clusters[i])
				{
					myfile << DATASET[doc_id] << std::endl;
				}
			}

			myfile.close();

	}

	void zone::K_Means_Clustering(const std::vector<std::string>& DATASET,int K,bool print_info)
	{	
		

		auto start = std::chrono::high_resolution_clock::now();

		centroids.clear();
		clusters.clear();
		if(print_info == true){
			std::cout << "\n#############################################################################################################" << std::endl;
			std::cout << "\n Number of clusters to be formed: " << K << std::endl;
			std::cout << "\n\n start cluserting for zone - " << zone_name << std::endl;
		    std::cout << "\n vocab size: " << matrix.cols << std::endl;
		}
		
		num_clusters = K;
		for(int i = 0; i < num_clusters;i++)
		{
			std::vector<float> temp(matrix.cols,0);
			centroids.push_back(temp);
			clusters.push_back(std::unordered_set<int>{});
		}




		 //std::cout << "\n\n allocated mem for centroids: " << zone_name  << std::endl;
		if(print_info == true)
			std::cout << "\n initializing centroids: " << zone_name <<  std::endl;
		
		initialize_centroids();

		if(print_info == true)
			std::cout << "\n start iterations to get clusters: " << zone_name << std::endl;
		//once centroids are initialized, start:

		std::vector<int> cluster_for_docs(matrix.rows);

		int iter = 0;
		while(true){

					//empty and fill cluster sets
					std::vector<std::unordered_set<int>> previous_clusters = clusters;

					for(int i=0;i<clusters.size();i++)
						clusters[i].clear();

					//assign cluster index to each doc
					//assign_docs_to_clusters_gpu(cluster_for_docs);
					assign_docs_to_clusters(cluster_for_docs);

					for(int i = 0; i< cluster_for_docs.size();i++)
						clusters[cluster_for_docs[i]].insert(i);

					iter++;

					if(print_info == true)
					 std::cout << "  Iteration number: " << iter << " for zone- " << zone_name <<  std::endl;

					if(previous_clusters == clusters )
					{	
						std::cout << "\nTotal Iterations happened: " << iter <<  " for zone- " << zone_name <<  std::endl;
						break;
					}

					//recompute centroids
					recompute_centroids();
					//recompute_centroids_gpu();
					//std::cout << "\n Recompute centorids:" << std::endl;
					

					//std::cout << "\n All centroids updated\n" << std::endl;
		}


		auto stop = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	

		if(print_info == true)
		{	
			
    		std::cout << " Time taken to cluster docs on basis of " << zone_name << " is: "  << (double)duration.count(); 
    		std::cout << " milliseconds " << std::endl; 
			std::cout << " Finally clustering over for zone: " << zone_name << std::endl;
			Print_cluster_info(DATASET);
			
		}
		

	}


	void zone::elbow_method(int start,int end,const std::vector<std::string>& DATASET)
	{

		std::ofstream myfile;
		myfile.open("RESULTS/Elbow_Method_zone-" + zone_name + ".txt");

		std::cout << "\n elbow method for zone: " << zone_name << std::endl;
		myfile << "\n elbow method for zone: " << zone_name << std::endl;


		//why can't we do stuff for different k in parallel here??? by using firstprivate(centroids,clusters) 
		//#pragma omp parallel for firstprivate(clusters) firstprivate(centroids) --->produces error: double free or corruption.
		//bcoz in K-means clustering, all threads modilfy the same centorid and clusters... I guess... How to prevent that???
		for(int k = start ;  k<= end ;k++)
		{	

			std::cout << "\n------------------------------------------------------------------------------\n K value = " << k << std::endl;
			myfile << "\n------------------------------------------------------------------------------\n K value = " << k << std::endl;
			
			float average_dissim = 0;
			this->K_Means_Clustering(DATASET,k,false);

			// #pragma omp parallel for     //also works...
			// for(int i = 0; i < clusters.size();i++)
			// {	
			// 	#pragma omp parallel
			// 	#pragma omp single
			// 	for(auto itr = clusters[i].begin(); itr != clusters[i].end();itr++)
			// 	{	
			// 		#pragma omp task
			// 			#pragma omp atomic
			// 			average_dissim += 1 - cosine_similarity(*itr,centroids[i]);
						
					
			// 	}

				
			// }

			#pragma omp parallel
			#pragma omp single
			for(int i = 0; i < clusters.size();i++)
			{	
				
				for(auto itr = clusters[i].begin(); itr != clusters[i].end();itr++)
				{	
					#pragma omp task
						#pragma omp atomic
						average_dissim += 1 - cosine_similarity(*itr,centroids[i]);
						
					
				}

				
			}



			// #pragma omp parallel for reduction(+: average_dissim)
			// for(int i = 0; i < clusters.size();i++)
			// {	
			// 	double cluster_dissim = 0;

			// 	#pragma omp parallel for reduction(+: cluster_dissim)  //This is producing an error
			// 	for(auto itr = clusters[i].begin(); itr != clusters[i].end();itr++)
			// 	{	
			// 			cluster_dissim += 1 - cosine_similarity(*itr,centroids[i]);
						
			// 	}

			// 	average_dissim += cluster_dissim;

			// }


			std::cout << "  average dissimilarity is: " << average_dissim << std::endl;
			myfile << "  average dissimilarity is: " << average_dissim << std::endl;
		}	
		
		myfile.close();
		
	}
