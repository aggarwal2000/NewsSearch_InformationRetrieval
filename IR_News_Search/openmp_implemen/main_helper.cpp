#include<iostream>
#include<string>
#include<vector>
#include "dirent.h"
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
#include <iomanip>
#include<chrono>  

#include"myheader.h"

std::vector<std::vector<std::string>> Tokenizer(const std::string& doc_name);

void Fill_dataset(std::vector<std::string>& DATASET)
{
    struct dirent *de;  // Pointer for directory entry 
  
    // opendir() returns a pointer of DIR type.  
    DIR *dr = opendir("../news_docs"); 
  
    if (dr == NULL)  // opendir returns NULL if couldn't open directory 
    { 
        printf("Could not open news_docs directory" ); 
        exit(0);
    } 
  
    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html 
    // for readdir() 
    while ((de = readdir(dr)) != NULL)
    {    
         if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0 )   
            continue;

         DATASET.push_back(std::string("../news_docs/") + std::string(de->d_name));
    }
           
    closedir(dr);   
}



void Fill_StopWordSet(std::unordered_set<std::string>& STOP_WORD_SET)
{
    std::ifstream fin;
    fin.open("../STOP_WORDS_LIST.txt");
    std::string line;
    while(std::getline(fin,line)) //read the file line by line...
    {   
        if(line == "" || line[0] == '#')
         continue;

        STOP_WORD_SET.insert(line);
    }    
    fin.close();
}



std::vector<zone> create_zonal_structures(const std::vector<std::string>& DATASET,const std::unordered_set<std::string>& STOP_WORD_SET,bool & clustering)
{	
	std::ofstream myfile;
	myfile.open("RESULTS/Doc-names-index-MAP.txt");
	myfile << " Doc names-index mapping " << std::endl;
	myfile << " Total number of docs- " << DATASET.size() << std::endl << std::endl;
	for(int i=0;i<DATASET.size();i++)
		myfile << DATASET[i] << " indexed as " << i << std::endl;
	myfile.close();


	auto start = std::chrono::high_resolution_clock::now();

	zone  title_zone{"TITLE",true,true};
	zone publication_zone{"PUBLICATION",false,false};
	zone author_zone{"AUTHOR",false,false};
	zone content_zone{"CONTENT",true,true};

	std::vector<zone> zonal_structures;

//parallelism here makes the execution slow
	// #pragma omp parallel
	// {

	// 	#pragma omp single 
	// 	{
	// 		for(int i = 0; i< DATASET.size();i++)
	// 		{
	// 			std::vector<std::vector<std::string>> tokens_all = Tokenizer(DATASET[i]);

	// 			#pragma omp task 
	// 			{	
	// 				title_zone.fill_mat(tokens_all[0],i,STOP_WORD_SET);
	// 			}
	// 			#pragma omp task 
	// 			{	
	// 				publication_zone.fill_mat(tokens_all[1],i,STOP_WORD_SET);	
	// 			}
	// 			#pragma omp task 
	// 			{	
	// 				author_zone.fill_mat(tokens_all[2],i,STOP_WORD_SET);
	// 			}
	// 			#pragma omp task 
	// 			{	
	// 				content_zone.fill_mat(tokens_all[3],i,STOP_WORD_SET);
	// 			}

	// 			#pragma omp taskwait	
				
	// 		}

	// 	}

	// }


	
			
//parallelism here makes the execution slow				
	// for(int i = 0; i< DATASET.size();i++)
	// {
	// 	std::vector<std::vector<std::string>> tokens_all = Tokenizer(DATASET[i]);

	// #pragma omp parallel 
	// {	
 //    	#pragma omp sections
 //    	{
	//         #pragma omp section
	//         {
	
	//             title_zone.fill_mat(tokens_all[0],i,STOP_WORD_SET);
	//         }
	        
	//         #pragma omp section
	//         {
	           
	//             publication_zone.fill_mat(tokens_all[1],i,STOP_WORD_SET);

	//         }

	//         #pragma omp section
	//         {
	           
	//             author_zone.fill_mat(tokens_all[2],i,STOP_WORD_SET);

	//         }

	//         #pragma omp section
	//         {
	           
	//             content_zone.fill_mat(tokens_all[3],i,STOP_WORD_SET);
	           

	//         }

 //    	}
	// }				

 //   }





	for(int i = 0; i< DATASET.size();i++)
	{
		std::vector<std::vector<std::string>> tokens_all = Tokenizer(DATASET[i]);

	        
    		title_zone.fill_mat(tokens_all[0],i,STOP_WORD_SET);
        
      
            publication_zone.fill_mat(tokens_all[1],i,STOP_WORD_SET);

        
            author_zone.fill_mat(tokens_all[2],i,STOP_WORD_SET);

        
            content_zone.fill_mat(tokens_all[3],i,STOP_WORD_SET);
   }

  
							

			

			
				
	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	

 
    std::cout << " Time taken to parse all docs and create zonal structures is:" << (double)duration.count()/(60); 
    std::cout << " minutes " << std::endl; 

	

	std::cout << "\n All docs parsed \n" << std::endl;
	std::cout << "Total number of docs in dataset: " << DATASET.size() << std::endl;
	std::cout << "Vocabulary size for zone- TITLE : "  <<  title_zone.vocabulary.size() <<  std::endl;
	std::cout << "Vocabulary size for zone- PUBLICATION : " << publication_zone.vocabulary.size() << std::endl;
	std::cout << "Vocabulary size for zone- AUTHOR : " << author_zone.vocabulary.size() << std::endl;
	std::cout << "Vocabulary size for zone- CONTENT : " << content_zone.vocabulary.size() << std::endl;
	title_zone.Print();
	publication_zone.Print();
	author_zone.Print();
	content_zone.Print();

	 
	int cluster_yes;
	std::cout << "Do you want to go for clustering:??? WARNING: It is very slow for cpu implementation(for dataset of over 1 lakh docs) If yes, enter 1: ";
	std::cin >> cluster_yes;

	if(cluster_yes == 1)
	{		
			clustering = true;


			int start_title = 1;
			int  end_title= 7;
			/* std::cout << "\n Enter starting k for elbow method - title:";
			std::cin >> start_title;
			std::cout << "\n Enter ending k for elbow method - title:";
			std::cin >> end_title;*/
			title_zone.elbow_method(start_title,end_title,DATASET);
			//title_zone.elbow_method_gpu(start_title,end_title,DATASET);
			int num_clusters_title;
			std::cout << "\nEnter number of clusters for zone-title: ";
			std::cin >> num_clusters_title;
			title_zone.K_Means_Clustering(DATASET,num_clusters_title,true);



			int start_publ = 1;
			int end_publ= 7;
			/*std::cout << "\n Enter starting k for elbow method -publication:";
			std::cin >> start_publ;
			std::cout << "\n Enter ending k for elbow method - publication:";
			std::cin >> end_publ;*/
			publication_zone.elbow_method(start_publ,end_publ,DATASET);
			//publication_zone.elbow_method_gpu(start_publ,end_publ,DATASET);
			int num_clusters_publ;
			std::cout << "\nEnter number of clusters for zone-publication: ";
			std::cin >> num_clusters_publ;
			publication_zone.K_Means_Clustering(DATASET,num_clusters_publ,true);



			int start_auth = 1;
			int end_auth = 7;
			/*std::cout << "\n Enter starting k for elbow method -author:";
			std::cin >> start_auth;
			std::cout << "\n Enter ending k for elbow method - author:";
			std::cin >> end_auth;*/
			author_zone.elbow_method(start_auth,end_auth,DATASET);
			//author_zone.elbow_method_gpu(start_auth,end_auth,DATASET);
			int num_clusters_author;
			std::cout << "\nEnter number of clusters for zone-author: ";
			std::cin >> num_clusters_author;
			author_zone.K_Means_Clustering(DATASET,num_clusters_author,true);

			/*
			int start_con, end_con;
			std::cout << "\n Enter starting k for elbow method -content:";
			std::cin >> start_con;
			std::cout << "\n Enter ending k for elbow method - content:";
			std::cin >> end_con;
			content_zone.elbow_method(start_con,end_con,DATASET);
			int num_clusters_content;
			std::cout <<"\nEnter number of clusters for zone-content:";
			std::cin >> num_clusters_content;
			content_zone.K_Means_Clustering(DATASET,num_clusters_content,true); */

			
			
			
			
			


	}
	else
	{
		clustering = false;
	}

	


	zonal_structures.push_back(title_zone);
	zonal_structures.push_back(publication_zone);
	zonal_structures.push_back(author_zone);
	zonal_structures.push_back(content_zone);

	return zonal_structures;

}




void Retrieve(const std::vector<std::string>& DATASET, std::vector<zone>& zonal_structures, const std::vector<std::string>& query,int k, bool cluster_yes,const std::string & fname )
{
	

	int cluster_based_retrieval = 0;

	if(cluster_yes == true){
		std::cout << "\nDo you want to go for cluster based retrieval ?? If yes, enter 1: ";
		std::cin >> cluster_based_retrieval;

	}
	
	std::ofstream myfile;
	myfile.open(fname);

	std::vector<std::pair<float,int>> total(DATASET.size());
	

	if(cluster_based_retrieval == 1)
		myfile << " Cluster based retrieval \n" << std::endl;
	else
		myfile << "Exact top k Retrieval\n" << std::endl;

	
	auto start = std::chrono::high_resolution_clock::now();

	for(int i = 0; i< zonal_structures.size();i++)
	{
		std::string query_for_zone = query[i];

		std::vector<float> temp(DATASET.size(),0);

		if(cluster_based_retrieval == 1 && zonal_structures[i].zone_name != std::string("CONTENT"))
		{
			
			temp = zonal_structures[i].query_handler_cluster_based(query_for_zone);
			//temp = zonal_structures[i].query_handler_cluster_based_gpu(query_for_zone);
			
		}
		else
			{	
				
		    temp = zonal_structures[i].query_handler(query_for_zone);
		    //temp =  zonal_structures[i].query_handler_gpu(query_for_zone);
		   

		   
		}

		myfile << " zone : " << zonal_structures[i].zone_name << ", free text query is: " << query_for_zone << std::endl;
		
		#pragma omp parallel for
			for(int k = 0; k < DATASET.size();k++)
			{
				//total_score[k] = total_score[k] + temp[k];
				total[k].first = total[k].first + temp[k]; //Note: Equally weighting all zones....
				total[k].second = k;
			}

			
			
	}

	std::sort(total.rbegin(),total.rend());

	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	

	myfile << "Top " << k << " docs are :" << std::endl;
	for(int i = 0 ; i < k ; i++)
	{
		std::cout << "doc  name: " << DATASET[total[i].second] << "  score: " << total[i].first << " out of 4.00 " << std::endl;
		myfile << "doc  name: " << DATASET[total[i].second] << "  score: " << total[i].first << " out of 4.00 " << std::endl;
	}


    std::cout << " The retrieval time is :" << (double)duration.count(); 
    std::cout << " milliseconds " << std::endl; 

    myfile << " The retrieval time is :" << (double)duration.count();
    myfile << " milliseconds " << std::endl; 


	myfile.close();
	
}




