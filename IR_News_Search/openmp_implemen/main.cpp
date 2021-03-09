#include<iostream>
#include<string>
#include<vector>
#include<cstring>
#include<unordered_set>
#include<stdlib.h>
#include <unordered_set>
#include <utility>

#include "myheader.h"


void Fill_dataset(std::vector<std::string>& DATASET);

void Fill_StopWordSet(std::unordered_set<std::string>& STOP_WORD_SET);

std::vector<zone> create_zonal_structures(const std::vector<std::string>& DATASET,const std::unordered_set<std::string>& STOP_WORD_SET,
	bool & clustering);


void Retrieve(const std::vector<std::string>& DATASET, std::vector<zone>& zonal_structures, const std::vector<std::string>& query,int k, bool cluster_yes,const std::string & fname );



int main()
{	

	std::vector<std::string> DATASET;
	std::unordered_set<std::string> STOP_WORD_SET;
	bool cluster_yes;
    Fill_dataset(DATASET);
	
    Fill_StopWordSet(STOP_WORD_SET);

    std::cout << "\n\n\n                                         NEWS SEARCH PROJECT                                      " << std::endl;


	std::vector<std::string> DATASET_short;
	for(int i=0;i< 50000 ;i++)
		DATASET_short.push_back(DATASET[i]); 

	std::vector<zone> zonal_structures = create_zonal_structures(DATASET ,STOP_WORD_SET,cluster_yes); //for each zone, creates csr matrix and vocab mapping, then applies k-mean clustering also.



	//all this later...
	std::string query_title{"mango juice favourite"};
	std::string query_publication{"buzzfeed new york news business"};
	std::string query_author{"david julius markus"};
	std::string query_content{"russia cricket match comments remarks house trump india"};

	int input;
	int k;

	int round = 0;

	while(1)
	{

		std::cout << "\nPress any number other than 0 to continue search: " ;
		std::cin >> input;
		if(input == 0)
			break;

		std::cin.ignore();

		std::cout << "\nEnter title: ";
		//std::cin >> query_title;
		std::getline(std::cin,query_title);

		std::cout << "\nEnter publication:";
		//std::cin >> query_publication;
		std::getline(std::cin,query_publication);


		std::cout << "\nEnter author: ";
		//std::cin >> query_author;
		std::getline(std::cin,query_author);


		std::cout << "\nEnter query_content:";
		//std::cin >> query_content;
		std::getline(std::cin,query_content);

		std::cout << "\nEnter number of top ranked docs to be retrieved:";
		std::cin >> k;

		std::vector<std::string> Query;

		Query.push_back(query_title);
		Query.push_back(query_publication);
		Query.push_back(query_author);
		Query.push_back(query_content);

		Retrieve(DATASET, zonal_structures, Query,k,cluster_yes,"RESULTS/RETRIEVAL_RESULTS-round-" + std::to_string(round) + ".txt"); //returns doc-indices.
		round++;
	}

	std::cout << "\n\n                                             Thank you                                                      \n\n\n";
	
	


}



