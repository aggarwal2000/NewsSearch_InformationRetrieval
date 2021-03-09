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






//for a zone, fill the csr matrix, and the vocabulary map.
	void zone::fill_mat(std::vector<std::string> tokens_doc, int doc_ref,const std::unordered_set<std::string>& STOP_WORD_SET)
	{	
		
		std::cout << zone_name << "  " << doc_ref << std::endl;
		
		matrix.rows++;
		int length_row = 0;
		for(auto token : tokens_doc){

			std::transform(token.begin(), token.end(), token.begin(), ::tolower);

			if(remove_stop_words == true)
			{
				if(STOP_WORD_SET.find(token) != STOP_WORD_SET.end()) //means present in the STOP WORDS SET
              		continue;   
			}

			if(stem == true)
			{
				Porter2Stemmer::trim(token);
        		Porter2Stemmer::stem(token);
			}

			
			
        	if( token == "" ) //means present in the STOP WORDS SET
              	continue;  

			//std::cout << token << "  ";

			//got the finished token
			//so insert in vocabulary if its not already there and insert in csr matrix too.

			if(vocabulary.find(token) == vocabulary.end())
			{
				vocabulary[token] = vocab_index;
				doc_freq.push_back(0);
				vocab_index++;
				matrix.cols++;
			}
			

			int token_index = vocabulary.at(token);

			int start_row_col_ind = matrix.row_ptrs[doc_ref];
			int flag = 0;


			for(int l = start_row_col_ind;l < matrix.col_ind.size();l++)
			{
				if(matrix.col_ind[l] == token_index)
					{
						matrix.term_freq[l]++;
						flag = 1;
						break;
					}
			}

			if(flag == 0){
				matrix.col_ind.push_back(token_index);
				doc_freq[token_index] += 1;
				matrix.term_freq.push_back(1);
				length_row++;
			}
			
		}

		length_current = length_current + length_row;
		matrix.row_ptrs.push_back(length_current);
		//std::cout << std::endl;

		//std::cout << " length:" << length_row << std::endl;

	}//each row of this matrix may not be sorted, have to sort it-OK-later

	//once the csr matrix is completely filled, do clustering.





void zone::Print(){


	std::ofstream myfile;
	myfile.open("RESULTS/doc_term_csr_matrix_zone-" + zone_name + ".txt");


	myfile << " zone - " << zone_name << std::endl;
	myfile << " Vocab size- " << vocabulary.size() << std::endl;

	myfile << " Vocab mapping:" << std::endl;
	 for (auto i : vocabulary)
	 {
		
		 myfile << i.first << "  -  " << " index:"<< i.second << " doc_freq:" <<  doc_freq[i.second] << std::endl;  
	 } 
	
	myfile << " csr matrix structure(matrix-->each row for a doc as per doc-index mapping) :" << std::endl;
	myfile << "num_rows:" << matrix.rows << "  num_cols:" << matrix.cols << std::endl;
	myfile << "row ptrs:" << std::endl;
	for(auto i : matrix.row_ptrs)
		myfile << i << "  ";
	myfile << "\n col ind:" << std::endl;
	for(auto i : matrix.col_ind)
		myfile<< i << "  ";
	myfile << "\n term freq:" << std::endl;
	for(auto i : matrix.term_freq)
		myfile << i << "  ";

	myfile.close();
        
}