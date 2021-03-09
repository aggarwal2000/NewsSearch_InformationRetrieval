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





void zone::Print_zonal_structure(){


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





__global__ void compute_doc_norms_kernel(int mat_rows, int mat_cols, int* mat_row_ptrs, int* mat_col_ind, 
	int* mat_term_freq, int* doc_freq, float* doc_norms)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

    int doc_ind = warp_index;

    if(doc_ind < mat_rows)
    {	
    	int id_within_warp = gid%32;

    	int start_col_ind_for_doc = mat_row_ptrs[doc_ind];
    	int end_col_ind_for_doc = mat_row_ptrs[doc_ind + 1];


    	float temp = 0;

    	for(int a = start_col_ind_for_doc + id_within_warp; a < end_col_ind_for_doc; a = a + 32)
    	{	
    		int tf = mat_term_freq[a];
    		int word_ind = mat_col_ind[a];
    		float weight = (1 + __log10f(tf))*(__log10f(mat_rows/doc_freq[word_ind]));
    		temp += weight*weight;
    	}

    	float val = temp;


	    for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }

        if(id_within_warp == 0)
        {
        	doc_norms[doc_ind] = sqrt(val);
        }

    }
}

void zone::Compute_doc_norms()
{
	dim3 block(THREADS_PER_BLOCK);

	dim3 grid(ceil((double)WARP_SIZE*matrix.rows/(double)THREADS_PER_BLOCK));

	cudaError_t err;
	compute_doc_norms_kernel<<< grid,block >>>(matrix.rows, matrix.cols, matrix.row_ptrs_gpu , matrix.col_ind_gpu, matrix.term_freq_gpu,
		doc_freq_gpu, doc_norms_gpu);

	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---dissim_kernel: %s\n", cudaGetErrorString(err));
}


void zone::Basic_Initialization() //doc_norms cpu, and all basic stuff(csr matrix, doc-freq arr, doc-norms ) on gpu
{

	//copy csr mat to gpu
	cudaError_t err;

	err = cudaMalloc((void**)&(this->matrix.row_ptrs_gpu),sizeof(int)*matrix.row_ptrs.size());
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));   


	err = cudaMalloc((void**)&(matrix.col_ind_gpu), sizeof(int)*matrix.col_ind.size());
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));   

	err = cudaMalloc((void**)&(matrix.term_freq_gpu), sizeof(int)*matrix.term_freq.size());
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));   

	err = cudaMalloc((void**)&doc_freq_gpu, sizeof(int)*matrix.cols);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));   
	




	err = cudaMemcpy( matrix.row_ptrs_gpu, &matrix.row_ptrs[0], sizeof(int)*matrix.row_ptrs.size(), cudaMemcpyHostToDevice);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err));   

	err = cudaMemcpy(matrix.col_ind_gpu, &matrix.col_ind[0], sizeof(int)*matrix.col_ind.size(), cudaMemcpyHostToDevice);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err));  

	err = cudaMemcpy(matrix.term_freq_gpu, &matrix.term_freq[0], sizeof(int)*matrix.term_freq.size(), cudaMemcpyHostToDevice);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err));  

	err = cudaMemcpy(doc_freq_gpu, &doc_freq[0], sizeof(int)*matrix.cols, cudaMemcpyHostToDevice );
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err));  



	doc_norms = std::vector<float>(matrix.rows,0);
	     	
	err = cudaMalloc((void**)&doc_norms_gpu, sizeof(float)*matrix.rows);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));   

	Compute_doc_norms();
	
}


void zone::Basic_Deallocation()
{
	cudaFree(matrix.row_ptrs_gpu);
	cudaFree(matrix.col_ind_gpu);
	cudaFree(matrix.term_freq_gpu);
	cudaFree(doc_freq_gpu);
	cudaFree(doc_norms_gpu);
}