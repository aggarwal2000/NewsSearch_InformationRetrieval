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
	for(auto query_word : query_tokens)
	{	
		//std::cout << query_word << std::endl;
		std::transform(query_word.begin(), query_word.end(), query_word.begin(), ::tolower);

		if(stem == true)
		{
			Porter2Stemmer::trim(query_word);
    		Porter2Stemmer::stem(query_word);
		}
	
		std::cout << query_word << "  ";

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

		query_vector[query_word_index] = query_vector[query_word_index] + 1;

	}
	std::cout << std::endl;
	


	//Now convert this raw tf to logtf-idf
	
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



//---------------------------------------------------------------------------------------------------------------

__device__ void block_reduce_query(float* data) //defined here due to linkage problems...
{
    int nt = blockDim.x;
    int tid = threadIdx.x;

    for (int k = nt / 2; k > 0; k = k / 2)
    {
        __syncthreads();
        if (tid < k)
        {
            data[tid] += data[tid + k];
        }
    }
}

__global__ void reduction_kernel_2(int length, float* array, float* ans); //already defined in K_Means_Clustering.cu

// __global__ void reduction_kernel_2(int length, float* array, float* ans)
// {
// 	int nt = blockDim.x;
//     int tid = threadIdx.x;

//     float temp = 0;
//     for (int i = tid; i < length; i += nt)
//     {
//         temp += array[i];
//     }

//     __shared__ float tmp_work[THREADS_PER_BLOCK];
//     tmp_work[tid] = temp;

//     __syncthreads();
//     block_reduce(tmp_work);

//     if (tid == 0)
//         *ans = tmp_work[0];

// }



__global__ void norm_sqr_kernel_1(int arr_length , float* Arr, float* buffer)
{	
	int dimgrid = gridDim.x;
    int dimblock = blockDim.x;

	int lid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    float tmp = 0;
    for (int i = gid; i < arr_length; i += dimgrid * dimblock)
    {
        tmp += Arr[i]*Arr[i];
    }

    __shared__ float tmp_work[THREADS_PER_BLOCK];
    tmp_work[lid] = tmp;

    __syncthreads();
    block_reduce_query(tmp_work);

    if (lid == 0)
        buffer[blockIdx.x] = tmp_work[0];
}



float zone::Compute_Query_Norm(float* query_vector_gpu, int length)
{	

	float ans;
    float* gpu_ans;
    cudaMalloc((void**)&gpu_ans, sizeof(float));

	dim3 block(THREADS_PER_BLOCK);
	int work_per_thread = 4;
	int gridsize = ceil((double)length / (double)(THREADS_PER_BLOCK * work_per_thread));
	dim3 grid(gridsize);


	float* gpu_buffer;
    int buffer_size = gridsize;
    cudaMalloc((void**)&gpu_buffer, buffer_size * sizeof(float));

	norm_sqr_kernel_1<<< grid, block >>>(length , query_vector_gpu, gpu_buffer);
	reduction_kernel_2 <<< 1 , block >>>(gridsize , gpu_buffer , gpu_ans);

	cudaMemcpy(&ans, gpu_ans, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_buffer);
    cudaFree(gpu_ans);
    return sqrt(ans);
	
}

//--------------------------------------------------------------------------------------------------------------


// __global__ void Compute_cluster_query_score_kernel(int num_clusters, float* centroids, float* centroid_norms, int length, float* query_vec, float* cluster_query_score)
// {
// 	int gid = blockDim.x * blockIdx.x + threadIdx.x;

//     int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

//     int centroid_ind = warp_index;

//     if(centroid_ind < num_clusters)
//     {
//     	int id_within_warp = gid%32;

//     	float temp = 0;

//     	for(int a = id_within_warp ; a < length; a = a + 32)
//     	{
//     		temp += query_vec[a]*centroids[a + centroid_ind*length];
//     	}

//     	float val = temp;

//     	for (int offset = 16; offset > 0; offset /= 2)
//         {
//             val += __shfl_down_sync(FULL_MASK, val, offset);
//         }
//         //0th thread of warp has the sum now

//         if(id_within_warp == 0)
//         {
//         	if(centroid_norms[centroid_ind] > 0)
//         		val = val/centroid_norms[centroid_ind];

//         	cluster_query_score[centroid_ind] = val;

//         }

//     }

// }



__global__ void Compute_cluster_query_score_kernel_1(int mat_cols,int num_clusters ,float* centroids , float* query_vector, float* buffer)
{
	int dimgrid_x = gridDim.x;
    int dimblock = blockDim.x;

	int lid_x = threadIdx.x;
    int gid_x = blockDim.x * blockIdx.x + threadIdx.x;

    float tmp = 0;

    int block_id_y = blockIdx.y;

    for (int i = gid_x; i < mat_cols; i += dimgrid_x * dimblock)
    {	
        tmp += centroids[block_id_y*mat_cols + i]*query_vector[i];
    }

    __shared__ float tmp_work[THREADS_PER_BLOCK];
    tmp_work[lid_x] = tmp;

    __syncthreads();
    block_reduce_query(tmp_work);

    if (lid_x == 0)
        buffer[ dimgrid_x*block_id_y +  blockIdx.x] = tmp_work[0];
}


__global__ void Compute_cluster_query_score_kernel_2(int size_x ,int num_clusters, float* buffer,float* centroid_norms, float* cluster_query_score)
{
	int nt = blockDim.x;
	int tid = threadIdx.x;

	int block_id_y = blockIdx.y;

	float temp = 0;
	for(int i = tid; i < size_x; i += nt)
	{
		temp += buffer[block_id_y*size_x + i ];
	}

	__shared__ float tmp_work[THREADS_PER_BLOCK];
	tmp_work[tid] = temp;

    __syncthreads();
    block_reduce_query(tmp_work);

    if(tid == 0)
    {	
    	float score = tmp_work[0];

    	if(centroid_norms[block_id_y] != 0)
    		score = score/centroid_norms[block_id_y];

    	cluster_query_score[block_id_y] = score;
    }
}


//In efficient: dot_product way later.....
int zone::Find_closest_cluster_index(float* query_vector_gpu)
{	
	cudaError_t err;

	std::vector<float> cluster_query_score(num_clusters,0);

	float* cluster_query_score_gpu;
	err = cudaMalloc((void**)&cluster_query_score_gpu , sizeof(float)*num_clusters);
	if ( err != cudaSuccess )
	       printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));

	// dim3 block(THREADS_PER_BLOCK);
	// dim3 grid(ceil((double)WARP_SIZE*num_clusters/(double)THREADS_PER_BLOCK));

	// //per centroid --> one warp
	// Compute_cluster_query_score_kernel<<< grid, block>>>(num_clusters,centroids_gpu,centroid_norms_gpu,matrix.cols,query_vector_gpu, cluster_query_score_gpu); //modifies cluster_query_score_gpu



	
	dim3 block1(THREADS_PER_BLOCK);
	int work_per_thread = 4;
	int gridsize_x = ceil((double)matrix.cols / (double)(THREADS_PER_BLOCK * work_per_thread));
	dim3 grid1(gridsize_x,num_clusters);

	float* gpu_buffer;
    int buffer_size = gridsize_x*num_clusters;
    cudaMalloc((void**)&gpu_buffer, buffer_size * sizeof(float));

    dim3 grid2(1,num_clusters);

    Compute_cluster_query_score_kernel_1<<< grid1 , block1 >>> (matrix.cols, num_clusters , centroids_gpu,  query_vector_gpu, gpu_buffer);
    Compute_cluster_query_score_kernel_2<<< grid2 , block1 >>>(gridsize_x, num_clusters, gpu_buffer, centroid_norms_gpu,cluster_query_score_gpu);

    cudaFree(gpu_buffer);

	err = cudaMemcpy(&cluster_query_score[0], cluster_query_score_gpu , sizeof(float)*num_clusters, cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess )
	       printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err));

	

	int closest_cluster = 0;
	float max_score = 0;
	for(int i=0; i < num_clusters;i++)
	{
		if(max_score < cluster_query_score[i])
		{
			max_score = cluster_query_score[i];
			closest_cluster = i;
		}
	}   

	cudaFree(cluster_query_score_gpu);

	return closest_cluster;
}


__global__ void Query_zone_scores_cluster_based_kernel(float* zone_scores ,int mat_rows, int mat_cols, int* mat_row_ptrs , 
		int* mat_col_ind, int* mat_term_freq, int* doc_freq,  float* query_vector,float* doc_norms ,float query_norm,int* cluster_ind_for_docs,int cluster_ind)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

    int doc_ind = warp_index;

    if(doc_ind < mat_rows)
    {	
    	int id_within_warp = gid%32;

    	if(cluster_ind_for_docs[doc_ind] == cluster_ind)
    	{
    		int start_ind = mat_row_ptrs[doc_ind];

    		int end_ind = mat_row_ptrs[doc_ind + 1];

    		float temp = 0;

    		for(int a = start_ind + id_within_warp ; a < end_ind; a = a + 32)
    		{
    			int tf = mat_term_freq[a];
    			int word_ind = mat_col_ind[a];

    			float weight = (1 + __log10f(tf))*(__log10f(mat_rows/doc_freq[word_ind]));

    			temp += weight*query_vector[word_ind];
    		}


    		float val = temp;

    	for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        //0th thread of warp has the sum now


        if(id_within_warp == 0)
        {
        	if(doc_norms[doc_ind] > 0 && query_norm > 0)
        	{
        		val = val/(doc_norms[doc_ind]*query_norm);
        	}

        	zone_scores[doc_ind] = val;
        }


    	}
    	else
    	{
    		if(id_within_warp == 0)
    		   zone_scores[doc_ind] = 0;
    	}	
    }

}

void zone::Compute_query_zone_scores_cluster_based(float* zone_scores_gpu,float* query_vector_gpu, float query_norm, int closest_cluster)
{
	//per doc: one warp
	//just zero for docs: not in cluster

	dim3 block(THREADS_PER_BLOCK);

	dim3 grid(ceil((double)WARP_SIZE*matrix.rows/(double)THREADS_PER_BLOCK));

	Query_zone_scores_cluster_based_kernel<<< grid, block >>>(zone_scores_gpu,matrix.rows,matrix.cols,matrix.row_ptrs_gpu
		,matrix.col_ind_gpu, matrix.term_freq_gpu, doc_freq_gpu, query_vector_gpu, doc_norms_gpu, query_norm, cluster_ind_for_docs_gpu, closest_cluster);

}

float* zone::query_handler_cluster_based(const std::string& query){

	cudaError_t err;


	std::vector<float> query_vector = query_transform_to_vector(query);

	//allocate on gpu
	float* query_vector_gpu;
	err = cudaMalloc((void**)&query_vector_gpu, sizeof(float)*matrix.cols);
	if ( err != cudaSuccess )
	       printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));


	err = cudaMemcpy(query_vector_gpu , &query_vector[0] , sizeof(float)*matrix.cols , cudaMemcpyHostToDevice);
	if ( err != cudaSuccess )
	       printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err));

	float query_norm = Compute_Query_Norm(query_vector_gpu, matrix.cols);
	
	int closest_cluster = Find_closest_cluster_index(query_vector_gpu);   
	//once closest cluster is found...

	//Now within that cluster, compare docs with query....

	std::cout << "\n The closest cluster is: " << closest_cluster << " for zone: " << zone_name << std::endl;

	
	float* zone_scores_gpu;
	err = cudaMalloc((void**)&zone_scores_gpu, sizeof(float)*matrix.rows);	
	if ( err != cudaSuccess )
	       printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));	   

	Compute_query_zone_scores_cluster_based(zone_scores_gpu,query_vector_gpu,query_norm, closest_cluster); //modifies zone_scores_gpu  

	

	cudaFree(query_vector_gpu);


	return zone_scores_gpu;
	
}




//parallel redn---> query_norm , and centroid-query score....


//----------------------------------------------------------------------------------------------------------------


__global__ void Query_zone_scores_exact_kernel(float* zone_scores ,int mat_rows, int mat_cols, int* mat_row_ptrs , 
		int* mat_col_ind, int* mat_term_freq, int* doc_freq,  float* query_vector,float* doc_norms ,float query_norm)
{
	//printf("\nHi There");

	int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

    int doc_ind = warp_index;

    if(doc_ind < mat_rows)
    {	
    	int id_within_warp = gid%32;

    	int start_ind = mat_row_ptrs[doc_ind];

    	int end_ind = mat_row_ptrs[doc_ind + 1];

    	float temp = 0;

		for(int a = start_ind + id_within_warp ; a < end_ind; a = a + 32)
		{
			int tf = mat_term_freq[a];
			int word_ind = mat_col_ind[a];

			float weight = (1 + __log10f(tf))*(__log10f(mat_rows/doc_freq[word_ind]));

			temp += weight*query_vector[word_ind];
		}


    	float val = temp;

    	for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        //0th thread of warp has the sum now

        //printf("   val = %f\n",val);

        if(id_within_warp == 0)
        {
        	//printf("   val = %f\n",val);
        	if(doc_norms[doc_ind] > 0 && query_norm > 0)
        	{
        		val = val/(doc_norms[doc_ind]*query_norm);
        	}
        	//printf("   val = %f\n",val);
        	zone_scores[doc_ind] = val;
        }

	
    }

}

void zone::Compute_query_zone_scores_exact(float* zone_scores_gpu,float* query_vector_gpu, float query_norm)
{
	//per doc: one warp
	//just zero for docs: not in cluster

	dim3 block(THREADS_PER_BLOCK);

	dim3 grid(ceil((double)WARP_SIZE*matrix.rows/(double)THREADS_PER_BLOCK));

	cudaError_t err;

	Query_zone_scores_exact_kernel<<< grid, block >>>(zone_scores_gpu,matrix.rows,matrix.cols,matrix.row_ptrs_gpu
		,matrix.col_ind_gpu, matrix.term_freq_gpu, doc_freq_gpu, query_vector_gpu, doc_norms_gpu, query_norm);

	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---launch--_query_zone_scores_exact: %s\n", cudaGetErrorString(err));
}




float* zone::query_handler_exact(const std::string& query){

	cudaError_t err;


	std::vector<float> query_vector = query_transform_to_vector(query);

	//allocate on gpu
	float* query_vector_gpu;
	err = cudaMalloc((void**)&query_vector_gpu, sizeof(float)*matrix.cols);
	if ( err != cudaSuccess )
	       printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));


	err = cudaMemcpy(query_vector_gpu , &query_vector[0] , sizeof(float)*matrix.cols , cudaMemcpyHostToDevice);
	if ( err != cudaSuccess )
	       printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err));

	float query_norm = Compute_Query_Norm(query_vector_gpu, matrix.cols);
	

	
	float* zone_scores_gpu;
	err = cudaMalloc((void**)&zone_scores_gpu, sizeof(float)*matrix.rows);	
	if ( err != cudaSuccess )
	       printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));	   

	Compute_query_zone_scores_exact(zone_scores_gpu,query_vector_gpu,query_norm); //modifies zone_scores_gpu  

	cudaFree(query_vector_gpu);

	return zone_scores_gpu;
	
	
}

