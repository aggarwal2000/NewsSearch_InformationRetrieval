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
#include<cmath>
#include<cassert>
#include <iomanip>  


#include"myheader.h"


void zone::Common_Allocation_for_clustering()
{
	cluster_ind_for_docs = std::vector<int>(matrix.rows, 0);

	cudaError_t err;

	err = cudaMalloc((void**)&cluster_ind_for_docs_gpu, sizeof(int)*matrix.rows);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));   
}

void zone::k_Allocation_for_clustering(int k) //basically depending on num_clusters
{	
	num_clusters = k;

	centroids = std::vector<float>(k*matrix.cols,0);

	cudaError_t err;
	err = cudaMalloc((void**)&centroids_gpu, sizeof(float)*matrix.cols*k);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));   


	centroid_norms = std::vector<float>(k,0);

	
	err = cudaMalloc((void**)&centroid_norms_gpu, sizeof(float)*k);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));  

	
	err = cudaMalloc((void**)&cardinality_clusters_gpu, sizeof(int)*k);
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err));  

	clusters.clear();
	for(int i=0; i < num_clusters;i++ )
	{
		clusters.push_back(std::unordered_set<int>{});
	}     	

}


void zone::k_Deallocation_for_clustering()
{	
	cudaFree(centroids_gpu);
	cudaFree(centroid_norms_gpu);
	cudaFree(cardinality_clusters_gpu);
}

void zone::Common_Deallocation_for_clustering()
{
	cudaFree(cluster_ind_for_docs_gpu);
}

//----------------------------------------------------------------------------------------------------------------

void zone::Elbow_method(const std::vector<std::string>& DATASET)
{

	int start = 1;
	int end = 7;

	float dissim = 0;

	std::ofstream myfile;
	myfile.open("RESULTS/Elbow_Method_zone-" + zone_name + ".txt");

	std::cout << "\n elbow method for zone: " << zone_name << std::endl;
	myfile << "\n elbow method for zone: " << zone_name << std::endl;


	for(int k = start; k <= end; k++)
	{	
		std::cout << "\n------------------------------------------------------------------------------\n K value = " << k << std::endl;
		myfile << "\n------------------------------------------------------------------------------\n K value = " << k << std::endl;	

		k_Allocation_for_clustering(k);

		Spherical_k_Means(k,DATASET,false);

		dissim = Average_Dissimilarity();

		
		std::cout << "  average dissimilarity is: " << dissim << std::endl;
		myfile << "  average dissimilarity is: " << dissim << std::endl;

		k_Deallocation_for_clustering();

	}

}

__global__ void dissimilarity_kernel(int mat_rows , int mat_cols, int* mat_row_ptrs ,int* mat_col_ind, int* mat_term_freq , 
		int* doc_freq ,float* centroids ,int num_clusters ,float* centroid_norms ,float* doc_norms ,int* cluster_ind_for_docs,float* doc_in_cluster_dissim)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

	int doc_ind = warp_index;

	if(doc_ind < mat_rows)
	{
		int id_within_warp = gid%32;

		int centroid_index = cluster_ind_for_docs[doc_ind];

		int start_ind_for_doc = mat_row_ptrs[doc_ind];

		int end_ind_for_doc = mat_row_ptrs[doc_ind + 1];

		float temp = 0;

		for(int i = start_ind_for_doc + id_within_warp ; i < end_ind_for_doc ; i = i + 32)
		{
			int tf = mat_term_freq[i];
			int word_ind = mat_col_ind[i];

			float weight = (1 + __log10f(tf))*(__log10f(mat_rows/doc_freq[word_ind]));

			temp += weight*centroids[centroid_index*mat_cols + word_ind];
		}

		float val = temp;

	    for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        //0th thread of warp has the sum now

	    if(id_within_warp == 0)
	    {		
	        if(centroid_norms[centroid_index] > 0 && doc_norms[doc_ind] > 0)
	    	  val = val/(centroid_norms[centroid_index]*doc_norms[doc_ind]);

        	float dissim = 1 - val;
        	doc_in_cluster_dissim[doc_ind] = dissim;
        }
	}
}


__device__ void block_reduce(float* data)
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

__global__ void reduction_kernel_1(int arr_length , float* doc_in_cluster_dissim, float* buffer)
{	
	int dimgrid = gridDim.x;
    int dimblock = blockDim.x;

	int lid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    float tmp = 0;
    for (int i = gid; i < arr_length; i += dimgrid * dimblock)
    {
        tmp += doc_in_cluster_dissim[i];
    }

    __shared__ float tmp_work[THREADS_PER_BLOCK];
    tmp_work[lid] = tmp;

    __syncthreads();
    block_reduce(tmp_work);

    if (lid == 0)
        buffer[blockIdx.x] = tmp_work[0];
}


__global__ void reduction_kernel_2(int length, float* array, float* ans)
{
	int nt = blockDim.x;
    int tid = threadIdx.x;

    float temp = 0;
    for (int i = tid; i < length; i += nt)
    {
        temp += array[i];
    }

    __shared__ float tmp_work[THREADS_PER_BLOCK];
    tmp_work[tid] = temp;

    __syncthreads();
    block_reduce(tmp_work);

    if (tid == 0)
        *ans = tmp_work[0];

}



float zone::Parallel_reduction(float* doc_in_cluster_dissim_gpu , int arr_length)
{
	float ans;
    float* gpu_ans;
    cudaMalloc((void**)&gpu_ans, sizeof(float));

	dim3 block(THREADS_PER_BLOCK);
	int work_per_thread = 4;
	int gridsize = ceil((double)arr_length / (double)(THREADS_PER_BLOCK * work_per_thread));
	dim3 grid(gridsize);


	float* gpu_buffer;
    int buffer_size = gridsize;
    cudaMalloc((void**)&gpu_buffer, buffer_size * sizeof(float));

	reduction_kernel_1<<< grid, block >>>(arr_length , doc_in_cluster_dissim_gpu, gpu_buffer);
	reduction_kernel_2 <<< 1 , block >>>(gridsize , gpu_buffer , gpu_ans);

	cudaMemcpy(&ans, gpu_ans, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_buffer);
    cudaFree(gpu_ans);
    return ans;

}



float zone::Average_Dissimilarity()
{	
	cudaError_t err;

	float* doc_in_cluster_dissim_gpu;


	err = cudaMalloc((void**)&doc_in_cluster_dissim_gpu, matrix.rows*sizeof(float));
	if ( err != cudaSuccess )
	     		printf("CUDA Error-memalloc: %s\n", cudaGetErrorString(err)); 

	//call kernel to fill it  
	dim3 block(THREADS_PER_BLOCK);
	dim3 grid(ceil( (double)WARP_SIZE*matrix.rows / (double)THREADS_PER_BLOCK ));
	
	//per doc- respective centroid: one warp
	dissimilarity_kernel<<< grid,block>>>( matrix.rows , matrix.cols, matrix.row_ptrs_gpu , matrix.col_ind_gpu , matrix.term_freq_gpu,
		doc_freq_gpu , centroids_gpu , num_clusters , centroid_norms_gpu , doc_norms_gpu , cluster_ind_for_docs_gpu , doc_in_cluster_dissim_gpu);

	
	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---dissim_kernel: %s\n", cudaGetErrorString(err));
	//Now add all dissims
		
    float sum = Parallel_reduction(doc_in_cluster_dissim_gpu, matrix.rows);
	
	cudaFree(doc_in_cluster_dissim_gpu);

	return sum;
}

//--------------------------------------------------------------------------------------------------------------

void zone::Clustering_docs(const std::vector<std::string>& DATASET)
{
	Common_Allocation_for_clustering();

	Elbow_method(DATASET);

	int k;
	std::cout << "\nPls enter the num of clusters for zone - " << zone_name << " : ";
	std::cin >> k;

	

	k_Allocation_for_clustering(k);

	Spherical_k_Means(k, DATASET , true);

}

//-----------------------------------------------------------------------------------------------------------

__global__ void make_centroids_zero_kernel(int num_clusters, int centroid_length, float* centroids)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

	int centroid_index = warp_index;

	if(centroid_index < num_clusters)
	{	
		int id_within_warp = gid % 32;
		for(int a = id_within_warp + centroid_index*centroid_length ; a < (centroid_index + 1)*centroid_length ; a = a + 32)
		{
			centroids[a] = 0;
		}
	}
	
}

void zone::Make_all_centroids_zero()
{
	dim3 block(THREADS_PER_BLOCK);

	dim3 grid( ceil(  (double)WARP_SIZE * num_clusters/(double)THREADS_PER_BLOCK ) );

	make_centroids_zero_kernel <<< grid, block >>> (num_clusters, matrix.cols, centroids_gpu); //per centroid -->one warp
	cudaError_t err;
	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---launch-make_zero: %s\n", cudaGetErrorString(err));

}





__global__ void initialize_centroids_kernel( int num_clusters,int mat_rows, int mat_cols, int* mat_row_ptrs,
 int* mat_col_ind, int* mat_term_freq, float* centroids, int* doc_freq)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

	int centroid_index = warp_index;

	if(centroid_index < num_clusters)
	{
		int doc_ind = centroid_index;

		int start_ind_for_doc = mat_row_ptrs[doc_ind];

		int end_ind_for_doc = mat_row_ptrs[doc_ind + 1];

		int id_within_warp = gid % WARP_SIZE;

		for(int k = start_ind_for_doc + id_within_warp; k < end_ind_for_doc; k = k + 32)
		{
			int tf = mat_term_freq[k];
			int word_ind = mat_col_ind[k];

			float weight = ( 1 + __log10f(tf))*(__log10f(mat_rows/doc_freq[word_ind]));

			centroids[centroid_index*mat_cols + word_ind] = weight;
		}
	}	
}



void zone::Initialize_centroids_with_docs()
{
	dim3 block(THREADS_PER_BLOCK);

	dim3 grid( ceil(  (double)WARP_SIZE * num_clusters/(double)THREADS_PER_BLOCK ) ); //per doc---> one warp

	initialize_centroids_kernel <<< grid, block >>>(num_clusters, matrix.rows, matrix.cols,matrix.row_ptrs_gpu, matrix.col_ind_gpu,
	 matrix.term_freq_gpu, centroids_gpu, doc_freq_gpu);

	cudaError_t err;
	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---launch-initialize_centroids: %s\n", cudaGetErrorString(err));
}

void zone::Choose_seed_points() //Choose first k docs as seed points
{
	//first make all the centroids_gpu zero

	//then a kernel for seed point initialization
	//std::cout << "Call make centrouds - 0 for seed points" << std::endl;
	Make_all_centroids_zero();



	Initialize_centroids_with_docs();

}

//--------------------------------------------------------------------------------------------------------------------


__global__ void centroid_norms_kernel1( int mat_cols,int  num_clusters ,float* centroids,float* buffer )
{	
	int dimgrid_x = gridDim.x;
    int dimblock = blockDim.x;

	int lid_x = threadIdx.x;
    int gid_x = blockDim.x * blockIdx.x + threadIdx.x;

    float tmp = 0;

    int block_id_y = blockIdx.y;

    for (int i = gid_x; i < mat_cols; i += dimgrid_x * dimblock)
    {
        tmp += centroids[block_id_y*mat_cols + i]*centroids[block_id_y*mat_cols + i];
    }

    __shared__ float tmp_work[THREADS_PER_BLOCK];
    tmp_work[lid_x] = tmp;

    __syncthreads();
    block_reduce(tmp_work);

    if (lid_x == 0)
        buffer[ dimgrid_x*block_id_y +  blockIdx.x] = tmp_work[0];
}


__global__ void centroid_norms_kernel2(int size_x ,int num_clusters, float* buffer,float* centroid_norms)
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
    block_reduce(tmp_work);

    if(tid == 0)
    	centroid_norms[block_id_y] = sqrt(tmp_work[0]);
}




void zone::Compute_centroid_norms()
{

	dim3 block(THREADS_PER_BLOCK);
	int work_per_thread = 4;
	int gridsize_x = ceil((double)matrix.cols / (double)(THREADS_PER_BLOCK * work_per_thread));
	dim3 grid(gridsize_x,num_clusters);

	float* gpu_buffer;
    int buffer_size = gridsize_x*num_clusters;
    cudaMalloc((void**)&gpu_buffer, buffer_size * sizeof(float));

    dim3 grid2(1,num_clusters);

    centroid_norms_kernel1 <<< grid , block >>> (matrix.cols, num_clusters , centroids_gpu, gpu_buffer);
    centroid_norms_kernel2 <<< grid2 , block >>>(gridsize_x, num_clusters, gpu_buffer, centroid_norms_gpu);

    cudaFree(gpu_buffer);

}



__global__ void assign_docs_to_clusters_kernel(int mat_rows, int mat_cols, int* mat_row_ptrs, int* mat_col_ind,
	int* mat_term_freq, int* doc_freq, int num_clusters, float* centroids, float* centroid_norms, int* cluster_ind_for_docs)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

	int doc_ind = warp_index;

	if(doc_ind < mat_rows)
	{
		float max_sim = 0;

		int id_within_warp = gid%32;
		int start_ind_for_doc = mat_row_ptrs[doc_ind];
		int end_ind_for_doc = mat_row_ptrs[doc_ind + 1];

		if(id_within_warp == 0)
		{
			cluster_ind_for_docs[doc_ind] = 0;
		}

		for(int centroid_index = 0; centroid_index < num_clusters; centroid_index ++)
		{	
			__syncwarp(); 

			float temp = 0;

			for(int i = start_ind_for_doc + id_within_warp; i < end_ind_for_doc; i = i + 32)
			{
				int tf = mat_term_freq[i];

				int word_ind = mat_col_ind[i];

				float weight = ( 1 + __log10f(tf))*(__log10f(mat_rows/doc_freq[word_ind]));

				temp += weight*centroids[centroid_index*mat_cols + word_ind]; 
			}

			float val = temp;

			__syncwarp();     // Is syncwarp really required ?                      


			
	        for (int offset = 16; offset > 0; offset /= 2)
	        {
	            val += __shfl_down_sync(FULL_MASK, val, offset);
	        }

	        //Now divide val by the norm of centroid vec
	        if(centroid_norms[centroid_index] > 0)
	        	val = val/centroid_norms[centroid_index];

	        if(id_within_warp == 0)
	        {
	        	if(val > max_sim)
	        	{
	        		max_sim = val;
	        		cluster_ind_for_docs[doc_ind] = centroid_index;
	        	}
	        }

		}

	}	
}



void zone::Assign_docs_to_clusters() //2 tasks: centroid_norms_gpu and then cluster_ind_for_docs_gpu
{
	Compute_centroid_norms();

	dim3 block(THREADS_PER_BLOCK);

	dim3 grid( ceil(  (double) WARP_SIZE*matrix.rows/(double)THREADS_PER_BLOCK ) );

	//per doc: one warp
	assign_docs_to_clusters_kernel <<< grid, block >>> (matrix.rows, matrix.cols, matrix.row_ptrs_gpu, matrix.col_ind_gpu,
		matrix.term_freq_gpu, doc_freq_gpu,  num_clusters, centroids_gpu, centroid_norms_gpu, cluster_ind_for_docs_gpu);
	
	cudaError_t err;
	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---launch-assign_docs_to_clusters: %s\n", cudaGetErrorString(err));
}





//----------------------------------------------------------------------------------------------------------------


__global__ void make_cardinality_clusters_zero_kernel(int* cardinality_clusters, int num_clusters)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gid < num_clusters)
	{
		cardinality_clusters[gid] = 0;
	}

}


__global__ void cardinality_clusters_kernel(int* cluster_ind_for_docs, int num_docs, int* cardinality_clusters )
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gid < num_docs)
	{
		int doc_ind = gid;

		int cluster_ind = cluster_ind_for_docs[doc_ind];

		atomicAdd(cardinality_clusters + cluster_ind, 1);

	}
}

void zone::Compute_cardinality_clusters()
{	cudaError_t err;

	dim3 block1(THREADS_PER_BLOCK);
	dim3 grid1(ceil( (double)num_clusters/(double)THREADS_PER_BLOCK ));
	make_cardinality_clusters_zero_kernel <<< grid1 , block1 >>>(cardinality_clusters_gpu, num_clusters);
	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---Launch cardinality_clusters_make_zero: %s\n", cudaGetErrorString(err));

	dim3 block2(THREADS_PER_BLOCK);

	dim3 grid2(ceil((double)matrix.rows/(double)THREADS_PER_BLOCK));

	
	cardinality_clusters_kernel<<< grid2, block2 >>>(cluster_ind_for_docs_gpu,matrix.rows,cardinality_clusters_gpu);
	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Error---Launch cardinality_clusters: %s\n", cudaGetErrorString(err));
}

__global__ void recompute_centroids_kernel(int mat_rows, int mat_cols, int* mat_row_ptrs, int* mat_col_ind, 
	int* mat_term_freq, int* doc_freq, int num_clusters , float* centroids , int* cluster_ind_for_docs , int* cardinality_clusters)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int warp_index = gid/WARP_SIZE;

	int doc_ind = warp_index;


	if(doc_ind < mat_rows)
	{
		int id_within_warp = gid%32;

		int start_ind_for_doc =  mat_row_ptrs[doc_ind];

		int end_ind_for_doc = mat_row_ptrs[doc_ind + 1];

		for(int a = start_ind_for_doc + id_within_warp; a < end_ind_for_doc; a = a + 32)
		{
			int tf = mat_term_freq[a];

			int word_ind = mat_col_ind[a];

			float weight = (1 + __log10f(tf))*(__log10f(mat_rows/doc_freq[word_ind]));



			int centroid_index = cluster_ind_for_docs[doc_ind];

			float weight_scaled = 0;

			if(cardinality_clusters[centroid_index] > 0)
				weight_scaled = weight/cardinality_clusters[centroid_index];
			
			//printf("\nweight_scaled: %f , weight: %f , mat_rows:%d , doc_freq[doc_ind]: %d ",weight_scaled , weight , mat_rows , doc_freq[word_ind] );

			atomicAdd(&centroids[centroid_index*mat_cols + word_ind], weight_scaled);
		}

	}
}

void zone::Recompute_centroids() // 2 tasks: cardinality_clusters_gpu and centroids_gpu
{	cudaError_t err;
	Compute_cardinality_clusters();
	//std::cout << "Call make centrouds - 0 from recomputation" << std::endl;
	Make_all_centroids_zero();

	dim3 block(THREADS_PER_BLOCK);

	// std::cout << WARP_SIZE << std::endl;
	// std::cout << matrix.rows << std::endl;
	// std::cout << WARP_SIZE*matrix.rows << std::endl;

	size_t num_bocks = ceil(  (double)WARP_SIZE*matrix.rows/ (double)THREADS_PER_BLOCK  );

	//std::cout << num_bocks << std::endl;

	dim3 grid(num_bocks);
	//std::cout << "Calling recomp kernel" << std::endl;
	//per row(doc) : one warp
	recompute_centroids_kernel <<< grid, block >>>( matrix.rows, matrix.cols, matrix.row_ptrs_gpu , matrix.col_ind_gpu ,
	matrix.term_freq_gpu, doc_freq_gpu, num_clusters , centroids_gpu , cluster_ind_for_docs_gpu , cardinality_clusters_gpu );
	
	// std::cout << block.x << " , " << block.y << "  , " << block.z << std::endl;
	// std::cout << grid.x << " , " << grid.y << " , " << grid.z << std::endl;
	err = cudaGetLastError();
	if ( err != cudaSuccess )
        printf("CUDA Launch Error---recompute_centroids_kernel: %s\n", cudaGetErrorString(err));
}


//----------------------------------------------------------------------------------------------------------------


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
					centroid_scores[l].first = centroids[i*matrix.cols + l];
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


//remember all centroids --> stored in order: one after the other. Does the other way storage have benefits???
void zone::Spherical_k_Means(int k,const std::vector<std::string>& DATASET, bool print_info)
{	


	auto start = std::chrono::high_resolution_clock::now();

		
	if(print_info == true){
		std::cout << "\n#############################################################################################################" << std::endl;
		std::cout << "\n Number of clusters to be formed: " << k << std::endl;
		std::cout << "\n\n start cluserting for zone - " << zone_name << std::endl;
	    std::cout << "\n vocab size: " << matrix.cols << std::endl;
	}
		
		

	if(print_info == true)
		std::cout << "\n start iterations to get clusters: " << zone_name << std::endl;
	//once centroids are initialized, start:




	num_clusters = k;
	cudaError_t err;

	std::vector<int>previous_cluster_ind_for_docs(matrix.rows,-1);

	if(print_info == true)
			std::cout << "\n initializing centroids: " << zone_name <<  std::endl;

	Choose_seed_points();//centroids_gpu is initialized

	int iter = 0;


	if(print_info == true)
		std::cout << "\n start iterations to get clusters: " << zone_name << std::endl;
	//once centroids are initialized, start:


	while(true)
	{
		

		Assign_docs_to_clusters(); // centroid_norms_gpu , cluster_ind_for_docs_gpu are modified

		//copy cluster_ind_for_docs_gpu to cluster_ind_for_docs : only one copy from gpu to cpu here
		err = cudaMemcpy(&cluster_ind_for_docs[0], cluster_ind_for_docs_gpu, sizeof(int)*matrix.rows, cudaMemcpyDeviceToHost);
		if ( err != cudaSuccess )
		   		printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err)); 


		iter++;		
		//std::cout << "Completed iter: " << iter << std::endl;

		if(print_info == true)
		 std::cout << "  Iteration number: " << iter << " for zone- " << zone_name <<  std::endl;

		if(previous_cluster_ind_for_docs == cluster_ind_for_docs)
		{
			std::cout << "\nFor k = " << num_clusters << "\n   Total number of iterations happened: " << iter << " for zone: " << zone_name <<  std::endl;
			break;
		}

		previous_cluster_ind_for_docs = cluster_ind_for_docs;

		Recompute_centroids(); //cardinality_clusters_gpu: obtain in prallel by traversing cluster_ind_for_docs_gpu
							   // centroids_gpu
		

	}

	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	//Once iterations are done, 
	//On cpu: only cluster_ind_for_docs updated, rest nothing else.//But we do need centroids for highly weighted terms, and 
	//Using that: we can form the clusters set.
	//No use centroid_norms on cpu, don't do anything with it. Let it be there, unupdated.

	//Now forming the clusters set.
	

	for(int i = 0; i < cluster_ind_for_docs.size();i++)
	{
		clusters[cluster_ind_for_docs[i]].insert(i);
	}


	err = cudaMemcpy(&centroids[0], centroids_gpu , sizeof(float)*num_clusters*matrix.cols , cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess )
		   		printf("CUDA Error-memcpy: %s\n", cudaGetErrorString(err)); 


	if(print_info == true)
	{	
		
		std::cout << " Time taken to cluster docs on basis of " << zone_name << " is: "  << (double)duration.count(); 
		std::cout << " milliseconds " << std::endl; 
		std::cout << " Finally clustering over for zone: " << zone_name << std::endl;
		Print_cluster_info(DATASET);
		
	}

	
}









//Now: centroid_norms in parallel, doc_norms in parallel and disisim array: sum in parallel.---->IMPORTANT
