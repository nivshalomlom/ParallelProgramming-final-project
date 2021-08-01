#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string.h>
#include "geneCuda.h"

#define CONSERVATIVE_GROUPS "NDEQ|NEQK|STA|MILV|QHRK|NHQK|FYW|HY|MILF"
#define SEMI_CONSERVATIVE_GROUPS "SAG|ATV|CSA|SGND|STPA|STNK|NEQHRK|NDEQHK|SNDEQK|HFY|FVLIM"

// a macro for quick aloocating and copying ro device memory 
#define cudeMemloc(dst, src, size, type) {\
				size_t arrSize = size * sizeof(type);\
				cudaMalloc((void**)&dst, arrSize);\
				cudaMemcpy(dst, src, arrSize, cudaMemcpyHostToDevice); }\
				

__device__ __host__ int checkConsGroups(char letter1, char letter2, const char* conservativeGroups)
{
	int counter = 0;
	for (int i = 0; conservativeGroups[i]; i++)
	{
		if (conservativeGroups[i] == '|')
		{
			counter = 0;
			continue;
		}
		if (letter1 == conservativeGroups[i] || letter2 == conservativeGroups[i])
			counter++;
		if (counter >= 2)
			return 1;
	}
	return 0;
}

__device__ __host__ int checkSemiConsGroups(char letter1, char letter2, const char* semiConservativeGroups)
{
	int counter = 0;
	for (int i = 0; semiConservativeGroups[i]; i++)
	{
		if (semiConservativeGroups[i] == '|')
		{
			counter = 0;
			continue;
		}
		if (letter1 == semiConservativeGroups[i] || letter2 == semiConservativeGroups[i])
			counter++;
		if (counter >= 2)
			return 1;
	}
	return 0;
}

// a method to compare a pair of letters, returns '*' if in same conservative group, ':' if in same semi conservative group, otherwise ' '
__device__ __host__ char comparePairInternal(char letter1, char letter2)
{
	// check if letters are equal
	if (letter1 == letter2)
		return '*';
	// else check groups
	if (checkConsGroups(letter1, letter2, CONSERVATIVE_GROUPS))
		return ':';
	if (checkSemiConsGroups(letter1, letter2, SEMI_CONSERVATIVE_GROUPS))
		return '.';
	// return ' ' if no common group was found
	return ' ';
}

char comparePair(char letter1, char letter2)
{
	return comparePairInternal(letter1, letter2);
}

__global__ void findOptimalMutationInternal(char** map, char* seq1, char* mutant, float* weights, float* results, int* params)
{
	// get the unique id(e.g assigned letter)
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	int bestIndex = -1;

	int offset = params[0];
	int direction = params[1];

	// iterate over every possiable mutation of the assigned letter and pick the optimal one
	for (int j = 0; map[i][j]; j++)
	{
		float newValue = direction;
		char sign = comparePairInternal(map[i][j], seq1[i + offset]);
		if (sign == '*')
			newValue *= weights[0];
		else if (sign == ':')
			newValue *= -weights[1];
		else if (sign == '.')
			newValue *= -weights[2];
		else
			newValue *= -weights[3];
		
		if (bestIndex == -1 || results[i] < newValue)
		{
			bestIndex = j;
			results[i] = newValue;
		}
	}
	
	mutant[i] = map[i][bestIndex];
	
}

float findOptimalMutation(char** map, char* seq1, int len1, char* mutant, int len2, float* weights, int offset, int direction)
{
	// copy the input data into device memory
	char* tempMap[len2];
	for (int i = 0; i < len2; i++)
		cudeMemloc(tempMap[i], map[i], len2, char);
	
	char** cudaMap;
	cudeMemloc(cudaMap, tempMap, len2, char*);
	
	char* cudaSeq1;
	cudeMemloc(cudaSeq1, seq1, len1, char);
	
	char* cudaMutant;
	cudaMalloc((void**)&cudaMutant, len2);
	
	float* cudaWeights;
	cudeMemloc(cudaWeights, weights, 4, float);
	
	float* cudaResults;
	size_t size = len2 * sizeof(float);
	cudaMalloc((void**)&cudaResults, size);
	
	int params[2] = {offset, direction};
	
	int* cudaParmas;
	cudeMemloc(cudaParmas, params, 2, int);
	
	// start the kernal
	int blocksPerGrid = 1;
	int threadsPerBlock = len2;
	findOptimalMutationInternal<<<blocksPerGrid, threadsPerBlock>>>(cudaMap, cudaSeq1, cudaMutant, cudaWeights, cudaResults, cudaParmas);

	// get the results from the kernal
	cudaMemcpy(mutant, cudaMutant, len2, cudaMemcpyDeviceToHost);
	
	mutant[len2] = '\0';
	
	float results[len2];
	cudaMemcpy(results, cudaResults, size, cudaMemcpyDeviceToHost);
	
	// compute sum of results and return value
	float output = 0;
	for (int i = 0; i < len2; i++)
		output += results[i];
		
	return output;
}




