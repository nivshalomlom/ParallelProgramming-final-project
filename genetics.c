#include "genetics.h"

// a method to generate all possiable mutations of a given letter
char* generateAllMutations(char letter)
{
	// create the output array
	int i = 0;	
	char* output = (char*)malloc(27 * sizeof(char));
	// iterate over each letter
	for (char c = 'A'; c <= 'Z'; c++)
	{
		// add it to the output if it's a mutation option
		char cmp = comparePair(letter, c);
		if (cmp != ':')
			output[i++] = c;
	}
	// trim the output string 
	output[i] = '\0';
	return output;
}

// a method to build a 'map' of all possiable mutations of a given sequence
// note: the map is defined as a 2d array the length of the sequence where each cell contains all possiable options for the letter
//       for example map[i] contains all possiable options for the i'th letter in the providede sequence 
char** generateMutationMap(char* sequence, int seqLen)
{
	char** output = (char**)malloc(seqLen * sizeof(char*));
#pragma omp parallel for
	for (int i = 0; i < seqLen; i++)
		output[i] = generateAllMutations(sequence[i]);
	return output;
}

// a method to find the sequence2 mutant that will provide the best result for the given sequence1, weights and direction
void evaluateAndMutate(char* sequence1, int len1, char* sequence2, int len2, float weights[], int direction, int proc_rank, int proc_num, MPI_Status status, const char* outputFilePath)
{

	// create mutation map and offset boundry
	char** map = generateMutationMap(sequence2, len2);
	
	// the mutent
	char* mutant = (char*)malloc(len2 * sizeof(char));

	if (proc_rank == MASTER)
	{
	
		// time stamp for measurement
		double timeStamp = MPI_Wtime();
	
		// the optimal values
		char* bestMutant = (char*)malloc(len2 * sizeof(char));
		int bestOffset = -1;
		float bestValue;

		int offset = 0;
		int maxOffset = len1 - len2 + 1;
		
		// start all slaves
		for (int i = 1; i < proc_num && offset < maxOffset; i++, offset++)
			MPI_Send(&offset, 1, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
		
		float result;
		int currentOffset;
		int answersRecived = 0;
		
		while (answersRecived < maxOffset)
		{
			
			// recive results from slaves
			MPI_Recv(mutant, len2, MPI_CHAR, MPI_ANY_SOURCE, WORK_TAG, MPI_COMM_WORLD, &status);
			int source = status.MPI_SOURCE;
			MPI_Recv(&result, 1, MPI_FLOAT, source, WORK_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(&currentOffset, 1, MPI_INT, source, WORK_TAG, MPI_COMM_WORLD, &status);
			
			answersRecived++;
			
			// check if better then top mutant
			if (bestOffset == -1 || bestValue < result)
			{
				strcpy(bestMutant, mutant);
				bestOffset = currentOffset;
				bestValue = result;
			}
			
			// send back work order if needed
			if (offset < maxOffset)
			{
				MPI_Send(&offset, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
				offset++;
			}
		}
		
		// kill all slave processess
		for (int i = 1; i < proc_num; i++)
			MPI_Send(&offset, 1, MPI_INT, i, KILL_TAG, MPI_COMM_WORLD);
	
		// trim best mutant to avoid printing garbage
		bestMutant[len2] = '\0';
	
		// output to file
		FILE *fp = fopen(outputFilePath, "w+");
   		fprintf(fp, "%s\n%d %lf\ntime: %lf", bestMutant, bestOffset, bestValue * direction, MPI_Wtime() - timeStamp);
   		fclose(fp);
   		
   		// free allocated space
   		free(bestMutant);
	}
	else
	{
		int offset;
		
		while (1)
		{
			// recive offset to work on
			MPI_Recv(&offset, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			
			// check if kill order
			if (status.MPI_TAG == KILL_TAG)
				break;
			
			// find the optimal mutation for the given offset, weights and direction
			float result = findOptimalMutation(map, sequence1, len1, mutant, len2, weights, offset, direction);
			
			// send back optimal mutant, result, and the offset this slave worked on
			MPI_Send(mutant, len2, MPI_CHAR, MASTER, WORK_TAG, MPI_COMM_WORLD);
			MPI_Send(&result, 1, MPI_FLOAT, MASTER, WORK_TAG, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, MASTER, WORK_TAG, MPI_COMM_WORLD);
		}
	}
	
	// free allocated space
	free(mutant);
   	for (int i = 0; i < len2; i++)
   		free(map[i]);
   	free(map);
	
}


