#include "genetics.h"

#define SEQ1_SIZE 10000
#define SEQ2_SIZE 5000

void readFromFile(float* weights, char* seq1, char* seq2, int* direction, const char* inputFilePath);
void removeDashes(char* sequence);

int main(int argc, char** argv)
{
	
	int num_proc, proc_rank;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);	
	
	char seq1[SEQ1_SIZE];
	char seq2[SEQ2_SIZE];
	
	int len1 = SEQ1_SIZE;
	int len2 = SEQ2_SIZE;
	
	float weights[4];
	
	// the variable to define if were looking for a min or a max, if direction = 1 max, if direction = -1 min
	int direction;
	
	if (proc_rank == MASTER)
	{
	
		// read input data
		readFromFile(weights, seq1, seq2, &direction, "input.txt");
	
		// send the input data to each process
#pragma omp parallel for
		for (int i = 1; i < num_proc; i++)
		{
			MPI_Send(seq1, SEQ1_SIZE, MPI_CHAR, i, WORK_TAG, MPI_COMM_WORLD);
			MPI_Send(seq2, SEQ2_SIZE, MPI_CHAR, i, WORK_TAG, MPI_COMM_WORLD);
			MPI_Send(weights, 4, MPI_FLOAT, i, WORK_TAG, MPI_COMM_WORLD);
			MPI_Send(&direction, 1, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
		}
	
	}
	else
	{
		// recive input data from master
		MPI_Recv(seq1, SEQ1_SIZE, MPI_CHAR, MASTER, WORK_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(seq2, SEQ2_SIZE, MPI_CHAR, MASTER, WORK_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(weights, 4, MPI_FLOAT, MASTER, WORK_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&direction, 1, MPI_INT, MASTER, WORK_TAG, MPI_COMM_WORLD, &status);
	}
	
	// get length of sequences
	len1 = strlen(seq1);
	len2 = strlen(seq2);
	
	// start the computation
	evaluateAndMutate(seq1, len1, seq2, len2, weights, direction, proc_rank, num_proc, status, "output.txt");
	
	// close everything
	MPI_Finalize();
	
	return 0;
}

// a method to read the input from the file
void readFromFile(float* weights, char* seq1, char* seq2, int* direction, const char* inputFilePath)
{
	char buffer[8];
	FILE *fp = fopen(inputFilePath, "r");
   	if (fp)
   	{
   		fscanf(fp, "%f %f %f %f", &weights[0], &weights[1], &weights[2], &weights[3]);
   		fscanf(fp, "%s %s %s", seq1, seq2, buffer);
   		removeDashes(seq1);
   		removeDashes(seq2);
   		if (strcmp(buffer, "maximum") == 0)
   			*direction = 1;
   		else *direction = -1;
   		fclose(fp);
   	}
}

// a method to remove dashes from a given sequence
void removeDashes(char* sequence)
{	
	for (int i = 0, j = 0; sequence[i]; i++)
	{
		if (sequence[i + j] == '-')
			j++;
		if (j > 0)
			sequence[i] = sequence[i + j];
	}
}
