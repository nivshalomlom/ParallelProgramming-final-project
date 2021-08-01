#ifndef GEN_H_INCLUDED
#define GEN_H_INCLUDED

#define MASTER 0

#define WORK_TAG 0
#define KILL_TAG 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "geneCuda.h"

char* generateAllMutations(char letter);
char** generateMutationMap(char* sequence, int seqLen);
void evaluateAndMutate(char* sequence1, int len1, char* sequence2, int len2, float weights[], int direction, int proc_rank, int proc_num, MPI_Status status, const char* outputFilePath);

#endif
