#ifndef GENCUDA_H_INCLUDED
#define GENCUDA_H_INCLUDED

char comparePair(char letter1, char letter2);
float findOptimalMutation(char** map, char* seq1, int len1, char* mutant, int len2, float* weights, int offset, int direction);

#endif
