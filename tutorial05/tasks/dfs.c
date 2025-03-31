#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "time_common.h"
#define MAX_COORD 1000000
#define MAX_DIST 100000;
#define SIZE 550000
// #define DEBUG

void dfs(long** G, long index, long* visited, long size);

void fillgraph(long* G) { return; };

void printGraph(long* G){ return; };

long** newGraph(long size) { long** v, i; v = (long**) malloc(sizeof(long*)*size); for (i = 0; i < size; i++) v[i] = (long*)malloc(sizeof(long)*size); return v; }

void euclidianDist(long src, long dst) { return; };

int main(int argc, char* argv[])
{
	long size = SIZE;
	long** G = newGraph(size);
	long* visited = malloc(sizeof(long)*size);

    double start = get_time();
	dfs(G, 0, visited, size);
    double end = get_time();
    print_time("Tempo de execução", start, end);

	return 0;
}

void dfs(long** G, long index, long* visited, long size)
{
	if (!visited[index])
	{
		visited[index] = 1;
		#pragma omp parallel
		#pragma omp single
		for (unsigned long i=0; i<size; i++)
		{
			if (G[index][i] != 0)
			{
				#pragma omp task depend(in: G) shared(visited) untied
				dfs(G, i, visited, size);			
				euclidianDist(index, i);
			}			
		}
	}
	return;
}