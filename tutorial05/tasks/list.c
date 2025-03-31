#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  // For sleep() function
#include "time_common.h"

#define TRIPCOUNT_CUTOFF 1000
#define TRIP_COUNT 20

/* Forward declarations */
typedef struct TYPE_4__ TYPE_1__;

/* Type definitions */
struct TYPE_4__ {int counter; int ans; struct TYPE_4__* next;};
typedef TYPE_1__ LIST;

void printList(LIST* L, int size);
void fillList(LIST* L, int size);

void foo(LIST* head) {
    double start = get_time();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            LIST* n = head;
            while (n != NULL) {
                LIST* current = n;  // Cria um local copy para evitar condição de corrida
                int N = current->counter;
                
                #pragma omp task firstprivate(current, N) if(N > TRIPCOUNT_CUTOFF) untied
                {
                    // sleep(1);
                    // int thread_id = omp_get_thread_num();
                    // printf("Thread %d está processando o elemento: %d\n", thread_id, current->counter);
                    
                    current->ans = 0;
                    for (int j = 0; j < N; j++) {
                        current->ans += rand() % j;
                    }
                }
                
                n = n->next;
            }
            // Espera por todas as tarefas terminarem
            #pragma omp taskwait
        }
    }
    
    double end = get_time();
    print_time("Processing time", start, end);
}

int main(int argc, char const *argv[]) {
    LIST* L = (LIST*) malloc(sizeof(LIST));
    L->next = NULL;
    fillList(L, TRIP_COUNT);
    
    printf("Executando com %d threads disponíveis\n", omp_get_max_threads());
    foo(L);
    
    // Free memory
    LIST* current = L;
    while (current != NULL) {
        LIST* temp = current;
        current = current->next;
        free(temp);
    }
    
    return 0;
}

//print list
void printList(LIST* L, int size) {
    while (L != NULL) {
        printf("%d ", L->counter);
        L = L->next;
    }
    printf("\n");
}

void fillList(LIST* L, int size) {  
    L->counter = size;
    L->next = NULL;

    LIST* current = L;
    for (int i = size-1; i > 0; i--) {
        LIST* new = (LIST*) malloc(sizeof(LIST));
        new->counter = i;
        new->next = NULL;
        current->next = new;
        current = new;
    }
}