// LICENSE : GPL 3, see file LICENSE_GPL-3.txt
#include <stdlib.h>
#include <stdio.h>

#ifndef N
#define N 5
#endif
#ifndef FS
#define FS 37 
#endif
#define MULTCOUNT 10000000

//  The goal of this lab is to demonstrate that unbounded loops such as while
//  loops can be parallelized with the omp task clause
//  There is overhead associated with omp task scheduler and so it is better for 
//  performance for each thread to have significant amount of calculation to perform
//  relative to the overhead of having a mastrer thread running thru the while loop
//  assigning threads
//
//  By varying the #define parameters FS & N
//  we can vary the amount of computation each thread performs

//  under the covers, process_work() calculates fibonacci numbers
//  fibonacci was chosen because the amount of time it takes to compute fib(38) 
//  is orders of magintude higher than the time is takes to compute fib(3) for example

//  this allows us to simply change a few parameters and quickly see the effect of 
//  omp task scheduling for the linked list exmaple

//  small values of FS mean we are computing small fib numbers and this takes little time
//  small values of N mean we are comuting fewer in a sequence of fib numbers
//  this also means that process_work() takes little time
//  small values such as FS=3, N=2 take so little time to compute, that the 
//  overhead of omp task scheduling dominates the execution time
//  so the parallel version runs slower
//
//  larger values for FS & N means that each thread is doing signicicant work
//  and the cost of the omp task scheduler if not as significant
//  in such cases, the parallel version of the code runs several times faster 
//  than the serial version

//  to experiment with this overhead, modify the values for FS & N 
//  in the ll.h inlcude file


struct node {
   int data;
   int fibdata;
   struct node* next;
};


struct node* init_list(struct node* p);
void processwork(struct node* p); 
int fib(int n); 
 
int fib(int n) 
{
   int x, y;
   if (n < 2) {
      return (n);
   } else {
      x = fib(n - 1);
      y = fib(n - 2);
      return (x + y);
   }
}


int fibiter(int n)
{
  int u = 0;
  int v = 1;
  int i, t;
  int j;

  for (j=0;j<MULTCOUNT;j++) {
    u=0;
    v=1;
    for(i = 2; i <= n; i++)
    {
      t = u + v;
      u = v;
      v = t;
    }
  }
  return v;
}

void processwork(struct node* p) 
{
   int n;
   n = p->data;
   p->fibdata = fib(n);
}


struct node* init_list(struct node* p) {
    int i;
    struct node* head = NULL;
    struct node* temp = NULL;
    
    head = malloc(sizeof(struct node));
    p = head;
    p->data = FS;
    p->fibdata = 0;
    for (i=0; i< N; i++) {
       temp  = malloc(sizeof(struct node));
       p->next = temp;
       p = temp;
       p->data = FS + i + 1;
       p->fibdata = i+1;
    }
    p->next = NULL;
    return head;
}


int main(int argc, char *argv[]) {
     struct node *q=NULL;
     struct node *temp=NULL;
     struct node *head=NULL;
	 struct node *p;

     printf("Process linked list\n");

     q = init_list(q);
     head = q;
	 p=head;

	 while (p != NULL) {
		processwork(p);
		p = p->next;
	 }

     q = head;   //print the values after the computation for validation check
     while (q != NULL) {
        printf("%d : %d\n",q->data, q->fibdata);
        temp = q->next;
        free (q);
        q = temp;
     }  
     free (q);

     return 0;
}

