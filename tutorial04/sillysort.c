#include <stdio.h>
#include <stdlib.h>

int main() 
{
   int i, j, n = 10000; 

   // Allocate input, output and position arrays
   int *in = (int*) calloc(n, sizeof(int));
   int *pos = (int*) calloc(n, sizeof(int));   
   int *out = (int*) calloc(n, sizeof(int));   

   // Initialize input array in the reverse order
   for(i=0; i < n; i++)
      in[i] = n-i;  

   // Silly sort
   for(i=0; i < n; i++) 
      for(j=0; j < n; j++)
	if(in[i] >= in[j]) 
           pos[j]++;	

   // Move elements to final position
   for(i=0; i < n; i++) 
      out[n-pos[i]] = in[i];
   
   // Check if answer is correct
   for(i=0; i < n; i++)
      if(i+1 != out[i]) 
      {
         printf("test failed\n");
         exit(0);
      }

   printf("test passed\n"); 
}  