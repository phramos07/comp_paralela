#include <stdio.h>

long long num_passos = 1000000000;
double passo;

int main() {
   int i;
   double x, pi, soma = 0.0;
   passo = 1.0 / (double) num_passos;
	
   for(i = 0; i < num_passos; i++){
      x = (i + 0.5) * passo;
      soma = soma + 4.0 / (1.0 + x * x);
   }

   pi = soma * passo;
	
   printf("O valor de PI é: %f\n", pi);
   
   return 0;
}