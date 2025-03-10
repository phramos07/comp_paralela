#include <stdio.h>
#include <math.h>

int primo(long num)
{
   long d;

   if(num <= 1) return 0;
   else if(num > 3){
      if(num % 2 == 0) return 0;
        long max_divisor = sqrt(num);
        for(d = 3; d <= max_divisor; d += 2){
	  if(num % d == 0) return 0;
        }
   }
   return 1;
}

int main()
{
  long max_num = 5000000;
  long cont_primo;
  long soma;
  int n;

  if(max_num <= 1) soma = 0;
  else {
    if(max_num == 2) soma = 1;
    else {
       soma = 1;
       for(n = 3; n < max_num; n += 2){ 
         cont_primo = primo(n);
         soma = soma + cont_primo;
       }
    }
  }
  printf("Número total de primos: %ld\n", soma);

  return 0;
}