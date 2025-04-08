# Tutorial 07: Auto-Vetorização e Vetorização com OpenMP

## Você vai aprender:

- Como utilizar a auto-vetorização do GCC.
- Como utilizar a diretiva "omp simd" para vetorizar códigos em C.
- Como combinar vetorização com paralelização em OpenMP.

## Pré-requisitos

- Tutorial 06

#### Auto-vetorização

Vamos revisitar o cálculo de Pi por meio de integração numérica.

```c
#include <stdio.h>

long long num_passos = 1000000000;
double passo;

int main(){
   int i;
   double x, pi, soma=0.0;
   passo = 1.0/(double)num_passos;
	
   for(i=0; i < num_passos; i++){
      x = (i + 0.5)*passo;
      soma = soma + 4.0/(1.0 + x*x);
   }

   pi = soma*passo;
	
   printf("O valor de PI é: %f\n", pi);
   return 0;
}
```

Primeiro copie o código acima para um arquivo chamado pi.c. 
Vamos então compilar o programa com o seguinte comando:

```bash
$ gcc -O3 -fopt-info-vec-optimized pi.c -o pi
```

A flag `-O3` habilita todas as otimizações de código do compilador, inclusive a auto-vetorização do código, se possível.
A flag `-fopt-info-vec-optimized` mostra quais laços foram vetorizados. Se nenhum laço foi vetorizado (a saída foi vazia), a flag `-fopt-info-vec-missed` informa os motivos pelos quais os laços não foram vetorizados. 

Vamos então recompilar o programa com este último comando:

```bash
$ gcc -O3 -fopt-info-vec-missed pi.c -o pi
```

O compilador mostra as várias tentativas de vetorização que foram realizadas, principalmente no laço principal na linha 14 do código (pi.c:14:3).

```txt
pi.c:14:3: note: step unknown.
pi.c:14:3: note: reduction: unsafe fp math optimization: soma_15 = _14 + soma_23;

pi.c:14:3: note: Unknown def-use cycle pattern.
pi.c:14:3: note: Unsupported pattern.
pi.c:14:3: note: not vectorized: unsupported use in stmt.
pi.c:14:3: note: unexpected pattern.
pi.c:10:15: note: not consecutive access num_passos.0_4 = num_passos;

pi.c:10:15: note: not consecutive access passo = passo.1_6;

pi.c:10:15: note: Failed to SLP the basic block.
pi.c:10:15: note: not vectorized: failed to find SLP opportunities in basic block.
pi.c:7:5: note: not vectorized: not enough data-refs in basic block.
pi.c:16:12: note: not vectorized: not enough data-refs in basic block.
pi.c:7:5: note: not vectorized: not enough data-refs in basic block.
pi.c:7:5: note: not vectorized: not enough data-refs in basic block.
pi.c:24:6: note: not vectorized: not enough data-refs in basic block.
```
Vamos nos concentrar na segunda linha: 

```txt
reduction: unsafe fp math optimization: soma_15 = _14 + soma_23;
```
Esta linha informa que apesar do compilador ter detectado uma operação de redução na variável soma, ele considerou que a vetorização desta operação de ponto flutuante era insegura. Por que um somatório paralelo é uma operação insegura?

<details>
<summary>Resposta</summary>
A adição de números de ponto flutuante não é uma operação associativa, portanto ao permitir que adições sejam feitas em paralelo, o compilador não pode garantir o mesmo resultado independente da ordem em que as adições são realizadas.
</details>

Vamos executar esta versão do programa sem vetorização.

```bash
$ time ./pi
```

Uma saída possível para o programa seria a seguinte:

```txt
O valor de PI é: 3.141593

real	0m7.743s
user	0m7.712s
sys	0m0.000s
```

Em muitas aplicações, tal como no cálculo do Pi, a imprecisão no resultado é insignificante em relação ao possível ganho de desempenho ao permitir que a operação de adição de números de ponto flutuante seja feita em paralelo. Neste caso, pode-se permitir que o compilador assuma que a adição seja associativa, por meio da flag `-ffast-math`.

Vamos então recompilar o programa com este último comando:

```bash 
$ gcc -O3 -fopt-info-vec-optimized -ffastmath pi.c -o pi
```

Note, que o compilador agora gera a seguinte saída, indicando que o laço principal foi vetorizado.

```txt
pi.c:14:3: note: loop vectorized
```

Vamos executar esta versão do programa agora com vetorização:

```bash
$ time ./pi
```

Uma saída possível para o programa seria a seguinte:

```txt
O valor de PI é: 3.141593

real	0m3.807s
user	0m3.780s
sys	0m0.000s
```

O tempo de execução foi reduzido pela metade (speedup = 7.74/3.80 = 2.03) com a vetorização. É importante notar que foi utilizado apenas um núcleo do processador, mas permitindo que mais de uma instrução fosse executada simultaneamente na ULA deste núcleo.

#### Vetorização com OpenMP

Quando o compilador não é capaz de vetorizar automaticamente, ou vetoriza de forma ineficiente, o OpenMP provê a diretiva `omp simd`, com a qual o programador pode indicar um laço explicitamente para o compilador vetorizar. 

Obs: Esta diretiva só está disponível a partir da versão 4.0 do OpenMP (gcc 4.9 em diante).

No código abaixo, a inclusão da diretiva reduction funciona de forma similar a flag -ffast-math, indicando que a redução na variável soma é segura e deve ser feita.

```c
#pragma omp simd reduction(+:soma)
for(i=0; i < num_passos; i++){
  x = (i + 0.5)*passo;
  soma = soma + 4.0/(1.0 + x*x);
}
```

Mas porque não foi necessário usar a diretiva private(x)?

<details>
<summary>Resposta</summary>
A vetorização é mais simples que a paralelização, pelos seguintes motivos: i) existe apenas uma thread em execução; ii) as instruções são executadas ao mesmo tempo na mesma ULA (SIMD). Portanto, não existe a possibilidade da redução na soma acontecer antes da atribuição em x e nem duas threads acessarem a variável soma ao mesmo tempo. Neste caso específico, para cada iteração, o compilador automaticamente atribui o cálculo <code>(i + 0.5)*passo</code>, para uma posição diferente do registrador vetorial de saída da variável x.
</details>

Vamos compilar o código acima com a diretiva omp simd:

```bash
$ gcc -O3 -fopt-info-vec-optimized -fopenmp pi.c -o pi
```

Note, que o compilador agora gera a seguinte saída, indicando que o laço principal foi vetorizado.

```txt
pi.c:14:3: note: loop vectorized
```

Vamos executar esta versão do programa agora com vetorização:

```bash
$ time ./pi
```

Uma saída possível para o programa seria a seguinte:

```txt
O valor de PI é: 3.141593

real	0m1.941s
user	0m1.920s
sys	0m0.000s
```

Note que o tempo de execução foi ainda menor que auto-vetorização, pois quando o programador indica explicitamente o que é permitido fazer, o compilador pode ser bem menos conservador e realizar mais otimizações ou vetorizações no código. O speedup foi igual a 7.74/1.94 = 3.98, ou seja, 2x mais rápido que a auto-vetorização e 4 vezes mais rápido que o sequencial utilizando-se apenas um núcleo.

#### Vetorização + Paralelização
É possível combinar a vetorização com a paralelização como mostrado no exemplo abaixo. 

```c
#pragma omp parallel for simd private(x) reduction(+:soma)
for(i=0; i < num_passos; i++){
  x = (i + 0.5)*passo;
  soma = soma + 4.0/(1.0 + x*x);
}
```

Note que ao utilizar-se da diretiva parallel for, foi necessária a inclusão da diretiva private(x).

Ao recompilar e executar o código modificado acima, uma saída possível é a seguinte:

```txt
O valor de PI é: 3.141593

real	0m1.067s
user	0m3.884s
sys	0m0.028s
```

Este código paralelo utiliza quatro núcleos, além de cada cada núcleo executar o laço vetorizado. O speedup foi igual a 7.74/1.06 = 7.3, ou seja, 2x mais rápido que apenas a vetorização e 7 vezes mais rápido que o sequencial utilizando-se apenas um núcleo.

## Links úteis

- [Compiler Autovectorization Guide](https://www.intel.com/content/dam/develop/external/us/en/documents/31848-compilerautovectorizationguide.pdf)
- [OpenMP](https://www.openmp.org/)
- [GCC](https://gcc.gnu.org/)
