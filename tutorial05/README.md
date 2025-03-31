# Tutorial 05: Paralelismo de Tarefas utilizando `sections`

### Você vai aprender: 
* Como executar trechos diferentes de códigos em paralelo em OpenMP.
* Como paralelizar funções recursivas. 
* Como utilizar o aninhamento de regiões paralelas em OpenMP.
* Como utilizar `sections` para disparar tarefas.

### Pré-requisitos
* Tutorial 01

### Executando trechos de códigos distintos

O programa sequencial abaixo faz com que dois laços de repetição sejam executados, em que o primeiro imprime N letras `i` e o segundo, N letras `j`.

```c
#include <stdio.h>

#define N 100

int main() {
  int i, j; 
  
  for(i=0; i < N; i++)
    printf("i");

  for(j=0; j < N; j++)
    printf("j");
  
}
```

Primeiro copie o código acima para um arquivo chamado isjs.c. Vamos então compilar e executar o programa com o seguinte comando:

```bash
$ gcc isjs.c -o isjs
```   

```bash
$ ./isjs
```

A saída esperada para esse programa é a seguinte:

```
iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiijjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj
```

Se a ordem em que os `i`'s e `j`'s não fosse importante, este programa poderia executar os dois laços ao mesmo tempo.

A diretiva `omp parallel for` é capaz apenas de paralelizar um mesmo laço de cada vez. Então como resolver este problema?

<details>
<summary>Resposta</summary>
O <code>omp parallel for</code> explora o paralelismo de dados, onde várias threads executam o mesmo trecho de código em dados (iterações) diferentes. Para executar os dois laços de repetição ao mesmo tempo é necessário utilizar o paralelismo de tarefas, onde cada thread pode executar um trecho de código diferente em paralelo.
</details>

O OpenMP possui a diretiva omp sections que especifica um escopo onde vários trechos de código diferentes são executados em paralelo, inclusive laços de repetição. Cada trecho de código deve ser definido pela diretiva omp section. No código abaixo, os dois laços são definidos como seções separadas e portanto podem ser executados em paralelo por diferentes threads.

```c
#include <stdio.h>

#define N 100

int main() {

  int i, j; 
  #pragma omp parallel sections private(i,j)
  {
    #pragma omp section
    for(i=0; i < N; i++)
      printf("i");

    #pragma omp section
    for(j=0; j < N; j++)
      printf("j");
  }
}
```

Vamos então compilar e executar o programa com os seguintes comandos:

```bash
$ gcc isjs.c -o isjs -fopenmp
```

```bash
$ ./isjs
```

Uma saída esperada para esse programa é a seguinte:

```
iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiijjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjiiiiiiii
```

É importante notar que não é possível determinar a ordem em que os `i`'s e `j`'s serão impressos.

Ao final do escopo delimitado por `omp sections`, as threads realizam um join e a thread mestre continua a execução sequencialmente.

### Utilizando paralelismo de tarefas em programas recursivos

A série de Fibonnaci é uma sequência de números onde o próximo número é a soma dos dois números anteriores: 0, 1, 1, 2, 3, 5, 8, 13, 21 etc. 

O código abaixo implementa um programa que calcula o n-ésimo número da série de Fibonnacci de forma recursiva.

```c
#include <stdio.h>

#define N 42

long fib(long n) {
  long i, j;

  if (n < 2) {
    return n;
  }
  else {
    i = fib(n-1);
    j = fib(n-2);

    return i + j;
  }
}

int main() {
  printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}
```

Primeiro copie o código acima para um arquivo chamado fib.c. 

Vamos então compilar e executar o programa com o seguinte comando:

```bash
$ gcc fib.c -o fib
```

```bash
$ time ./fib
```

Uma possível saída é a seguinte:

```
Fibonacci(42) = 267914296

real	0m2.253s
user	0m2.244s
sys	0m0.004s
```

Neste exemplo, o décimo número da série de Fibonacci é o número 55. Mas como podemos paralelizar este código?

<details>
<summary>Resposta</summary>
É possível utilizar o padrão paralelo Divide and Conquer, onde cada chamada recursiva se torna uma tarefa, que pode ser delegada para threads diferentes executarem.
</details>

O código abaixo implementa o Fibonacci paralelo utilizando o padrão Divide and Conquer utilizando a diretiva omp sections, onde cada chamada recursiva da função fib é executada como uma seção paralela. 

```c
#include <stdio.h>

#define N 42

long fib(long n) {
  long i, j;

  if (n < 2) {
    return n;
  }
  else {
    #pragma omp parallel sections 
    { 
      #pragma omp section 
      i = fib(n-1);
      #pragma omp section 
      j = fib(n-2);
    }
    return i + j;
  }
}

int main() {
  printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}
```

Vamos então compilar e executar o programa com o seguinte comando:

```bash
$ gcc fib.c -o fib -fopenmp
```

```bash
$ time ./fib
```

Uma possível saída é a seguinte:

```
Fibonacci(42) = 267914296

real	1m58.808s
user	1m58.544s
sys	1m17.680s
```

O tempo paralelo ficou muito pior. Por que?

<details>
<summary>Resposta</summary>
Cada seção criada é como um chunk (bloco de iterações) em um laço de repetição, também chamado de tarefa. Existe um custo para a criação e distribuição de tarefas entre threads. Como a recursão é muito profunda, é criado um número excessivo de tarefas, causando uma sobrecarga no acesso as tarefas, aumentando muito o tempo de execução.
</details>

Para resolver este problema, deve-se limitar a profundidade da recursão, ou seja, **a partir de uma determinada profundidade, o código deve ser executado de forma sequencial, evitando-se a criação desnecessária de mais tarefas.**

O código abaixo implementa esta otimização, limitando a criação de tarefas até a profundidade de 30 na árvore de recursão (ver o `else if`).

```c
#include <stdio.h>

#define N 42

long fib(long n) {
  long i, j;

  if (n < 2)
    return n;
  else if (n < 30) {
    return fib(n-1) + fib (n-2);
  }
  else {
    #pragma omp parallel sections
    { 
      #pragma omp section 
      i = fib(n-1);
      #pragma omp section 
      j = fib(n-2);
    }
    return i + j;
  }
}

int main() {
  printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}
```

Um possível saída é a seguinte:

```
Fibonacci(42) = 267914296

real	0m1.875s
user	0m3.160s
sys	0m0.000s
```

Esta versão paralela teve um speedup de 1.3 (2.25/1.73) em dois núcleos.

> ⚠️ **Regra 10 do Paralelismo:** O paralelismo de tarefas pode ter sua escalabilidade limitada ​​​​​​​pelo número pequeno de tarefas existentes ou pelo excesso de tarefas criadas.

### Aninhamento de regiões paralelas (nesting)

Ao iniciar uma região paralela (omp parallel), por default, o OpenMP cria uma quantidade de threads igual ao número de núcleos disponíveis no processador.

Mas o que acontece quando uma região paralela é criada dentro de outra região paralela? Quantas threads são criadas?

<details>
<summary>Resposta</summary>
Por default, nenhuma nova thread é criada, além das cridas na região paralela inicial. Apesar disso, o aninhamento de regiões paralelas ou nesting, pode ser ativado, possibilitando que novas threads sejam criadas a cada nova região paralela.
</details>

Vamos então testar o exemplo abaixo onde habilitamos o aninhamento com a função `omp_set_nested(1)`.

Desta forma, cada nova chamada a diretiva `omp parallel sections` criará n novas threads, onde n é o número de núcleos.

```c
#include <stdio.h>
#include <omp.h>

#define N 42

long fib(long n) {
  long i, j;

  if (n < 2)
    return n;
  else if (n < 30) {
    return fib(n-1) + fib (n-2);
  }
  else {
    #pragma omp parallel sections
    { 
      #pragma omp section 
      i = fib(n-1);
      #pragma omp section 
      j = fib(n-2);
    }
    return i + j;
  }
}

int main() {
  omp_set_nested(1);
  printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}
```

Abaixo, uma possível saída para o programa acima.

```
Fibonacci(42) = 267914296

real	0m1.459s
user	0m4.420s
sys	0m0.748s
```

É possível notar que o tempo reduziu em relação a utilização de apenas duas threads, pois a criação de novas threads pode ajudar no balanceamento de carga, ou seja, a melhoria na utilização dos núcleos. Mas isso nem sempre é verdade, já que a criação de um número excessivo de threads pode levar até mesmo a interrupção da execução do programa ou causar uma sobrecarga onde o programa fica muito mais lento. Teste passar o limite de 30 para 20.