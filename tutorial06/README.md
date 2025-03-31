# Tutorial 06: Paralelizando e Sincronizando Unbounded Loops em OpenMP

## Você vai aprender:

- Como serializar trechos de regiões paralelas.
- Como paralelizar funções recursivas com a diretiva task.
- Como paralelizar uma estrutura WHILE.

### Pré-requisitos

- Tutorial 05: Paralelismo de Tarefas em OpenMP

### Gerando números aleatórios

O programa abaixo faz com que cada thread gere e imprima um número aleatório, além do seu próprio identificador.

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main()
{
  srand(time(NULL));

  #pragma omp parallel 
  {
    int id = omp_get_thread_num();
    int numero_secreto = rand() % 20; 

    #pragma omp single    
    printf("Vamos revelar os números secretos!\n");   

    printf("Thread %d escolheu o número %d\n",id,numero_secreto);   

  }
}
```

Primeiro copie o código acima para um arquivo chamado `num_secretos.c`. Vamos então compilar e executar o programa com os seguintes comandos:

```bash
$ gcc num_secretos.c -o num_secretos -fopenmp
```

```bash
$ ./num_secretos
```

Uma saída esperada para esse programa em um processador com quarto núcleos é a seguinte:

```bash
Vamos revelar os números secretos!
Thread 0 escolheu o número 11
Thread 3 escolheu o número 18
Thread 2 escolheu o número 11
Thread 1 escolheu o número 16
```

Execute várias vezes o código. O que você pode notar?

<details>
<summary>Resposta</summary>
A frase "Vamos revelar os números!" aparece apenas uma vez e sempre é a primeira frase impressa na tela. 
</details>

A diretiva `omp single` serializa um trecho de código, permitindo que este trecho seja executado por apenas uma thread.
Além disso, ela serve como uma barreira que sincroniza todas as threads, ou seja, as threads só podem continuar suas execuções, após o `omp single` ter sido executado.

Para retirar-se a barreira do `omp single`, basta acrescentar o comando `nowait`. Desta forma, as threads não precisam mais esperar a conclusão do `single` para continuarem. Realize a seguinte mudança, compile e execute o código novamente.

```c
#pragma omp single nowait   
printf("Vamos revelar os números secretos!\n");   
```

Uma saída esperada para este programa é a seguinte:

```bash
Thread 0 escolheu o número 9
Thread 3 escolheu o número 13
Thread 2 escolheu o número 11
Vamos revelar os números secretos!
Thread 1 escolheu o número 15
```

Equivalente ao `omp single nowait` é o comando `omp master`, mas ele exige que apenas a thread com identificador 0 execute a seção sequencial. Modifique e teste o programa conforme o seguinte código.

```c
#pragma omp master   
printf("Vamos revelar os números secretos!\n");   
```

Uma saída esperada para esse programa é a seguinte. 

```bash
Thread 3 escolheu o número 4
Thread 2 escolheu o número 4
Vamos revelar os números secretos!
Thread 0 escolheu o número 15
Thread 1 escolheu o número 16
```

Execute este código várias vezes. O que é possível notar?

<details>
<summary>Resposta</summary>
Após a frase "Vamos revelar os números secretos!" sempre vem a mensagem da thread 0, pois esta thread é a responsável por executar a seção serializada.
</details>

Também é possível simular o comportamento do `omp single`, utilizando-se as diretivas `omp master` e `omp barrier`. Esta última diretiva faz com que todas as threads só continuem suas execuções depois de todas terem atingido a barreira.

Modifique e teste o programa conforme o seguinte código.

```c
#pragma omp master    
printf("Vamos revelar os números secretos!\n");   
#pragma omp barrier
```

### Paralelizando o Fibonacci com `omp task`

A diretiva `omp task` é muito similar a diretiva `omp section`, mas ela é bem mais leve, pois a região paralela é criada apenas uma vez, evitando-se a criação excessiva de threads usando nesting, além de permitir que as tarefas sejam desvinculadas da thread que as criou com a diretiva `untied`.

Uma thread apenas fica responsável pela criação de tarefas, por meio da diretiva `omp single`, enquanto as demais threads executam as tarefas criadas (como no escalonamento dinâmico). Ao terminar a geração de tarefas, esta thread também passa a executar as tarefas restantes.

O trecho de código sequencial abaixo mostra uma função recursiva para calcular o n-ésimo número da série de Fibonacci.

```c
#include <stdio.h>

#define N 42

long fib(long n) {
  long i, j;

  if (n<2)
    return n;
  else if (n < 30) {
    return fib(n-1) + fib (n-2);
  }
  else {
    #pragma omp task shared(i) untied
    i=fib(n-1);
    #pragma omp task shared(j) untied
    j=fib(n-2);
    #pragma omp taskwait

    return i+j;
  }
}
```

```c
int main() {

#pragma omp parallel 
#pragma omp single
  printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}
```

Note que ao contrário do `omp sections`, o `omp task` exige uma barreira explícita para a sincronização das threads com a diretiva `omp taskwait`, pois as tarefas são executadas de forma assíncrona. Além disso, as variáveis i e j precisam ser declaradas como shared, por que?

<details>
<summary>Resposta</summary>
Porque as variáveis <code>i</code> e <code>j</code> são declaradas dentro da região paralela, portanto são privadas. Por default, elas também serão privadas dentro da tarefa, ou seja, o resultado será jogado fora caso não se atribua <code>shared</code> explicitamente para as variáveis <code>i</code> e <code>j</code> para que elas sejam compartilhadas dentro do escopo da região paralela.
</details>

Vamos então compilar e executar o programa acima com os seguintes comandos:

```bash
$ gcc fib.c -o fib -fopenmp
```

```bash
$ time ./fib
```

Um saída possível para o programa é a seguinte:

```bash
Fibonacci(42) = 267914296

real	0m1.321s
user	0m4.120s
sys	0m0.016s
```

Note que o tempo de execução foi um pouco melhor que a versão com omp sections. E neste caso, se você reduzir o limite de 30 para 20, o programa continuará funcionando, pois a criação de tarefas não causa tanta sobrecarga quanto a criação de threads. Volte no tutorial Paralelismo de Tarefas em OpenMP e compare o tempo de sistema (sys) para visualizar esta sobrecarga.

### Paralelizando o WHILE com `omp task`

Além de funções recursivas, a diretiva `omp task` é adequada para a paralelização de _unbounded loops_ como a estrutura `WHILE`.

Ela permite que tarefas sejam criadas e executadas sob demanda, ou seja, ao contrário do `parallel for`, não é necessário saber a priori quantas iterações (tarefas) serão executadas. 

O trecho de código sequencial abaixo mostra um while para percorrer uma lista, onde cada elemento da lista é um número inteiro que contém o n-ésimo número da série de Fibonacci a ser calculado. Como o tamanho da lista não é conhecido, não é possível usar `parallel for` ou `sections` para paralelizar o código.

```c
q = init_list(q);
head = q;
p = head;

while (p != NULL) {
  processwork(p);
  p = p->next;
}
```

O código completo pode ser visto em [linked.c](linked.c).

Vamos então compilar e executar o programa baixado com os seguintes comandos:

```bash
$ gcc linked.c -o linked -fopenmp
```

```bash
$ time ./linked
```

Uma saída possível do programa que mostra o Fibonacci para 6 valores contidos na lista (37 a 42).

```bash
Process linked list
37 : 24157817
38 : 39088169
39 : 63245986
40 : 102334155
41 : 165580141
42 : 267914296

real	0m6.269s
user	0m6.268s
sys	0m0.004s
```

Abaixo a versão paralela deste trecho de código utilizando-se a diretiva omp task.

```c
q = init_list(q);
head = q;
p=head;

#pragma omp parallel 
{
  #pragma omp single nowait
  {
    while (p != NULL) {
      #pragma omp task firstprivate(p)
      processwork(p);
      
      p = p->next;
    }
  }
}
```

Por que foram usadas as diretivas nowait e firstprivate? Pense um pouco.

<details>
<summary>Resposta</summary>
A diretiva nowait foi utilizada para que as outras threads pudessem processar tarefas assim que elas fossem geradas pela thread executando o single. Já a diretiva firstprivate foi usada para que a variável p fosse inicializada com o último valor de p antes de entrar na região da tarefa, caso contrário, se fosse usado o private apenas, o valor de p seria indefinido.
</details>

O código completo pode ser baixado em [linked_par.c](linked_par.c).

Vamos então compilar e executar o programa baixado com os seguintes comandos:

```bash
$ gcc linked_par.c -o linked_par -fopenmp
```

```bash
$ time ./linked_par
```


Uma saída possível do programa paralelo para duas threads.

```bash
Process linked list 
37 : 24157817
38 : 39088169
39 : 63245986
40 : 102334155
41 : 165580141
42 : 267914296

real	0m3.883s
user	0m6.192s
sys	    0m0.040s
```

Esta aplicação se mostrou bastante escalável, pois não existe dependência de dados entre as tarefas, alcançando um speedup de 6.2/3.8 = 1.6. Apesar disso, ela é uma aplicação muito desbalanceada, pois possui poucas tarefas que gastam muito tempo.