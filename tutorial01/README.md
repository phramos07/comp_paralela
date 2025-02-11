# Tutorial 01: Introdução a Programação Paralela com OpenMP

### Disciplina: Computação Paralela

#### Você vai aprender:

* Como paralelizar um código sequencial em OpenMP.
* Como usar os padrões FORK-JOIN e MAP em OpenMP.
* Como lidar com variáveis compartilhadas em OpenMP.

#### Pré-requisitos:

* Noções básicas de programação em C
* Noções básicas de comandos bash em Linux
* Noções básicas de processos e threads
* Sistema operacional Linux

#### Contando de 1 a 4

O programa sequencial abaixo que conta de 1 a 4 é executado por apenas uma thread.

#include <stdio.h>

```c
int main()
{
    int i;

    printf("Vamos contar de 1 a 4\n");
    
    for(i = 1; i <= 4; i++)
        printf("%d\n",i);
}
```

Primeiro copie o código acima para um arquivo chamado contar.c. 

Vamos então compilar e executar o programa com os seguintes comandos:

```bash
$ gcc -o contar contar.c
```

```bash
$ ./contar
```

A saída esperada para essse programa é a seguinte:

```
Vamos contar de 1 a 4
1
2
3
4
```

Não importa a quantidade de núcleos do seu processador, este código sempre será executado uma vez em apenas um núcleo.

#### Criando threads em OpenMP

O OpenMP é uma ferramenta que auxilia na paralelização automática de código.

Ele cria e gerencia a sincronização entre threads automaticamente.

Para fazer o mesmo trecho de código executar em todos os núcleos do seu processador, basta adicionar a diretiva:

```c
#pragma omp parallel {...}
```

... no respectivo trecho de código que deseja paralelizar.

O OpenMP vai automaticamente criar várias threads (o número *default* de threads é o número de núcleos ou threads reais. Por ex, em um quad-core, ele cria 4 threads por *default*) que executam exatamente o mesmo código. Este padrão de programação paralela é conhecido como **FORK-JOIN**.

```c
#include <stdio.h>

int main() 
{

  #pragma omp parallel
  {
     int i;

     printf("Vamos contar de 1 a 4\n");

     for(i = 1; i <= 4; i++)
       printf("%d\n",i);
  }
}
```

Vamos compilar e executar o programa com os seguintes comandos:

```bash
$ gcc -o contar contar.c -fopenmp
```

```bash
$ ./contar
```

**Obs:  Obs: Note que é necessário adicionar a diretiva *-fopenmp* para habilitar o uso do OpenMP.**

Uma possível saída seria a seguinte:

```
Vamos contar de 1 a 4
1
2
3
4
Vamos contar de 1 a 4
1
2
3
4
```

Ao acrescentar um #pragma omp parallel, o que vc espera na saída do programa?

<details> 
<summary>Resposta (clique para abrir)</summary>
<b>Provavelmente que ele replique o código para todas as threads. Se o seu processador é um dual core, seria esperado que a contagem de 1 a 4 fosse feita duas vezes seguidas, em ordem.</b>
</details>

<br/><br/>
Mas foi isso que aconteceu? Execute o código mais 10 vezes. 

<details>
<summary>Resposta (clique para abrir)</summary>
<b>É possível notar que ele gerou outras saídas diferentes. Como as duas threads executam ao mesmo tempo, não tem como garantir qualquer ordem de execução.</b>
</details>

<br/><br/>

> ⚠️ **Regra 1 do Paralelismo:** Não se pode garantir a ordem de execução de threads em paralelo.

#### Replicando apenas parte do código

Para replicar apenas a contagem de 1 a 4, basta adicionar a diretiva `#pragma omp parallel` antes da estrutura de repetição (for).

Quando as chaves `{}`, ou seja, o escopo do bloco de código não é especificado, o OpenMP automaticamente paraleliza o comando logo abaixo da diretiva.

```c
#include <stdio.h>

int main() 
{

  int i;

  printf("Vamos contar de 1 a 4\n");

  #pragma omp parallel
  for(i = 1; i <= 4; i++)
    printf("%d\n",i);
}
```

Vamos compilar e executar o programa com os seguintes comandos:

```bash
$ gcc -o contar contar.c -fopenmp
```

```bash
$ ./contar
```

Uma possível saída seria a seguinte:

```
Vamos contar de 1 a 4
1
2
3
4
1
```

Ao acrescentar a diretiva #pragma omp parallel, o que você espera na saída do programa?

<details>
<summary>Resposta (clique para abrir)</summary>
<b>Provavelmente que ele imprima a contagem de 1 a 4 duas vezes, no caso de um dual core. Ou seja, o número 1 deve aparecer 2 vezes, o número 2 duas vezes, etc., independente da ordem.</b>
</details>

<br/><br/>

Execute o código mais 10 vezes. Por que estão faltando alguns números na saída?

<details>
<summary>Resposta (clique para abrir)</summary>
<b>Quando duas ou mais threads compartilham uma mesma variável, neste exemplo a variável "i", o valor da variável pode ficar imprevisível.</b>
</details>

<br/><br/>

No exemplo acima, a thread 0 leu o valor 1 de "i" ao mesmo tempo que a thread 1, mas a thread 0 foi mais rápida e executou todas as iterações incrementando o contador "i" até 5, terminando sua execução. Quando a thread 1 decide continuar sua execução, ela imprime o valor 1 e quando vai conferir o valor de "i" na condição do loop, ela descobre que ele vale 5, fazendo com que ela também saia do loop (for).

> ⚠️ **Regra 2 do Paralelismo:**  Quando threads compartilham uma mesma variável, o valor da variável pode ficar imprevisível. 

#### Lidando com variáveis compartilhadas

Para resolver o problema de ter apenas um contador "i" para as threads, basta replicar o contador em cada thread. Para isso é necessário declarar o contador dentro da região paralela, ou seja, dentro da estrutura for.

```c
#include <stdio.h>

int main() 
{

  printf("Vamos contar de 1 a 4\n");

  #pragma omp parallel
  for(int i = 1; i <= 4; i++)
    printf("%d\n",i);
}
```

Vamos compilar e executar o programa com os mesmos comandos anteriores.

Uma saída possível seria a seguinte:

```
Vamos contar de 1 a 4
1
2
3
4
1
2
3
4
```

Ao declarar o contador "i" dentro da região paralela, cada thread agora tem a sua cópia privada. Ou seja, não existe mais o compartilhamento de variáveis.

Outra forma de resolver este problema, é usar o comando private que transforma variáveis compartilhada em variáveis privadas, como no exemplo abaixo.

```c
#include <stdio.h>

int main() 
{

  int i;

  printf("Vamos contar de 1 a 4\n");

  #pragma omp parallel private(i)
  for(i = 1; i <= 4; i++)
    printf("%d\n",i);
}
```

#### Distribuindo o trabalho entre threads

Mas se ao invés de replicar o código, eu dividir o trabalho entre as threads?

Para isso existe o comando #pragma omp parallel for especificamente para distribuir as iterações do for entre as threads. Ou seja, cada iteração do for é executada apenas uma vez, mas cada thread executa uma iteração diferente. Este padrão de programação é conhecido como MAP.

```c
#include <stdio.h>

int main() 
{

  printf("Vamos contar de 1 a 4\n");

  #pragma omp parallel for
  for(int i = 1; i <= 4; i++)
    printf("%d\n",i);
}
```

Vamos compilar e executar o programa com os mesmos comandos anteriores.

Uma saída possível seria a seguinte:

```
Vamos contar de 1 a 4
1
3
4
2
```

Execute várias vezes e você notará que a ordem pode sempre variar, ​​​​mas necessariamente cada iteração ocorrerá uma única vez.

A maioria dos programas paralelos são feitos através da paralelização de um for, que geralmente é onde o programa gasta mais tempo e portanto deve ser paralelizado.

> ⚠️ **Regra 3 do Paralelismo:** Estruturas de repetição são as melhores candidatas para a paralelização. 

### Links úteis

Tutorial OpenMP: https://hpc-tutorials.llnl.gov/openmp/