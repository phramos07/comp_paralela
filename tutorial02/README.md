# Tutorial 02: Balanceamento de carga

### Disciplina: Computação Paralela

#### Você vai aprender:

* Como medir o tempo de aplicações e calcular o ganho (speedup).
* Como permitir que várias threads utilizem uma mesma variável compartilhada.
* Como usar o padrão REDUCE em OpenMP.

#### Pré-requisitos:

* Tutorial 01

#### Calculando o valor de PI

O cálculo de Pi por meio de integração numérica é uma aplicação que consome muito tempo de execução. Este tempo é proporcional ao aumento da precisão (número de casas decimais). Vamos então medir este tempo na prática.

```c
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
```

Vamos compilar o programa acima com o seguinte comando:

```bash
$ gcc pi.c -o pi
```

Para medir o tempo de execução do programa, podemos usar o comando time.

```bash
$ time ./pi
```

Uma saída esperada para esse programa é a seguinte:

```txt
O valor de PI é: 3.141593

real   0m20.805s
user   0m20.783s
sys    0m0.024s
```

Os valores real, user e sys representam respectivamente:

- **real**: tempo total de execução do programa, considerando o tempo decorrido desde o início da execução até o término do programa.
- **user**: tempo de CPU gasto pelo programa em código modo usuário (código que não é executado pelo kernel do sistema operacional). Se houver mais de um processador, o tempo de CPU é a soma dos tempos de CPU de todos os processadores.
- **sys**: tempo de sistema (tempo gasto pelo sistema operacional para executar o programa).

### Paralelizando o laço de repetição

Geralmente, a primeira ideia que temos é paralelizar a laço de repetição usando a diretiva #pragma omp parallel for {...}. Vamos fazer isso e ver o que acontecerá.

```c
#include <stdio.h>

long long num_passos = 1000000000;
double passo;

int main(){
   int i;
   double x, pi, soma=0.0;
   passo = 1.0/(double)num_passos;
	
   #pragma omp parallel for	
   for(i=0; i < num_passos; i++){
      x = (i + 0.5)*passo;
      soma = soma + 4.0/(1.0 + x*x);
   }

   pi = soma*passo;
	
   printf("O valor de PI é: %f\n", pi);
   return 0;
}
```

Vamos compilar e executar o programa com os seguintes comandos:

```bash
$ gcc pi.c -o pi -fopenmp
```

```bash
$ time ./pi
```

Uma saída possível seria a seguinte em um processador com dois núcleos:

```txt
O valor de PI é: 2.003199

real  0m17.788s
user  0m35.100s
sys   0m0.160s
```

Neste resultado existem três pontos interessantes. Tente identificá-los.

<details>
<summary>Resposta (clique para abrir)</summary>

* O valor de PI está incorreto.

* O tempo de execução (real) reduziu um pouco.

* O tempo de usuário (user) dobrou.
</details>

<br>

Quais são as causas destes pontos?

<details>
<summary>Resposta (clique para abrir)</summary>

* O valor de PI está incorreto porque o laço de repetição não foi paralelizado corretamente.

* O tempo de execução (real) reduziu um pouco porque o OpenMP conseguiu distribuir o laço de repetição entre os núcleos do processador.

* O tempo de usuário também é esperado dobrar, pois estão somados os tempos da execução de cada thread. Todas rodam o mesmo código de forma paralela.
</details>

<br>

> :warning: **Regra 4 do Paralelismo:** Resultados incorretos geralmente indicam a existência de conflitos por variáveis compartilhadas não resolvidos.







=========== draft from here downwards ===========

**Obs:  Obs: Note que é necessário adicionar a diretiva *-fopenmp* para habilitar o uso do OpenMP.**


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