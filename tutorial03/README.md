# Tutorial 03: Balanceamento de Carga em OpenMP

### Você vai aprender

* Como usar seções críticas de forma eficiente.
* Como lidar com balanceamento de carga em laços de repetição.
* Como mudar a política de escalonamento do OpenMP.

### Pré-requisitos

* Tutorial 02

### Calculando números primos

O programa sequencial abaixo calcula quantos números primos existem no intervalo de 0 a 5 milhões.

A função primo() é responsável por avaliar se um número é primo ou não. Já a função principal conta o número de primos encontrados.

```c
#include <stdio.h>
#include <math.h>

int primo(long num)
{
   long d;

   if(num <= 1) return 0;
   else if(num > 3){
      if(num % 2 == 0) return 0;
        long max_divisor = sqrt(num);
        for(d = 3; d <= max_divisor; d+=2){
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
```

Primeiro copie o código acima para um arquivo chamado primo.c. Vamos então compilar o programa com o seguinte comando:

```bash
$ gcc primo.c -o primo -lm
```

Note que o flag -lm é necessário para utilizar a biblioteca math do C.

Para medir o tempo de execução da aplicação, utilize o seguinte comando:

```bash
$ time ./primo
```

Uma saída esperada para esse programa é a seguinte:

```txt
Número total de primos: 348513

real  0m4.215s
user  0m4.204s
sys   0m0.000s
```

Este programa possui dois laços de repetição. Qual dos dois laços deve ser então paralelizado?

<details>
<summary>Resposta</summary>

Uma dica é sempre tentar paralelizar o laço mais externo, pois a tarefa fica maior, ou seja, a quantidade de trabalho feita por cada iteração do laço requer mais computação. No caso de processadores com múltiplos núcleos, quando maior a tarefa, geralmente é melhor, pois as threads são melhor aproveitadas.
</details>

<br>

Então vamos paralelizar o laço da função principal (laço mais externo) com no código abaixo:

```c
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
       #pragma omp parallel for private (cont_primo) reduction(+:soma)
       for(n = 3; n < max_num; n += 2){ 
         cont_primo = primo(n);
         soma = soma + cont_primo;
       }
    }
  }
  printf("Número total de primos: %ld\n", soma);

  return 0;
}
```

Note que a paralelização deste código é muito similar a do programa de cálculo do Pi. A variável `cont_primo` deve ser privada, pois ela é usada apenas dentro de uma mesma iteração. Já a variável `soma` passa por uma redução, acumulando um somatório para contar a quantidade total de primos.

Vamos então compilar e executar:

```bash
$ gcc primo.c -o primo -lm -fopenmp
$ time ./primo
```


Uma saída esperada para esse programa é a seguinte:

```txt
Número total de primos: 348513

real  0m2.893s
user  0m4.665s
sys   0m0.004s
```

A aplicação foi paralelizada corretamente e obteve um speedup de aproximadamente 1,45 (4,2/2,9), em um processador com dois núcleos.

### Usando seções críticas de forma eficiente

Apesar da paralelização deste programa ser parecida com o Pi, ela se diferencia em dois aspectos. O primeiro, o speedup não foi linear, ou seja, próximo de 2. Além disso, no Pi, a inclusão de um critical tornava o tempo de execução do programa inaceitável.

Vamos então testar primeiro a substituição de uma redução por um critical como no código abaixo.

```c
#pragma omp parallel for private (cont_primo) 
for(n = 3; n < max_num; n += 2)
 { 
   cont_primo = primo(n);
   #pragma omp critical
   soma = soma + cont_primo;
 }
```

Vamos compilar e executar o programa utilizando os comandos anteriores.

Uma saída possível seria a seguinte:

```txt
Número total de primos: 348513

real  0m2.961s
user  0m4.866s
sys   0m0.008s
```

Você pode notar que o tempo de execução com a redução e com o `critical` foram praticamente iguais.

Mas a inclusão de uma seção crítica não torna o código muito lento? Pense um pouco sobre o motivo pelo qual o código executou rápido.

<details>
<summary>Resposta</summary>

Depende. Se fosse sempre o caso, talvez o critical nem existiria. Nesta contagem de números primos, ao contrário do Pi, cada iteração do laço de repetição tem quantidade de trabalho diferente, pois para descobrir se um número pequeno é primo, gasta-se muito menos tempo do que para calcular se um número grande é primo. Para observar isso, basta olhar na função primo(), que você observará que o laço de repetição depende do máximo divisor encontrado. Ou seja, o número de iterações do laço varia de acordo com o número primo.
</details>

<br>

```c
int primo(long num){
   long d;

   if(num <= 1) return 0;
   else if(num > 3){
      if(num % 2 == 0) return 0;
        long max_divisor = sqrt(num);
        for(d = 3; d <= max_divisor; d+=2){
	  if(num % d == 0) return 0;
        }
   }
   return 1;
}
```

A conclusão é que as duas threads dificilmente vão entrar na seção crítica ao mesmo tempo. Ou seja, elas não ficarão dormindo e acordando o tempo todo como no exemplo do Pi. Desta forma, o uso de uma seção crítica é bastante razoável e tem desempenho comparável a redução.

Apesar disso, esse desbalanceamento da quantidade de trabalho por iteração causa um outro problema, conhecido como balanceamento de carga.

Se as iterações do laço são divididas igualmente entre as _threads_ (metade pra cada uma), o desbalanceamento pode fazer com que uma thread tenha muito mais trabalho que a outra. Isso faz com que uma thread acabe seu trabalho muito cedo e fique ociosa até que a outra termine. Esta é uma possível causa para que a nossa versão paralela não tenha alcançado um speedup linear.

> :warning: **Regra 8 do Paralelismo:** Seções críticas são eficientes desde que utilizadas em situações nas quais a probabilidade de múltiplas threads utilizarem-na ao mesmo tempo seja pequena.

### Balanceamento de carga de trabalho

Para resolver o problema de balanceamento, o OpenMP tem diferentes políticas de distribuição de trabalho (escalonamento de tarefas) entre as threads através do comando schedule(pol,chunk). Neste comando é possível especificar três políticas: estática (_static_), dynamic (_dinâmica_) e guiada (_guided_). Além disso, é possível especificar o tamanho das tarefas ou bloco de iterações (chunk) que cada thread deve executar.

O OpenMP geralmente utiliza o escalonamento estático que no nosso exemplo pode ser ruim, pois neste caso ele distribui as iterações em dois blocos, um para cada thread. Ou seja, a thread que pegar o bloco com números maiores vai demorar mais pra terminar, deixando a outra thread ociosa.

Vamos mudar a política para dinâmica com blocos de tamanho 100 como no código abaixo.

```c
#pragma omp parallel for private (cont_primo) reduction(+:soma) schedule (dynamic,100)
 for(n = 3; n < max_num; n += 2){ 
   cont_primo = primo(n);
   soma = soma + cont_primo;
 }
```

Vamos compilar e executar o programa com os mesmos comandos utilizados anteriormente.

Uma saída possível seria a seguinte:

```txt
Número total de primos: 348513

real  0m2.553s
user  0m5.035s
sys   0m0.012s
```

O tempo de execução da aplição caiu de 2,9 para 2,5, ou seja, o speedup foi para 1,7. Qual o motivo para essa melhoria?

<details>
<summary>Resposta</summary>

A política dinâmica, ao contrário da estática, vai atribuindo para as threads um bloco de 100 iterações de cada vez. Assim que uma thread termina um bloco, ela busca o próximo. Se uma thread executa um bloco com números grandes, elas provavelmente computarão menos iterações. Uma thread que executa blocos com número pequenos, provavelmente executará mais blocos e as threads terminarão quase ao mesmo tempo, minimizando a ociosidade e consequentemente o desbalanceamento.
</details>

<br>

A política de escalonamento guiada tem comportamento igual a dinâmica, mas o tamanho dos blocos vai diminuindo ao longo da execução até atingirem o tamanho do bloco especificado. Ela tende a reduzir ainda mais o desbalaceamento, evitando que uma grande quantidade de trabalho reste para apenas uma thread no fim da execução.

Varie as políticas e o tamanho do bloco para ver se você consegue encontrar um tempo de execução ainda menor.

> :warning: **Regra 9 do Paralelismo:** O balanceamento de carga pode ser resolvido utilizando-se políticas de escalonamento de tarefas.

### Conclusão

Neste tutorial, você aprendeu:

* Como usar seções críticas de forma eficiente.
* Como lidar com balanceamento de carga em laços de repetição.
* Como mudar a política de escalonamento do OpenMP.