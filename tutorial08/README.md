# Tutorial 08: Programação para GPU com OpenMP

## Você vai aprender:

* Como as diretivas OpenMP 4.5 target, map, teams e distribute funcionam.
* Como portar um código OpenMP para executar em GPU.
* Como otimizar esse código para GPU.

## Pré-requisitos
* Tutorial 07: Auto-Vetorização e Vetorização com OpenMP​​​​​​​

### Definindo um baseline
Vamos revisitar a versão paralela do cálculo de Pi por meio de integração numérica.

```c
#include <stdio.h>

long long num_passos = 1000000000;
double passo;

int main(){
   int i;
   double x, pi, soma=0.0;
   passo = 1.0/(double)num_passos;

#pragma omp parallel for private(x) reduction(+:soma)	
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
Vamos então compilar e executar o programa com o seguinte comando:

```bash
$ gcc -O3 pi.c -o pi -fopenmp
```
```bash
$ time ./pi
```

Uma saída possível para o programa seria a seguinte:

```txt
O valor de PI é: 3.141593

real    0m3.580s
user    0m14.172s
sys     0m0.004s
```

Esta versão paralela gastou 3.58 segundos em um processador com 4 núcleos. O que é necessário para que este código execute em uma GPU (acelerador gráfico)?

### Comunicando com uma GPU

O primeiro passo para portar um código em CPU para GPU é lidar com a transferência de dados.

O OpenMP provê a diretiva target map para mapear os dados da memória da CPU para a GPU. Ou seja, cada dado na CPU deve ser replicado na GPU. 

O OpenMP possui os comandos: `to`, `from` e `tofrom`. Eles especificam respectivamente o dado que deve apenas ser copiado da CPU para a GPU (usado apenas para leitura), o dado que deve ser apenas copiado da GPU para CPU (apenas escrita), ou o dado que deve ser copiado para GPU e depois retornar para a CPU (leitura e escrita).

No exemplo do Pi, todas as variáveis são apenas de leitura (que é o default). Então é necessário apenas especificar que a variável soma deve ser copiada com seu valor inicial para a GPU e ao final da execução ela deve retornar para a CPU com o valor final. 

```c
#pragma omp target map(tofrom:soma) 
#pragma omp parallel for private(x) reduction(+:soma)	
for(i=0; i < num_passos; i++){
  x = (i + 0.5)*passo;
  soma = soma + 4.0/(1.0 + x*x);
}
```

### Criando e executando times de threads

Para a criação e distribuição da carga de trabalho na GPU, o OpenMP oferece as diretivas teams e distribute. A diretiva teams cria times de threads, compostas por uma thread mestre e threads trabalhadoras. Já a diretiva distribute distribui as iterações de um laço entre as threads mestres.

```c
#pragma omp target map(tofrom:soma) 
#pragma omp teams distribute parallel for private(x) reduction(+:soma)	
for(i = 0; i < num_passos; i++) {
    x = (i + 0.5)*passo;
    soma = soma + 4.0/(1.0 + x*x);
}
```

Com a inclusão das diretivas teams e distribute, o código acima já está pronto para a execução em GPU, pois o `parallel for` será executado para cada thread mestre, dividindo as iterações recebidas pelo mestre entre as threads trabalhadoras.

Para compilar o código acima, não é necessária nenhuma flag adicional além do `-fopenmp`.

Vamos então compilar e executar o código como anteriormente. Uma saída possível para o programa seria a seguinte:

```txt
O valor de PI é: 3.141593

real    0m17.824s
user    0m13.500s
sys     0m4.220s
```

Note que o programa em GPU ficou em torno de 5x mais lento comparado com o baseline (CPU). Qual será o motivo?

### Otimizando a paralelização para GPU

A diretiva `teams distribute parallel for` cria M times e N threads por time, ou seja, MxN threads. Cada thread é executada por um SM (_Streaming Multiprocessor_) da GPU, que possui capacidade de executar **32 threads simultaneamente**. 

:warning: **Entretanto, o OpenMP não tem acesso completo à arquitetura da GPU (código proprietário), e consegue apenas replicar a mesma thread 32 vezes, ou seja, 31 threads estão fazendo trabalho redundante.**

Para amenizar este problema, ao incluir a diretiva `simd`, o OpenMP agrupa as iterações em operações SIMD que são executadas ao mesmo tempo no SM. Dependendo do número de operações vetorizadas, cada SM passa a executar mais threads diferentes ao mesmo tempo, geralmente entre 4 e 16 threads, enquanto as demais threads continuam fazendo trabalho redundante.

```c
#pragma omp target map(tofrom:soma) 
#pragma omp teams distribute parallel for simd private(x) reduction(+:soma)	
for(i=0; i < num_passos; i++){
  x = (i + 0.5)*passo;
  soma = soma + 4.0/(1.0 + x*x);
}
```

Acrescente a diretiva simd como no código acima, recompile e execute. Uma possível saída é a seguinte:

```txt
O valor de PI é: 3.141593

real    0m1.541s
user    0m0.568s
sys     0m0.860s
```

O tempo de execução foi em torno de 11x mais rápido que a versão anterior, e 2.3x mais rápido que o baseline em CPU. Isso mostra que a inclusão da diretiva `simd`, pode aumentar bastante o desempenho por utilizar melhor os SMs da GPU, mas a versão atual do OpenMP não consegue automaticamente utilizar todas as threads disponíveis na GPU.

### Links úteis

[OpenMP 4.5: Diretivas de paralelismo para GPUs da NVIDIA](https://www.openmp.org/wp-content/uploads/openmp-4.5-1.pdf)


