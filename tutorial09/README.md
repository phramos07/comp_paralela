# Tutorial 09: Programação para GPU com CUDA

## Você vai aprender

* Como converter um código em OpenMP 4.5 para CUDA
* Os principais comandos de alocação e movimentação de dados entre host e device
* Os principais comandos para implementação e execução de um kernel

## Pré-requisitos

Programação para GPU com OpenMP (https://web.archive.org/web/20181224061209/http://www.eitas.com.br/tutorial/12/57)

### Adição de vetores em OpenMP 4.5

A adição de vetores abaixo foi implementada com o OpenMP 4.5, onde: i) os vetores de entrada `a` e `b` são mapeados;
ii) o vetor de saída `c` é mapeado; iii) as iterações do laço de repetição são distribuídas entre as threads de cada time.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void vecadd(double* a, double* b, double* c, int width)
{
    #pragma omp target map(to:a[0:width],b[0:width]) map(tofrom:c[0:width])
    #pragma omp teams distribute parallel for simd
    for (int i = 0; i < width; i++)
        c[i] = a[i] + b[i];
}

int main()
{
    int width = 10000000;
    double *a = (double*) malloc (width * sizeof(double));
    double *b = (double*) malloc (width * sizeof(double));
    double *c = (double*) malloc (width * sizeof(double));
    for(int i = 0; i < width; i++) {
        a[i] = i;
        b[i] = width-i;
        c[i] = 0;
    }

    vecadd(a,b,c,width);
    for(int i = 0; i < width; i++)
        printf("\n c[%d] = %f",i,c[i]);
}
```

Primeiro copie o código acima para um arquivo chamado `vecadd.c`.

Vamos então compilar e executar o programa com o seguinte comando:

```bash
$ gcc -O3 vecadd.c -o vecadd -fopenmp
```

```bash
$ ./vecadd > openmp.txt
```

Note que a saída do vetor `c` foi redirecionada para o arquivo `openmp.txt`. Para ver parte do conteúdo do arquivo
gerado, execute o seguinte comando e veja a saída:

```bash
$ head openmp.txt
c[0] = 10000000.000000
c[1] = 10000000.000000
c[2] = 10000000.000000
c[3] = 10000000.000000
c[4] = 10000000.000000
c[5] = 10000000.000000
c[6] = 10000000.000000
c[7] = 10000000.000000
c[8] = 10000000.000000
```

### Alocando e transferindo dados em CUDA

Em OpenMP, a transferência explícita de dados é toda realizada pelo seguinte comando

```c
map(to:a[0:width],b[0:width]) map(tofrom:c[0:width])
```

Em CUDA, é necessário o uso de dois comandos:

```c
cudaMalloc (apontador, tamanho (em bytes)) - aloca uma quantidade de bytes no dispositivo
cudaMemcpy (destino, origem, tamanho (em bytes), direção) - copia uma quantidade de bytes da origem para o
```

### Destino na direção especificada (`HostToDevice` ou `DeviceToHost`)

O `map` do OpenMP em CUDA ficaria como no trecho de código abaixo.

```c
int size = width*sizeof(double);
double *d_a, *d_b, *d_c;
cudaMalloc((void **) &d_a, size);
cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

cudaMalloc((void **) &d_b, size);
cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

cudaMalloc((void **) &d_c, size);

vecadd(a,b,c,width);

cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

// ...
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

Neste código são alocados 3 vetores também no dispositivo (`d_a`, `d_b` e `d_c`). Além disso, o tamanho em bytes dos
vetores são armazenados na variável `size`.

Então os vetores `d_a`, `d_b` e `d_c` são alocados com o `cudaMalloc`, sendo os vetores de entrada `a` e `b` copiados para
as suas respectivas cópias no dispositivo `d_a` e `d_b`.

E por último o vetor de saída `d_c`, depois de calculado na GPU,
ele é copiado para o host por meio do `cudaMemcpy`.

O `cudaFree` é usado no final do programa para desalocar os dados da GPU.

### Criando o kernel em cuda

Em CUDA, um kernel é o núcleo da aplicação que será executado na GPU. Neste caso, a função `vecadd`.

```c
void vecadd(double* a, double* b, double* c, int width)
{
    #pragma omp target map(to:a[0:width],b[0:width]) map(tofrom:c[0:width])
    #pragma omp teams distribute parallel for simd
    for (int i = 0; i < width; i++)
        c[i] = a[i] + b[i];
}
```

Em CUDA, geralmente não estamos mais lidando com um laço, mas com a granularidade de uma iteração.

Isso significa que potencialmente cada thread deve calcular o mínimo possível, pois existem milhares de threads disponíveis.

Neste caso, cada thread fica responsável por executar uma única iteração.

O kernel portado para o CUDA é apresentado abaixo.

```c
__global__ void vecadd_cuda(double* a, double* b, double* c, int width) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < width)
        c[i] = a[i] + b[i];
}
```

O que mudou em relação ao dois códigos? Pense um pouco.

<details>
<summary>Resposta</summary>
Existem alguns pontos importantes a serem destacados:

i) `__global__`

Este comando indica para o compilador que esta função deve ter o código gerado para a GPU, mas que a
função pode ser chamada pelo host.

ii) `blockIdx.x*blockDim.x+threadIdx.x` - estas variáveis são equivalentes a chamada `omp_get_thread_num()`, onde
`blockIdx.x` representa o índice do bloco (equivalente ao time em que a thread pertence no OpenMP), `blockDim.x` é o
tamanho do bloco (número de threads em um time) e `threadIdx.x` é o índice da thread (identificador da thread).

iii) `if (i < width)` - o vetor pode não ser um múltiplo da quantidade de threads disponíveis, portanto algumas threads
podem tentar acessar posições do vetor que não existem e gerarem uma falha de segmentação.
</details>

Note que o laço de repetição desapareceu, pois este kernel será executado por cada uma das threads, uma iteração por
thread. Nem sempre isso é possível e um kernel pode conter laços de repetição dependendo da aplicação.

### Executando o kernel

Em OpenMP, a definição do número de times e threads é feita de forma implícita. No CUDA, é necessário especificar
explicitamente o número de blocos (times) e threads por time como no código abaixo.

```c
int block_size = 1024;
dim3 dimGrid((width-1)/block_size + 1, 1, 1);
dim3 dimBlock(block_size,1,1);
vecadd_cuda<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, width);
```

Note que o `block_size` é o número de threads por bloco, portanto o tamanho do vetor `width` deve ser dividido pelo
tamanho do bloco (e arredondado para cima) para garantir que o código tenha um número total de threads no mínimo
igual ao número de iterações. As estruturas `dimGrid` e `dimBlock` servem respectivamente para definir-se a quantidade
de blocos e a quantidade de threads por bloco em cada dimensão (x, y e z). No caso da adição de vetores, apenas a
dimensão x é especificada (vetor de uma dimensão), as outras são então preenchidas com 1.

Por último, a chamada do kernel exige que o programador passe como argumentos o `dimGrid` e `dimBlock` para que o
CUDA saiba quantos blocos e threads serão criados (você pode pensar que eles indicam quantas iterações teria o laço
que não existe mais). Os vetores passados como argumentos devem ser os vetores alocados no dispositivo (`d_a`, `d_b` e
`d_c`).

### Compilando e executando o código

O código completo em CUDA é mostrado abaixo.

```c
#include <stdio.h>
#include <stdlib.h>
__global__ void vecadd_cuda(double* a, double* b, double* c, int width) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < width)
        c[i] = a[i] + b[i];
}

int main()
{
    int width = 10000000;
    double *a = (double*) malloc (width * sizeof(double));
    double *b = (double*) malloc (width * sizeof(double));
    double *c = (double*) malloc (width * sizeof(double));

    for(int i = 0; i < width; i++) {
        a[i] = i;
        b[i] = width-i;
        c[i] = 0;
    }
    int size = width*sizeof(double);

    double *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_b, size);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_c, size);

    int block_size = 1024;
    dim3 dimGrid((width-1)/block_size + 1, 1, 1);
    dim3 dimBlock(block_size,1,1);

    vecadd_cuda<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, width);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < width; i++)
        printf("\n c[%d] = %f",i,c[i]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

Para compilar o código em CUDA é necessário utilizar o compilador nvcc da Nvidia. Copie o código acima para um
arquivo chamado vecadd.cu e execute os comandos abaixo para compilar e executar.

```bash
$ nvcc -O3 vecadd.cu -o vecadd
$ ./vecadd > cuda.txt
```

Para checar se a saída está correta, execute o seguinte comando:

```bash
$ diff openmp.txt cuda.txt
```

Para medir os tempos de execução das duas versões, utilize o comando time.

```bash
$ time ./vecadd > cuda.txt
```

## Links úteis

[Introduction to cuda](https://www.youtube.com/watch?v=IzU4AVcMFys&ab_channel=NVIDIA)