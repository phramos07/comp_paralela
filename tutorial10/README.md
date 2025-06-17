# Tutorial 10: Introdução a Programação em MPI

### Você vai aprender:

* Como implementar um programa hello world em MPI.
* Como trocar mensagens com as funções send/recv entre processos.

### Pré-requisitos:

* Programação em C/Linux

### Hello World em MPI

O **MPI** (Message Passing Interface) permite que um programa paralelo seja executado de forma distribuída em diferentes máquinas. O mesmo código é executado em cada um dos processos que formam a aplicação paralela. Mas ao contrário de multicores, que utilizam memória compartilhada para as threads se comunicarem, as máquinas com memórias distribuídas utilizam passagem de mensagens para os processos se comunicarem.

Abaixo, o código de um hello world em MPI, onde cada processo imprime o seu identificador (`rank`) e o número total de processos executados em paralelo. Note que, obrigatoriamente, todo programa em MPI precisa pelo menos das chamadas `MPI_Init(...)` e `MPI_Finalize(...)`.

```c
#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int rank, size;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  printf( "Hello world from process %d of %d\n", rank, size );
  MPI_Finalize();
  return 0;
}
```

Primeiro copie o código acima para um arquivo chamado `hello_mpi.c`. 
Vamos então compilar e executar o programa com o seguinte comando:

```bash
$ mpicc hello_mpi.c -o hello_mpi
```

```bash
$ mpirun -np 4 ./hello_mpi
```

O comando `mpirun` é necessário para criar os processos, enquanto a opção `-np` indica a quantidade de processos. A saída esperada para esse programa é a seguinte:

```txt
Hello world from process 0 of 4
Hello world from process 1 of 4
Hello world from process 2 of 4
Hello world from process 3 of 4
```

### Enviando e Recebendo Mensagens

No exemplo do hello world não existe comunicação entre os processos. Para que essa comunicação aconteça, o MPI provê duas chamadas básicas, `MPI_Send(...)` e `MPI_Recv(...)`, respectivamente para enviar e receber dados. Para cada `MPI_Send(...)` deve existir um respectivo `MPI_Recv(...)`, ou o programa pode, por exemplo, travar esperando por um dado que nunca foi enviado.

O código abaixo mostra um programa onde o processo com `rank` 0 envia um número (o valor de pi) para o processo com `rank` 1.

```c
#include <stdio.h>
#include <mpi.h>
#define TAG 1

void main(int argc, char* argv[]) {
  int p, rank;
  double val;    
  MPI_Status stat;
    
  MPI_Init(&argc, &argv) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    printf("Processor 0 sends a message to 1\n");
    val = 3.14 ;
    MPI_Send(&val, 1, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD);
  
  } else if (rank == 1) {
    printf("Processor 1 receives a message from 0\n");
    MPI_Recv(&val, 1, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &stat);
    printf("I received the value: %.2lf \n", val);
  }
  
  MPI_Finalize();    
}
```

Como todos os processos executam o mesmo código, é necessário diferenciar as funções de cada um dos processos por meio dos seus `ranks` (identificadores).

A `TAG` é utilizada como um identificador da mensagem e pode ser qualquer número inteiro.

Primeiro copie o código acima para um arquivo chamado `send_recv_mpi.c`. 
Vamos então compilar e executar o programa com o seguinte comando:

```bash
$ mpicc send_recv_mpi.c -o send_recv_mpi
```

```bash
$ mpirun -np 2 ./send_recv_mpi
```

Note que apenas dois processos são suficientes (`-np 2`).

```txt
Processor 0 sends a message to 1
Processor 1 receives a message from 0
I received the value: 3.14
```

#### Tutorial MPI:

https://hpc-tutorials.llnl.gov/mpi/ 