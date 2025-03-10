# Tutorial 02: Resolvendo condições de disputa

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

### Identificando os conflitos

Para localizar conflitos ou condições de disputa por variáveis compartilhadas, vamos observar o programa. Existe alguma variável dentro da região paralela que está sendo modificada por várias threads ao mesmo tempo?

Lembre-se: o comportamento de um programa multithread é não determinístico, ou seja, não sabemos qual thread vai executar primeiro.

```c
 #pragma omp parallel for	
 for(i=0; i < num_passos; i++){
    x = (i + 0.5)*passo;
    soma = soma + 4.0/(1.0 + x*x);
 }
```

No código acima, podemos identificar 5 variáveis (`i`, `num_passos`, `x`, `passo` e `soma`).

As variáveis `num_passos` e `passo` são apenas de leitura, ou seja nunca são modificadas, logo não geram nenhum conflito.

A variável `i` apesar de ser modificada (`i=0`, `i++`), ela é contador do laço de repetição, então ela é privatizada automaticamente pelo `#pragma omp parallel for {...}.`, ou seja, cada thread tem uma cópia local de `i`.

Restam então as variáveis `x` e `soma` que são modificadas. Vamos avaliar a variável `x`.

Substitua as variáveis `x` pela expressão atribuída a ela e veja se o seu programa continua correto como no exemplo abaixo.

```c
 #pragma omp parallel for	
 for(i=0; i < num_passos; i++){
   soma = soma + 4.0/(1.0 + ((i + 0.5)*passo)*((i + 0.5)*passo));
 }
```

<details>
<summary>Resposta (clique para abrir)</summary>

Sim, seu programa continua correto!. Isso significa que `x` é uma variável privada, pois o valor atribuído a `x` é apenas utilizado dentro da mesma iteração.

Ou seja, não existe dependência de dados entre iterações do laço em relação a variável `x`, portanto `x` pode ser privado (private(x)) como no exemplo abaixo.
</details>

```c
#pragma omp parallel for private(x)
for(i = 0; i < num_passos; i++){
   x = (i + 0.5) * passo;
   soma = soma + 4.0 / (1.0 + x * x);
}
```

Vamos modificar o código conforme o código acima e executar os comandos abaixo.

```bash
$ gcc pi.c -o pi -fopenmp
$ time ./pi
```

```txt
Uma saída possível seria a seguinte:

O valor de PI é: 2.108069

real  0m17.977s
user  0m31.341s
sys   0m0.136s
```

O resultado mudou pouco, pois apesar de termos resolvido o conflito pela variável `x`, ao criar uma cópia privada de `x` para cada uma das threads, ainda não resolvemos o conflito pela variável `soma`.


> :warning: **Regra 5 do Paralelismo:** Variáveis que são utilizadas em uma mesma iteração não são dependências de dados reais, portanto devem ser privatizadas.

### Criando seções críticas

Então porque também não privatizamos a variável `soma`? Por que não? Pense um pouco sobre isso.

<details> 
<summary>Resposta (clique para abrir)</summary>

Porque ela acumula um resultado que é realmente compartilhado por todas as threads.
</details>

<br>

Pense ... ela é o resultado de um somatório. Imagine que estivéssemos somando `1 + 2 + 3 + 4`. Se a soma fosse privatizada e cada thread ficasse responsável por somar metade dos elementos, a thread 0 somaria `1 + 2 = 3` e a thread 1 somaria `3 + 4 = 7`.

Qual seria a resposta certa do somatório?

<details> 
<summary>Resposta (clique para abrir)</summary>

Não seria nem `3` e nem `7`, a resposta correta seria a soma dos dois valores, ou seja, `3 + 7 = 10`.
</details>

<br>

Então como resolver este problema?

No OpenMP existe um comando chamado `#pragma omp critical`. Ele cria uma seção crítica ao redor de um comando. Uma seção crítica impede que duas threads executem um mesmo código ao mesmo tempo. Ou seja, se colocarmos uma seção crítica ao redor da linha que modifica a variável `soma`, a soma continua compartilhada, mas não existe a possibilidade de duas ou mais threads modificarem o seu valor ao mesmo tempo. Isso elimina a condição de disputa.

```c
#pragma omp parallel for private(x) 
for(i=0; i < num_passos; i++){
   x = (i + 0.5)*passo;
   #pragma omp critical
   soma = soma + 4.0/(1.0 + x*x);
}
```

Vamos compilar e executar o programa acima com a diretiva `#pragma omp critical`.

Uma saída possível seria a seguinte:

```txt
O valor de PI é: 3.141593

real  0m51.101s
user  1m39.295s
sys   0m0.527s
```

Você vai notar dois resultados interessantes. Tente pensar quais são eles.

<details> 
<summary>Resposta (clique para abrir)</summary>

O primeiro é que o valor de PI agora está correto, ou seja, não existem mais condições de disputa (conflitos).

A segunda notícia menos animadora. O tempo de execução ficou muito maior.
</details>

<br>

Será que paralelismo vale a pena então?

> :warning: **Regra 6 do Paralelismo:** Variáveis que são realmente compartilhadas devem ser protegidas por seções críticas.

### Realizando uma redução

Por que o programa ficou tão lento? Pense um pouco antes de conferir a resposta.

<details> 
<summary>Resposta (clique para abrir)</summary>

Porque as threads passam a maioria do tempo esperando a outra na seção crítica. Quando uma thread já se encontra seção crítica, e uma segunda thread tenta acessá-la, esta segunda thread dorme e só acorda quando a thread anterior sai da seção crítica. Este processo de dormir e acordar consome muito mais tempo do que apenas realização a operação do somatório.
</details>

<br>

Mas existe uma solução melhor pra este problema?

No OpenMP, o padrão REDUCE é implementado por meio do chamado `reduction(op:var)`, onde `var` é a variável onde a operação `op` é aplicada. Uma redução consiste em somar os resultados parciais de cada thread até gerar um único valor. Imagine a situação descrita anteriormente onde a soma foi privatizada. O único problema foi não somar o 3 + 7, ou seja, somar os resultados parciais de cada thread.

É exatamnete isso que a redução faz, ela cria uma variável privada para cada thread, mas no final ela agrupa todas elas em apenas uma variável, como no exemplo abaixo.

```c
#pragma omp parallel for private(x) reduction(+:soma)
 for(i=0; i < num_passos; i++){
   x = (i + 0.5)*passo;
   soma = soma + 4.0/(1.0 + x*x);
 }
```

Vamos compilar e executar o programa.

Uma saída possível seria a seguinte:

```txt
O valor de PI é: 3.141593

real  0m10.620s
user  0m20.732s
sys   0m0.056s
```

Você pode notar que neste resultado, o valor de PI está correto e além disso o tempo caiu pela metade.

Ou seja, a aplicação teve um speedup de aproximadamente 2.

O speedup é a métrica utilizada para avaliar o ganho de desempenho, dividindo-se o tempo da versão sequencial pelo tempo da versão paralela, no nosso exemplo: 20.8/10.6 = 1.96, ou aproximadamente 2 de speedup, próximo ao speedup linear.

Com a redução, a seção crítica é eliminada e as threads não perdem mais tempo dormindo. Apenas ao final da região paralela, existe a sincronização das threads para realizar a redução. 

> :warning: **Regra 7 do Paralelismo:** Variáveis compartilhadas que acumulam resultados podem ser mapeadas no padrão REDUCE, evitando a inserção de seções críticas.

### Conclusão

Neste tutorial, você aprendeu:

* Como medir o tempo de aplicações com a ferramenta `time`.
* Como permitir que várias threads utilizem uma mesma variável compartilhada.
* Como usar o padrão REDUCE em OpenMP.