# Códigos exemplos: Paralelismo de Tarefas utilizando a diretiva `task`

Nesta página você encontra alguns exemplos de paralelismo de tarefas utilizando a diretiva `task`.

A grande vantagem de se utlizar `task` e não `sections` está no fato de que `task`s podem ser `untied`, ou seja, não estão vinculadas a um núcleo específico, enquanto que `sections` estão vinculadas ao núcleo onde a thread principal foi criada.