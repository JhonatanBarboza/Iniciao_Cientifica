# Fundamentos de Algoritmos Evolutivos

## Base Biológica

A ideia central é utilizar os princípios básicos da teoria da evolução genética para desenvolver algoritmos capazes de se adaptar e resolver uma ampla variedade de problemas.

### Funcionamento Básico

1. **População Inicial**: Gera-se uma população inicial de soluções candidatas de forma aleatória.
2. **Avaliação**: Cada indivíduo é avaliado em relação a uma função objetivo, recebendo um valor de fitness que indica sua qualidade.
3. **Seleção e Cruzamento**: Os indivíduos mais aptos são selecionados para cruzamento, gerando descendentes.
4. **Mutação**: Aplica-se uma taxa de mutação para introduzir variações nos descendentes.
5. **Iteração**: O processo é repetido, buscando soluções cada vez melhores.

O objetivo é encontrar soluções ótimas para o problema em questão.

### Pseudocódigo de um Algoritmo Evolutivo Típico

```plaintext
Entrada: Parâmetros típicos (De Jong, 2006).
Saída: População final de soluções.

1. INICIALIZA população com soluções candidatas aleatórias.
2. AVALIA cada candidata.
3. repita
4.     SELECIONA pais.
5.     RECOMBINA pares de pais.
6.     MUTA os descendentes resultantes.
7.     AVALIA novas candidatas.
8.     SELECIONA indivíduos para a nova geração.
9. até CONDIÇÃO DE PARADA satisfeita.
```

### Pontos Importantes

**Mutação**:  
A mutação é o processo biológico pelo qual os genes de um indivíduo sofrem alterações, resultando em características diferentes das dos pais. Essas mudanças podem ser benéficas ou prejudiciais para a adaptação do indivíduo ao ambiente. No entanto, a mutação é essencial para manter a variabilidade genética dentro de uma população.

**Seleção Natural**:  
A seleção natural ocorre devido à variação entre os membros de uma população e à pressão ambiental pela sobrevivência. Indivíduos mais adaptados ao ambiente têm maior probabilidade de sobreviver e deixar descendentes, perpetuando suas características.

**Cromossomo**:  
O cromossomo é a estrutura que codifica as características de um indivíduo. Em algoritmos evolutivos, ele é utilizado para representar as soluções candidatas dentro da população.

**Fitness**:  
O fitness é um valor numérico que indica o quão bem uma solução está adaptada ao problema. Indivíduos com maior fitness têm mais chances de serem selecionados para reprodução, propagando suas características para a próxima geração.

**Geração e Seleção**:  
Uma geração é o conjunto de indivíduos que competem entre si para determinar quais são os mais adaptados. Os indivíduos mais aptos têm maior probabilidade de gerar descendentes. Existem diversas técnicas para realizar a seleção, como elitismo, método da roleta e torneio, cada uma com suas vantagens e desvantagens.

## Algoritmos Evolutivos

Utilizando os conceitos apresentados, desenvolvem-se algoritmos genéticos para a criação de inteligências artificiais.

### Programação Evolutiva

Na programação evolutiva, cada indivíduo é representado como uma máquina de estados finitos. Após cada iteração do algoritmo, os indivíduos mais destacados são selecionados. Favorecer apenas esses indivíduos durante a seleção é chamado de **elitismo**. Essa abordagem converge rapidamente para um máximo, mas não necessariamente para o máximo global, além de reduzir a diversidade da população rapidamente.

#### Métodos de Seleção

1. **Elitismo**:  
    A geração de novos indivíduos ocorre por meio de um operador de mutação com distribuição de probabilidade Gaussiana (média zero e desvio padrão baseado no gene correspondente do pai). O elitismo completo preserva sempre o melhor indivíduo, garantindo sua sobrevivência para a próxima geração.

2. **Método da Roleta**:  
    Nesse método, os indivíduos são sorteados com uma probabilidade proporcional ao seu fitness. Indivíduos com maior fitness têm mais chances de serem escolhidos. Esse método mantém maior diversidade na população, permitindo uma exploração mais ampla do espaço de soluções, mas pode convergir mais lentamente.

    **Pseudocódigo do Método da Roleta**:
    ```plaintext
    Entrada: µ, pais.
    Saída: lista_reprodutores.

    1. cont ← 1;
    2. Calcular a probabilidade acumulada ai para cada indivíduo;
    3. enquanto cont ≤ µ faça
    4.     Obter um valor aleatório r em [0, 1];
    5.     i ← 1;
    6.     enquanto ai < r faça
    7.         i ← i + 1;
    8.         ai ← ai + P(i);
    9.     fim
    10.    lista_reprodutores[cont] ← pais[i];
    11.    cont ← cont + 1;
    12. fim
    ```

3. **Torneio**:  
    Seleciona-se um grupo de `n` indivíduos aleatoriamente. Entre eles, o indivíduo com maior fitness é escolhido para reprodução. Esse método mantém a diversidade da população e explora o espaço de soluções de forma eficiente.

    **Pseudocódigo do Algoritmo de Seleção por Torneio**:
    ```plaintext
    Entrada: População, µ.
    Saída: lista_reprodutores.

    1. cont ← 1;
    2. enquanto cont ≤ µ faça
    3.     Obter k indivíduos aleatoriamente;
    4.     Selecionar o indivíduo com maior fitness dentre k;
    5.     lista_reprodutores[cont] ← indivíduo selecionado;
    6.     cont ← cont + 1;
    7. fim
    ```

### Outros Modelos

1. **Propagação Genética**:  
    Baseia-se na transmissão direta de características genéticas dos pais para os descendentes, com pouca ou nenhuma modificação. É útil para problemas onde a exploração do espaço de soluções já foi bem delimitada.

2. **Micro-AG**:  
    Algoritmos genéticos de pequena escala que utilizam populações reduzidas. São mais rápidos e eficientes para problemas específicos, mas podem sofrer com a falta de diversidade genética.

3. **Algoritmos de Estimação de Distribuição**:  
    Substituem os operadores tradicionais de cruzamento e mutação por modelos probabilísticos que estimam a distribuição das melhores soluções, gerando novos indivíduos com base nessas estimativas.

### Algoritmos para Otimização Multiobjetivo

Nos problemas de otimização multiobjetivo, busca-se atender a múltiplos objetivos que, muitas vezes, entram em conflito entre si. Por exemplo, em um problema de engenharia, pode ser necessário minimizar o custo de produção enquanto se maximiza a qualidade do produto. Esses objetivos conflitantes tornam a otimização mais complexa, pois melhorar um objetivo pode significar piorar outro.

#### Conceito de Fronteira de Pareto

Uma abordagem comum para resolver problemas multiobjetivo é utilizar o conceito de **Fronteira de Pareto**. A Fronteira de Pareto é o conjunto de soluções que são consideradas ótimas no sentido de Pareto, ou seja, soluções onde não é possível melhorar um objetivo sem piorar pelo menos um outro. Essas soluções são chamadas de **não-dominadas**.

#### Métodos de Otimização Multiobjetivo

Existem diversas estratégias para lidar com problemas multiobjetivo. Algumas das mais comuns incluem:

1. **Método de Agregação com Pesos**:  
    Nesse método, os diferentes objetivos são combinados em uma única função objetivo por meio de pesos atribuídos a cada objetivo. A função resultante é então otimizada como um problema de objetivo único. Apesar de ser simples, essa abordagem depende fortemente da escolha dos pesos, que podem não refletir adequadamente as preferências do tomador de decisão.

2. **Aproximação da Fronteira de Pareto**:  
    Em vez de reduzir o problema a um único objetivo, busca-se encontrar um conjunto de soluções que representem a Fronteira de Pareto. Algoritmos evolutivos, como o NSGA-II (Non-dominated Sorting Genetic Algorithm II), são amplamente utilizados para esse propósito. Esses algoritmos mantêm a diversidade das soluções e permitem explorar diferentes trade-offs entre os objetivos.

3. **Método de Restrições**:  
    Um dos objetivos é otimizado enquanto os outros são tratados como restrições com limites aceitáveis. Esse método é útil quando há uma clara prioridade entre os objetivos.

4. **Método de Escalarização**:  
    Transforma o problema multiobjetivo em um problema de otimização escalar, utilizando funções de escalarização que atribuem diferentes pesos ou prioridades aos objetivos. Exemplos incluem a soma ponderada e a programação por metas.

#### Aplicações

Os algoritmos de otimização multiobjetivo são amplamente aplicados em diversas áreas, como:

- **Engenharia**: Projetos de aeronaves, automóveis e sistemas de energia.
- **Economia**: Planejamento de investimentos e alocação de recursos.
- **Ciências da Computação**: Treinamento de redes neurais e ajuste de hiperparâmetros.
- **Logística**: Roteirização de veículos e planejamento de cadeias de suprimentos.

#### Vantagens e Desafios

- **Vantagens**:  
  - Permitem explorar soluções que atendem a múltiplos critérios.
  - Oferecem maior flexibilidade na tomada de decisão.
  - Promovem a diversidade de soluções.

- **Desafios**:  
  - Alta complexidade computacional, especialmente para problemas com muitos objetivos.
  - Dificuldade em interpretar e escolher entre soluções na Fronteira de Pareto.
  - Dependência de métodos eficientes para manter a diversidade das soluções.

A otimização multiobjetivo é uma área rica e desafiadora, que continua a evoluir com o desenvolvimento de novos algoritmos e técnicas para lidar com problemas cada vez mais complexos.


capitulo 4