from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

Grafico = True # Variável para controle de exibição do gráfico
# Configurações do algoritmo genético
IND_SIZE = 5     # Tamanho do indivíduo (número de bits)
POP_SIZE = 20    # Tamanho da população inicial
CXPB = 0.0       # Probabilidade de crossover (recombinação)
MUTPB = 0.5      # Probabilidade de mutação
NGEN = 10        # Número de gerações
SEARCH_SPACE = [0, 5]  # Intervalo de busca para visualização

# Definição da estrutura de fitness e indivíduo
creator.create("FitnessSum", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessSum)

# Inicialização do toolbox (conjunto de ferramentas do DEAP)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Função armadilha para avaliação
def funcao_armadilha(x):
    """
    Função de avaliação personalizada.
    Retorna valores diferentes dependendo da soma dos bits.
    """
    if isinstance(x, list):  # Se for uma lista, soma os bits
        x = sum(x)
    # Avaliação condicional
    return np.where(x < 4, -x + 4, np.where(x < 5, 5*x - 20, 5))

# Função de avaliação do indivíduo
def evaluate(individual):
    """
    Avalia o indivíduo usando a função armadilha.
    """
    fitness_value = funcao_armadilha(individual)  # Avaliação pela função armadilha
    return fitness_value,  # Retorna o fitness como uma tupla

# Registro dos operadores genéticos no toolbox
toolbox.register("mate", tools.cxTwoPoint)  # Crossover de dois pontos
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # Mutação flip bit com probabilidade de 20%
toolbox.register("select", tools.selTournament, tournsize=2)  # Seleção por torneio (tamanho 2)
toolbox.register("evaluate", evaluate)  # Função de avaliação

# Função para visualização da população
def plot_population(population, generation):
    """
    Plota a população atual em relação à função armadilha.
    """
    plt.clf()  # Limpa o gráfico anterior
    x = np.linspace(0, 5, 100)  # Valores para plotar a função
    y = funcao_armadilha(x)  # Avaliação da função
    plt.plot(x, y, 'b-', label='Função Armadilha')  # Plota a função armadilha
    
    # Converte os indivíduos para valores decimais (soma dos bits)
    x_values = [sum(ind) for ind in population]
    y_values = [funcao_armadilha(ind) for ind in population]
    
    # Plota os indivíduos
    plt.scatter(x_values, y_values, c='red', label='Indivíduos', zorder=3)
    # Identifica o melhor indivíduo
    best_ind = max(population, key=lambda ind: ind.fitness.values[0])
    plt.scatter(sum(best_ind), funcao_armadilha(best_ind), 
                c='green', s=100, label='Melhor (Fitness)', zorder=4)
    
    # Configurações do gráfico
    plt.title(f'Geração {generation}\n')
    plt.xlabel('Soma dos bits')
    plt.ylabel('Valor da função')
    plt.grid(True)
    plt.legend()
    plt.xlim(SEARCH_SPACE)
    plt.pause(0.3)  # Pausa para atualizar o gráfico

# Função principal do algoritmo genético
def main():
    # Criação da população inicial
    pop = toolbox.population(n=POP_SIZE)
    # Avaliação inicial da população
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    if Grafico:
        # Plota a população inicial
        plot_population(pop, 0)
    
    # Loop principal do algoritmo genético
    for gen in range(1, NGEN+1): 
        # Seleção dos indivíduos para a próxima geração
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Aplicação do crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]): 
            if random.random() < CXPB:  # Probabilidade de crossover
                toolbox.mate(child1, child2)
                del child1.fitness.values  # Invalida o fitness
                del child2.fitness.values
        
        # Aplicação da mutação
        for mutant in offspring:
            if random.random() < MUTPB:  # Probabilidade de mutação
                toolbox.mutate(mutant)
                del mutant.fitness.values  # Invalida o fitness
        
        # Reavaliação dos indivíduos com fitness inválido
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        if Grafico:
            # Substituição da população pela nova geração
            pop[:] = offspring
            # Plota a nova população
            plot_population(pop, gen)
    
    if Grafico:
        plt.show()  # Mostra o gráfico final

    if Grafico==False:
        # Exibe o gráfico apenas no final
        plt.figure(figsize=(10, 6))
        x = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], 100)
        y = funcao_armadilha(x)
        plt.plot(x, y, 'b-', label='Função Armadilha')
        
        # Converter indivíduos binários para valores decimais para plotagem
        x_values = [sum(ind) for ind in pop]
        y_values = [funcao_armadilha(ind) for ind in pop]
        
        plt.scatter(x_values, y_values, c='red', label='População Final', zorder=3)
        best_ind = max(pop, key=lambda ind: ind.fitness.values[0])
        plt.scatter(sum(best_ind), funcao_armadilha(best_ind), 
                    c='green', s=200, label='Melhor Indivíduo (Fitness)', zorder=4)
        
        plt.title('Resultado Final - Algoritmo Genético')
        plt.xlabel('Soma dos bits do indivíduo')
        plt.ylabel('Valor da função armadilha')
        plt.grid(True)
        plt.legend()
        plt.show()

    return pop

# Execução do script
if __name__ == "__main__":
    final_pop = main()  # Executa o algoritmo genético
    # Identifica o melhor indivíduo da população final
    best_ind = max(final_pop, key=lambda ind: ind.fitness.values[0])
    
    # Exibe os resultados finais
    print("\n=== Resultados Finais ===")
    print(f"Melhor indivíduo: {best_ind}")
    print(f"Soma dos bits (Fitness): {sum(best_ind)}")
    print(f"Valor na função armadilha: {funcao_armadilha(best_ind):.2f}")