from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

# Configurações
IND_SIZE = 5     # Tamanho do indivíduo binário
POP_SIZE = 10    # Tamanho da população
CXPB = 0.7       # Probabilidade de crossover
MUTPB = 0.2      # Probabilidade de mutação
NGEN = 10        # Número de gerações
SEARCH_SPACE = [0, 5]  # Intervalo para visualização

# Definição da estrutura (para maximização da soma)
creator.create("FitnessSum", base.Fitness, weights=(1.0,))  # Maximizar soma
creator.create("Individual", list, fitness=creator.FitnessSum)

# Inicialização do toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def funcao_armadilha(x):
    """Função original (para avaliação)"""
    if isinstance(x, list):
        x = sum(x)  # Soma dos elementos para avaliação
    return np.where(x < 4, -x + 4, np.where(x < 5, 5*x - 20, 5))

def evaluate(individual):
    """Avaliação pela função armadilha, mas fitness é a soma dos bits"""
    return sum(individual),  # Fitness = soma dos bits

# Operadores genéticos
toolbox.register("mate", tools.cxTwoPoint)  # Crossover para binários
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # Mutação flip bit
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate)

def main():
    # Executa o algoritmo genético
    pop = toolbox.population(n=POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    for gen in range(1, NGEN+1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
    
    # Plot apenas no final
    plt.figure(figsize=(10, 6))
    x = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], 100)
    y = funcao_armadilha(x)
    plt.plot(x, y, 'b-', label='Função Armadilha')
    
    # Converter indivíduos binários para valores decimais para plotagem
    x_values = [sum(ind) for ind in pop]
    y_values = [funcao_armadilha(ind) for ind in pop]
    
    plt.scatter(x_values, y_values, c='red', label='População Final')
    best_ind = max(pop, key=lambda ind: ind.fitness.values[0])
    plt.scatter(sum(best_ind), funcao_armadilha(best_ind), 
                c='green', s=200, label='Melhor Indivíduo (Fitness)')
    
    plt.title('Resultado Final - Algoritmo Genético')
    plt.xlabel('Soma dos bits do indivíduo')
    plt.ylabel('Valor da função armadilha')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return pop

if __name__ == "__main__":
    final_pop = main()
    best_ind = max(final_pop, key=lambda ind: ind.fitness.values[0])
    
    print("\n=== Resultados Finais ===")
    print(f"Melhor indivíduo: {best_ind}")
    print(f"Soma dos bits (Fitness): {sum(best_ind)}")
    print(f"Valor na função armadilha: {funcao_armadilha(best_ind):.2f}")