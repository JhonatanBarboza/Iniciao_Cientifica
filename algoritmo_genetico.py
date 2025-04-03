from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

# Configurações do algoritmo genético
IND_SIZE = 1  # Otimização unidimensional
POP_SIZE = 50
CXPB, MUTPB, NGEN = 0.7, 0.3, 100
SEARCH_SPACE = [0, 5]  # Intervalo de busca

# Definição da estrutura
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimização
creator.create("Individual", list, fitness=creator.FitnessMin)

# Inicialização do toolbox
toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, SEARCH_SPACE[0], SEARCH_SPACE[1])
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def funcao_armadilha(x):
    """Função armadilha personalizada conforme especificação"""
    if isinstance(x, list):
        x = x[0]  # Extrai o valor do indivíduo
    
    return np.where(x < 4,
                    -x + 4,        # Segmento decrescente 0-4
                    np.where(x < 5,
                             5*x - 20,  # Segmento crescente 4-5
                             -1))       # Constante para x > 5

def evaluate(individual):
    """Função de avaliação"""
    return funcao_armadilha(individual),

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields
    
    for gen in range(NGEN):
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
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    
    return pop, logbook

def plot_function():
    """Visualização da função personalizada"""
    x = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], 1000)
    y = funcao_armadilha(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Função Armadilha Personalizada')
    
    # Marcando características importantes
    plt.scatter(0, 4, color='green', label='Máximo Global (0,4)')
    plt.scatter(4, 0, color='red', label='Mínimo Global (4,0)')
    plt.scatter(5, 5, color='orange', label='Máximo Local (5,5)')
    
    plt.title('Função Armadilha Especificada')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_function()  # Mostra a função antes de executar
    
    final_pop, log = main()
    best_ind = tools.selBest(final_pop, 1)[0]
    best_x = best_ind[0]
    
    print("\n=== Resultados ===")
    print(f"Melhor solução encontrada: x = {best_x:.4f}")
    print(f"Valor da função: f(x) = {funcao_armadilha(best_x):.4f}")
    print(f"Valor no mínimo global (x=4): {funcao_armadilha([4])[0]:.4f}")