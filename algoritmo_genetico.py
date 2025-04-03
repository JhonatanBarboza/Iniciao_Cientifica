from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

# Configurações do algoritmo genético
IND_SIZE = 5     # Tamanho do indivíduo binário
POP_SIZE = 10    # Tamanho da população
CXPB = 0.7       # Probabilidade de crossover
MUTPB = 0.2      # Probabilidade de mutação
NGEN = 10        # Número de gerações

# Definição da estrutura (agora para maximização)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximização
creator.create("Individual", list, fitness=creator.FitnessMax)

# Inicialização do toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    """Agora o fitness é simplesmente a soma dos bits"""
    return sum(individual),

# Operadores genéticos (simplificados para representação binária)
toolbox.register("mate", tools.cxTwoPoint)  # Crossover de dois pontos
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # Mutação flip bit
toolbox.register("select", tools.selTournament, tournsize=2)
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
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
        print(f"Geração {gen}: Melhor fitness = {max(pop, key=lambda ind: ind.fitness.values[0]).fitness.values[0]}")
    
    return pop, logbook

if __name__ == "__main__":
    final_pop, log = main()
    best_ind = tools.selBest(final_pop, 1)[0]
    
    print("\n=== Resultados Finais ===")
    print(f"Melhor indivíduo encontrado: {best_ind}")
    print(f"Fitness (soma dos bits): {best_ind.fitness.values[0]}")