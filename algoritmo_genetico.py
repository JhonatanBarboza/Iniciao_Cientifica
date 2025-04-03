from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configurações do algoritmo genético
IND_SIZE = 10    # Otimização unidimensional
POP_SIZE = 10   # Reduzido para melhor visualização
CXPB = 0.7      # Probabilidade de crossover
MUTPB = 0.2     # Probabilidade de mutação
NGEN = 10       # Número de gerações
SEARCH_SPACE = [0, 5]  # Intervalo de busca

# Definição da estrutura
creator.create("FitnessMin", base.Fitness, weights=(1.0,))  # Minimização
creator.create("Individual", list, fitness=creator.FitnessMin)

# Inicialização do toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def funcao_armadilha(x):
    """Função armadilha personalizada"""
    if isinstance(x, list):
        x = x[0]
    return np.where(x < 4, -x + 4, np.where(x < 5, 5*x - 20, -1))

def evaluate(individual):
    return funcao_armadilha(individual),

def clip_value(value, min_val, max_val):
    """Garante que o valor esteja dentro dos limites"""
    return min(max(value, min_val), max_val)

def cxBlend_clipped(ind1, ind2, alpha):
    """Crossover blend com clipping para manter dentro dos limites"""
    for i in range(len(ind1)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = clip_value(gamma * ind1[i] + (1. - gamma) * ind2[i], SEARCH_SPACE[0], SEARCH_SPACE[1])
        ind2[i] = clip_value(gamma * ind2[i] + (1. - gamma) * ind1[i], SEARCH_SPACE[0], SEARCH_SPACE[1])
    return ind1, ind2

def mutGaussian_clipped(individual, mu, sigma, indpb):
    """Mutação gaussiana com clipping para manter dentro dos limites"""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = clip_value(random.gauss(mu, sigma) + individual[i], SEARCH_SPACE[0], SEARCH_SPACE[1])
    return individual,

# Operadores genéticos modificados
toolbox.register("mate", cxBlend_clipped, alpha=0.5)
toolbox.register("mutate", mutGaussian_clipped, mu=0, sigma=0.3, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Configuração do plot
plt.ion()
fig, ax = plt.subplots(figsize=(12, 7))
x_plot = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], 1000)
y_plot = funcao_armadilha(x_plot)
ax.plot(x_plot, y_plot, 'b-', label='Função Armadilha')
ax.set_xlim(SEARCH_SPACE)
ax.set_ylim(-2, 6)
ax.grid(True)
ax.set_title('Evolução do Algoritmo Genético')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# Elementos do plot que serão atualizados
pop_scatter = ax.scatter([], [], c='red', label='Indivíduos')
best_scatter = ax.scatter([], [], c='green', s=100, label='Melhor Indivíduo')
ax.legend()

def update_plot(population, generation):
    """Atualiza o plot com os indivíduos atuais"""
    x_values = [ind[0] for ind in population]
    y_values = [funcao_armadilha(ind) for ind in population]
    
    # Atualiza os scatter plots
    pop_scatter.set_offsets(np.column_stack((x_values, y_values)))
    
    # Encontra e destaca o melhor indivíduo
    best_ind = tools.selBest(population, 1)[0]
    best_scatter.set_offsets([best_ind[0], funcao_armadilha(best_ind)])
    
    ax.set_title(f'Evolução do Algoritmo Genético - Geração {generation}')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Pausa para visualização

def main():
    pop = toolbox.population(n=POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Plot inicial
    update_plot(pop, 0)
    
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
        
        # Atualiza o plot a cada geração
        update_plot(pop, gen)
    
    plt.ioff()
    return pop, logbook

if __name__ == "__main__":
    final_pop, log = main()
    best_ind = tools.selBest(final_pop, 1)[0]
    
    print("\n=== Resultados Finais ===")
    print(f"Melhor solução encontrada: x = {best_ind[0]:.4f}")
    print(f"Valor da função: f(x) = {funcao_armadilha(best_ind):.4f}")
    
    # Plot final estático
    plt.figure(figsize=(12, 7))
    plt.plot(x_plot, y_plot, 'b-', label='Função Armadilha')
    
    x_final = [ind[0] for ind in final_pop]
    y_final = [funcao_armadilha(ind) for ind in final_pop]
    plt.scatter(x_final, y_final, c='red', label='População Final')
    plt.scatter(best_ind[0], funcao_armadilha(best_ind), 
                c='green', s=200, label='Melhor Solução')
    
    plt.title('Resultado Final do Algoritmo Genético')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()