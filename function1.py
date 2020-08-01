import numpy as np
import  matplotlib.pyplot  as plt
import sys



"""
input: x is a boolean list. Example [1,0,1,0,1]
ouput: It converts  the boolean list to integer and square it.
input=[0,0,1,0,1] converted to integer -> 5 
output=square(5)->25
"""
def f1(x):
    #convert binary to integer
    res = 0
    for ele in x: 
        res = (res << 1) | ele 

    out=(res*res)       
    return(out)     


"""
input= population with shape=(num of genes,8) 
output= fitness value of each indiv genes with; output shape=((num of genes,)
"""
def fitness(population):
     F=np.array([f1(x) for x in population])
     F=F.astype("float32")
     return(F)


"""
Initalizing the population with random boolean values(0 or 1)
"""
def initialization(size=(10,8)):
    population = np.random.randint(low=0, high=2, size=size) #make a list
    return(population)


#select population randomly which has highest finess probability
def select(pop, pop_fitness,num=1):   
    size=pop.shape[0]
    idx = np.random.choice(np.arange(size), size=num, replace=True,
                           p=(pop_fitness/pop_fitness.sum()))
    return pop[idx]


#crossover between two genes ind_0,ind_1
def crossover(ind_0, ind_1):
    point = np.random.randint(len(ind_0))   #crossover point is selected randomly
    new_0 = np.hstack((ind_0[:point], ind_1[point:]))
    new_1 = np.hstack((ind_1[:point], ind_0[point:]))
    return new_0, new_1



#mutatate individual gene
def mutation(indiv):
    point = np.random.randint(len(indiv))   
    indiv[point] = 1 - indiv[point]  #complement the bit value
    return indiv



def rws(size, fitness):
    fitness= 1.0 / fitness
    idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True,p=fitness_/fitness_.sum())
    return idx




"""
@ input parameters 
num_gen         : number of generation to run iteration
pop_size        : population size
pc              : crossover probability
pm              : mutation probabiliy

@output : returns (fitness value over time,best gene)
"""
def run_GA(num_gen=100,pop_size=4,pc=0.7,pm=0.2):
    #To keep track of the best individuals
    best_idx = -1           #best index in the copulation
    best_indiv=-1           #best gene in the population
    best_fitness_val = -999   #fitness value of best gene

    """
    random initilization  of population.since maximum
    range is 255 hence we have taken 8 bit representation 
    """
    pop=initialization(size=(pop_size,8))

    F_list=[]
    for i in range(num_gen):  
        sys.stdout.write('\r running generation %d '%(i,))
        sys.stdout.flush()

        pop_fitness=fitness(pop)  #calcutate fitness of individual genes 
        #pop=select(pop, pop_fitness,num=pop.shape[0])

        next_gene = []
        for n  in range(int(pop.shape[0]/2)):
            child1=select(pop, pop_fitness,num=1)[0] #select child randomly with probability proportional to fitness value
            child2=select(pop, pop_fitness,num=1)[0]

            #do crossover with probability pc
            if np.random.rand() < pc:
                child1,child2 = crossover(child1, child2)

            #do mutation with probability pc
            if np.random.rand() < pm:
                child1 = mutation(child1)
                child2 = mutation(child2)
            next_gene.append(child1)
            next_gene.append(child2)

        pop = np.array(next_gene)   #save new genes to the next population
        pop_fitness=fitness(pop)  #calcutate fitness of individual genes 
        #print(best_indiv)

        #track the best gene
        if np.max(pop_fitness) > best_fitness_val:
            best_idx = np.argsort(pop_fitness)[-1]   
            best_indiv=pop[best_idx]
            best_fitness_val = pop_fitness[best_idx]
            count = 0
            #print(best_indiv)
        F_list.append(best_fitness_val)

    return(F_list,best_indiv)



#run GA
print("running with population size=2")
f1_list,best_gene=run_GA(num_gen=1000,pop_size=2,pc=0.7,pm=0.2)
print("| Best gene ",best_gene)
print("running with population size=4")
f2_list,best_gene=run_GA(num_gen=1000,pop_size=4,pc=0.7,pm=0.2)
print("| Best gene ",best_gene)
print("running with population size=8")
f3_list,best_gene=run_GA(num_gen=1000,pop_size=8,pc=0.7,pm=0.2)
print("| Best gene ",best_gene)



#ploting graph  
plt.show()
plt.plot(f1_list)
plt.plot(f2_list)
plt.plot(f3_list)
l=["population size= "+str(x) for x in [2,4,8]]
plt.legend(l)
plt.xlabel("# of Iteration")
plt.ylabel("Fitness value")
print("figure saved at  out/function1_graph.png")
plt.savefig("out/function1_graph.png")





