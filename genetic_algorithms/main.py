"""
A
B
C
"""
from random import random, randint, choices
from typing import List

def random_gen():
    return [1 if random() < 0.5 else 0 for i in range(24)]

def create_population(size = 30):
    return [random_gen() for _ in range(size)]

def encode(states: List[List[int]], writes):
    genome = []
    for i in range(len(states)):
        for j in range(len(states[i])):
            genome.append(states[i][j] // 2)
            genome.append(states[i][j] % 2)
            genome.append(writes[i][j])
    return genome

def grow(genome):
    states = [[]]
    writes = [[]]
    k = 0
    j = 0
    for i in range(0, len(genome), 3):
        states[k].append(2 * genome[i] + genome[i + 1])
        writes[k].append(genome[i + 2])
        j += 1
        if j == 2:
            j = 0
            k += 1
            if k < (len(genome) // 6):
                states.append([])
                writes.append([])
    return states, writes


def adapt(states, writes, inputs):
    output = []
    start = 0
    for i in range(len(inputs)):
        output.append(writes[start][inputs[i]])
        start = states[start][inputs[i]]
    c = 0
    for i in range(len(inputs) - 1):
        if output[i] == inputs[i + 1]:
            c += 1
    return c


def mutate(genome):
    genome = genome.copy()
    p = 1 / len(genome)
    for i in range(len(genome)):
        if random() < p:
            genome[i] = 1 - genome[i]
    return genome


def cross(genome1, genome2):
    p = randint(1, len(genome1) - 1)
    return genome1[:p] + genome2[p:], genome2[:p] + genome1[p:]

def eval_population(population, inputs):
    f = []
    for x in population:
        s, e = grow(x)
        f.append(adapt(s, e, inputs))
    return f

def select(p, fitness):
    p1 = []
    n = len(fitness)
    for i in range(n):
        k = randint(0, n - 1)
        for j in range(3):
            k1 = randint(0, n - 1)
            if fitness[k1] > fitness[k]:
                k = k1
        p1.append(p[k])
    return p1

def generational(n, iters, inputs):
    p = create_population(n)
    f = eval_population(p, inputs)
    for i in range(iters):
        p1 = select(p, f)
        p = []
        for j in range(0, n, 2):
            g1, g2 = cross(p1[j], p1[j+1])
            p.append(mutate(g1))
            p.append(mutate(g2))
        f = eval_population(p, inputs)
        print(i+1, min(f), sum(f)/n, max(f))
    return p[f.index(max(f))]


if __name__ == '__main__':
    inputs = [1,0,0,1,0,0,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,0,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,0,1,0,1]
    best = generational(100, 1000, inputs)
    print(f'Genotype: {best}')
    s, e = grow(best)
    print(f'Fenotype: {s} *\n {e}')
    print(f'Adaptability: {adapt(s, e, inputs)}')