# deprecated: módulo legado conservado para evaluación manual.

import numpy as np

def count_to_prob(counts):
    if '1' not in counts.keys():
        return 1
    return counts['0']/(counts['0'] + counts['1'])

def dictionary_to_probabilities(dict):

    for k in dict.keys():
        pos, val = k.split(" ")
        print(pos, val)

def similitudTradicional(u, v, max_val):
    pr = 0

    sims = [abs((int(u[i])-int(v[i]))/max_val) for i in range(len(u))]
    # print("Debug******************************************")
    # print(u)
    # print(v)
    # print(u-v)
    # print(sims)
    # print(sum(sims))
    # print(len(sims))
    # print("DebugEND***************************************")

    return 1- sum(sims)/len(sims)

def obtenerProbabilidadEsperada_Full(u, v, max_val):

    pr = 0

    for i in range(len(u)):
        theta = np.pi*u[i]/max_val
        phi = np.pi*v[i]/max_val

        pr += (np.cos((theta-phi)/2))
    
    return 1/2 + (1/2)*((pr/len(u))**2)

def obtenerProbabilidadEsperada_Parcial(u, v, max_val):

    pr = 0

    for i in range(len(u)):
        theta = np.pi*u[i]/max_val
        phi = np.pi*v[i]/max_val

        pr += (np.cos(theta-phi)+1)
    
    return 1/2 + (1/2)*((pr/2)/len(u))

def obtenerProbabilidadesEsperadaMultivector_Full(u, vs, max_val):

    prbs = []

    M = len(vs)
    N = len(u)

    for v in vs:
        pr = 0

        for i in range(N):
            theta = np.pi*u[i]/max_val
            phi = np.pi*v[i]/max_val

            pr += (np.cos((theta-phi)/2))

        prbs.append(1/(2*M) + (1/(2*M))*(     (   pr/N   )**2     ))
    
    return np.array(prbs)

def obtenerProbabilidadesEsperadaMultivector_Parcial(u, vs, max_val):

    prbs = []

    M = len(vs)
    N = len(u)

    for v in vs:
        pr = 0

        for i in range(N):
            theta = np.pi*u[i]/max_val
            phi = np.pi*v[i]/max_val

            pr += (np.cos(theta-phi) + 1)

        prbs.append(1/(2*M) + (1/(2*M))*(     (   pr/2   )/N     ))
    
    return np.array(prbs)