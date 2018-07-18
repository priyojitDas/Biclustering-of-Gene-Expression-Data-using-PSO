import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
from sklearn.cluster import KMeans

def MSR(bc):
    global data
    X = []
    Y = []
    r_size = data.shape[0]
    for i,j in enumerate(bc):
        if int(j):
            if i >= r_size:
                Y.append(i - r_size)
            else:
                X.append(i)
    sub_matrix = data.iloc[X,Y].values
    ms_b, ns_b = sub_matrix.shape
    AIJ = np.sum(sub_matrix) / float(ms_b * ns_b)
    SAI = np.sum(sub_matrix,axis=1) / float(ns_b)
    SAI = SAI.reshape((len(X),1))
    SAJ = np.sum(sub_matrix,axis=0) / float(ms_b)
    SAI_stack = np.tile(SAI,(1,len(Y)))
    SAJ_stack = np.tile(SAJ,(len(X),1))
    R = sub_matrix + AIJ - SAI_stack - SAJ_stack
    MSR = np.sum(R ** 2) / float(ms_b * ns_b)
    return len(X),len(Y),MSR

def fitness(bc):
    global data
    X = []
    Y = []
    r_size = data.shape[0]
    for i,j in enumerate(bc):
        if int(j):
            if i >= r_size:
                Y.append(i - r_size)
            else:
                X.append(i)
    Z = list(set(range(data.shape[1])) - set(Y))
    sub_matrix = data.iloc[X,Y].values
    sub_matrix_nt = data.iloc[X,Z].values
    avgin = np.abs(np.sum(sub_matrix,axis=1)) / float(sub_matrix.shape[1])
    avgout = np.abs(np.sum(sub_matrix_nt,axis=1)) / float(sub_matrix_nt.shape[1])
    gscore = avgin ** 2 + avgin * np.abs(avgin - avgout)
    return np.sum(gscore)

def seed_combination(rowC,rC,colC,cC):
    global data
    index = []
    r_size = data.shape[0]
    for i in xrange(rC):
        for j in xrange(cC):
            i_index = [a for a, b in enumerate(rowC) if b == i]
            j_index = [r_size + a for a, b in enumerate(colC) if b == j]
            index.append(i_index + j_index)
    return deepcopy(index)

def population_genration(rowC,rC,colC,cC):
    seed_matrix = np.zeros((rC * cC, rowC.shape[0] + colC.shape[0]))
    velocity = np.zeros((rC * cC, rowC.shape[0] + colC.shape[0]))
    index_matrix = seed_combination(rowC,rC,colC,cC)
    for i in xrange(seed_matrix.shape[0]):
        seed_matrix[i][index_matrix[i]] = 1
        velocity[i] = np.random.uniform(0,1,rowC.shape[0] + colC.shape[0])
    return deepcopy(seed_matrix),deepcopy(velocity)

def init(population):
    particle_best = deepcopy(population)
    f_value = []
    for ibc in population:
        f_value.append(fitness(ibc))
    f_best = f_value.index(max(f_value))
    swarm_best = deepcopy(population[f_best])
    return particle_best,swarm_best

def PSO(init_pop,velocity,rC,cC,iter=30):
    pdata = deepcopy(init_pop)
    vdata = deepcopy(velocity)
    pbdata,gbest = init(init_pop)
    for it in xrange(iter):
        print "Iteration ",it+1
        for i in xrange(pdata.shape[0]):
            C1 = C2 = 1.49
            r1 = np.random.uniform(0,1,pbdata.shape[1])
            r2 = np.random.uniform(0,1,pbdata.shape[1])
            W = 0.72
            D = (W * vdata[i]) + (C1 * r1 * (pbdata[i] - pdata[i])) + (C2 * r2 * (gbest - pdata[i]))
            #D = sigmoid(D)
            r3 = np.random.uniform(0,1,pbdata.shape[1])
            NW = np.array(r3 < D, dtype='int32')
            while not sum(NW[data.shape[0]:]):
                r1 = np.random.uniform(0,1,pbdata.shape[1])
                r2 = np.random.uniform(0,1,pbdata.shape[1])
                D = (W * vdata[i]) + (C1 * r1 * (pbdata[i] - pdata[i])) + (C2 * r2 * (gbest - pdata[i]))
                #D = sigmoid(D)
                r3 = np.random.uniform(0,1,pbdata.shape[1])
                NW = np.array(r3 < D, dtype='int32')
            ftns_new = fitness(NW)
            if (ftns_new < fitness(pbdata[i])):
                pbdata[i] = deepcopy(NW)
                if (fitness(pbdata[i]) < fitness(gbest)):
                    gbest = deepcopy(pbdata[i])
        mu = int(0.1 * pdata.shape[0])
        for i in xrange(mu):
            r = np.random.randint(pdata.shape[0])
            rand_bit = np.random.randint(pdata.shape[1])
            if pbdata[i,rand_bit]:
                pbdata[i,rand_bit] = 0
            else:
                pbdata[i,rand_bit] = 1
    return pbdata

data = pd.read_csv("Yeast.csv")
data.set_index('Gene',inplace=True)
#data = data.fillna(0.0)
#data = data.replace(999,0)

#x = data.values
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#data = pd.DataFrame(x_scaled)

delta = 300
rC = 30
cC = 3
rowC = KMeans(n_clusters=rC, random_state=0).fit(data.values).labels_
colC = KMeans(n_clusters=cC, random_state=0).fit(data.values.T).labels_

init_population, velocity_data = population_genration(rowC,rC,colC,cC)
BC = PSO(init_population,velocity_data,rC,cC,iter=50)

print 
for ib in BC:
    a,b,m = MSR(ib)
    if m <= delta:
        print "Bicluster Size :",a,"*",b,"=",a*b,"; MSR : ",m,
        ib_flag = np.array(ib[:data.shape[0]],dtype='bool')
        gene_set = np.array(data.index)[ib_flag]
        print gene_set
        print
