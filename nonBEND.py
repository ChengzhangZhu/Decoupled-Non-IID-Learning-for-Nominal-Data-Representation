# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:53:13 2017

@author: Chengzhang Zhu (kevin.zhu.china@gmail.com)

This code provides the nonBEND class for decoupled non-IID nominal data representation

Two fitting methods are provides:
(1) Collapsed Gibbs Sampling (fit_gbs)

(2) Variational inference (fit_vi)
"""

import numpy as np
import pickle
#from scipy.special import gamma
from scipy.stats import mode
import var_dpmm as vi
import time
from tqdm import tqdm

log_gamma_dict = dict()
for i in range(5000):
    log_gamma_dict[i+1] = np.sum(np.log(np.r_[1:i+2]))



def logsum(x, log_gamma_dict=log_gamma_dict):
    if x > 0 and x <= 5000:
        y = log_gamma_dict[x]
    elif x <= 0:
        y = 0
    else:
        y = logsumcalc(x)
    return y


def logsumcalc(x):
    return (x+0.5)*np.log(x+1) + 0.5*np.log(2*np.pi) + 1/(12*(x+1))

def logGammaC(beta):
    prop = 0
    for b in beta:
        prop += logsum(b-1)
    c = prop - logsum(np.sum(beta)-1)
    return c


def calcProb(data, kappa, zOthers, obj, otherData, z, alpha, alpha0, uniValue_list):
    sub_space_loc = np.where(zOthers == kappa)
    prob = len(sub_space_loc[0])/(len(data) - 1 + alpha) #CRP
    gammaLog = 0
    kappaData = otherData[sub_space_loc]
    for i in range(data.shape[1]):
        uniValue = uniValue_list[i]
        ti = np.zeros(len(uniValue))
        tni = np.zeros(len(uniValue))
        ti[np.where(uniValue == obj[i])] = 1
        for j in range(len(uniValue)):
            tni[j] = len(np.where(kappaData[:,i] == uniValue[j])[0])
        gammaLog += logGammaC(ti+tni+alpha0[i])- logGammaC(tni+alpha0[i])
    prob = prob * np.exp(gammaLog)
    return prob


def calcProbNew(data, obj, alpha, alpha0):
    prob = alpha/(len(data) - 1 + alpha) #CRP
    gammaLog = 0
    for i in range(data.shape[1]):
        uniValue = np.unique(data[:,i])
        ti = np.zeros(len(uniValue))
        ti[np.where(uniValue == obj[i])] = 1
        gammaLog += logGammaC(ti+alpha0[i]) - logGammaC(alpha0[i])
    prob = prob * np.exp(gammaLog)
    return prob


def updateClassList(zList, z):
    clearList = list()
    for i in range(len(zList)):
        if len(np.where(z == zList[i])[0]) == 0:
            clearList.append(i)
    zList = np.delete(zList, clearList) #delet empty class
    zList = np.array(list(set(list(z)+list(zList))))  #add new class
    return zList


class nonBEND(object):
    def __init__(self, name = None, alpha = 1, numOfZ = 10, maxEpoch = 2000, burnin = 1000, alpha0 = 1, prt = False):
        self.alpha = alpha
        self.numOfZ = numOfZ #number of the initial random subspaces
        self.maxEpoch = maxEpoch
        self.burnin = burnin
        self.alpha0value = alpha0
        self.alpha0 = list()
        self.prt = prt
        self.name = name
        
    def fit_gbs(self, data, embedding_method='naive', weighted=True):
        # Gibbs sampling
        self.data = data
        for i in range(data.shape[1]):
            self.alpha0.append(self.alpha0value * np.ones(len(np.unique(data[:,i]))))
        self.GibbsSampling()
        self.characterEstimate()
        self.embed(embedding_method=embedding_method, weighted=weighted)

    def fit_vi(self, data, T=50, n_iter=50, embedding_method='naive', weighted=True):
        # Variational inference
        self.data = data
        for i in range(data.shape[1]):
            self.alpha0.append(self.alpha0value * np.ones(len(np.unique(data[:, i]))))
        self.VI(T=T, n_iter=n_iter)
        self.characterEstimate()
        self.embed(embedding_method=embedding_method, weighted=weighted)

    def VI(self,T=50, n_iter=50):
        data = vi.transform_data(self.data)
        g1, g2, tau, phi, ll, held_out = vi.var_dpmm_multinomial(data, self.alpha0value, T, n_iter=n_iter, Xtest=None)
        self.z = vi.get_cat(phi)
        self.ll = ll

    def GibbsSampling(self):
        numOfData = len(self.data)
        uni_value_list = list()
        for i in range(self.data.shape[1]):
            uni_value_list.append(np.unique(self.data[:, i]))
        #step 1, initial subspace with random assignment
        zList = np.arange(1,self.numOfZ + 1) #the subspace list
        zProb = np.ones([self.numOfZ])/self.numOfZ
        z = np.random.choice(zList, numOfData, p = zProb) #the assignment of each object
        #step 2, sampling
        self.zHistory = list()
        for epoch in range(self.maxEpoch):
            for i in range(numOfData):
                zProb = np.zeros(len(zList)+1)
                for j in range(len(zList)):
                    zProb[j] = calcProb(self.data, zList[j], np.concatenate((z[:i],z[i+1:])), self.data[i,:], np.concatenate((self.data[:i,:],self.data[i+1:,:])), z, self.alpha, self.alpha0, uni_value_list)
                zProb[len(zList)] = calcProbNew(self.data, self.data[i,:], self.alpha, self.alpha0)
                zProb = zProb/np.sum(zProb)
                z[i] = np.random.choice(np.concatenate((zList,np.array([np.max(zList)+1]))), 1, p = zProb)
                zList = updateClassList(zList, z) #clear empty class and add new class
            self.zHistory.append(z.copy())
            if self.prt == True:
                print('The',epoch+1, '-th runs class distribution:', dict((c, list(z).count(c)) for c in list(z)))
        z = mode(np.array(self.zHistory)[self.burnin:,:])[0][0]
        self.z = np.array([np.where(i == np.unique(z))[0][0] for i in z])
    
    def characterEstimate(self):
        #estimate pi
        self.numOfClass = len(np.unique(self.z))
        self.pi = np.ones(self.numOfClass)
        self.theta = dict()
        self.weight = list()
        for i,c in enumerate(np.unique(self.z)):
            self.pi[i] = len(np.where(self.z == c)[0])/len(self.z)
        # estimate theta
            for j in range(self.data.shape[1]):
                uniqueValue = np.unique(self.data[:,j])
                updateLambda = np.ones(len(uniqueValue))
                update_lambda = dict()
                for loc, value in enumerate(uniqueValue):
                    updateLambda[loc] = self.alpha0value + len(np.where(self.data[np.where(self.z == c), j] == value)[0])
                    update_lambda[value] = self.alpha0value + len(np.where(self.data[np.where(self.z == c), j] == value)[0])
                for value in update_lambda:
                    update_lambda[value] = update_lambda[value]/np.sum(updateLambda)
                self.theta[(i, j)] = update_lambda
        # estimate attribute weight
        for j in range(self.data.shape[1]):
            freq_dict = dict()
            for value in np.unique(self.data[:, j]):
                freq_dict[value] = list()
                for i, c in enumerate(np.unique(self.z)):
                    freq_dict[value].append(self.theta[(i, j)][value])
                freq_dict[value] = np.std(freq_dict[value])
            self.weight.append(np.exp(np.mean(list(freq_dict.values()))))

    def embed(self, embedding_method='naive', weighted=True):
        if embedding_method == 'naive':
            self.embedding = np.zeros([self.data.shape[0], self.numOfClass * self.data.shape[1]])
            for j in range(self.data.shape[1]):
                for i,c in enumerate(np.unique(self.z)):
                    uniqueValue = np.unique(self.data[np.where(self.z == c), j])
                    for loc, value in enumerate(range(len(uniqueValue))):
                        self.embedding[np.where(self.data[:,j] == value), self.numOfClass*j + i] = self.pi[i]*self.theta[(i,j)][value]
        else:  # one-zero embedding multiplies the exp(-frequency) with the distribution std as weight
            embedding_dict = dict()
            for j in range(self.data.shape[1]):
                one_zero_dict = dict()
                for i, c in enumerate(np.unique(self.z)):
                    unique_value = np.unique(self.data[:, j])
                    # one zero embedding
                    one_zero_dict[i] = dict()
                    for value_index, value in enumerate(unique_value):
                        one_zero_dict[i][value] = np.ones(len(unique_value))
                        one_zero_dict[i][value][value_index] = 0
                    # multiply exp(-frequency) and subspace probability
                    for value in one_zero_dict[i]:
                        one_zero_dict[i][value] = self.pi[i] * one_zero_dict[i][value] * np.exp(-self.theta[(i, j)][value])
                embedding_dict[j] = one_zero_dict
            data_embedding_list = []
            for obj in self.data:
                data_embedding = []
                for j, value in enumerate(obj):
                    feature_embedding = []
                    for z in embedding_dict[j]:
                        feature_embedding.append(embedding_dict[j][z][value])
                    if weighted:
                        data_embedding.append(self.weight[j] * np.concatenate(feature_embedding, axis=0))
                    else:
                        data_embedding.append(np.concatenate(feature_embedding, axis=0))
                data_embedding_list.append(np.concatenate(data_embedding, axis=0))
            self.embedding = np.stack(data_embedding_list)

# data_set = 'soybeansmall'
# epochs = 50
# burnin = 5
# data_package = pickle.load(open('./Data/'+data_set+'.pkl', 'rb'))  # load data and label
# data = data_package['data']
# label = data_package['label']
# model = nonBEND(prt = True, name = data_set, maxEpoch = epochs, burnin = burnin)  # model initialization
# model.fit_vi(data, embedding_method='one-zero')
# print(model.embedding)
# print(model.weight)