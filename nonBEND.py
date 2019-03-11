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

log_gamma_dict = None

def logsum(x):
    global log_gamma_dict
    if log_gamma_dict is None:
        log_gamma_dict = dict()
        for i in range(5000):
            log_gamma_dict[i+1] = np.sum(np.log(np.r_[1:i+2]))
    if x > 0 and x <= 5000:
        y = log_gamma_dict[x]
    elif x <= 0:
        y = 0
    else:
        y = (x+0.5)*np.log(x+1) + 0.5*np.log(2*np.pi) + 1/(12*(x+1))
    return y


def logGammaC(beta):
    prop = 0
    for b in beta:
        prop += logsum(b-1)
    c = prop - logsum(np.sum(beta)-1)
    return c

def calcProb(data, kappa, zOthers, obj, otherData, z, alpha, alpha0, uniValue_list):
    prob = len(np.where(zOthers == kappa)[0])/(len(data) - 1 + alpha) #CRP
    gammaLog = 0
    kappaData = otherData[np.where(zOthers == kappa)]
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
        self.alpha0 = dict()
        self.prt = prt
        self.name = name
        
    def fit_gbs(self, data):
        # Gibbs smapling
        self.data = data
        for i in range(data.shape[1]):
            self.alpha0[i] = self.alpha0value * np.ones(len(np.unique(data[:,i])))
        self.GibbsSampling()
        self.characterEstimate()
        self.embed()

    def fit_vi(self, data, T=50, n_iter=50):
        # Variational inference
        self.data = data
        for i in range(data.shape[1]):
            self.alpha0[i] = self.alpha0value * np.ones(len(np.unique(data[:, i])))
        self.VI(T=T, n_iter=n_iter)
        self.characterEstimate()
        self.embed()

    def VI(self,T=50, n_iter=50):
        data = vi.transform_data(self.data)
        g1, g2, tau, phi, ll, held_out = vi.var_dpmm_multinomial(data, self.alpha0value, T, n_iter=n_iter, Xtest=None)
        self.z = vi.get_cat(phi)
        self.ll = ll


    def GibbsSampling(self):
        numOfData = len(self.data)
        uni_value_list = list()
        for i in range(self.data.shape[1]):
            uni_value_list.append(np.unique(data[:, i]))
        #step 1, initial subspace with random assignment
        zList = np.arange(1,self.numOfZ + 1) #the subspace list
        zProb = np.ones([self.numOfZ])/self.numOfZ
        z = np.random.choice(zList, numOfData, p = zProb) #the assignment of each object
        #step 2, sampling
        self.zHistory = list()
        for epoch in range(self.maxEpoch):
            for i in tqdm(range(numOfData)):
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
        for i,c in enumerate(np.unique(self.z)):
            self.pi[i] = len(np.where(self.z == c)[0])/len(self.z)
        #estimate theta
            for j in range(self.data.shape[1]):
                uniqueValue = np.unique(self.data[np.where(self.z == c),j])
                updateLambda = np.ones(len(uniqueValue))
                for loc, value in enumerate(uniqueValue):
                    updateLambda[loc] = self.alpha0value + len(np.where(self.data[np.where(self.z == c), j] == value)[0])
                self.theta[(i,j)] = updateLambda/np.sum(updateLambda)
    
    def embed(self):
        self.embedding = np.zeros([self.data.shape[0], self.numOfClass * self.data.shape[1]])
        for j in range(self.data.shape[1]):
            for i,c in enumerate(np.unique(self.z)):
                uniqueValue = np.unique(self.data[np.where(self.z == c),j])
                for loc, value in enumerate(range(len(uniqueValue))):
                    self.embedding[np.where(self.data[:,j] == value), self.numOfClass*j + i] = self.pi[i]*self.theta[(i,j)][loc]

data_set = 'mushroom'
epochs = 30
burnin = 5
data_package = pickle.load(open('./Data/'+data_set+'.pkl', 'rb'))  # load data and label
data = data_package['data']
label = data_package['label']
model = nonBEND(prt = True, name = data_set, maxEpoch = epochs, burnin = burnin)  # model initialization
model.fit_gbs(data)