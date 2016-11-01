#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:00:31 2016

@author: gromit
"""

import numpy as np
import random
import time


def load_data(filepath):
    #load the dataset as a ndarray.
    data = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=0)
    data = np.array(data)
    print("item {0} {1}".format("dimension is",data.shape))
    return data

def initial_centroid(data, k, random_seed):
    #initial k random centroids from the dataset by different seed
    sample_num, sample_dim = data.shape
    random.seed(random_seed)
    centroids_index = random.sample(range(0, sample_num), k)
    centroids = np.zeros((k, sample_dim))
    for i in range(k):
        centroids[i,:] = data[centroids_index[i],:]
    #print centroids
    return centroids

def init_covariance(data,k):
    covariance = [0] * k
    for i in range(k):
        covariance[i] = np.cov(data.T)
    return covariance    
    
def gaussian(data,mean,covariance):
        dim = np.shape(covariance)[0]
        determinant = np.linalg.det(covariance + np.eye(dim)*0.01)
        inverse = np.linalg.inv(covariance + np.eye(dim)*0.01)
        difference = data - mean
        probability = 1.0/np.power(2*np.pi,1.0*dim/2)/np.sqrt(np.abs(determinant))*np.exp(-1.0/2*np.dot(np.dot(difference,inverse),difference))
        return probability
        
class GMM(object):
    
    def __init__(self,data,k,threshold = 1e-15,random_seed = 1):
        self.data = data
        self.num,self.dim = data.shape
        self.k = k
        self.threshold = threshold
        self.random_seed = random_seed
     
    def gmm_fit(self):
        data = self.data
        N = self.num
        dim = self.dim
        K = self.k
        means = initial_centroid(data,K,self.random_seed)
        convs = init_covariance(data,K)
        pis = [1.0/K]*K
        gammas = [np.zeros(K) for i in range(N)]
        likelyhood = 0
        pre_likelyhood = 1
        deviation = np.abs(likelyhood - pre_likelyhood)
        while deviation > self.threshold:
            pre_likelyhood = likelyhood
            for n in range(N):
                res = [pis[k]*gaussian(data[n],means[k],convs[k]) for k in range(K)]
                sum_res = np.sum(res)
                for k in range(K):
                    gammas[n][k] = res[k]/sum_res

            for k in range(K):
                nk = np.sum([gammas[n][k] for n in range(N)])
                means[k] = 1.0/nk * np.sum([gammas[n][k]*data[n] for n in range(N)],axis=0)
                diffs = data - means[k]
                convs[k] = 1.0/nk * np.sum([gammas[n][k]*diffs[n].reshape(dim,1)*diffs[n] for n in range(N)],axis=0)
                pis[k] = 1.0*nk/N
            likelyhood = np.sum([np.log(np.sum([pis[k]*gaussian(data[n],means[k],convs[k]) for k in range(K)])) for n in range(N)])
            deviation = np.abs(likelyhood - pre_likelyhood)
        
        return [pis,means,convs]

class GMMClassifier(object):
    def __init__(self,data,para):
        self.data = data
        self.para = para
        self.gmm_num = len(para)
        self.cluster_table = np.zeros((data.shape[0],len(para)))
        
    def classify(self):
        data = self.data
        N,dim = data.shape
        K = len(self.para[0][0])
        for n in range(N):
            prob = []
            for i in range(self.gmm_num):
                pis,means,convs = self.para[i]
                temp_prob = np.sum([pis[k]*gaussian(data[n],means[k],convs[k]) for k in range(K)])
                prob.append(temp_prob)
            cluster_index = prob.index(max(prob))
            self.cluster_table[n,cluster_index] = 1
        return self.cluster_table
    
    def get_accuracy(self,tag):
        N,K = self.cluster_table.shape
        predtag = np.zeros((N,1))
        for n in range(N):
            flag = np.where(self.cluster_table[n,:])
            predtag[n,0] = int(flag[0])
        match_index = [int(predtag[n] - tag[n]) for n in range(N)]
        count = match_index.count(0)
        accuracy = 1.0*count/N       
        return [count,accuracy]


def main():
    training_data = load_data("TrainSamples.csv")
    training_tag = load_data("TrainLabels.csv")
    group = []
    for group_id in range(10):
        group_index = np.where(training_tag == group_id)
        group.append(list(group_index[0]))
    GMMBox = []
    PARABox = []
    k = 3
    threshold = 1e-15
    random_seed = 1
    for i in range(10):
        start = time.clock()
        trainset = training_data[group[i],]
        gmm = GMM(trainset,k,threshold,random_seed)
        para = gmm.gmm_fit()
        GMMBox.append(gmm)
        PARABox.append(para)
        end = time.clock() - start
        print("para%d done %s sec" %(i,end))
        
    print "classification loading..."
    start = time.clock()
    testset = load_data("TestSamples.csv")
    testtag = load_data("TestLabels.csv")
    classifier = GMMClassifier(testset,PARABox)
    pred = np.mat(classifier.classify())
    filename = "pred_GMM_10_k{:d}.csv".format(k)
    np.savetxt(filename,pred,fmt="%d",delimiter=",")
    result = classifier.get_accuracy(testtag)
    end = time.clock() - start
    print("classification done %s sec"%(end))
    print result
    
    
    '''
    data1 = load_data("Train1.csv")     
    data2 = load_data("Train2.csv")
    print '**'*10
    k = 2
    threshold = 1e-15
    random_seed = 2
    gmm1 = GMM(data1,k,threshold,random_seed)
    para1 = gmm1.gmm_fit()
    gmm2 = GMM(data2,k,threshold,random_seed)
    para2 = gmm2.gmm_fit()
    para = [para1,para2]
    print 'classification...'
    testset1 = load_data("Test1.csv")
    testset2 = load_data("Test2.csv")
    c1 = GMMClassifier(testset1,para)
    result1 = np.mat(c1.classify())
    np.savetxt("Result1.csv",result1,fmt="%d",delimiter=",")
    c2 = GMMClassifier(testset2,para)
    result2 = np.mat(c2.classify())
    np.savetxt("Result2.csv",result2,fmt="%d",delimiter=",")
    '''
        
if __name__ == '__main__':
    main()