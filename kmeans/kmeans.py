#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:17:29 2016

@author: gromit
"""

import numpy as np
import random
import copy
from matplotlib import pyplot as plt

def load_data(filepath):
    #load the dataset as a ndarray.
    dataset = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=0)
    print("item {0} {1}".format("dimension is",dataset.shape))
    return dataset

def euclidean_metric(v1,v2):
    #compute the euclidean metric of two samples
    return pow(sum(pow(v2 - v1, 2)), 0.5)

def initial_centroid(dataset, k, random_seed):
    #initial k random centroids from the dataset by different seed
    sample_num, sample_dim = dataset.shape
    random.seed(random_seed)
    centroids_index = random.sample(range(0, sample_num), k)
    centroids = np.zeros((k, sample_dim))
    for i in range(k):
        centroids[i,:] = dataset[centroids_index[i],:]
    print centroids
    return centroids

def kmean(dataset, k, random_seed):
    #kmean algorithm implement
    centroids = initial_centroid(dataset, k, random_seed)
    sample_num, sample_dim = dataset.shape
    sample_predict = np.zeros((sample_num, 1))
    iterations = 0
    while(True):
        previous_centroids = copy.deepcopy(centroids)
        iterations += 1
        for i in range(sample_num):
            nearest_tag = 0
            min_distance = float("inf")
            for j in range(k):
                distance = euclidean_metric(centroids[j,], dataset[i,])
                if distance < min_distance:
                    min_distance = distance
                    nearest_tag = j
            if sample_predict[i,] != nearest_tag:
                sample_predict[i,] = nearest_tag
            
        for j in range(k):
            sub_cluster = dataset[[index for index in range(sample_num) if sample_predict[index,0] == j],]
            centroids[j,] = np.mean(sub_cluster, axis = 0)
        if(previous_centroids == centroids).all():
            break
    print("{0} {1}".format("clustering finished, the iterations is",iterations))
    return centroids, sample_predict

def show_cluster(dataset, k, centroids, sample_predict):
    #plot the demo sample clustering
    sample_num = dataset.shape[0]
    marker = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(sample_num):
        marker_type = int(sample_predict[i,0])
        plt.plot(dataset[i, 0], dataset[i, 1], marker[marker_type])
    marker = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], marker[i], markersize = 12)
    plt.title("k-mean cluster")
    plt.savefig('cluster.jpg')
    plt.show()

def show_comparison(dataset, k, prediction, tag):
    #obtain the result of clustering
    sample_num = dataset.shape[0]
    prediction.shape = -1,1
    print prediction
    tag.shape = -1,1
    print tag
    cluster_counter = []
    for i in range(k):
        sub_cluster = list(tag[[index for index in range(sample_num) if prediction[index] == i],0])
        for j in range(k):
            cluster_counter.append(sub_cluster.count(j))
    result_table = np.asarray(cluster_counter)
    result_table.shape = k,-1
    print result_table
    
    #for i in range(sample_num):
    #    result_table[int(prediction[i]),int(tag[i])] += 1
    np.savetxt('cluster_result.csv', result_table, delimiter = ',') 
    print("comparision result has been written")

def main():
    dataset = load_data("ClusterSamples.csv")
    data_tag = load_data("SampleLabels.csv")
    k = 10
    random_seed = 1
    centroids, prediction = kmean(dataset, k, random_seed)
    show_comparison(dataset, k, prediction, data_tag)
    print centroids
    print prediction


if __name__ == '__main__':
    main()


