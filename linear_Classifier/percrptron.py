#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:12:00 2016

@author: gromit
"""

import numpy as np
from matplotlib import pyplot as plt

def load_data(filepath):
    #load the dataset as a ndarray.
    dataset = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=0)
    print("item {0} {1}".format("dimension is",dataset.shape))
    return dataset

class Percrptron(object):
    def __init__(self, step = 0.1, iterations = 10):
        self.step = step
        self.iterations = iterations
    
    def compute_value(self, data):
        return np.dot(data, self.w) + self.d
    
    def predicted_tag(self, data):
        return np.where(self.compute_value(data) >= 0, 1, -1)
    
    def training(self, training_set, training_tag):
        sample_num, attribute_num = training_set.shape
        self.w = np.zeros(attribute_num)
        self.d = 0
        count = 0
        while count < self.iterations:
            for sample, tag in zip(training_set, training_tag):
                result = self.compute_value(sample)*self.predicted_tag(sample)
                #print result
                if result <= 0:
                    #increment = self.step * (tag-self.predicted_tag(sample))
                    self.w += self.step * tag * sample
                    self.d += self.step * tag
                    print self.w,self.d
            count += 1
        return self
      
def corrected_value(w, d):
    return -w[0]/w[1], -d/w[1]

def show_plot(dataset, data_tag, w, d):
    data_num = dataset.shape[0]
    marker = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(data_num):
        marker_type = int(data_tag[i,] + 1)
        plt.plot(dataset[i, 0], dataset[i, 1], marker[marker_type], markersize = 7)
    plt.xlim(xmax=3,xmin=-1)
    plt.ylim(ymax=3,ymin=-1)
    w, d = corrected_value(w, d)
    print w,d
    x = np.array(xrange(-5, 5))
    plt.plot(x, w*x + d, color = 'b')
    plt.title("percrptron")
    plt.savefig('percrptron.jpg')
    plt.show()
      
def main():
    training_set = load_data("DemoSamples.csv")
    training_tag = load_data("DemoLabels.csv")
    percrptron = Percrptron(step = 0.01, iterations = 30)
    percrptron.training(training_set, training_tag)
    show_plot(training_set,training_tag, percrptron.w, percrptron.d)
    print percrptron.w
    print percrptron.d
    
    
      
      
if __name__ == '__main__':
    main()       