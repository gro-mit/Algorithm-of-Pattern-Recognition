#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:13:09 2016

@author: gromit
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    #load the dataset as a ndarray.
    dataset = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=0)
    print("item {0} {1}".format("dimension is",dataset.shape))
    return dataset
    
class LMSE(object):
    def __init__(self,data,tag):
        self.data = data
        self.tag = tag
        
    def fit_line(self):
        Y = np.column_stack((self.tag, self.data))
        Y = np.mat(Y)
        pseudo_inverse_Y = np.dot((Y.T*Y).I,Y.T)
        b = np.mat([1,1,1,1,1,1]).T
        a = pseudo_inverse_Y*b
        print a
        self.line_para = a
        
    def corrected_value(self):
        return float(self.line_para[0]/self.line_para[2]),float(self.line_para[1]/self.line_para[2])

def show_plot(dataset, data_tag, line_para_k,line_para_b):
    data_num = dataset.shape[0]
    marker = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(data_num):
        marker_type = int(data_tag[i,] + 1)
        plt.plot(dataset[i, 0], dataset[i, 1], marker[marker_type], markersize = 7)
    plt.xlim(xmax=3,xmin=-1)
    plt.ylim(ymax=3,ymin=-1)
    x = np.array(xrange(-5, 5))
    plt.plot(x, line_para_k*x+line_para_b, color = 'b')
    plt.title("LMSE")
    plt.savefig('LMSE.jpg')
    plt.show()
    
def main():
    data = load_data("DemoSamples.csv")
    tag = load_data("DemoLabels.csv")
    lmse = LMSE(data,tag)
    lmse.fit_line()
    k,b = lmse.corrected_value()
    show_plot(data,tag,k,b)
    print type(k)
    print k,b
    
  
if __name__ == '__main__':
    main() 