#coding=utf-8
import math
import csv
import random
import operator

class KNN(object):
    def __init__(self):
        pass
    
    def loadData(self,filename,split,trainingSet,testSet):
        with open(filename,'r') as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            for x in range(len(dataset)-1):
                for y in range(4):
                    dataset[x][y]=float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
    
    def calDistance(self,testdata,traindata,length):
        distance = 0
        for x in range(length):
            distance += pow((testdata[x]-traindata[x]),2)
        return math.sqrt(distance)
    
    def getNeighbors(self,trainingSet,testInstance,k):
        distance = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
            dist = self.calDistance(testInstance,trainingSet[x],length)
            distance.append((trainingSet[x],dist))
            print('{}--{}'.format(trainingSet[x],dist))
        distance.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distance[x][0])
        print(neighbors)
        return neighbors
    
    def getResponse(self,neighbor):
        classVotes = {}
        for x in range(len(neighbor)):
            response = neighbor[x][-1]
            if response in classVotes:
                classVotes[response]+=1
            else:
                classVotes[response]=1
            print(classVotes.items())
        sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]
    
    def Run(self):
        trainingSet = []
        testSet = []
        predictions = []
        k = 3
        split = 0.75
        self.loadData(r'testdata.txt',split,trainingSet,testSet)
        for x in range(len(testSet)):
            neighbors = self.getNeighbors(trainingSet,testSet[x],k)
            result = self.getResponse(neighbors)
            predictions.append(result)
        acc = self.getAccuracy(testSet,predictions)
        print('Acc:'+repr(acc)+'%')
    
    def getAccuracy(self,testSet,predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct+=1
        print('right to predict:{},number of testSet:{}'.format(correct,len(testSet)))
        return (100.0*correct/float(len(testSet)))
    
if __name__ == '__main__':
    a = KNN()
    a.Run()