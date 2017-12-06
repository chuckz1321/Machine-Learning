#coding=UTF-8
from numpy import *
from KNN.KNNmain import training
import operator

def txtToMatrix(filename) :
    """
    convert the data in txt to matrix
    :param filename: as defined...
    :return:
    """
    fr = open(filename)
    textFormat = fr.readlines()
    textLine = len(textFormat)
    originalData = zeros((textLine,3))
    labels = []
    index = 0
    for line in textFormat:
        line = line.strip().split('\t')
        originalData[index,:] = line[0:3]
        labels.append(line[-1])
        index += 1
    return originalData,labels

def normalize(originalData):
    """
    regnerate the Data
    :param originalData:
    :return: new Matrix
    """
    minParam = originalData.min(0)
    maxParam = originalData.max(0)
    range = maxParam-minParam
    normData = zeros(shape(originalData))
    m = originalData.shape[0]
    normData = originalData - tile(minParam,(m,1))
    normData = normData/tile(range,(m,1))
    return normData

def dateClassTest():
    """
    :return: the result
    """
    # define the ratio of the data for judgement
    hoRatio = 0.05
    datingDataMat, datingLabels = txtToMatrix('dateData1.txt')  # load data set from file
    normMat = normalize(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        print(normMat[i])
        # 5 is knn's k
        classifierResult = training(normMat[i], normMat[numTestVecs:m], datingLabels[numTestVecs:m], 5)
        print("the classifier came back with: %s, the real answer is: %s, result is :%s" % (
            classifierResult, datingLabels[i], classifierResult == datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

if __name__ == '__main__':
    dateClassTest()
