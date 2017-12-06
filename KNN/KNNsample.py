#codeing=UTF-8
from numpy import *
import operator

def createOriData():
    """
    :return:
    """
    #init the original data array(called model)
    data = array([
        [0.0,1.0],
        [2.0,1.0],
        [-1.,1.0],
        [-2.,4.0]
    ])
    #tap these labels on the data above
    dataLabels = ['A','A','B','B']
    return data, dataLabels

def training(point, oriData, labels, k):
    """
    :param point: index under classify
    :param oriData: original DataSet
    :param labels: original labels
    :param k: KNN's k
    :return: classifacation of the point
    """

    #return the number of the origianl data
    oriDataSize = oriData.shape[0]
    #tile() to generate matrix which has the same column and row with the origianl data
    #get the physical distance individually
    trainingData = tile(point,(oriDataSize,1)) - oriData
    #square the phy-distance
    sqTrainingData = trainingData ** 2
    #sum the row getting the sqDistance
    sqDistance = sqTrainingData.sum(axis=1)
    # extract the sqDistance getting the distance
    distance = sqDistance ** 0.5
    # sort the distance and return the original index
    sortIndex = distance.argsort()
    classCount = {}

    for i in range(k):
        label = labels[sortIndex[i]]
        # record the count of classifation seperately
        classCount[label] = classCount.get(label,0) + 1
    # sort the count
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == "__main__":
    oriData, labels = createOriData()
    point = [-4.0, -1.0]
    className = training(point,oriData,labels,3)
    print('result %s' %className)