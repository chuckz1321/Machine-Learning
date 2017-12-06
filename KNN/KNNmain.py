from numpy import *
import operator
def training(underClassifyData, oriData, labels, k):
    """
    :param underClassifyData: index under classify
    :param oriData: original DataSet
    :param labels: original labels
    :param k: KNN's k
    :return: classifacation of the point
    """

    #return the number of the origianl data
    oriDataSize = oriData.shape[0]
    #tile() to generate matrix which has the same column and row with the origianl data
    #get the physical distance individually
    trainingData = tile(underClassifyData,(oriDataSize,1)) - oriData
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