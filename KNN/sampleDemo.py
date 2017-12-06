#codeing=UTF-8
from numpy import *
import operator
from KNN.KNNmain import training

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



if __name__ == "__main__":
    oriData, labels = createOriData()
    point = [-4.0, -1.0]
    className = training(point,oriData,labels,3)
    print('result %s' %className)