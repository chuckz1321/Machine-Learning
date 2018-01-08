#coding=UTF-8

from numpy import *


def loadDataSet():
    """
    prepare for the data
    :return: data and cate
    """
    orginal = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['work', 'patience', 'fabulous', 'so', 'food', 'with', 'team']]
    cate = ['normal', 'garbage', 'normal', 'garbage', 'normal', 'good']    # 0 noraml, 1 garbage, 2 good
    return orginal, cate


def createElementList(dataSet):
    """
    return all element from dataset
    :param dataSet: orginal data
    :return: list of elements
    """
    elementSet = set([])
    for document in dataSet:
        elementSet = elementSet | set(document)
    return list(elementSet)


def setOfElements2Vec(elementList, aDataSet):
    """
    whether the element emerge in this row
    :param elementList: from the createElementList(dataSet)
    :param aDataSet: a row of original data
    :return:
    """
    aVec = [0]*len(elementList)
    for element in aDataSet:
        if element in elementList:
            aVec[elementList.index(element)] = 1
        else: print("the word: %s is not in elementry dictionary!" % element)
    return aVec


def bagOfWords2VecMN(elementList, aDataSet):
    """
    times of the element emerge
    :param elementList:
    :param aDataSet:
    :return:
    """
    returnVec = [0]*len(elementList)
    for word in aDataSet:
        if word in elementList:
            returnVec[elementList.index(word)] += 1
    return returnVec


def trainNB(trainMatrix,trainCategory):
    """
    train the original data and return the probabilities individually
    :param trainMatrix: the array of training data
    :param trainCategory: categories of original data
    :return: each cate's probability
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    cataSet = set(trainCategory)
    allCataNums = len(trainCategory)
    cataNums = len(cataSet)
    catap = []  # each cata's probably
    for standCata in cataSet:
        emerge = 0.0
        for aCata in trainCategory:
            if standCata == aCata : emerge += 1
        catap.append(emerge/float(allCataNums))
    #all element init to 1 ,in case of the log(0)
    nums = []  # each element's chance
    pList = []
    for i in range(cataNums):
        nums.append(ones(numWords))
        pList.append(2.0)
    for i in range(numTrainDocs):
        for j in range(cataNums):
            if trainCategory[i] == list(cataSet)[j]:
                nums[j] += trainMatrix[i]
                pList[j] += sum(trainMatrix[i])
    #in case of out of range
    pVec = []
    for i in range(cataNums):
        pVec.append(log(nums[i]/pList[i]))
    return pVec, catap


def classifyNB(testData, pVec, cateP, originalCate):
    """
    find the proper cate
    :param testData:
    :param pVec: training probabilities
    :param cateP: cate's prob```
    :param originalCate: cates
    :return:
    """
    finalP = -999999999999999999.0
    cateIndex = -1;
    for i in range(len(pVec)):
        temp = sum(testData * pVec[i]) + log(cateP[i])
        if temp > finalP:
            cateIndex = i
            finalP = temp
    return list(set(originalCate))[cateIndex]


