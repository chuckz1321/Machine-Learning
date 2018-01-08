from NaiveBayesian.mainAlgorithm import *

import turtle

def testNB():
    orginalData, orginalCate = loadDataSet()
    elementsList = createElementList(orginalData)
    trainMat = []
    for adata in orginalData:
        trainMat.append(setOfElements2Vec(elementsList, adata))
    pVec,cateP = trainNB(array(trainMat), array(orginalCate))
    testEntry = ['love', 'my', 'dalmation']
    thisEleArray = array(setOfElements2Vec(elementsList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisEleArray, pVec, cateP, orginalCate))
    testEntry = ['stupid', 'garbage']
    thisEleArray = array(setOfElements2Vec(elementsList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisEleArray, pVec, cateP, orginalCate))
    testEntry = ['patience', 'fabulous', 'garbage']
    thisEleArray = array(setOfElements2Vec(elementsList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisEleArray, pVec, cateP, orginalCate))


if __name__ == '__main__':
    testNB()


