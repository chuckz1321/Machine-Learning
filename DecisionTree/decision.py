from math import log

def calcShannon(data):
    """
    caculate the shannon entropy
    :param data:
    :return: ShannonEnt
    ent = - sum(p(xi)*log(p(xi),2))
    """
    num = len(data)
    labelData = {}
    #get the result map
    for aData in data:
        tempLabel = aData[-1]
        if tempLabel not in labelData.keys():
            labelData[tempLabel] = 0
        labelData[tempLabel] += 1
    shannon = 0.0
    for key in labelData:
        pro = float(labelData[key])/num
        shannon -= pro * log(pro,2)
    return shannon

def getSubInfo(data,index,value):
    """
    data:[
        ['1','1','2'],
        ['1','2','2'],
        ['1','3','1']
    ]
    index: 2
    value: 2
    output:
    {['1','1'],['1','2']
    :param data:
    :param index: which column
    :param value: the value
    :return: subSet
    """
    subSet = []
    for aData in data:
        if aData[index] == value:
            tempData = aData[:index]
            tempData.extend(aData[index+1:])
            subSet.append(tempData)
    return subSet
def getBestFeature(data):
    """
    get the best feature which make the shannon entropy largest
    :param data:
    :return:
    """
    setFeaturesRange = len(data[0]) -1
    shannonPara = calcShannon(data)
    bestIndex = 0.0;bestFeature = -1
    for i in range(setFeaturesRange):
        featList = [aData[i] for aData in data]
        featSet = set(featList)
        newEntropy = 0.0
        for value in featSet:
            subSet = getSubInfo(data,i,value)
            pro = len(subSet)/float(len(data))
            newEntropy += pro * calcShannon(subSet)
        temp = shannonPara - newEntropy
        if(temp > bestIndex):
            bestIndex = temp
            bestFeature = i
    return bestFeature

def testData():
    data=[
        [1, 1, 1, '√'],
        [1, 0, 0, '√'],
        [0, 0, 1, '√'],
        [0, 0, 0, '×'],
        [0, 1, 0, '×'],
        [1, 1, 0, '×']
    ]

    labels = ['long','fee','six']
    return data,labels

