from DecisionTree.decision import *
import operator

def majorCnt(classList):
    """
    if all the features are classified, select the most emerging result to be the final res
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(data,labels):
    classList = [example[-1] for example in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(data[0]) == 1:
        return majorCnt(classList)
    bestFeature = getBestFeature(data)
    bestFeatureLabel = labels[bestFeature]
    tree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featureValues = [example[bestFeature] for example in data]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:]
        tree[bestFeatureLabel][value] = createTree(getSubInfo(data,bestFeature,value),subLabels)
    return tree

if __name__ == "__main__":
    data,labels = testData()
    print(createTree(data,labels))
