from DecisionTree.decision import *
from DecisionTree.geneTree import *
def classify(tree, labels, testData):
    firstSides = list(tree.keys())
    firstStr = firstSides[0]
    secondDict = tree[firstStr]
    featureIndex = labels.index(firstStr)
    key = testData[featureIndex]
    valueOfFeature = secondDict[key]
    if isinstance(valueOfFeature,dict):
        classLabel = classify(valueOfFeature,labels,testData)
    else: classLabel = valueOfFeature
    return classLabel

if __name__=='__main__':
    data,labels = testData()
    tree = createTree(data,labels)
    data,labels = testData()
    print(classify(tree,labels,[0, 0, 1]))