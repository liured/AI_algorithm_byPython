from math import log
import operator
from utils.plotTree import createPlot
import pprint

def calcuShannonEnt(dataSet):
 '''
    输入数据集：[[feat1, feat2,..., label], ... ]
    返回：该数据集的信息熵
:param dataSet:
:return:
    '''
N = len(dataSet)  # 样本数量
    labelCounts = {}  # 每个类别的样本数量
    for sample in dataSet:  # 遍历所有的样本
        currentLabel = sample[-1]  # 数据集的最后一维是样本的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:  # 计算信息熵
        prob = float(labelCounts[key]) / N
        shannonEnt = shannonEnt - prob * log(prob, 2)

    return shannonEnt

def splitDataSet(dataSet, axis, value):
 '''
    返回：第axis个特征，特征取值为value的子数据集
:param dataSet:
:param axis:
:param value:
:return:
    '''
subDataSet = []
    sample = []
    for sample in dataSet:
        if sample[axis] == value:
            reducedFeat = sample[:axis]
            reducedFeat.extend(sample[axis+1:])
            subDataSet.append(reducedFeat)

    return subDataSet

def chooseBestFeature(dataSet, model):
    numFeat = len(dataSet[0]) -1  # 计算特征的个数
    baseEntropy = calcuShannonEnt(dataSet)
    bestInforGain = 0.0
    bestFeatAxis = -1  # 将要返回的最佳分裂特征所在的维度
    for axis in range(numFeat):  # 对每个特征计算信息增益
        featList = [sample[axis] for sample in dataSet]
        uniqueValues = set(featList)  # 该特征的取值集合
        newEntropy = 0.0
        IV = 0.0
        for value in uniqueValues:  # 遍历每一种特征的取值value,计算信息增益
            subDataSet = splitDataSet(dataSet, axis, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcuShannonEnt(subDataSet)
            IV -= prob * log(prob, 2)
            # print('IV',IV)
        infoGain = baseEntropy - newEntropy
        infoGainRadio = (baseEntropy - newEntropy) / (IV+1e-8)
        if model == 'ID3':
            if infoGain > bestInforGain:  # 选择信息增益最大的特征
                bestInforGain = infoGain
                bestFeatAxis = axis
        if model == 'C4.5':
            if infoGainRadio > bestInforGain:  # 选择信息增益率最大的特征
                bestInforGain = infoGainRadio
                bestFeatAxis = axis

    return bestFeatAxis

def majorCount(classList):
 '''
    给定类别列表，返回出现次数最大的类别。
:param classList:
:return:
    '''
classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)  # operator.itemgetter(1)指定义函数，获取对象的第1个域的值

    return sortedClassCount[0][0]

def createTree(dataSet, FeatList, model='ID3'):
 '''
    计算信息增益，得到当前节点的最佳分裂特征。
    对最佳分裂特征的不同取值建立子节点，对子节点递归构建决策树。
:param dataSet:
:param FeatList:
:return:
    '''
classList = [sample[-1] for sample in dataSet] # 类别向量
    if classList.count(classList[0]) == len(classList):  # 如果只有一个类别，返回
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        return majorCount(classList)
    bestFeatAxis = chooseBestFeature(dataSet, model)  # 最优划分特征的下标
    bestFeatLabel = FeatList[bestFeatAxis]
    Tree = {bestFeatLabel:{}}
    del (FeatList[bestFeatAxis])  # 删除已经选择的特征，不再参与分类
    featValues = [sample[bestFeatAxis] for sample in dataSet]
    uniqueValue = set(featValues)  # 该特征的所有可能取值，也就是节点的分支
    for value in uniqueValue:  # 对每一个分支，递归构建树
        subLabels = FeatList[:]  # 删除了已经选择的特征后，剩下的特征
        Tree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeatAxis, value), subLabels
        )

    return Tree

def get_data():
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍陷', '软沾', '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍陷', '软沾', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍陷', '硬滑', '是'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍陷', '硬滑', '否'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软沾', '否'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软沾', '否'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍陷', '软沾', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍陷', '硬滑', '否']
    ]
    FeatList = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

    return dataSet, FeatList

def print_d(d):
    for k,v in d.items():
        print(k, v)
        if type(v) == dict:
            print_d(v)


if __name__ == "__main__":
    DataSet, FeatList = get_data()
    myTree = createTree(DataSet, FeatList, 'C4.5')
    # print(myTree)
    # createPlot(myTree)
    dic =  {'纹理': {'清晰': {'根蒂': {'硬挺': '否', '稍蜷': {'色泽': {'青绿': '是', '乌黑': {'触感': {'硬滑': '是', '软沾': '否'}}}}, '蜷缩': '是'}},
            '稍糊': {'触感': {'硬滑': '否', '软沾': '是'}}, '模糊': '否'}}
    # createPlot({'a':{'c':'1', 'd':'2'}})
    # dic = {'wl': {'qx': {'gd': {'yt': 'No', 'sq': {'sz': {'ql': 'Yes', 'wh': {'cg': {'yh': 'Yes', 'rz': 'No'}}}}, 'qs': 'Yes'}},
    #               'sh': {'qg': {'yh': 'No', 'rz': 'Yes'}}, 'mh': 'No'}}
    # createPlot(dic)
    print_d(myTree)
