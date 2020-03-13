'''
@Author: your name
@Date: 2020-03-11 19:05:30
@LastEditTime: 2020-03-12 21:56:54
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \\undefinedc:\\Users\\dongj\\Desktop\\Adaboost.py
'''
import numpy as np


def getAlpha(error):
    alpha = 0.5 * np.math.log((1.0 - error) / max(error, 1e-16))
    return alpha



#通过阈值对数据分类+1 -1
#dimen为dataMat的列索引值，即特征位置；threshIneq为阈值对比方式，大于或小于
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((dataMatrix.shape[0],1))
    #阈值的模式，将小于某一阈值的特征归类为-1
    if threshIneq=='lt':#less than
        retArray[dataMatrix[:,dimen] <= threshVal]=-1.0
    #将大于某一阈值的特征归类为-1
    else:#greater than
        retArray[dataMatrix[:,dimen] > threshVal]=-1.0
    return retArray



#单层决策树生成函数
def buildStump(dataArr, labels, D):
    dataMatrix = dataArr
    labels = np.squeeze(labels.T)
    m, n = dataMatrix.shape #m是样本数，n是特征数
    #步长或区间总数 最优决策树信息 最优单层决策树预测结果
    numSteps=10.0
    bestStump={}
    bestClasEst = np.zeros((m,1))
    minError = np.inf
     #遍历数据集的每个特征：遍历特征的每个步长：遍历步长的每个阈值对比方式
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / 2.
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                thresV = rangeMin + float(j) * stepSize
                predV = stumpClassify(dataMatrix, i, thresV, inequal)
                errArr = np.ones((m, 1))
                errArr[np.squeeze(predV.T) == labels] = 0
                
                weightError = np.dot(D.T, errArr)
                if weightError < minError:
                    minError = weightError
                    bestClasEst = predV.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=thresV
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst


def Adaboost(dataArr, labels, num=40):
    weakClassArr = []
    m = dataArr.shape[0]
    D = np.ones((m, 1)) / m
    aggClassEst = np.zeros((m, 1))
    labels = labels[np.newaxis, :].T
    for i in range(num):
        bestStump, minError, bestClasEst = buildStump(dataArr, labels, D)
        alpha = getAlpha(minError)
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        
        expon = np.multiply(-1 * alpha * labels, bestClasEst)
        D = np.multiply(np.exp(expon), D) / np.sum(D)
        # D = np.squeeze(D)
        if len(D.shape) > 2:
            D = np.squeeze(D, 0)
        aggClassEst += alpha * bestClasEst
        aggError = np.multiply(np.sign(aggClassEst) != labels.T, np.ones((m, 1)))
        errorRate = aggError.sum() / m
        if errorRate == 0:
            break
    return weakClassArr, errorRate, aggClassEst


data = np.matrix([[ 1. ,  2.1],[ 2. ,  1.1],[ 1.3,  1. ],[ 1. ,  1. ],[ 2. ,  1. ]])
label = [1.0,1.0,-1.0,-1.0,1.0]
D = np.ones((5,1))/5
weakClassArr, errorRate, aggClassEst = Adaboost(np.array(data), np.array(label))
print(weakClassArr)