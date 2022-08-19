# -*- coding: utf-8 -*-
# @Time : 2022/8/17 15:34
# @Author : zhuyu
# @File : 编程作业.py
# @Project : Python菜鸟教程

import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

#查看训练集中某张样本图片
# index=25
# plt.imshow(train_set_x_orig[index])
# plt.show()

print(train_set_x_orig.shape) #(209,64,64,3)
print(train_set_x_orig.shape[1])
print(train_set_y.shape) #(1,209)

m_train=train_set_y.shape[1] #训练集样本数量
m_test=test_set_y.shape[1] #测试集样本数量
num_px=train_set_x_orig.shape[1] #单张样本像素大小

#将数据集的维度降低并转置  二维：特征数*样本数
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
print(train_set_x_flatten.shape) #(12288, 209)
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print(test_set_x_flatten.shape)#(12288, 50)

#数据集输入数据归一化  x属于[0,1]
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

def sigmoid(z):
    """
    sigmoid网络结构最后的非线性激活函数，用来输出预测的概率结果，在0-1之间
    :param z:任何大小的标量或numpy数组
    :return: sigmoid(z)
    """
    return 1/(1+np.exp(-z))

#测试sigmoid()
print("="*10+"测试sigmoid函数"+"="*10)
print("sigmoid(0)=",sigmoid(0))
print("sigmoid([0-8]矩阵)=",sigmoid(np.arange(0,9).reshape(3,3)))

#初始化参数w和b
def initialize_with_zeros(dim):
    """
    创建一个(dim,1)的0向量初始化w，并将b初始化为0
    :param dim: 参数数量
    :return: (w,b)
    """
    w=np.zeros(shape=(dim,1))
    b=0
    assert w.shape==(dim,1)
    assert (isinstance(b,float)or isinstance(b,int))

    return (w,b)

#测试initialize_with_zeros()
print("="*10+"测试initialize_with_zeros函数"+"="*10)
w_test,b_test=initialize_with_zeros(2)
print("w_test=",w_test)
print("b_test=",b_test)

#计算样本的损失cost和梯度grad
def propagate(w,b,X,Y):
    """
    实现前向传播和反向传播的成本函数及其梯度
    :param w: 权重 （num_px*num_px*3,1）
    :param b: 偏差 标量
    :param X: 样本的输入特征 (num_px*num_px*3,训练样本数量)
    :param Y: 样本的标签矢量-二分类0和1 矩阵维度(1,训练样本数量)
    :return:(grads,cost)
            cost - 逻辑回归的负对数似然成本
            以字典的形式存储grad
            dw - w的损失梯度，与w的形状相同
            db - b的损失梯度，与b的形状相同
    """
    m=X.shape[1] #样本数量

    #正向传播 - 计算损失函数cost
    z=np.dot(w.T,X)+b #计算sigmoid函数参数z
    A=sigmoid(z) #计算sigmoid函数
    cost=(-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A))) #计算损失函数 逻辑回归的负对数似然成本-交叉熵损失函数

    #反向传播 - 计算梯度grad
    dw=(1/m)*np.dot(X,(A-Y).T) #参考视频中公式
    db=(1/m)*np.sum(A-Y)

    #使用assert断言 确定数据是正确的，从而方便调试避免在运行时出错
    assert dw.shape==w.shape
    assert db.dtype==float

    #创建一个字典，将dw和db存起来
    grads={
        "dw":dw,
        "db":db
    }

    return (grads,cost)

#测试propagate()
print("="*10+"测试propagate函数"+"="*10)
w_test,b_test=np.array([[1],[2]]),2
X_test,Y_test=np.array([[1,2],[3,4]]),np.array([1,0])
grads_test,cost_test=propagate(w_test,b_test,X_test,Y_test)
print("test cost:",cost_test)
print("test grads:")
print("dw:{0}  db:{1}".format(grads_test["dw"],grads_test["db"]))

#由反向传播获得的grads梯度来更新优化参数
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    """
    通过运行梯度下降算法来优化w和b
    :param w:权重 (num_px*num_px*3,1)
    :param b: 标量
    :param X: 维度(num_px*num_px_3,训练样本数量)
    :param Y: 维度(1,num_px*_num_px*3)
    :param num_iterations: 优化循环的迭代次数 ； 这里没有采用mini-batch，因此样本全部正向计算一遍后即循环遍历完整的一遍样本才更新一次参数
    :param learning_rate:学习率
    :param print_cost: 每100步打印一次损失值
    :return:(params,grads,costs)
            params - 包含w和b的字典
            grads - 包含dw和db的字典
            costs - 优化期间所有成本列表，用于绘制学习曲线
    """

    costs=[] #记录每次迭代循环的损失函数

    # 这里没有采用mini - batch，因此样本全部正向计算一遍后即循环遍历完整的一遍样本才更新一次参数
    for epoch in range(1,num_iterations+1):
        #正向传播
        grads,cost=propagate(w,b,X,Y)

        dw=grads["dw"]
        db=grads["db"]

        #梯度下降法更新
        w=w-learning_rate*dw
        b=b-learning_rate*db

        #记录成本
        if epoch %100 == 0:
            costs.append(cost)
        #打印损失函数
        if print_cost and epoch % 100 == 0:
            print("迭代的次数：{0} . loss = {1}".format(epoch,cost))

    #创建params和grads字典记录参数及其梯度
    params={
        "w":w,
        "b":b
    }
    grads={
        "dw":dw,
        "db":db
    }
    return (params,grads,costs)

#测试optimize
print("="*10+"测试optimize函数"+"="*10)
w_test,b_test,X_test,Y_test=np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([1,0])
params,grads,costs=optimize(w_test,b_test,X_test,Y_test,num_iterations=200,learning_rate=0.1,print_cost=True)
print("w_test=",params["w"])
print("b_test=",params["b"])
print("dw_test=",grads["dw"])
print("db_test=",grads["db"])
print("costs = ",costs)

#测试optimize
print("====================测试optimize====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

def predict(w,b,X):
    """
    使用训练好的logistic模型(w,b)来预测测试集的标签是0or1
    :param w: 最终优化后的权重 维度(num_px*num_px*3,1)
    :param b: 最终优化后的偏差 标量
    :param X: 输入的测试集X 维度(num_px*num_px*3,测试样本数量)
    :return: Y_prediction - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组
    """
    m=X.shape[1] #测试集样本数量
    Y_predications=np.zeros((1,m)) #最终输出的列表
    w.reshape(X.shape[0],1)

    #计算预测概率
    A=sigmoid(np.dot(w.T,X)+b)
    # A中预测概率大于0.5的输出结果为1 ； 小于0.5的输出结果为0
    # y_gr=np.where(A>=0.5) #返回A中大于0.5的索引
    # y_ls=np.where(A<0.5) #返回A中小于0.5的索引
    for i in range(A.shape[1]):
        Y_predications[0,i]=1 if A[0,i]>0.5 else 0
    #使用断言
    assert (Y_predications.shape==(1,m))

    return Y_predications

#测试predict
print("="*10+"测试predict函数"+"="*10)
Y_predictions_test=predict(np.array([[1],[2]]),2,X=np.array([[1,2],[3,4]]))
print("Y_predictions_test : ",Y_predictions_test)

#将前面的函数整合到一个函数中model
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    :param X_train: numpy数组 维度(num_px*num_px*3,m_train)的训练集
    :param Y_train: numpy数组 维度(1，num_px*num_px*3)的训练集标签
    :param X_test: numpy数组 维度(num_px*num_px*3,m_test)的测试集
    :param Y_test: numpy数组 维度(1,num_px*num_px*3)的测试集标签
    :param num_iterations: 超参数 - 权重迭代优化的次数
    :param learning_rate: 超参数 - 学习率
    :param print_cost:  设置是否打印每100次训练迭代的损失函数
    :return: d - 返回有关模型信息的字典
    """

    #初始化模型参数
    w,b =initialize_with_zeros(X_train.shape[0])

    #正向传播并利用梯度下降优化参数
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations=num_iterations,learning_rate=learning_rate,print_cost=print_cost)
    #从字典中检索优化后的参数w和b
    w,b=parameters["w"],parameters["b"]

    #预测训练集/测试集的例子
    Y_predictions_train=predict(w,b,X_train)
    Y_predictions_test=predict(w,b,X_test)

    #打印训练后的准确性
    print("训练集准确性accuracy：",format(100 - np.mean(np.abs(Y_predictions_train - Y_train)) * 100) ,"%")
    print("测试集准确性accuracy：",format(100 - np.mean(np.abs(Y_predictions_test - Y_test)) * 100) ,"%")

    #返回模型基本信息的字典
    d={
        "costs":costs,
        "Y_predictions_train":Y_predictions_train,
        "Y_predictions_test":Y_predictions_test,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations
    }

    return d

#实际数据测试model
print("="*10+"实际数据测试model函数"+"="*10)
d=model(X_train=train_set_x,Y_train=train_set_y,
        X_test=test_set_x,Y_test=test_set_y,
        num_iterations=2000,learning_rate=0.005,
        print_cost=True)

#绘制图
print(d["costs"])
costs=np.squeeze(d["costs"]) #np.squeeze(arr,axis) 从arr中删除一维条目 [[[]]] -> [[]]
plt.plot(costs)
plt.xlabel("iterations(per hundreds)")
plt.ylabel("cost")
plt.title("Learning rate = "+str(d["learning_rate"]))
plt.show()

#分析不同学习率α的选择对梯度下降的作用效果
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
