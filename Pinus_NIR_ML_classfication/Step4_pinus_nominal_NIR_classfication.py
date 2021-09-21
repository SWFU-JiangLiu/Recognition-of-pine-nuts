import pandas as pd
import numpy as np
# 计算精确度/召回率/F1值
def acc_f1_recall(y_test,y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    from sklearn.metrics import classification_report
    cv = classification_report(y_test, y_pred)
    print('精确度/召回率/F1值\n', cv)
    return cm
# 决策树
from sklearn import preprocessing
def decisiontree(x_train,y_train,x_test,y_test):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=1)
    clf = model.fit(x_train, y_train)
    # print('决策树节点数：', clf.n_estimators)
    result = model.score(x_test, y_test)
    print('决策树结果为：', result)
    y_pred = model.predict(x_test)
    cm=acc_f1_recall(y_test,y_pred)
    return cm.reshape(1,49),clf.score(x_test,y_test)
    # print('决策树混淆矩阵\n', cm)
# 神经网络
def neural_network(x_train,y_train,x_test,y_test):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(random_state=1)
    clf1 = model.fit(x_train, y_train)
    # print(clf1)
    print('神经网络结果为：', model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    cm=acc_f1_recall(y_test,y_pred)
    return cm.reshape(1,49),clf1.score(x_test,y_test)
# 随机森林
def randomforest(x_train,y_train,x_test,y_test):
    from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(random_state=1, n_estimators=100)
    model = RandomForestClassifier(random_state=43, n_estimators=100)
    clf1 = model.fit(x_train, y_train)
    print('随机森林结果为：', model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    cm=acc_f1_recall(y_test,y_pred)
    return cm.reshape(1,49),clf1.score(x_test,y_test)
# SVM SVM中有SVC和SVR 其中SVC一般用于分类SVR一般用于回归
def SVM(x_train,y_train,x_test,y_test):
    from sklearn.svm import SVC
    clf=SVC()
    clf.fit(x_train,y_train)
    print('SVM分类结果',clf.score(x_test,y_test))
    y_pred=clf.predict(x_test)
    cm=acc_f1_recall(y_test,y_pred)
    return cm.reshape(1,49),clf.score(x_test,y_test)
# Multinomial Naive Bayes Classifier  朴素贝叶斯
def Bayes(x_train,y_train,x_test,y_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    print('Bayes分类结果为:',clf.score(x_test,y_test))
    y_pred = clf.predict(x_test)
    cm=acc_f1_recall(y_test,y_pred)
    return cm.reshape(1,49),clf.score(x_test,y_test)
if __name__ == '__main__':
    # 1、读取数据文件
    # ----------------------------------------------------------------------------------
    df = pd.read_csv('./../data/NIR_data/3_Pinus_nominal_NIR_data_all.csv')
    dt_acc1=np.empty((0, 1), int)
    bp_acc1=np.empty((0, 1), int)
    rf_acc1=np.empty((0, 1), int)
    svm_acc1=np.empty((0, 1), int)
    bayes_acc1=np.empty((0, 1), int)
    for i in range(10):
        # 2、数据划分（调用sklearn的方法进行划分）
        # ----------------------------------------------------------------------------------
        from sklearn.model_selection import train_test_split
        # 随机选择20%作为验证集
        train, test = train_test_split(df, test_size=0.2)
        x_train = train.drop('class', axis=1)
        y_train = train['class']
        x_test = test.drop('class', axis=1)
        y_test = test['class']
        # 3、调用方法进行训练
        # ----------------------------------------------------------------------------------
        dt,dt_acc= decisiontree(x_train, y_train, x_test, y_test)
        # dt, dt_acc = decisiontree(x_train_norm3, y_train, x_test_norm3, y_test)
        print('决策树混淆矩阵\n', dt.reshape(7, 7))
        bp,bp_acc = neural_network(x_train, y_train, x_test, y_test)
        # bp, bp_acc = neural_network(x_train_norm3, y_train, x_test_norm3, y_test)
        print('神经网络混淆矩阵\n', bp.reshape(7, 7))
        rf,rf_acc=randomforest(x_train,y_train,x_test,y_test)
        # rf, rf_acc = randomforest(x_train_norm3, y_train, x_test_norm3, y_test)
        print('随机森林混淆矩阵\n',rf.reshape(7,7))
        # svm, svm_acc=SVM(x_train_norm3, y_train, x_test_norm3, y_test)
        svm,svm_acc=SVM(x_train,y_train,x_test,y_test)
        print('SVM混淆矩阵\n',svm.reshape(7,7))
        bayes,bayes_acc = Bayes(x_train, y_train, x_test, y_test)
        # bayes, bayes_acc = Bayes(x_train_norm3, y_train, x_test_norm3, y_test)
        print('bayes混淆矩阵\n', bayes.reshape(7, 7))
        dt_acc1 = np.vstack((dt_acc1, dt_acc))
        bp_acc1 = np.vstack((bp_acc1, bp_acc))
        rf_acc1 = np.vstack((rf_acc1, rf_acc))
        svm_acc1= np.vstack((svm_acc1, svm_acc))
        bayes_acc1 = np.vstack((bayes_acc1, bayes_acc))
    print(dt_acc1.T)
    print(bp_acc1.T)
    print(rf_acc1.T)
    print(svm_acc1.T)
    print(bayes_acc1.T)