# 导入库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import preprocessing


def get_metrics(y_test, y_predicted):
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


# 声明文本特征提取方法 TF-IDF
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer


# 训练并保存模型
def createAndSaveModel():
    clf_tfidf = LogisticRegression()
    clf_tfidf.fit(X_train_tfidf, y_train)
    joblib.dump(clf_tfidf, modelFile)


# 加载模型并预测，打印预测结果，给出混淆矩阵
def loadModel():
    clf_tfidf = joblib.load(modelFile)
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.6f, precision = %.6f, recall = %.6f, f1 = %.6f" % (
        accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))
    # 评估
    print("Precision, Recall, F1-Score and support")
    print(metrics.classification_report(y_test, y_predicted_tfidf, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test, y_predicted_tfidf)
    print(cm)


if __name__ == '__main__':
    # 通过pandas读入数据
    data = pd.read_excel('../1data.xls', encoding='utf-8', header=None)
    data.columns = ['class_label', 'text', 'tokens']
    label = data['class_label']
    categories = label.unique()
    # categories = []
    # for i in label:
    #     if i in categories:
    #         pass
    #     else:
    #         categories.append(i)
    print(categories)

    le = preprocessing.LabelEncoder().fit_transform(data['class_label'])
    print("标签形状:", le.shape)
    data["class_label"] = le
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data["tokens"],
                                                        data["class_label"],
                                                        test_size=0.2,
                                                        random_state=1)
    # 文本特征提取
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)

    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    modelFile = "LogicRegressionModel.m"

    # 训练保存模型
    createAndSaveModel()

    # 预测测试集结果
    loadModel()
    """
    结果： accuracy = 0.849908, precision = 0.866846, recall = 0.849908, f1 = 0.842734
    """
