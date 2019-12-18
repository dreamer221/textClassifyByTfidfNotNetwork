import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.models import load_model


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


def createAndSaveModel(inputPoint):
    net = Sequential()
    net.add(Dense(input_dim=inputPoint, output_dim=10))
    net.add(Activation('relu'))
    net.add(Dense(input_dim=10, output_dim=14))
    net.add(Activation('sigmoid'))
    net.compile(optimizer='adam', loss='binary_crossentropy')
    net.fit(X_train_tfidf, y_train, batch_size=32, epochs=20)
    # 保存模型
    net.save(modelFile)


def loadModelAndPredict():
    # 载入模型
    model = load_model('model.h5')
    # 评估模型
    y_predicted_tfidf = model.predict(X_test_tfidf)
    y_predicted_tfidf = list(y_predicted_tfidf)
    y_predicted_tfidf = [list(i).index(max(list(i))) for i in y_predicted_tfidf]
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.6f, precision = %.6f, recall = %.6f, f1 = %.6f" % (
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))
    print("Precision, Recall, F1-Score and support")
    print(metrics.classification_report(y_test, y_predicted_tfidf, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test, y_predicted_tfidf)
    print(cm)


if __name__ == '__main__':
    data = pd.read_excel('../1data.xls', encoding='utf-8', header=None)
    # data = data[0:10000]
    data.columns = ['class_label', 'text', 'tokens']
    label = data['class_label']
    categories = []
    for i in label:
        if i in categories:
            pass
        else:
            categories.append(i)
    print(categories)  # 混淆矩阵需要

    le = preprocessing.LabelEncoder().fit_transform(data['class_label'])
    data["class_label"] = le
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data["tokens"],
                                                        data["class_label"],
                                                        test_size=0.2,
                                                        random_state=1)

    # y_train = tf.one_hot(y_train, 10)
    y_train = np_utils.to_categorical(y_train, num_classes=14)
    # y_test = np_utils.to_categorical(y_test, num_classes=14)

    # 文本特征提取
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    inputPoint = X_train_tfidf.shape[1]
    print("形状：", inputPoint)
    modelFile = 'model.h5'

    createAndSaveModel(inputPoint)
    loadModelAndPredict()
    """
    accuracy = 0.895754, precision = 0.896559, recall = 0.895754, f1 = 0.894497
    """
