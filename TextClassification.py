import pandas as pd
from model.svm_model import SVMModel
from model.naive_bayes_model import NaiveBayesModel
import json

class TextClassificationPredict(object):
    def __init__(self, text):
        self.text = text
      

    def get_train_data(self):
        #  train data file:
        with open('Data.json', encoding='utf8') as myfile:
             obj = json.load(myfile)
             
       # print(train_data)
        df_train = pd.DataFrame(obj)

        #  test data
        test_data = []
        test_data.append({"feature": self.text})
        df_test = pd.DataFrame(test_data)

        # init model naive bayes
        model = SVMModel()

        clf = model.clf.fit(df_train["feature"], df_train.target)

        predicted = clf.predict(df_test["feature"])

        # Print predicted result
        print (predicted)
        print (clf.predict_proba(df_test["feature"]))
        return predicted


if __name__ == '__main__':
    tcp = TextClassificationPredict('')
    tcp.get_train_data()