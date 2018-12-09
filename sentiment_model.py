#!/usr/bin/env python3

import pandas as pd
import collections
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

model = svm.SVC() 

df_training = pd.read_csv("training.txt")
df_testing = pd.read_csv("testing.txt")
training_data = df_training.loc[:, df_training.columns != 'label']
training_label = df_training['label']

testing_data = df_testing.loc[:, df_testing.columns != 'label']
testing_label = df_testing['label']

model.fit(training_data, training_label)
predictions = model.predict(testing_data)
correct = 0

data = collections.Counter()
total_articles = collections.Counter()
wrong_guesses = collections.defaultdict(collections.Counter)

for i, ans in enumerate(testing_label):
    total_articles[ans] += 1
    predicted = predictions[i]
    if ans == predicted:
        correct += 1
        data[ans] += 1
    else:
        wrong_guesses[ans][predicted] += 1
    # print("{}->{}".format(ans, predicted))

print(correct / len(predictions))

guess_data = []
for source in total_articles:
    percent = data[source] / total_articles[source]
    guess_data.append((percent, source))

guess_data.sort(reverse=True)
print('='*20)
for percent, source in guess_data:
    print("Source: {} Accuracy: {}".format(source, percent * 100))

print('='*20)
for source in wrong_guesses:
    tot = sum(wrong_guesses[source].values())
    other = wrong_guesses[source].most_common(3)
    other_percentages = [round(x[1] * 100, 2) / tot for x in other]
    other_sources = [x[0] for x in other]
    print("Source: {} Most confused with:".format(source))
    output = "\t"
    for i, other_s in enumerate(other_sources):
        output += "{}: {} ".format(other_s, round(other_percentages[i], 2))
    print(output)

