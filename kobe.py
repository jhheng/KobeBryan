import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

df = pd.read_csv('data.csv', parse_dates = ['game_date'], infer_datetime_format = True)
df.set_index('shot_id', inplace=True)

print df.isnull().sum()

df = df[df['shot_made_flag']>=0] # remove null rows

X = df.copy()
# drop columns with only 1 feature
X.drop(['shot_made_flag'], axis = 1, inplace = True) #target
X.drop(['game_event_id'], axis = 1, inplace = True) #independent
X.drop(['game_id'], axis = 1, inplace = True) #independent
X.drop(['team_id'], axis = 1, inplace = True) #independent
X.drop(['team_name'], axis = 1, inplace = True) #single value
X.drop(['matchup'], axis = 1, inplace = True) #duplicate with opponent
X.drop(['season'], axis = 1, inplace = True) #duplicate to date/independent

y = df['shot_made_flag']

#make dummy Variable
pd.set_option('display.max_rows', 200)
X = pd.get_dummies(X[['action_type', 'combined_shot_type','shot_type','shot_zone_area','shot_zone_basic','opponent' ]], drop_first=True)

#check balance data. if inbalance, just duplicate to make it balance?
sns.countplot(x = 'shot_made_flag', data = df)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

y_predict = knn.fit(X_train,y_train).predict(X_test)

#knn.predict_proba(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (y_predict.size, ((y_predict != y_test)==True).count()))
(y_predict != y_test).value_counts()

#Accuracy 

def accuracy_report(_clf):
    training_accuracy = _clf.score(X_train, y_train)
    test_accuracy = _clf.score(X_test, y_test)
    print "Accuracy on test data: %0.2f%%" % (100 * test_accuracy)
    print "Accuracy on training data: %0.2f%%" % (100 * training_accuracy)

#mapreduce?

#Naive Bayes (week 4)
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

print "Multinomial:"
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)
accuracy_report(mnb)

print ""
print "GaussianNB:"
gnb = GaussianNB()
gnb.fit(X_train, y_train)
accuracy_report(gnb)

print ""
print "BernoulliNB:"
clf = BernoulliNB()
clf.fit(X_train, y_train)
accuracy_report(clf)

print ""
print "LogisticRegression:"
#logistic regression?? (week 5)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
accuracy_report(logreg)

from sklearn.cross_validation import cross_val_score
print cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()
print cross_val_score(mnb, X, y, cv=10, scoring='accuracy').mean()
print cross_val_score(gnb, X, y, cv=10, scoring='accuracy').mean()
print cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
print cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean()

#SVM
#random forest
from sklearn.svm import SVC
sv = SVC(kernel='rbf')
sv.fit(X, y)
accuracy_report(sv)

svl = SVC(kernel='linear')
svl.fit(X, y)
accuracy_report(svl)