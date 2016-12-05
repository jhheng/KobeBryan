import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

df = pd.read_csv('data.csv')
df.set_index('shot_id', inplace=True)

# drop columns with only 1 feature
X = df.drop('shot_made_flag') #target
X = df.drop('game_event_id') #indepedent
X = df.drop('game_id')
X = df.drop('team_id')
X = df.drop('team_name') #single value
X = df.drop('matchup')

y = df['shot_made_flag']

print X.dtypes
print ' ==============='
print df.describe()

#check balance data. if inbalance, just duplicate to make it balance?
shot_count = sns.countplot(x = 'shot_made_flag', data = df)

#work on some visualisations using seaborn. scatter plot and correlation?


from sklearn.cross_validation import train_test_split

#KNN
from sklearn.neighbors import KNeighboursClassifier

knn = KNeighboursClassifier(n_neighbors = 5)

knn.fit
knn.predict

#mapreduce?

#Naive Bayes (week 4)
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import metrics

mnb = MultinomialNB()
gnb = GaussianNB()
clf = BernoulliNB()


#logistic regression?? (week 5)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

#SVM

#random forest?


#cross validation
