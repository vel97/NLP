import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import FreqDist
from scipy.stats import chi2_contingency  
from plotnine import ggplot, aes,geom_histogram,geom_boxplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score


airline = pd.read_csv("C://Users//SriramvelM//Downloads//archive (3)//airline_data.csv")
airline.head()
airline.columns
# airline = airline[['rates', 'date', 'country','verified','comments']]
# airline = airline.rename(columns={'comments':'reviews'})

################################ cleaning ################################################
airline.dtypes
airline.isnull().sum()
airline.describe()

airline['rates'] = airline['rates'].replace('\n\t\t\t\t\t\t\t\t\t\t\t\t\t5', '5')
airline['rates'] = airline['rates'].replace(['None'], '1')
airline['rates'] = airline['rates'].astype(float)
airline['date'] = airline['date'].astype('datetime64[ns]')

#Drop unwanted column
airline = airline.drop(['Unnamed: 0'], axis=1)

#creating verfied column
airline['verified'] = airline['reviews'].str.contains('Trip Verified')

#cleaning the reviews
airline['reviews'] = airline['reviews'].str.strip('✅ Trip Verified')
airline['reviews'] = airline['reviews'].str.replace('[^a-zA-Z0-9]', ' ')
airline['reviews'] = airline['reviews'].str.replace('[^?<!\w)\d+]', ' ')
airline['reviews'] = airline['reviews'].str.replace('[^-(?!\w)|(?<!\w)-]', ' ')
airline['reviews'] = airline['reviews'].apply(lambda x:' '.join([x for x in x.split() if len(x)>2]))

#c hanging the reviews to lowercase
airline['reviews'] = [i.lower() for i in airline['reviews']]

#stop words reomval
stop_words = stopwords.words('english')
stop_words

extra_words = ['airways', 'review', 'airline', 'flight', 'seat', 'service', 'london', 'british']
stop_words.extend(extra_words)

def Stopword_rmv(x):
    tokenized_review = word_tokenize(x)
    rem_words = ' '.join([i for i in tokenized_review if i not in stop_words])
    return rem_words

airline['reviews'] = airline['reviews'].apply(Stopword_rmv)

#Lemmetization
lemm = WordNetLemmatizer()
def nltk_wordnet_tag(x):
    if x.startswith ('J'):
        return wordnet.ADJ
    elif x.startswith ('V'):
        return wordnet.VERB
    elif x.startswith ('N'):
        return wordnet.NOUN
    elif x.startswith ('R'):
        return wordnet.ADV
    else:
        return None

def lemm_sentence(x):
    tagged = nltk.pos_tag(word_tokenize(x))
    wordnet_tag = map(lambda x: (x[0], nltk_wordnet_tag(x[1])), tagged)
    lemmetized = []
    for word, tag in wordnet_tag:
        if tag is None:
            lemmetized.append(word)
        else:
            lemmetized.append(lemm.lemmatize(word,tag))
    return ' '.join(lemmetized)

airline['reviews'] = airline['reviews'].apply(lemm_sentence)

######### Creating sentiment scores using polarity#############
def polarity(x):
    return TextBlob(x).sentiment.polarity

def sentiment(score):
    if score <=0:
        return 'negative'
    else:
        return 'positive'
    
airline['scores'] = airline['reviews'].apply(polarity)
airline['sentiment'] = airline['scores'].apply(sentiment)

#positive data subset
pos_df = airline[airline['sentiment'] == 'positive'] 

pos_words = ' '.join([i for i in pos_df['reviews']])
pos_words = pos_words.split()
frequency_pos = FreqDist(pos_words)

#Extracting words and frequency from frequncy object
df = pd.DataFrame({'word':list(frequency_pos.keys()),'count':list(frequency_pos.values())})
df

#subset words by frequency
df = df.nlargest(columns='count', n = 30)
df.sort_values('count', inplace=True)

#Plotting the graph for positive words
plt.figure(figsize=(20,5))
ax = plt.barh(df['word'], width = df['count'])
# ax = plt.pie(df['count'], labels=df['word'])
plt.show()

#negative data subset
neg_df = airline[airline['sentiment'] == 'negative'] 

neg_words = ' '.join([i for i in neg_df['reviews']])
neg_words = neg_words.split()
frequency_neg = FreqDist(neg_words)

#Extracting words and frequency from frequncy object
df1 = pd.DataFrame({'word':list(frequency_neg.keys()),'count':list(frequency_neg.values())})
df1

#subset words by frequency
df1 = df1.nlargest(columns='count', n = 30)
df1.sort_values('count', inplace=True)

#Plotting the graph for negative words
plt.figure(figsize=(20,5))
ax = plt.barh(df1['word'], width = df1['count'])
plt.show()

#chi-square test to understand relationship between categorical variables and target variable 
cs=chi2_contingency(pd.crosstab(airline['sentiment'], airline['verified']))
print("P-value: ",cs[1])

cs1=chi2_contingency(pd.crosstab(airline['sentiment'], airline['reviews']))
print("P-value: ",cs1[1])

#EDA
#Bar count plot
fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='sentiment', data=airline, hue='verified')
# ax.set_ylim(0,500)
plt.title("Verified vs Sentiment")
plt.show()

fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='rates', data=airline, hue='verified')
plt.title("Verified vs rates")
plt.show()

airline.groupby('country').size().sort_values(ascending = True).plot(kind='bar') 
plt.title("Count of reviews for different country")
plt.show()

#Box plot
# col = airline.select_dtypes(include='object')
ggplot(airline) + aes(x='sentiment', y='scores') + geom_boxplot()

#Histpogram
col1 = airline.select_dtypes(include='number')
for i in col1:
    ggplot(airline) + aes(x=i) + geom_histogram()
    plt.show()

#Bag of words model
#Creating matrix
tfidf = TfidfVectorizer(max_features=2000)
x= tfidf.fit_transform(airline.reviews).toarray()
y = airline.sentiment.map({'positive': 1, 'negative': 0}).values
featurenames = tfidf.get_feature_names_out

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

#Building ML model
from sklearn.tree import DecisionTreeClassifier
Dt = DecisionTreeClassifier()

#Fitting random model
Dt.fit(x_train, y_train)
y_pred = Dt.predict(x_test)
#Computing test data accuracy
acc = accuracy_score (y_test, y_pred)
print(acc)
auc = roc_auc_score(y_test, y_pred)
print(auc)

#since Y is imbalanced we use imblearn to balance the data
#Imbalanced learning
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros = RandomOverSampler(sampling_strategy=1)
x_sm, y_sm = ros.fit_resample(x_train,y_train)
counter = Counter(y_sm)
print(counter)

#pipeline building
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline
from numpy import mean
# from numpy import std
steps = [('sampling', RandomOverSampler(sampling_strategy=1)), ('model', DecisionTreeClassifier())]
pipe = Pipeline(steps=steps)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipe, x_train, y_train, cv=cv)
print('Mean ACCURACY: %.3f' % mean(scores))

#GridsearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2 , 4 , 5 , 6 ,7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_leaf_nodes': [None, 5, 10, 20],
}

dtree_cv = GridSearchCV(Dt,  param_grid=param_grid, cv=10, verbose=2, n_jobs=4)
dtree_cv.fit(x_train, y_train)

#best patameters from decision tree
dtree_cv.best_params_

# def Decision_Tree(a):
#     for depth in a:
#         dt = DecisionTreeClassifier(max_depth=depth)
#         #Fitting dt to training data
#         dt.fit(x_train, y_train)
#         #Accuracy
#         Train_acc = accuracy_score(y_train, dt.predict(x_train))
#         dt = DecisionTreeClassifier(max_depth=depth)
#         Val_acc = cross_val_score(dt, x_train, y_train, cv = cv)
#         print("Depth :", depth, "Training accuracy :", Train_acc, "Cross Val Score :", mean(Val_acc))

# Decision_Tree([1,2,3,4,5,6,7,8,9,10])

#Fitting the best model
steps = [('sampling', RandomOverSampler(sampling_strategy=1)), ('model', DecisionTreeClassifier(max_depth = None, criterion='entropy', max_leaf_nodes=20, min_samples_leaf=4, min_samples_split=2))]
pipe = Pipeline(steps=steps)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
accuracy_dt = pipe.score(x_test, y_test)
f1_dt = f1_score(y_test, y_pred, average='macro')
roc_dt = roc_auc_score(y_test, y_pred)

#Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm1=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm1,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm1[0,0]
TP=cm1[1,1]
FN=cm1[1,0]
FP=cm1[0,1]
sensitivity_dt=TP/float(TP+FN)
sensitivity_dt
specificity_dt=TN/float(TN+FP)
specificity_dt

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',
'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
'sensitivity_dt or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
'specificity_dt or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',
'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',
'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',
'Positive Likelihood Ratio = sensitivity_dt/(1-specificity_dt) = ',sensitivity_dt/(1-specificity_dt),'\n',
'Negative likelihood Ratio = (1-sensitivity_dt)/specificity_dt = ',(1-sensitivity_dt)/specificity_dt)

#################### random forest #########################################
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

#Creating pipeline
steps = [('sampling', RandomOverSampler(sampling_strategy=1)), ('model', RandomForestClassifier())]
pipe = Pipeline(steps=steps)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipe, x_train, y_train, cv=cv)
print('Mean ACCURACY: %.3f' % mean(scores))

#Fitting random model
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
roc_auc_score(y_test, y_pred)

#Hyper parameter tuning for random forest
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

#Gridsearch
from sklearn.model_selection import GridSearchCV
rf_grid = GridSearchCV(rfc, param_grid=param_grid, cv=10, verbose=2, n_jobs=4)
try:
    rf_grid.fit(x_train, y_train)
except ValueError:
    pass

#Best parameters
rf_grid.best_params_
# rf_grid.score(x_train, y_train)
# rf_grid.score(x_test, y_test)

#Fitting the best model from best_params
steps = [('sampling', RandomOverSampler(sampling_strategy=1)), ('model', RandomForestClassifier(n_estimators = 200,criterion = 'gini',max_depth=8, max_features='sqrt'))]
pipe = Pipeline(steps=steps)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
accuracy_rf = pipe.score(x_test, y_test)
f1_rf = f1_score(y_test, y_pred, average='macro')
roc_rf = roc_auc_score(y_test, y_pred)

#Confusion matrix
cm2=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm2,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm2[0,0]
TP=cm2[1,1]
FN=cm2[1,0]
FP=cm2[0,1]
sensitivity_rf=TP/float(TP+FN)
sensitivity_rf
specificity_rf=TN/float(TN+FP)
specificity_rf

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',
'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
'sensitivity_rf or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
'specificity_rf or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',
'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',
'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',
'Positive Likelihood Ratio = sensitivity_rf/(1-specificity_rf) = ',sensitivity_rf/(1-specificity_rf),'\n',
'Negative likelihood Ratio = (1-sensitivity_rf)/specificity_rf = ',(1-sensitivity_rf)/specificity_rf)

############# XGBoost calssifier ########################
from xgboost import XGBClassifier
xgb = XGBClassifier()

#Creating pipeline
steps = [('sampling', RandomOverSampler(sampling_strategy=1)), ('model', XGBClassifier())]
pipe = Pipeline(steps=steps)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipe, x_train, y_train, cv=cv)
print('Mean ACCURACY: %.3f' % mean(scores))

param_grid = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6],
#  'gamma':[i/10.0 for i in range(0,5)],
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)],
#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140,  
gamma=0,subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_grid, scoring='roc_auc',n_jobs=4, verbose = 2, cv=10)
gsearch2.fit(x_train,y_train)
gsearch2.best_params_, gsearch2.best_score_

#Fitting the model
steps = [('sampling', RandomOverSampler(sampling_strategy=1)), ('model', XGBClassifier(learning_rate=0.1, n_estimators=140,  
gamma=0,subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27,max_depth= 6,min_child_weight = 4))]
pipe = Pipeline(steps=steps)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
accuracy_xgb = pipe.score(x_test, y_test)
f1_xgb = f1_score(y_test, y_pred, average='macro')
roc_xgb = roc_auc_score(y_test, y_pred)

#Confusion matrix
cm3=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm3,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()

TN=cm3[0,0]
TP=cm3[1,1]
FN=cm3[1,0]
FP=cm3[0,1]
sensitivity_xgb=TP/float(TP+FN)
sensitivity_xgb
specificity_xgb=TN/float(TN+FP)
specificity_xgb

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',
'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
'sensitivity_xgb or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
'specificity_xgb or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',
'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',
'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',
'Positive Likelihood Ratio = sensitivity_xgb/(1-specificity_xgb) = ',sensitivity_xgb/(1-specificity_xgb),'\n',
'Negative likelihood Ratio = (1-sensitivity_xgb)/specificity_xgb = ',(1-sensitivity_xgb)/specificity_xgb)

#summary of models
df=pd.DataFrame()
df['Model']=pd.Series(['Decision Tree','Random Forest','XG Boosting'])
df['Accuracy']=pd.Series([accuracy_dt, accuracy_rf, accuracy_xgb])
df['ROC_AUC_Score']=pd.Series([roc_dt, roc_rf, roc_xgb])
df['F1-Score'] = pd.Series([f1_dt, f1_rf, f1_xgb])
df['Sensitivity'] = pd.Series([sensitivity_dt, sensitivity_rf, sensitivity_xgb])
df['Specificity'] = pd.Series([specificity_dt, specificity_rf, specificity_xgb])
df.set_index('Model')