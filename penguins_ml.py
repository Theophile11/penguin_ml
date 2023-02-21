import pandas as pd 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle
import streamlit as st 

penguin_df = pd.read_csv('penguins.csv')
print(penguin_df.head())

penguin_df.dropna(inplace = True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)


print('This is the output : ', output.head())
print('These are the features : ', features.head())

output, uniques = pd.factorize(output)
print('uniques : ', uniques)
print('output : ', output)


X_train, X_test, y_train, y_test = train_test_split(features, output, test_size = 0.2)

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy : ', accuracy_score(y_test, y_pred))

rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(clf, rf_pickle)
rf_pickle.close()

output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

fig, ax = plt.subplots()
ax = sns.displot(x = clf.feature_importances_, y = features.columns)
plt.title('Importance of features')
plt.xlabel('Importance')
plt.ylabel('Feature')
fig.savefig('feature_importance.png')
plt.show()
