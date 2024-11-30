#Import necessary libaries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

#Load the dataset
url = 'https://github.com/Safa1615/Dataset--loan/blob/main/bank-loan.csv?raw=true'
data = pd.read_csv(url, nrows=700)

# Save to Excel
data.to_excel('dataset.xlsx', index=False)
current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'dataset.xlsx')
print(f"The file is saved at: {file_path}")

#Split the data into features (independent variables) and the target variable (default or not)
X = data.drop('default', axis=1)
y = data['default']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize a classification model (in this case, a Random Forest classifier)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the classifier on the training data
classifier.fit(X_train, y_train)

#Make prediction on the test data
y_pred = classifier.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

#Print the results
print(f"Accuracy: {accuracy: .2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)



print(data)

!pip install xgboost

#Import necessary libaries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

#Load the dataset
url = 'https://github.com/Safa1615/Dataset--loan/blob/main/bank-loan.csv?raw=true'
data = pd.read_csv(url, nrows=700)

#Exploratory Data Analysis
print(data.head())
print(f"The shape of this dataframe: {data.shape}")
print(data.info)
print(data.describe())

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
plt.hist(data["age"], bins=20)

#Object-oriented interface
fig, ax = plt.subplots() #Create a figure and an axes
x = gdp_data.gdpPercap 
y = gdp_data.lifeExp
ax.plot(x,y,'o')   #Plot data on the axes
ax.set_xlabel('GDP per capita')    #Add an x-label to the axes
ax.set_ylabel('Life expectancy')   #Add a y-label to the axes
ax.set_title('Life expectancy vs GDP per capita from 1952 to 2007');

#Stacked bar plot showing number of complaints at different months of the year, and from different locations
ax = complaints_location.plot.bar(stacked=True,ylabel = 'Number of complaints',figsize=(15, 10), xlabel = 'Month')
ax.tick_params(axis = 'both',labelsize=15)
ax.yaxis.set_major_formatter('{x:,.0f}')


flowers_df = sns.load_dataset('iris')
tips_df = sns.load_dataset('tips')
flights_df = sns.load_dataset("flights").pivot(index="month", columns="year", values="passengers")


fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# Pass the axes into seaborn
axes[0,0].set_title('Sepal Length vs. Sepal Width')
sns.scatterplot(x=flowers_df.sepal_length, 
                y=flowers_df.sepal_width, 
                hue=flowers_df.species, 
                s=100, 
                ax=axes[0,0])

# Use the axes for plotting
axes[0,1].set_title('Distribution of Sepal Width')
sns.histplot(flowers_df.sepal_width, ax=axes[0,1])
axes[0,1].legend(['Setosa', 'Versicolor', 'Virginica'])

# Pass the axes into seaborn
axes[1,0].set_title('Restaurant bills')
sns.barplot(x='day', y='total_bill', hue='sex', data=tips_df, ax=axes[1,0])

# Pass the axes into seaborn
axes[1,1].set_title('Flight traffic')
sns.heatmap(flights_df, cmap='Blues', ax=axes[1,1])

plt.tight_layout(pad=2)


# Save to Excel
data.to_excel('dataset.xlsx', index=False)
current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'dataset.xlsx')
print(f"The file is saved at: {file_path}")

#Split the data into features (independent variables) and the target variable (default or not)
X = data.drop('default', axis=1)
y = data['default']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize a classification model (in this case, a Random Forest classifier)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the classifier on the training data
classifier.fit(X_train, y_train)

#Make prediction on the test data
y_pred = classifier.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

#Print the results
print(f"Accuracy: {accuracy: .2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)
