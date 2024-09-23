#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Checking the version of Python packages (libraries)
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
print('Python: {}'.format(sys.version),"\n",
     'Scipy: {}'.format(scipy.__version__),"\n",
     'numpy: {}'.format(numpy.__version__),"\n",
     'matplotlib: {}'.format(matplotlib.__version__),"\n",
     'pandas: {}'.format(pandas.__version__),"\n",
     'sklearn: {}'.format(sklearn.__version__))


# In[3]:


## Load the needed Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import lightgbm as lgb


# In[4]:


# Reading the CSV file 'ipl.csv' and storing the data in a DataFrame called 'data'
data = pd.read_csv('IPL_match.csv')

# Displaying the first 5 rows of the dataset
data.head()


# In[5]:


data.tail(5)


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


# Displaying the columns in our dataset
data.columns


# In[9]:


# Get unique values in the 'team1' column
data['team1'].unique()


# In[10]:


# Get unique values in the 'team2' column
data['team2'].unique()


# In[11]:


# Get unique values in the 'winner' column
data['winner'].unique()


# In[12]:


# Get unique values in the 'toss_winner' column
data['toss_winner'].unique()


# In[13]:


# Replacing 'Royal Challengers Bangalore' with 'Royal Challengers Bengaluru' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
data.team1.replace({'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'}, regex=True, inplace=True)
data.team2.replace({'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'}, regex=True, inplace=True)
data.winner.replace({'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'}, regex=True, inplace=True)
data.toss_winner.replace({'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'}, regex=True, inplace=True)

# Replacing 'Kings XI Punjab' with 'Punjab Kings' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
data.team1.replace({'Kings XI Punjab': 'Punjab Kings'}, regex=True, inplace=True)
data.team2.replace({'Kings XI Punjab': 'Punjab Kings'}, regex=True, inplace=True)
data.winner.replace({'Kings XI Punjab': 'Punjab Kings'}, regex=True, inplace=True)
data.toss_winner.replace({'Kings XI Punjab': 'Punjab Kings'}, regex=True, inplace=True)


# In[14]:


# To show statistical summary of the columns of our data
data.describe(include ='all')


# In[15]:


# To count the null values
data.isnull().sum()


# In[16]:


# Fill missing values
data['city'].fillna('Unknown', inplace=True)
cols_to_fill = ['player_of_match', 'eliminator']
data[cols_to_fill] = data[cols_to_fill].fillna('Not Available')
mean_result_margin = data['result_margin'].mean()  # Calculate the mean
data['result_margin'].fillna(mean_result_margin, inplace=True)


# In[17]:


# To drop the unwanted columns and rows 
data.drop(['id','method'],axis=1,inplace=True)
data.dropna(subset=['winner'], inplace=True)


# In[18]:


# To count the null values
data.isnull().sum()


# In[19]:


# Checking the shape of our data after handling null values
data.shape


# In[20]:


# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract the year from the 'date' column and create a new 'season' column
data['season'] = pd.DatetimeIndex(data['date']).year


# In[21]:


# displaying our data
data.head()


# <H1> EDA </H1>

# In[22]:


# Get the unique venues present in the 'venue' column
data['venue'].unique()


# In[23]:


# Get the top 10 venues with the highest number of matches played
venue_counts = data['venue'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=venue_counts.index, y=venue_counts.values)
plt.xlabel('Venue')
plt.ylabel('Count')
plt.title('Top 10 Venues')
plt.xticks(rotation=90)
plt.show()


# In[24]:


# Get the top 10 players with the highest number of "Player of the Match" awards
plt.figure(figsize=(10, 6))
top_players = data['player_of_match'].value_counts().head(10)
sns.barplot(x=top_players.index, y=top_players.values)
plt.xticks(rotation=90)
plt.xlabel('Player')
plt.ylabel('Count')
plt.title('Top 10 Players of the Match')
plt.show()


# In[25]:


# Extracting day, month, and year from the 'date' column
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year


# In[26]:


# Number of matches played over the years
plt.figure(figsize=(10, 6))
data['year'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Matches')
plt.title('Number of Matches Played Over the Years')
plt.show()


# In[27]:


# Number of matches played over the months
plt.figure(figsize=(10, 6))
data['month'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Number of Matches')
plt.title('Number of Matches Played Over the Months')
plt.show()


# In[28]:


# Calculate win percentage for each team
team_wins = data['winner'].value_counts()
team_matches = data['team1'].value_counts() + data['team2'].value_counts()
win_percentage = (team_wins / team_matches).sort_values(ascending=False)

# Plot win percentage for each team
plt.figure(figsize=(10, 6))
sns.barplot(x=win_percentage.index, y=win_percentage.values)
plt.xticks(rotation=90)
plt.xlabel('Team')
plt.ylabel('Win Percentage')
plt.title('Win Percentage of Each Team')
plt.show()


# In[29]:


# Toss Decision Frequency
toss_decision_counts = data['toss_decision'].value_counts()

# Plot the frequency of toss decisions
plt.figure(figsize=(6, 4))
sns.countplot(x='toss_decision', data=data)
plt.xlabel('Toss Decision')
plt.ylabel('Count')
plt.title('Frequency of Toss Decisions')
plt.show()


# In[30]:


# Toss Winner
toss_winner_counts = data['toss_winner'].value_counts()
plt.figure(figsize=(10, 6))
sns.countplot(x='toss_winner', data=data, order=toss_winner_counts.index)
plt.xlabel('Toss Winner')
plt.ylabel('Count')
plt.title('Matches Count by Toss Winner')
plt.xticks(rotation=90)
plt.show()


# In[31]:


# Pie Chart
plt.pie(data['result'].value_counts(), labels=data['result'].value_counts().index, autopct='%1.1f%%')
plt.title('Match Results Distribution')
plt.show()


# <h1> Data Preparation </h1>

# In[32]:


# To display the columns of the data
data.columns


# In[33]:


## Get the unique venues present in the 'winner' column 
data['winner'].unique()


# In[34]:


# Create a dictionary to map team names to unique numbers
team_mapping = {
    'Chennai Super Kings': 1,
    'Delhi Capitals': 2,
    'Royal Challengers Bengaluru': 3,
    'Rajasthan Royals': 4,
    'Mumbai Indians': 5,
    'Punjab Kings': 6,
    'Kolkata Knight Riders': 7,
    'Sunrisers Hyderabad': 8,
    'Gujarat Titans': 9,
    'Lucknow Super Giants':10
}

# Replace team names in 'team1' and 'team2' columns with unique numbers
data['team1'] = data['team1'].map(team_mapping)
data['team2'] = data['team2'].map(team_mapping)

# Replace winner names in 'winner' column with unique numbers
data['winner'] = data['winner'].map(team_mapping)
data['toss_winner'] = data['toss_winner'].map(team_mapping)


# In[35]:


# Get the unique venues present in the 'venue' column
data['venue'].unique()


# In[36]:


# Create a dictionary to map each unique venue name to a unique number
venue_mapping = {venue: i for i, venue in enumerate(data['venue'].unique())}

# Replace the venue names in the 'venue' column with the corresponding unique numbers
data['venue'] = data['venue'].map(venue_mapping)


# In[37]:


# Get the unique venues present in the 'toss_decsion' column
data['toss_decision'].unique()


# In[38]:


# Create a dictionary to map 'toss_decision' values to numerical values
temp = {'field': 0, 'bat': 1}

# Use the map() function to replace 'toss_decision' values with numerical values
data['toss_decision'] = data['toss_decision'].map(temp)


# In[39]:


# Create a set of unique umpires
umpires_set = set(data['umpire1'].unique()).union(set(data['umpire2'].unique()))

# Create a dictionary to map umpire names to unique numbers
umpire_dict = {umpire: i for i, umpire in enumerate(umpires_set, 1)}

# Apply the dictionary to create new encoded columns for 'umpire1' and 'umpire2'
data['umpire1'] = data['umpire1'].map(umpire_dict)
data['umpire2'] = data['umpire2'].map(umpire_dict)


# In[40]:


# Create a dictionary to map each unique venue name to a unique number
player_of_match_mapping = {venue: i for i, venue in enumerate(data['player_of_match'].unique())}

# Replace the venue names in the 'venue' column with the corresponding unique numbers
data['player_of_match'] = data['player_of_match'].map(player_of_match_mapping)


# In[41]:


# Create a dictionary to map each unique venue name to a unique number
city_mapping = {venue: i for i, venue in enumerate(data['city'].unique())}

# Replace the venue names in the 'venue' column with the corresponding unique numbers
data['city'] = data['city'].map(city_mapping)


# In[42]:


# to display our data
data.head()


# In[43]:


# List of unwanted columns
unwanted_columns = ['date','result','eliminator','season','day','month','year']

# Drop the unwanted columns from the DataFrame
data.drop(columns=unwanted_columns, inplace=True)


# In[44]:


data.head()


# In[45]:


# checking for the null values in updated dataframe
sns.heatmap(data.isnull(),cmap='rainbow',yticklabels=False)


# <h1> Splitting our data </h1>

# In[46]:


# Split the data into features (X) and the target variable (y)
X = data.drop(['winner'], axis=1)
y = data['winner']


# In[47]:


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# <h1> Identifying Important features </h1>

# In[48]:


# Create an instance of the RandomForestClassifier with hyperparameters
forest = RandomForestClassifier(n_estimators=500, random_state=1)

# Train the RandomForestClassifier on the training data
forest.fit(X_train, y_train.values.ravel())


# In[49]:


# Get the feature importances from the trained RandomForestClassifier
importances = forest.feature_importances_

# Loop over each feature and its importance
for i in range(X_train.shape[1]):
    # Print the feature number, name, and importance score
    print("%2d) %-*s %f" % (i + 1, 30, data.columns[i], importances[i]))


# In[50]:


# Plotting the feature importances as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances, align='center')
plt.title('Feature Importance')
plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()


# <h1> Training our Model </h1>

# <h3> 1. Logistic Regression </h3>

# In[51]:


model = LogisticRegression()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# <h3> 2. SVM </h3>

# In[52]:


model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# <h3> 3. Random Forest </h3>

# In[53]:


model = RandomForestClassifier(n_estimators=13)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# <h3> 4. Decision Tree Regressor </h3>

# In[54]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# <h3> 5. LGBMClassifier </h3>

# In[55]:


model = lgb.LGBMClassifier()
model.fit(X_train, y_train)


# In[56]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




