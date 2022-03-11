import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


"""
Load and inspect data
"""
path = "folder_path/research_data.xlsx"
research_data = pd.read_excel(path)

# print(research_data.head())
# print(research_data.tail())
# print(research_data.info())


"""
Ensure that all DNA sequences have same length
(important for deep learning)
"""


def equal_sequence_lengths():
    for i in research_data.dna_sequence:
        if len(i) != 60:
            print(len(i))
            # you should get no result


# equal_sequence_lengths()


"""
Create function to get k-mers
"""


def get_Kmers(sequence, size=2):
    result = [sequence[
              x:x+size].lower() for x in range(
        len(sequence) - size + 1
    )]
    return result


"""
Insert K-mers in a new research_data column
called 'words' and drop 'dna_sequence' column
since 'words' is replacing it
"""
research_data['words'] = research_data.apply(
    lambda x: get_Kmers(x['dna_sequence']), axis=1)

research_data_2 = research_data.drop(
    ['dna_sequence'], axis=1)

# print(research_data_2.head())
# print(research_data_2.tail())
# print(research_data_2.info())


"""
Store the 'words' column items in a list,
then join each item
"""
research_data_text = list(
    research_data_2['words'])

for item in range(len(research_data_text)):
    research_data_text[
        item] = ' '.join(research_data_text[item])

# print(research_data_text[2])


"""
Define dna_label data as y_data for machine learning
"""
y_data = research_data_2.iloc[:, 0].values

# print(y_data)


"""
Transform feature set
"""

cv = CountVectorizer(ngram_range=(2, 2))

X = cv.fit_transform(research_data_text)

# print(X.shape)
# print(y_data.shape)


"""
Train-Test split
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y_data, test_size=0.20, random_state=42, stratify=y_data)

# print(X_train.shape, X_test.shape,
#       y_train.shape, y_test.shape)


"""
Instantiate model
"""
classifier = RandomForestClassifier(random_state=42)


"""
Create hyper-parameter grid to 
choose from during splitting
(based on the results of random search
RandomSearchCV)
"""
# Number of trees in random forest
n_estimators = [1100, 1200, 1300, 2000]
# Number of features to consider at every split
max_features = [2, 3]
# Maximum number of levels in tree
max_depth = [700, 800, 900, 1000]
# Minimum number of samples required to split a node
min_samples_split = [3, 5, 7]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3]
# Method of selecting samples for training each tree
bootstrap = [False]

# Create the random grid
param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

# print(param_grid)


"""
Grid-Searching
"""
grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1)

# Fit the random search model
grid_search.fit(X_train, y_train)

# Show best params
print(grid_search.best_params_)
