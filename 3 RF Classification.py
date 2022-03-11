import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


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
Fit model and derive predicted y values
"""
classifier = RandomForestClassifier(
    n_estimators=1100,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features=3,
    max_depth=700,
    bootstrap=False
)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


"""
Results
"""
# Confusion Matrix
def tabular_confusion_matrix():
    print("Confusion Matrix\n")
    print(pd.crosstab(
        pd.Series(y_test, name="Actual"),
        pd.Series(y_pred, name="Predicted")))


# Evaluation metrics
def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,
                                average='weighted')
    recall = recall_score(y_test, y_pred,
                          average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1


def print_evaluation_metrics():
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)

    print(
        "Accuracy = %.3f \n Precision = %.3f \n recall = %.3f \n f1 = %.3f" % (
            accuracy, precision, recall, f1
        ))


# Visualize confusion Matrix
def visual_confusion_matrix():
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix,
                annot=True,
                annot_kws={'size': 10},
                cmap=plt.cm.get_cmap('Greens', 10),
                linewidths=0.2)

    # Add labels to the plot
    class_names = ['generated_DNA',
                   'real_DNA']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()


# classification report
def model_classification_report():
    report = classification_report(y_test, y_pred)
    print(report)


# ROC Curve display
def roc_curve_plot():
    # predict probabilities
    fr_probs = classifier.predict_proba(X_test)
    # keep probabilities for the real_DNA sequences only
    fr_probs = fr_probs[:, 1]
    # calculate scores
    rf_auc = roc_auc_score(y_test, fr_probs)
    # summarize scores
    print('Random Forest: ROC AUC=%.3f' % (rf_auc))
    # calculate roc curves
    rf_fpr, rf_tpr, _ = roc_curve(y_test, fr_probs)
    # plot the roc curve for the model
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (Worst Configuration)')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


# ROC AUC Score
def auc_score():
    # predict probabilities
    fr_probs = classifier.predict_proba(X_test)
    # keep probabilities for the real_DNA sequences only
    fr_probs = fr_probs[:, 1]
    # calculate scores
    rf_auc = roc_auc_score(y_test, fr_probs)
    # summarize scores
    print('Random Forest: ROC AUC=%.3f' % (rf_auc))


# Print Results

# tabular_confusion_matrix()
# print("=================================")
# print_evaluation_metrics()
# print("=================================")
# auc_score()
# print("=================================")
# roc_curve_plot()
# visual_confusion_matrix()
# model_classification_report()
