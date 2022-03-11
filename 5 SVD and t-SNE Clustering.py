import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


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
Train-Test split:
test_size is 'usually' much larger,
because t-SNE is computationally 
very expensive
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y_data, test_size=0.20, random_state=42, stratify=y_data)

# print(X_train.shape, y_train.shape,
#       X_test.shape, y_test.shape)


"""
Truncated SVD
Perform Truncated SVD and reduce dimensions to 2
(PCA does not accept sparse input)
"""
svd = TruncatedSVD(n_components=2)
X_SVD = svd.fit_transform(X_train)

"""
TSNE
"""
t_sne = TSNE(n_components=2,
             perplexity=100,
             random_state=42)

X_100 = t_sne.fit_transform(X_train)


"""
Plot
"""
colors = ['royalblue', 'red', 'deeppink',
          'maroon', 'mediumorchid', 'tan',
          'forestgreen', 'olive', 'goldenrod',
          'lightcyan', 'navy']

vectorizer = np.vectorize(
    lambda x: colors[x % len(colors)])


plt.figure(figsize=(16, 5))

ax1 = plt.subplot(121)
plt.gca().set_title('SVD')
scatter = plt.scatter(
    X_SVD[:, 0],
    X_SVD[:, 1],
    c=vectorizer(y_train),
    label=colors)
# plt.gca().legend(('generated_dna', 'real_dna'))

ax2 = plt.subplot(122)
plt.gca().set_title('tSNE')
plt.scatter(
    X_100[:, 0],
    X_100[:, 1],
    c=vectorizer(y_train),
    label=y_train)
# plt.gca().legend(('generated_dna', 'real_dna'))

plt.show()
