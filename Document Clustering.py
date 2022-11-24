# import modules
import re
import ast
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

###############################################################################

# Prepare our Doc2Vec model:

# store file path for ease of use
file_path = "/Users/zanemazorbrown/Desktop/reviews_Tools_and_Home_Improvement_5.json"

# create empty list to append to
review_list = []

# open file and store review data in review_list
with open(file_path, "r") as infile:
    for line in infile.readlines():
        review = ast.literal_eval(line)
        review_list.append(review)

# Define function to remove html tags


def remove_html_tags(text):
    p = re.compile('<.*?>')
    return p.sub(' ', text)


# Contraction dictionary from: https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not",
                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                    "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                    "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                    "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                    "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                    "you'll've": "you will have", "you're": "you are", "you've": "you have"}


def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


contractions, contractions_re = _get_contractions(contraction_dict)

# define function to deconstruct contractions


def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# define function to tokenize review text


def tokenize(text):
    tokenizer = RegexpTokenizer("[\w']+")
    return tokenizer.tokenize(text)

# define function to remove stopwords from tokenized text
# Retrieved list form of stopwords from a comment under the following github:
# https://gist.github.com/sebleier/554280


def rv_stopwords(tokenized_text):
    sw_list = [",", ".", "'", '"', "i", "me", "my", "myself", "we", "our",
               "ours", "ourselves", "you", "your", "yours", "yourself",
               "yourselves", "he", "him", "his", "himself", "she", "her",
               "hers", "herself", "it", "its", "itself", "they", "them",
               "their", "theirs", "themselves", "what", "which", "who",
               "whom", "this", "that", "these", "those", "am", "is", "are",
               "was", "were", "be", "been", "being", "have", "has", "had",
               "having", "do", "does", "did", "doing", "a", "an", "the", "and",
               "but", "if", "or", "because", "as", "until", "while", "of",
               "at", "by", "for", "with", "about", "against", "between",
               "into", "through", "during", "before", "after", "above",
               "below", "to", "from", "up", "down", "in", "out", "on", "off",
               "over", "under", "again", "further", "then", "once", "here",
               "there", "when", "where", "why", "how", "all", "any", "both",
               "each", "few", "more", "most", "other", "some", "such", "no",
               "nor", "not", "only", "own", "same", "so", "than", "too",
               "very", "s", "t", "can", "will", "just", "don", "should",
               "now"]
    return [word for word in tokenized_text if word not in sw_list]

# preprocess function combining all preprocessing measures


def preprocess(text):
    text = remove_html_tags(text)
    text = replace_contractions(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = rv_stopwords(tokens)
    return tokens


# create empty list to append to
raw_text = []

# store review text in a list for ease of use
for key in review_list:
    text = key["reviewText"]
    raw_text.append(text)

# create empty list to append to
corpus = []

# preprocess the review text and store in the corpus list
for idx, text in enumerate(raw_text):
    if(idx % 10000 == 0):
        print(f"{idx:,} files have been preprocessed.")
    corpus.append(preprocess(text))

# display message at the end of data preprocessing
print("Preprocessing Complete")

# create Doc2Vec model using corpus
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

docvecs = []
# had to downgrade from gensim 4.2.0 to gensim 3.8.3 to use .vectors_docs
for idx, doc in enumerate(model.docvecs.vectors_docs):
    docvecs.append(doc)

###############################################################################

# Document Clustering: K-Means:

# specify k-means
k_val = 4
kmeans = KMeans(n_clusters=k_val).fit(docvecs)

# retrieve cluster labels for each doc
cluster_label = kmeans.labels_

# retrieve cluster-document frequencies
label_count = [0] * k_val
for label in cluster_label:
    label_count[int(label)] += 1
print(f"label count: {label_count}")

# dimension reduction with PCA
data_dim = 2
pca = PCA(n_components=data_dim)
principalComponents = pca.fit_transform(docvecs)
pca1 = principalComponents[:, 0]
pca2 = principalComponents[:, 1]

# visualize data
# we want k number of colors
colors = ['red', 'green', 'blue', 'yellow']
fig = plt.figure(figsize=(8, 8))
plt.scatter(pca1, pca2, c=cluster_label,
            cmap=matplotlib.colors.ListedColormap(colors))

# plt.savefig('kmeans_scatter.jpg')

###############################################################################

# Document Clustering: Hierarchical

hc = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, affinity="euclidean", linkage="average")
hc.fit(docvecs)

# The following code also needs to be run when distance_threshold is not equal
# to "None", as the ".distances_" argument in the following code would not work

# def plt_dendro(model, **kwargs):
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1
#             else:
#                 current_count += counts[child_idx - n_samples]
#             counts[i] = current_count
#         linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
#         dendrogram(linkage_matrix, **kwargs)

# plt_dendro(hc, truncate_mode="none")

fig = plt.figure(figsize=(8, 8))
plt.scatter(pca1, pca2, c=hc.labels_,
            cmap=matplotlib.colors.ListedColormap(colors))

# plt.savefig('hierarchical_scatter.jpg')
