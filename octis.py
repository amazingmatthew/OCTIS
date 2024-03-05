"""Main module."""
"""
import os
import string
from octis.preprocessing.preprocessing import Preprocessing
os.chdir(os.path.pardir)

# Initialize preprocessing
preprocessor = Preprocessing(vocabulary=None, max_features=None,
                             remove_punctuation=True, punctuation=string.punctuation,
                             lemmatize=True, stopword_list='english',
                             min_chars=1, min_words_docs=0)
# preprocess
dataset = preprocessor.preprocess_dataset(documents_path=r'/Users/mayanan/Library/CloudStorage/OneDrive-TheUniversityofManchester/Project/02_Text Mining/01_Corpora/Reddit/data/OCTIS_Reddit.tsv')

# save the preprocessed dataset
dataset.save('Reddit_dataset') """


from octis.models.LDA import LDA
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

# Define dataset
dataset = Dataset()
dataset.load_custom_dataset_from_folder("/Users/mayanan/PycharmProjects/TopicModeling/OCTIS/Reddit_dataset")

print (len(dataset._Dataset__corpus))
