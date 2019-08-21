import pandas as pd
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.augmentation import transformation_function
import nltk
from nltk.corpus import wordnet as wn
from snorkel.augmentation import RandomPolicy
from snorkel.augmentation import PandasTFApplier
import numpy as np
import random

nltk.download("wordnet")

import names

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)


def main():
  df = pd.read_csv('../airbnb/reviews.tsv', sep='\t')

  newdf = df[['comments', 'Great (1) Not Great (0)']]
  newdf.columns = ['text', 'label']
  chunks = []
  labels = []
  buffer = []
  for i, row in newdf.iterrows():
    sents = nltk.sent_tokenize(row['text'])
    for sent in sents:
      buffer.append(sent)
      if (len(buffer)) % 3 == 0:
        chunks.append(" ".join(buffer))
        labels.append(row['label'])
        buffer = [buffer[random.randint(0,2)]]
    if len(buffer) > 1:
      chunks.append(" ".join(buffer))
      labels.append(row['label'])
      buffer = []

  chunkedDf = pd.DataFrame({'text' : chunks, 'label': labels})

  random_policy = RandomPolicy(
    len(tfs), sequence_length=4, n_per_original=1, keep_original=True
  )
  tf_applier = PandasTFApplier(tfs, random_policy)
  newdf_augmented = tf_applier.apply(chunkedDf)
  print(len(newdf))
  print(len(newdf_augmented))
  newdf_augmented.to_csv('airbnb_augmented.csv')

def train_split():
  df = pd.read_csv('airbnb_augmented.csv')
  msk = np.random.rand(len(df)) < 0.8
  train = df[msk]
  test = df[~msk]
  train.to_csv('train.tsv',sep='\t')
  test.to_csv('test.tsv', sep='\t')


# Pregenerate some random person names to replace existing ones with
# for the transformation strategies below
replacement_names = [names.get_full_name() for _ in range(50)]


# Replace a random named entity with a different entity of the same type.
@transformation_function(pre=[spacy])
def change_person(x):
  person_names = [ent.text for ent in x.doc.ents if ent.label_ == "PERSON"]
  # If there is at least one person name, replace a random one. Else return None.
  if person_names:
    name_to_replace = np.random.choice(person_names)
    replacement_name = np.random.choice(replacement_names)
    x.text = x.text.replace(name_to_replace, replacement_name)
    return x


# Swap two adjectives at random.
@transformation_function(pre=[spacy])
def swap_adjectives(x):
  adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
  # Check that there are at least two adjectives to swap.
  if len(adjective_idxs) >= 2:
    idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))
    # Swap tokens in positions idx1 and idx2.
    x.text = " ".join(
      [
        x.doc[:idx1].text,
        x.doc[idx2].text,
        x.doc[1 + idx1: idx2].text,
        x.doc[idx1].text,
        x.doc[1 + idx2:].text,
      ]
    )
    return x


def get_synonym(word, pos=None):
  """Get synonym for word given its part-of-speech (pos)."""
  synsets = wn.synsets(word, pos=pos)
  # Return None if wordnet has no synsets (synonym sets) for this word and pos.
  if synsets:
    words = [lemma.name() for lemma in synsets[0].lemmas()]
    if words[0].lower() != word.lower():  # Skip if synonym is same as word.
      # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
      return words[0].replace("_", " ")


def replace_token(spacy_doc, idx, replacement):
  """Replace token in position idx with replacement."""
  return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx:].text])


@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x):
  # Get indices of verb tokens in sentence.
  verb_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
  if verb_idxs:
    # Pick random verb idx to replace.
    idx = np.random.choice(verb_idxs)
    synonym = get_synonym(x.doc[idx].text, pos="v")
    # If there's a valid verb synonym, replace it. Otherwise, return None.
    if synonym:
      x.text = replace_token(x.doc, idx, synonym)
      return x


@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x):
  # Get indices of noun tokens in sentence.
  noun_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
  if noun_idxs:
    # Pick random noun idx to replace.
    idx = np.random.choice(noun_idxs)
    synonym = get_synonym(x.doc[idx].text, pos="n")
    # If there's a valid noun synonym, replace it. Otherwise, return None.
    if synonym:
      x.text = replace_token(x.doc, idx, synonym)
      return x


@transformation_function(pre=[spacy])
def replace_adjective_with_synonym(x):
  # Get indices of adjective tokens in sentence.
  adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
  if adjective_idxs:
    # Pick random adjective idx to replace.
    idx = np.random.choice(adjective_idxs)
    synonym = get_synonym(x.doc[idx].text, pos="a")
    # If there's a valid adjective synonym, replace it. Otherwise, return None.
    if synonym:
      x.text = replace_token(x.doc, idx, synonym)
      return x


tfs = [
  change_person,
  replace_noun_with_synonym,
  replace_adjective_with_synonym
]

if __name__ == '__main__':
  main()
  train_split()
