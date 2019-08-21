from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

import nltk
import random


MODEL_DIR = './airbnb_train/'
CHECKPOINT_FILE = 'checkpoint_best.pt'
CLASSES = ['GREAT', 'NOT_GREAT']

class Roberta (object):
  def __init__(self,model_dir=MODEL_DIR,ckpt_file=CHECKPOINT_FILE,
                 use_gpu=False):
    self.model = RobertaModel.from_pretrained(model_dir, checkpoint_file=ckpt_file)
    self.model.eval() # disable dropout
    if use_gpu: self.model.cuda()

  def classify(self, review):
    reviews = self.batch_review(review)
    roberta = self.model
    batch = collate_tokens([roberta.encode(r) for r in reviews], pad_idx=1)
    label = roberta.predict('sentence_classification_head', batch)
    return CLASSES[label.sum(dim=0).argmax()]

  def batch_review(self, review):
    sents = nltk.sent_tokenize(review)
    buffer = []
    chunks = []
    for sent in sents:
      buffer.append(sent)
      if (len(buffer)) % 3 == 0:
        chunks.append(" ".join(buffer))
        buffer = [buffer[random.randint(0,2)]]
    if len(buffer) > 1:
      chunks.append(" ".join(buffer))
    return chunks

  #
  # def get_embedding(self, sentences, pooling_strategy='cls'):
  #   roberta = self.model
  #   batch = collate_tokens([roberta.encode(sentence) for sentence in sentences], pad_idx=1)
  #   last_layer_features = roberta.extract_features(batch)
  #   if pooling_strategy == 'cls':
  #     return last_layer_features[0]
  #   elif pooling_strategy == 'mean':
  #     return last_layer_features.mean(dim=1)

