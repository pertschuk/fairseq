from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

import nltk
import random

# DOWNLOAD:
# EXTRACT: tar -xvzf airbnb_model.tar.gz
# MAKE SURE MODEL_DIR is correct

# DEPENDENCIES:
# pip install fairseq
# pip install nltk
# import nltk
# nltk.download('punkt')

## USAGE:
# from run_inf import Roberta
# model = Roberta(use_gpu=False, model_dir='./airbnb_train/')
# label = model.classify(row['text'])


MODEL_DIR = './airbnb_train/'
CHECKPOINT_FILE = 'checkpoint_best.pt'
CLASSES = ['NOT_GREAT', 'GREAT']

CHUNK_SIZE=3


class Roberta (object):
  def __init__(self,model_dir=MODEL_DIR,ckpt_file=CHECKPOINT_FILE,
                 use_gpu=False):
    self.model = RobertaModel.from_pretrained(model_dir, checkpoint_file=ckpt_file)
    self.model.eval() # disable dropout
    if use_gpu: self.model.cuda()

  def classify(self, review, logits=False):
    reviews = self.batch_review(review)
    roberta = self.model
    tokens = map(lambda x: x if len(x) < 512 else x[:511], [roberta.encode(r) for r in reviews])
    batch = collate_tokens(list(tokens), pad_idx=1)
    label = roberta.predict('sentence_classification_head', batch)
    if logits:
        return label.sum(dim=0).tolist()
    else:
        return CLASSES[label.sum(dim=0).argmax()]

  def batch_review(self, review):
    sents = nltk.sent_tokenize(review)
    buffer = []
    chunks = []
    for sent in sents:
      buffer.append(sent)
      if (len(buffer)) % CHUNK_SIZE == 0:
        chunks.append(" ".join(buffer))
        buffer = [buffer[random.randint(0,CHUNK_SIZE-1)]]
    chunks.append(" ".join(buffer))
    return chunks

