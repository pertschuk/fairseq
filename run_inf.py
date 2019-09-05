from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

import nltk
import random

# DOWNLOAD: wget https://storage.googleapis.com/poloma-models/airbnb_model.tar.gz
# EXTRACT: tar -xvzf airbnb_model.tar.gz

# MAKE SURE model directory points to where you downloaded the model
MODEL_DIR = './airbnb_train/'

# DEPENDENCIES:
# pip install fairseq
# pip install nltk
# import nltk
# nltk.download('punkt')

## USAGE:
# from run_inf import Roberta
# model = Roberta(use_gpu=False, model_dir='./airbnb_train/')
# label = model.classify(review)

CHECKPOINT_FILE = 'checkpoint_best.pt'
CLASSES = ['NOT_GREAT', 'GREAT']

# how many sentences to run through at the same time. Tweak if running out of memory
CHUNK_SIZE=4

# set bias based on excel spreadsheet
BIAS = 10


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
        logits = label.sum(dim=0).tolist()
        return CLASSES[0] if logits[0] > logits[1] + BIAS else CLASSES[1]

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

