from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens


MODEL_DIR = './checkpoints/'
CHECKPOINT_FILE = 'checkpoint_best.pt'
CLASSES = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

class Roberta (object):
  def __init__(self,model_dir=MODEL_DIR,ckpt_file=CHECKPOINT_FILE,
                 use_gpu=False):
    self.model = RobertaModel.from_pretrained(model_dir, checkpoint_file=ckpt_file)
    self.model.eval() # disable dropout
    if use_gpu: self.model.cuda()

  def classify_fever(self, claims, evidences):
    roberta = self.model
    batch = collate_tokens([self.trim_sentence(roberta.encode(c, e), max_len=500) for c, e in zip(claims,evidences)], pad_idx=1)
    labels = roberta.predict('sentence_classification_head', batch).argmax(dim=1)
    labels = [CLASSES[label] for label in labels]
    return labels

  def trim_sentence(self, sent, max_len):
    return sent if len(sent) < max_len else sent[:max_len]

  def encode(self, sentences, pooling_strategy='cls', layer=-4, max_len=400):
    roberta = self.model
    batch = collate_tokens([self.trim_sentence(roberta.encode(sentence), max_len)
                            for sentence in sentences], pad_idx=1)
    features = roberta.extract_features(batch,return_all_hiddens=True)[layer]
    if pooling_strategy == 'cls':
      return features[:,0]
    elif pooling_strategy == 'mean':
      return features.mean(dim=1)
    elif pooling_strategy == 'max':
      return features.max(dim=1)[0]
    else:
      raise NotImplementedError()