from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens


MODEL_DIR = './airbnb_training/'
CHECKPOINT_FILE = 'checkpoint_best.pt'
CLASSES = ['GREAT', 'NOT_GREAT']

class Roberta (object):
  def __init__(self,model_dir=MODEL_DIR,ckpt_file=CHECKPOINT_FILE,
                 use_gpu=False):
    self.model = RobertaModel.from_pretrained(model_dir, checkpoint_file=ckpt_file)
    self.model.eval() # disable dropout
    if use_gpu: self.model.cuda()

  def classify_fever(self, claims, evidences):
    roberta = self.model
    batch = collate_tokens([roberta.encode(c, e) for c, e in zip(claims,evidences)], pad_idx=1)
    labels = roberta.predict('sentence_classification_head', batch).argmax(dim=1)
    labels = [CLASSES[label] for label in labels]
    return labels

  def get_embedding(self, sentences, pooling_strategy='cls'):
    roberta = self.model
    batch = collate_tokens([roberta.encode(sentence) for sentence in sentences], pad_idx=1)
    last_layer_features = roberta.extract_features(batch)
    if pooling_strategy == 'cls':
      return last_layer_features[0]
    elif pooling_strategy == 'mean':
      return last_layer_features.mean(dim=1)