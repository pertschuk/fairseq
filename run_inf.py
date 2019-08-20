from fairseq.models.roberta import RobertaModel

MODEL_DIR = './fever_output/'
CHECKPOINT_FILE = 'checkpoint_best.pt'
CLASSES = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

def load_model(model_dir=MODEL_DIR,ckpt_file=CHECKPOINT_FILE):
  roberta = RobertaModel.from_pretrained(model_dir, checkpoint_file=ckpt_file)
  roberta.eval() # disable dropout
  return roberta

def classify_fever(roberta, claim, evidence):
  tokens = roberta.encode(claim, evidence)
  res = roberta.predict('sentence_classification_head', tokens).argmax()
  return CLASSES[res]
