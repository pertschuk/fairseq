from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained('./fever_output/', checkpoint_file='checkpoint_best.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
res = roberta.predict('fever', tokens).argmax()

print(res)