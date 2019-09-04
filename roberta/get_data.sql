\copy (select claim, line_text as evidence, label from fever.train_set)
to './fever/train.tsv'