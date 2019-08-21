import argparse
import os
import random
import csv

random.seed(0)

def main(args):
  for split in ['train', 'test']:
    samples = []
    fname = os.path.join(args.datadir, split + '.tsv')
    labels = ['not great', 'great']
    labelMap = dict()
    for i, label in enumerate(labels):
      labelMap[label] = i
    with open(fname) as file:
      for row in csv.reader(file, delimiter='\t'):
        samples.append((row[2], row[3]))

    random.shuffle(samples)
    out_fname = 'train' if split == 'train' else 'dev'
    f1 = open(os.path.join(args.datadir, out_fname + '.input0'), 'w+')
    f2 = open(os.path.join(args.datadir, out_fname + '.label'), 'w+')
    for sample in samples:
      f1.write(sample[0] + '\n')
      if not sample[1] in ['0', '1']:
        print(sample[1])
      assert(sample[1] in ['0', '1'])
      f2.write(sample[1] + '\n')

    f1.close()
    f2.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='.')
  args = parser.parse_args()
  main(args)
