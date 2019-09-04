import argparse
import os
import random
import csv

random.seed(0)

def main(args):
  for split in ['train', 'test']:
    samples = []
    fname = os.path.join(args.datadir, split + '.tsv')
    labels = ['supports', 'refutes', 'not enough info']
    labelMap = dict()
    for i, label in enumerate(labels):
      labelMap[label] = i
    with open(fname) as file:
      for row in csv.reader(file, delimiter='\t'):
        samples.append((row[0], row[1], labelMap[row[2]]))

    random.shuffle(samples)
    out_fname = 'train' if split == 'train' else 'dev'
    f1 = open(os.path.join(args.datadir, out_fname + '.input0'), 'w+')
    f2 = open(os.path.join(args.datadir, out_fname + '.input1'), 'w+')
    f3 = open(os.path.join(args.datadir, out_fname + '.label'), 'w+')
    for sample in samples:
      f1.write(sample[0] + '\n')
      f2.write(sample[1] + '\n')
      f3.write(str(sample[2]) + '\n')

    f1.close()
    f2.close()
    f3.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datadir', default='fever')
  args = parser.parse_args()
  main(args)