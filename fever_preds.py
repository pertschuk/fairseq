from roberta import Roberta
from polomapy import PolomaConn, PolomaBuff

PSQL_HOST = "54.172.249.216"

def get_claims(n=None):
  p = PolomaConn(host=PSQL_HOST)
  limit = "limit " + str(n) if n != None else ""
  rows = p.iter_rows('''
      select claim_id, claim, line_text as evidence from fever.test_set order by random() {}
  '''.format(limit))
  return rows

def main():
  model = Roberta(model_dir='./fever_train')
  b = PolomaBuff('fever.test_preds',
                    workers=4,  # set number of processes
                    maxconn=8,  # set maximum postgres connections
                    maxbuff=50000,  # set buffer size to be held in memory
                    batchsize=100,
                 host=PSQL_HOST)  # set batchsize to send to postgres
  for id, claim, evidence in get_claims(1000):
    label = model.classify_fever([claim], [evidence])[0]
    print(label)
    b.append((id, label))


if __name__ == '__main__':
  main()