import csv
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

data = []
for inp_file in Path('TK').glob("*.csv"):
  print(inp_file)
  gl = inp_file.name[:-4].split('_')[-1]
  reader = csv.reader(inp_file.open())
  print(next(reader), gl)
  c = 0
  l = 0
  for row in reader:
    c += 1
    doc = row[-1]
    doc = doc.strip('"').strip()
    tokens = doc.split()
    l += len(tokens)
    row.append(gl)
    data.append(row)
  print(c, l/c)

for inp_file in Path('GODO').glob("*.csv"):
  print(inp_file)
  gl = 'dg_' + inp_file.name[:-4].split('_')[-1]
  reader = csv.reader(inp_file.open())
  print(next(reader), gl)
  c = 0
  l = 0
  for row in reader:
    c += 1
    doc = row[-1]
    doc = doc.strip('"').strip()
    tokens = doc.split()
    l += len(tokens)
    row.append(gl)
    myrow = [row[0], 'placeholder', 'placeholder', doc, gl]
    data.append(myrow)
  print(c, l/c)
print(len(data), len(data[0]), data[0])

train_rows, test_rows = train_test_split(data, random_state=42)
train_rows, dev_rows = train_test_split(train_rows, random_state=42)

print(len(train_rows), len(dev_rows), len(test_rows))

with open('tk_train.csv', 'w') as tf:
  writer = csv.writer(tf, delimiter='\t')
  writer.writerow(['Index', 'Genre', 'Topics', 'Text', 'GL'])
  for row in train_rows:
    text = row[-2].strip().strip('\n')
    text = ' '.join(text.split())
    row[-2] = text
    writer.writerow(row)

with open('tk_dev.csv', 'w') as tf:
  writer = csv.writer(tf, delimiter='\t')
  writer.writerow(['Index', 'Genre', 'Topics', 'Text', 'GL'])
  for row in dev_rows:
    writer.writerow(row)

with open('tk_test.csv', 'w') as tf:
  writer = csv.writer(tf, delimiter='\t')
  writer.writerow(['Index', 'Genre', 'Topics', 'Text', 'GL'])
  for row in test_rows:
    writer.writerow(row)
