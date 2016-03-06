import os
import re
import numpy as np

# TODO: (set1,set2) x (train, test, valid)
filenames = ['set1.test.txt']
pathRoot = './'
lines = []

FEATURE_COUNT = 700
input_data = []

for filename in filenames:
  path = pathRoot + filename
  with open(path, 'r') as hFile:
    lines = hFile.readlines()

# Fettermania: Note - there is a blank zero index in every row here.
# Looks like features go "1: - 699:"
i = 1
for line in lines:
  line_matches = re.match('([0-9]*) qid:([0-9]*) (.*)', line)
  input_features = np.zeros(FEATURE_COUNT)
  for obj in re.finditer('([0-9]*):([0-9\.]*)', line_matches.group(3)):
    input_features[int(obj.group(1))] = float(obj.group(2))
  if i % 10000 == 0:
    print (i)
  i = i + 1
  input_data.append(
    [int(line_matches.group(1)), int(line_matches.group(2)), input_features, np.linalg.norm(input_features)])


# >>> min(map(lambda x: x[3], input_data))
# 5.8357033809644969

# >>> max(map(lambda x: x[3], input_data))
# 18.741407769034719


# for obj in re.finditer('([0-9]*):([0-9\.]*)', line_matches.group(3)):
#   input_features[int(obj.group(1))] = obj.group(2)

# NOTE: Better 
# from sklearn.datasets import load_svmlight_file
# (X, y) = load_svmlight_file("./set2.test.txt")
#  