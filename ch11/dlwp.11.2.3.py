# 11.2.3 Vocabulary indexing
print("11.2.3 Vocabulary indexing")

import numpy as np

dataset = ["Such a set is called a bag"]
vocabulary = {}
for text in dataset:
    standardized = text.lower()
    print(standardized)
    tokens = standardized.split()
    print(tokens)
    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
            print(vocabulary)

def one_hot_encode_token(token):
    vector = np.zeros((len(vocabulary) + 2,))
    if token in vocabulary:
        token_index = vocabulary[token] + 2
    else:
        token_index = vocabulary.get(token, 1)
    vector[token_index] = 1
    return vector

for token in tokens:
    print(one_hot_encode_token(token))

print(one_hot_encode_token("junk"))

