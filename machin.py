import embeddings
import logging
import numpy
import torch

embeddings.EmbeddingsDictionary.neighborhood('tree')

embeddings.EmbeddingsDictionary(250000).analogy('Tokyo', 'Spain', 'Japan')
