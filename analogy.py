import embeddings

emb = embeddings.EmbeddingsDictionary (100000)
geek = emb.embed('geek')
neighbors = emb.emb2neighbors(geek)
for i in neighbors[1]:
  print(emb.words[i])
