# embedding-retrieval-tradeoffs

Which embedding choices matter for retrieval, measured on a real benchmark: pooling, similarity, approximate
search, domain fine-tuning, vector size, and chunking.

The task is NFCorpus from the BEIR benchmark: 3,633 medical abstracts and 323 test questions, each with graded
relevance labels (1 = relevant, 2 = highly relevant). The main score is NDCG@10, which rewards relevant documents
near the top of the ranking. The default model is `all-MiniLM-L6-v2` (22M parameters, 384 dimensions); its
published NDCG@10 on this set is about 0.32, which the first table reproduces.

```console
$ uv run el data
NFCorpus test: 3633 documents, 323 queries, 12334 relevance labels (38.2 per query)
  example query: 'Do Cholesterol Statin Drugs Cause Breast Cancer?'
  a grade-2 match: 'Elevated Levels of Cholesterol-Rich Lipid Rafts in Cancer Cells Are Correlated with Apoptosis Sensitivity Induced by Cholesterol-Depleting Agents'
```

## Evidence

**Pooling and similarity.** Use what the model was trained with. MiniLM was trained with mean pooling and cosine
similarity; the CLS vector loses 0.025 NDCG@10 and a raw dot product loses 0.031.

```console
$ uv run el pooling
sentence-transformers/all-MiniLM-L6-v2, unit-length vectors, exact cosine search on NFCorpus test
  setting                 NDCG@10  Recall@100     MRR
  mean pooling             0.3186      0.3148  0.5158
  cls pooling              0.2938      0.2890  0.4757
$ uv run el similarity
sentence-transformers/all-MiniLM-L6-v2, mean pooling, raw (not unit-length) vectors, exact search on NFCorpus test
  setting                 NDCG@10  Recall@100     MRR
  cosine similarity        0.3186      0.3148  0.5158
  dot similarity           0.2880      0.2978  0.4905
```

**Approximate search.** Exact search time grows with the corpus; HNSW stays near a millisecond and trades some
recall for it. `ef` sets how many candidates each query explores. Past 3,633 documents the corpus is padded with
random unit vectors, not real documents, so the overlap at 50k and 200k may differ on real data. Timings are
single-threaded on an M1 and vary a little between runs. The flat index scores 0.3179, not 0.3186: NFCorpus
holds 40 duplicate abstracts, and FAISS breaks their ties differently.

```console
$ uv run el ann
exact (flat) vs HNSW, 323 real queries, top 10; past 3633 docs the corpus is padded with random unit vectors; one CPU thread
     docs index           build s  ms/query  overlap with exact  NDCG@10
     3633 flat (exact)       0.00     0.109               1.000   0.3179
     3633 HNSW ef=16         0.39     0.016               0.883   0.3026
     3633 HNSW ef=64         0.39     0.049               0.976   0.3163
     3633 HNSW ef=256        0.39     0.172               0.999   0.3182
    50000 flat (exact)       0.02     1.540               1.000        -
    50000 HNSW ef=16        57.60     0.066               0.795        -
    50000 HNSW ef=64        57.60     0.185               0.904        -
    50000 HNSW ef=256       57.60     0.681               0.965        -
   200000 flat (exact)       0.03     6.140               1.000        -
   200000 HNSW ef=16       331.60     0.103               0.745        -
   200000 HNSW ef=64       331.60     0.281               0.865        -
   200000 HNSW ef=256      331.60     0.942               0.921        -
```

**Domain fine-tuning.** Neither run changed NDCG@10 or Recall@100 beyond noise: all four intervals below
include zero. Each run trains 3 seeds, and each question's score is averaged over the seeds before the paired
interval is computed. Title-to-abstract pairs teach MiniLM nothing new; its model card lists scientific
title-abstract pairs (S2ORC) among its training data. Training on NFCorpus's train-split questions moved every
seed the same way, NDCG@10 down and Recall@100 up, but by less than question-to-question variation. Train and
test questions share no text (three repeated texts are dropped from training), though they search the same
corpus.

A one-seed version of this experiment reported an NDCG@10 drop of 0.0145 with an interval that excluded zero.
It did not survive more seeds and a slightly different sample of training pairs. Train several seeds before
believing a fine-tuning result.

```console
$ uv run el finetune
MiniLM before and after 1 epoch(s) of contrastive training, 3 seeds per run; scored on the 323 test queries, whose texts no training run sees
  setting                 NDCG@10  Recall@100     MRR
  no fine-tune             0.3186      0.3148  0.5158
  title -> abstract        0.3152      0.3122  0.5163
  train queries            0.3131      0.3238  0.5055
gain over no fine-tune, seed-averaged per query, paired 95% bootstrap CI
  title -> abstract  ndcg@10    -0.0035 [-0.0094, +0.0021]  per seed: -0.0039, -0.0040, -0.0026
  title -> abstract  recall@100 -0.0026 [-0.0126, +0.0062]  per seed: -0.0013, -0.0033, -0.0031
  train queries      ndcg@10    -0.0056 [-0.0170, +0.0059]  per seed: -0.0050, -0.0074, -0.0043
  train queries      recall@100 +0.0090 [-0.0055, +0.0211]  per seed: +0.0141, +0.0050, +0.0078
train-query batches of 32: 31% of queries have a labeled-relevant doc among their negatives
```

**Vector size.** Projecting onto the corpus's top singular directions (SVD) beats both PCA and cutting at every
size. At 256 dimensions SVD keeps 99% of MiniLM's NDCG@10 and all of mpnet's. mpnet at 128 dimensions scores
about the same as full MiniLM in a third of the memory. PCA loses even with no dimension dropped (MiniLM 0.2916
against 0.3186): it subtracts the corpus mean, which moves the origin and so changes every cosine similarity.

```console
$ uv run el dims
NDCG@10 on NFCorpus test as vectors shrink; SVD and PCA are fit on the corpus vectors only
  model       dims method     NDCG@10  bytes/doc
  MiniLM        32 svd         0.1934        128
  MiniLM        32 pca         0.1710        128
  MiniLM        32 truncate    0.1368        128
  MiniLM        64 svd         0.2712        256
  MiniLM        64 pca         0.2376        256
  MiniLM        64 truncate    0.2124        256
  MiniLM       128 svd         0.3020        512
  MiniLM       128 pca         0.2694        512
  MiniLM       128 truncate    0.2741        512
  MiniLM       256 svd         0.3166       1024
  MiniLM       256 pca         0.2906       1024
  MiniLM       256 truncate    0.3044       1024
  MiniLM       384 full        0.3186       1536
  MiniLM       384 pca         0.2916       1536
  mpnet         32 svd         0.2209        128
  mpnet         32 pca         0.1871        128
  mpnet         32 truncate    0.1603        128
  mpnet         64 svd         0.2806        256
  mpnet         64 pca         0.2430        256
  mpnet         64 truncate    0.2163        256
  mpnet        128 svd         0.3170        512
  mpnet        128 pca         0.2822        512
  mpnet        128 truncate    0.2822        512
  mpnet        256 svd         0.3351       1024
  mpnet        256 pca         0.3027       1024
  mpnet        256 truncate    0.3152       1024
  mpnet        768 full        0.3352       3072
  mpnet        768 pca         0.3086       3072
```

**Chunking.** Each long document hides one gold abstract among nine abstracts that no test question is relevant
to, and no question is relevant to another question's gold. Each strategy is compared with fixed 8-sentence
chunks on the same 100 questions. Large 32-sentence chunks find the right document far less often. Auto-merging
does better than fixed 8-sentence chunks while returning fewer sentences; the sentence window's interval just
reaches zero.

```console
$ uv run el chunks
100 long documents (110 sentences each on average), one gold abstract hidden in each; top 3 returned spans per query
  strategy                chunks  doc hit   vs fixed 8, paired 95% CI  gold covered  sentences returned
  fixed, 8 sentences        1424    0.310                    baseline         0.224                23.0
  fixed, 32 sentences        394    0.160     -0.150 [-0.230, -0.080]         0.113                80.9
  sentence window +-3      11050    0.390     +0.080 [+0.000, +0.160]         0.245                20.5
  auto-merge 4 -> 16        2799    0.380     +0.070 [+0.010, +0.140]         0.220                14.4
```

## What is here
- **Data** (`data/`): the NFCorpus loader, and long documents built from it for the chunking test.
- **Encoding** (`encode/`): a transformer encoder with mean or CLS pooling written out, an embedding cache, and
  vector reduction (PCA or cutting).
- **Retrieval** (`retrieval/`): Recall@k, NDCG@k and MRR; exact search; FAISS flat and HNSW indexes; three
  chunking strategies.
- **Fine-tuning** (`finetune/`): contrastive training with in-batch negatives, from title-abstract pairs or
  train-split questions, and the in-batch false-negative rate.

NDCG and Recall are checked against sklearn, mean pooling against `sentence-transformers`, PCA against sklearn,
and the FAISS flat index against exact search. Paired confidence intervals come from
[metrics-from-scratch](https://github.com/bhavya94/metrics-from-scratch).

## Run it
```console
uv sync
uv run pytest
uv run el --help
```
The first run downloads NFCorpus and the models from Hugging Face. Embeddings and fine-tuned models are cached in
`.cache/`, so later runs are fast. `el finetune` trains 3 seeds of two runs, about 10 minutes on an M1's GPU.
`el ann` takes about 8 minutes, mostly building the 200k-document HNSW graph.

## Charts
`docs/views/index.html` explains the retrieval pipeline and draws each result above. Every command saves its
numbers with `uv run el --results-dir docs/results <command>`; `uv run python scripts/build_views.py` rebuilds
the page from those files.

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py` enforces both.

1. `data`: NFCorpus and the long-document test set.
2. `encode`: text to vectors, the cache, and reduction.
3. `retrieval`: metrics, search, indexes and chunking.
4. `finetune`: contrastive training.
5. `cli.py`: the `el` command.

## Limits
- **One dataset.** NFCorpus is medical and small. Gains and losses here may not carry to other domains.
- **One epoch, three seeds, 323 questions.** That is enough to rule out large fine-tuning effects, not small
  ones: the intervals are 0.01 to 0.03 wide.
- **Synthetic padding.** The ANN scaling rows pad the corpus with random vectors, not real documents.
- **Synthetic long documents.** The chunking test stitches unrelated abstracts together, so its boundaries are
  cleaner than in real long documents.
- **8 GB laptop.** Training uses MPS with 128-token texts and batch 32; larger settings ran out of memory.
