# ETM - R package for Topic Modelling in Embedding Spaces

This repository contains an R package which is an implementation of ETM

- ETM is a generative topic model combining traditional topic models (LDA) with word embeddings (word2vec)
    - It models each word with a categorical distribution whose natural parameter is the inner product between a word embedding and an embedding of its assigned topic
    - The model is fitted using an amortized variational inference algorithm on top of libtorch (https://torch.mlverse.org)
- The techniques are explained in detail in the paper: "Topic Modelling in Embedding Spaces" by Adji B. Dieng, Francisco J. R. Ruiz and David M. Blei, available at https://arxiv.org/pdf/1907.04907.pdf 

### Installation

This R package not on CRAN, for now, you can install it with `remotes::install_github('bnosac/ETM')`, once the package has some plotting functionalities, I'll push it on CRAN.

### Example

Examples are provided in the help of the ETM function

```
library(ETM)
?ETM
```

## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

