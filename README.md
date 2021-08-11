# ETM - R package for Topic Modelling in Embedding Spaces

This repository contains an R package which is an implementation of ETM

- ETM is a generative topic model combining traditional topic models (LDA) with word embeddings (word2vec)
    - It models each word with a categorical distribution whose natural parameter is the inner product between a word embedding and an embedding of its assigned topic
    - The model is fitted using an amortized variational inference algorithm on top of libtorch (https://torch.mlverse.org)
- The techniques are explained in detail in the paper: "Topic Modelling in Embedding Spaces" by Adji B. Dieng, Francisco J. R. Ruiz and David M. Blei, available at https://arxiv.org/pdf/1907.04907.pdf 

### Installation

This R package is not on CRAN (yet), for now, you can install it as follows 

```
install.packages("torch")
install.packages("word2vec")
install.packages("doc2vec")
install.packages("udpipe")
install.packages("remotes")
library(torch)
remotes::install_github('bnosac/ETM', INSTALL_opts = '--no-multiarch')
```

Once the package has some plotting functionalities, I'll push it on CRAN.

### Example

Build a topic model on questions answered in Belgian parliament in 2020 in Dutch.

#### a. Get data 

- Example text of +/- 6000 questions asked in the Belgian parliament (available in R package doc2vec).
- Standardise the text a bit

```
library(torch)
library(ETM)
library(doc2vec)
library(word2vec)
data(be_parliament_2020, package = "doc2vec")
x      <- data.frame(doc_id           = be_parliament_2020$doc_id, 
                     text             = be_parliament_2020$text_nl, 
                     stringsAsFactors = FALSE)
x$text <- txt_clean_word2vec(x$text)
```

#### b. Build a word2vec model to get word embeddings and inspect it a bit

```
set.seed(1234)
w2v        <- word2vec(x = x$text, dim = 25, type = "skip-gram", iter = 10, min_count = 5, threads = 2)
embeddings <- as.matrix(w2v)
predict(w2v, newdata = c("migranten", "belastingen"), type = "nearest", top_n = 4)
$migranten
      term1               term2 similarity rank
1 migranten              lesbos  0.9434163    1
2 migranten               chios  0.9334459    2
3 migranten vluchtelingenkampen  0.9269973    3
4 migranten                kamp  0.9175452    4

$belastingen
        term1                term2 similarity rank
1 belastingen            belasting  0.9458982    1
2 belastingen          ontvangsten  0.9091899    2
3 belastingen              geheven  0.9071115    3
4 belastingen            ontduiken  0.9029559    4
```

#### c. Build the embedding topic model

- Create a bag of words document term matrix (using the udpipe package but other R packages provide similar functionalities)
- Keep only the top 50% terms with the highest TFIDF
- Make sure document/term/matrix and the embedding matrix have the same vocabulary
    
```
library(udpipe)
dtm   <- strsplit.data.frame(x, group = "doc_id", term = "text", split = " ")
dtm   <- document_term_frequencies(dtm)
dtm   <- document_term_matrix(dtm)
dtm   <- dtm_remove_tfidf(dtm, prob = 0.50)

vocab        <- intersect(rownames(embeddings), colnames(dtm))
embeddings   <- dtm_conform(embeddings, rows = vocab)
dtm          <- dtm_conform(dtm,     columns = vocab)
dim(dtm)
dim(embeddings)
```
    
- Learn 25 topics with a 100-dimensional hyperparameter for the variational inference

```
torch_manual_seed(4321)
model          <- ETM(k = 25, dim = 100, embeddings = embeddings, dropout = 0.5)
optimizer      <- optim_adam(params = model$parameters, lr = 0.005, weight_decay = 0.0000012)
loss_evolution <- model$fit(data = dtm, optimizer = optimizer, epoch = 20, batch_size = 1000)
plot(loss_evolution$loss_test, xlab = "Epoch", ylab = "Loss", main = "Loss evolution on test set")
```

![](tools/loss-evolution.png)


#### d. Inspect the model 

```
terminology  <- predict(model, type = "terms", top_n = 5)
terminology
[[1]]
          term       gamma
2221  agressie 0.017512944
1567    wapens 0.011357320
1956    beslag 0.008855813
2002      unia 0.007408552
2388 foltering 0.007176779

[[2]]
           term      gamma
1234   gebouwen 0.01591281
1244 voertuigen 0.01182650
1725   infrabel 0.01099777
2048      bpost 0.01027404
1984 bestuurder 0.00681403

[[3]]
              term      gamma
1385 gedetineerden 0.02460375
1906 gerepatrieerd 0.01325942
1347   landgenoten 0.01323096
1587    militairen 0.01186746
2047      hongkong 0.01039845

[[4]]
       term      gamma
1297    app 0.08897609
2101   favv 0.02519019
1968    gba 0.01936190
1837   apps 0.01890722
2176 tiktok 0.01491214

[[5]]
         term      gamma
1725 infrabel 0.04996496
1918   dieren 0.04363983
1414 bushmeat 0.02696244
2410    vlees 0.01729637
2048    bpost 0.01599066

[[6]]
               term       gamma
1193 geneesmiddelen 0.008525930
1701          tests 0.005812821
1385  gedetineerden 0.005222120
1637 woonzorgcentra 0.004903472
2026     aandoening 0.004264550

[[7]]
                    term      gamma
1491            facturen 0.01670218
1833                 btw 0.01503910
1989 zekerheidsbijdragen 0.01353257
1946                 rsz 0.01305994
1870                 pod 0.01061896

[[8]]
               term       gamma
1193 geneesmiddelen 0.004858544
2101           favv 0.003693702
1701          tests 0.003470343
1965             5g 0.003284918
1442           bipt 0.003238067

[[9]]
                    term       gamma
1495                 vzw 0.009609610
2748   beroepsvereniging 0.006409373
1833                 btw 0.005791733
1870                 pod 0.005371029
1558 beroepsverenigingen 0.004683901

[[10]]
              term       gamma
1865     bezoekers 0.008799853
1587    militairen 0.007766290
1922     beperking 0.006185776
1385 gedetineerden 0.006049119
1906 gerepatrieerd 0.005425004

[[11]]
                    term       gamma
2095 belastingplichtigen 0.009138852
2179             mandaat 0.007034023
1946                 rsz 0.006237295
1833                 btw 0.006221564
1984          bestuurder 0.005178992

[[12]]
                   term       gamma
1385      gedetineerden 0.006461640
2221           agressie 0.006268502
1865          bezoekers 0.004667756
1919     cyberaanvallen 0.004332834
1666 cybercriminaliteit 0.004241356

[[13]]
                    term       gamma
1833                 btw 0.006126372
1495                 vzw 0.005881680
2095 belastingplichtigen 0.004853509
1946                 rsz 0.003556216
1289     gepensioneerden 0.003081724

[[14]]
                term       gamma
1603        telewerk 0.011072752
1999         kabinet 0.007473562
2048           bpost 0.007210678
1833             btw 0.006780275
1289 gepensioneerden 0.006000138

[[15]]
                    term       gamma
1833                 btw 0.010264894
2483  overbruggingsrecht 0.007632098
1931 tweedelijnsbijstand 0.007382195
2095 belastingplichtigen 0.005279044
1289     gepensioneerden 0.004825821

[[16]]
         term      gamma
1692    spits 0.02207421
1999  kabinet 0.01708163
1603 telewerk 0.01514265
1378 turnhout 0.01506173
2385  daluren 0.01276572

[[17]]
          term      gamma
1999   kabinet 0.10355280
1603  telewerk 0.05764178
1870       pod 0.02747864
1834      reis 0.02058239
1865 bezoekers 0.01466277

[[18]]
                  term       gamma
1495               vzw 0.007413817
2748 beroepsvereniging 0.003792811
2332              jobs 0.003681063
1357              nace 0.003485834
1623        artistieke 0.003441319

[[19]]
                 term       gamma
1193   geneesmiddelen 0.006785220
1800  nationaliteiten 0.005246738
1741              dvz 0.004748864
1196             visa 0.004452683
2018 betalingstermijn 0.003933043

[[20]]
           term      gamma
1234   gebouwen 0.10468882
1725   infrabel 0.09569670
1244 voertuigen 0.05763330
2158      regie 0.04608878
1851       site 0.03914404

[[21]]
                  term       gamma
1833               btw 0.018040504
1454 personenbelasting 0.007000353
2048             bpost 0.006666712
1603          telewerk 0.006580409
1946               rsz 0.005649193

[[22]]
              term       gamma
1587    militairen 0.019814594
2150 tewerkgesteld 0.014522048
1385 gedetineerden 0.010257471
1999       kabinet 0.009206563
1865     bezoekers 0.007076799

[[23]]
               term       gamma
1922      beperking 0.007870421
1893     chronische 0.007610516
2026     aandoening 0.005695772
1919 cyberaanvallen 0.005679780
1193 geneesmiddelen 0.003775434

[[24]]
               term      gamma
1193 geneesmiddelen 0.02275880
1999        kabinet 0.01833401
2150  tewerkgesteld 0.01487170
2058            kce 0.01459871
1603       telewerk 0.01203416

[[25]]
                   term       gamma
1193     geneesmiddelen 0.007481673
1666 cybercriminaliteit 0.004415261
2048              bpost 0.003457238
1919     cyberaanvallen 0.003405995
1956             beslag 0.003151005
```

#### e. Predict alongside the model

```
newdata <- head(dtm, n = 5)
scores  <- predict(model, newdata, type = "topics")
scores
```

> More examples are provided in the help of the ETM function see `?ETM`


## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

