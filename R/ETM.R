
#' @title Topic Modelling in Embedding Spaces
#' @description ETM is a generative topic model combining traditional topic models (LDA) with word embeddings (word2vec). 
#' It models each word with a categorical distribution whose natural parameter is the inner product between
#' a word embedding and an embedding of its assigned topic.\cr
#' The model is fitted using an amortized variational inference algorithm on top of libtorch.
#' @param k the number of topics to extract
#' @param embeddings either a matrix with pretrained word embeddings or an integer with the dimension of the word embeddings. Defaults to 50 if not provided.
#' @param dim dimension of the variational inference hyperparameter theta (passed on to \code{\link[torch]{nn_linear}}). Defaults to 800.
#' @param activation character string with the activation function of theta. Either one of 'relu', 'tanh', 'softplus', 'rrelu', 'leakyrelu', 'elu', 'selu', 'glu'. Defaults to 'relu'.
#' @param dropout dropout percentage on the variational distribution for theta (passed on to \code{\link[torch]{nn_dropout}}). Defaults to 0.5.
#' @param vocab a character vector with the words from the vocabulary. Defaults to the rownames of the \code{embeddings} argument.
#' @references \url{https://arxiv.org/pdf/1907.04907.pdf}
#' @return an object of class ETM which is a torch \code{nn_module} containing o.a
#' \itemize{
#'   \item{num_topics: }{the number of topics}
#'   \item{TODO.: }{TODO.}
#' }
#' @section Methods:
#' \describe{
#'   \item{\code{get_beta()}}{softmax-transformed inner product of word embedding and topic embeddings}
#'   \item{\code{fit(TODO)}}{fit the model on a document term matrix}
#' }
#' @section Arguments:
#' \describe{
#'  \item{x}{TODO.}
#'  \item{x}{TODO. In \code{dgCMatrix} format}
#' }
#' @export
#' @examples
#' library(torch)
#' library(word2vec)
#' library(ETM)
#' data(ng20, package = "ETM")
#' vocab  <- ng20$vocab
#' tokens <- ng20$bow_tr$tokens
#' counts <- ng20$bow_tr$counts
#' 
#' torch_manual_seed(123456789)
#' model     <- ETM(k = 4, vocab = vocab, dim = 5, embeddings = 25)
#' model
#' optimizer <- optim_adam(params = model$parameters, lr = 0.005, weight_decay = 0.0000012)
#' 
#' traindata <- list(tokens = tokens, counts = counts, vocab = vocab)
#' test1     <- list(tokens = ng20$bow_ts_h1$tokens, counts = ng20$bow_ts_h1$counts, vocab = vocab)
#' test2     <- list(tokens = ng20$bow_ts_h2$tokens, counts = ng20$bow_ts_h2$counts, vocab = vocab)
#' 
#' out <- model$fit(data = traindata, test1 = test1, test2 = test2, epoch = 4, 
#'                  optimizer = optimizer, batch_size = 1000, 
#'                  lr_damping_factor = 4, nonmono = 10)
#' test <- subset(out$loss, out$loss$batch_is_last == TRUE)
#' plot(test$epoch, test$loss)
#' 
#' x <- as.matrix(model$parameters$alphas.weight)
#' x <- as.matrix(model$parameters$rho.weight)
#' x <- as.matrix(model$get_beta())
#' 
#' terminology <- model$topwords(top_n = 4)
#' terminology
#' 
#' 
#' ##
#' ## Toy example with pretrained embeddings
#' ##
#' library(udpipe)
#' library(word2vec)
#' data(brussels_reviews_anno, package = "udpipe")
#' x          <- subset(brussels_reviews_anno, language %in% "nl")
#' x          <- paste.data.frame(x, term = "lemma", group = "doc_id") 
#' set.seed(123456789)
#' model      <- word2vec(x = x$lemma, dim = 15, iter = 20)
#' embeddings <- as.matrix(model)
#' 
#' dtm <- subset(brussels_reviews_anno, language %in% "nl" & upos %in% c("NOUN", "ADJ"))
#' dtm <- document_term_frequencies(dtm, document = "doc_id", term = "lemma")
#' dtm <- document_term_matrix(dtm)
#' dtm <- dtm_conform(dtm, columns = rownames(embeddings))
#' 
#' model     <- ETM(k = 8, dim = 100, embeddings = embeddings, dropout = 0.5)
#' optimizer <- optim_adam(params = model$parameters, lr = 0.005, weight_decay = 0.0000012)
#' 
#' traindata <- as_tokencounts(dtm[-c(1:50), ])
#' test1     <- as_tokencounts(dtm[1:25, ])
#' test2     <- as_tokencounts(dtm[26:50, ])
#' out <- model$fit(data = traindata, test1 = test1, test2 = test2, epoch = 40, 
#'                  optimizer = optimizer, batch_size = 1000, 
#'                  lr_damping_factor = 4, nonmono = 10)
#' plot(out$loss_test)
#' test <- subset(out$loss, out$loss$batch_is_last == TRUE)
#' plot(test$epoch, test$loss)
#' plot(test$epoch, test$nelbo)
#' terminology <- model$topwords(top_n = 7)
#' terminology
ETM <- nn_module(
  classname = "ETM",
  initialize = function(k = 20,
                        embeddings, 
                        dim = 800, 
                        activation = c("relu", "tanh", "softplus", "rrelu", "leakyrelu", "elu", "selu", "glu"), 
                        dropout = 0.5,
                        vocab = rownames(embeddings)) {
    if(missing(embeddings)){
      rho           <- 50
    }else{
      rho           <- embeddings  
    }
    num_topics    <- k
    t_hidden_size <- dim
    activation    <- match.arg(activation)
    if(is.matrix(rho)){
      stopifnot(length(vocab) == nrow(rho))
      stopifnot(all(vocab == rownames(rho)))
      train_embeddings <- FALSE
      rho_size         <- ncol(rho)
    }else{
      if(!is.character(vocab)){
        stop("provide in vocab a character vector")
      }
      train_embeddings <- TRUE
      rho_size         <- rho    
    }
    enc_drop           <- dropout
    
    vocab_size         <- length(vocab)
    self$vocab         <- vocab
    self$num_topics    <- num_topics
    self$vocab_size    <- vocab_size
    self$t_hidden_size <- t_hidden_size
    self$rho_size      <- rho_size
    self$enc_drop      <- enc_drop
    self$t_drop        <- nn_dropout(p = enc_drop)
    
    self$activation    <- activation
    self$theta_act     <- get_activation(activation)
    
    ## define the word embedding matrix \rho
    if(train_embeddings){
      self$rho           <- nn_linear(rho_size, vocab_size, bias = FALSE)    
    }else{
      #rho = nn.Embedding(num_embeddings, emsize)
      #self.rho = embeddings.clone().float().to(device)
      self$rho           <- nn_embedding(num_embeddings = vocab_size, embedding_dim = rho_size, .weight = torch_tensor(rho))
      #self$rho           <- torch_tensor(rho)
    }
    
    ## define the matrix containing the topic embeddings
    self$alphas        <- nn_linear(rho_size, self$num_topics, bias = FALSE)#nn.Parameter(torch.randn(rho_size, num_topics))
    
    ## define variational distribution for \theta_{1:D} via amortizartion
    self$q_theta       <- nn_sequential(
      nn_linear(vocab_size, t_hidden_size), 
      self$theta_act,
      nn_linear(t_hidden_size, t_hidden_size),
      self$theta_act
    )
    self$mu_q_theta       <- nn_linear(t_hidden_size, self$num_topics, bias = TRUE)
    self$logsigma_q_theta <- nn_linear(t_hidden_size, self$num_topics, bias = TRUE)
  },
  # print = function(...){
  #   cat("Embedding Topic Model", sep = "\n")
  #   cat(sprintf(" - topics: %s", self$num_topics), sep = "\n")
  #   cat(sprintf(" - vocabulary size: %s", self$vocab_size), sep = "\n")
  #   cat(sprintf(" - embedding dimension: %s", self$rho_size), sep = "\n")
  #   cat(sprintf(" - variational distribution dimension: %s", self$t_hidden_size), sep = "\n")
  #   cat(sprintf(" - variational distribution activation function: %s", self$activation), sep = "\n")
  # },
  encode = function(bows){
    # """Returns paramters of the variational distribution for \theta.
    # 
    # input: bows
    #         batch of bag-of-words...tensor of shape bsz x V
    # output: mu_theta, log_sigma_theta
    # """
    q_theta <- self$q_theta(bows)
    if(self$enc_drop > 0){
      q_theta <- self$t_drop(q_theta)
    }
    mu_theta <- self$mu_q_theta(q_theta)
    logsigma_theta <- self$logsigma_q_theta(q_theta)
    kl_theta <- -0.5 * torch_sum(1 + logsigma_theta - mu_theta$pow(2) - logsigma_theta$exp(), dim = -1)$mean()
    list(mu_theta = mu_theta, logsigma_theta = logsigma_theta, kl_theta = kl_theta)
  },
  decode = function(theta, beta){
    res            <- torch_mm(theta, beta)
    preds          <- torch_log(res + 1e-6)
    preds
  },
  get_beta = function(){
    logit <- try(self$alphas(self$rho$weight)) # torch.mm(self.rho, self.alphas)
    if(inherits(logit, "try-error")){
      logit <- self$alphas(self$rho)
    }
    #beta <- nnf_softmax(logit, dim=0)$transpose(1, 0) ## softmax over vocab dimension
    beta <- nnf_softmax(logit, dim = 1)$transpose(2, 1) ## softmax over vocab dimension
    beta  
  },
  get_theta = function(normalized_bows){
    reparameterize = function(self, mu, logvar){
      if(self$training){
        std <- torch_exp(0.5 * logvar) 
        eps <- torch_randn_like(std)
        eps$mul_(std)$add_(mu)
      }else{
        mu
      }
    }
    msg            <- self$encode(normalized_bows)
    mu_theta       <- msg$mu_theta
    logsigma_theta <- msg$logsigma_theta
    kld_theta      <- msg$kl_theta
    z              <- reparameterize(self, mu_theta, logsigma_theta)
    theta          <- nnf_softmax(z, dim=-1) 
    list(theta = theta, kld_theta = kld_theta)
  },
  forward = function(bows, normalized_bows, theta = NULL, aggregate = TRUE) {
    ## get \theta
    if(is.null(theta)){
      msg       <- self$get_theta(normalized_bows)
      theta     <- msg$theta
      kld_theta <- msg$kld_theta
    }else{
      kld_theta <- NULL
    }
    ## get \beta
    beta       <- self$get_beta()
    ## get prediction loss
    preds      <- self$decode(theta, beta)
    recon_loss <- -(preds * bows)$sum(2)
    #print(dim(recon_loss))
    if(aggregate){
      recon_loss <- recon_loss$mean()
    }
    list(recon_loss = recon_loss, kld_theta = kld_theta)
  },
  topwords = function(top_n = 10){
    self$eval()
    out <- list()
    with_no_grad({
      gammas <- self$get_beta()
      for(k in seq_len(self$num_topics)){
        gamma <- gammas[k, ]
        gamma <- as.numeric(gamma) 
        gamma <- data.frame(term = self$vocab, gamma = gamma, stringsAsFactors = FALSE)
        gamma <- gamma[order(gamma$gamma, decreasing = TRUE), ]
        out[[k]] <- head(gamma, n = top_n)
      }
    })
    out
  },
  train_epoch = function(data, optimizer, epoch, batch_size, normalize = TRUE, clip = 0){
    train_tokens   <- data$tokens
    train_counts   <- data$counts
    vocab_size     <- length(data$vocab)
    num_docs_train <- length(train_tokens)
    self$train()
    acc_loss          <- 0
    acc_kl_theta_loss <- 0
    cnt               <- 0
    indices           <- torch_randperm(num_docs_train) + 1
    #indices <- torch_tensor(seq_len(num_docs_train))
    indices           <- torch_split(indices, batch_size)
    losses            <- list()
    for(i in seq_along(indices)){
      ind <- indices[[i]]
      optimizer$zero_grad()
      self$zero_grad()
      data_batch <- get_batch(train_tokens, train_counts, ind, vocab_size)
      sums <- data_batch$sum(2)$unsqueeze(2)
      if(normalize){
        normalized_data_batch <- data_batch / sums
      }else{
        normalized_data_batch <- data_batch
      }
      #as.matrix(self$q_theta(data_batch[1:2, , drop = FALSE]))
      out <- self$forward(data_batch, normalized_data_batch)
      total_loss <- out$recon_loss + out$kld_theta
      total_loss$backward()
      
      if(clip > 0){
        nn_utils_clip_grad_norm_(self$parameters, max_norm = clip)
      }
      optimizer$step()
      
      acc_loss          <- acc_loss + torch_sum(out$recon_loss)$item()
      acc_kl_theta_loss <- acc_kl_theta_loss + torch_sum(out$kld_theta)$item()
      cnt               <- cnt + 1
      
      cur_loss <- round(acc_loss / cnt, 2) 
      cur_kl_theta <- round(acc_kl_theta_loss / cnt, 2) 
      cur_real_loss <- round(cur_loss + cur_kl_theta, 2)
      
      losses[[i]] <- data.frame(epoch = epoch, 
                                batch = i,
                                batch_is_last = i == length(indices), 
                                lr = optimizer$param_groups[[1]][['lr']], 
                                loss = cur_loss,
                                kl_theta = cur_kl_theta,
                                nelbo = cur_real_loss,
                                batch_loss = acc_loss,
                                batch_kl_theta = acc_kl_theta_loss,
                                batch_nelbo = acc_loss + acc_kl_theta_loss)
      #cat(
      #    sprintf('Epoch: %s .. batch: %s/%s .. LR: %s .. KL_theta: %s .. Rec_loss: %s .. NELBO: %s',
      #            epoch, i, length(indices), optimizer$param_groups[[1]][['lr']], cur_kl_theta, cur_loss, cur_real_loss), sep = "\n")
    }
    losses <- do.call(rbind, losses)
    losses
  },
  evaluate = function(data1, data2, batch_size, normalize = TRUE){
    self$eval()
    vocab_size <- length(data1$vocab)
    tokens1    <- data1$tokens
    counts1    <- data1$counts
    tokens2    <- data2$tokens
    counts2    <- data2$counts
    
    indices    <- torch_split(torch_tensor(seq_along(tokens1)), batch_size)
    ppl_dc     <- 0
    with_no_grad({
      beta     <- self$get_beta()
      acc_loss <- 0
      cnt      <- 0
      for(i in seq_along(indices)){
        ## get theta from first half of docs
        ind          <- indices[[i]]
        data_batch_1 <- get_batch(tokens1, counts1, ind, vocab_size)
        sums         <- data_batch_1$sum(2)$unsqueeze(2)
        if(normalize){
          normalized_data_batch <- data_batch_1 / sums
        }else{
          normalized_data_batch <- data_batch_1
        }
        msg   <- self$get_theta(normalized_data_batch)
        theta <- msg$theta
        
        ## get prediction loss using second half
        data_batch_2 <- get_batch(tokens2, counts2, ind, vocab_size)
        sums         <- data_batch_2$sum(2)$unsqueeze(2)
        res          <- torch_mm(theta, beta)
        preds        <- torch_log(res)
        recon_loss   <- -(preds * data_batch_2)$sum(2)
        
        loss         <- recon_loss / sums$squeeze()
        loss         <- loss$mean()$item()
        acc_loss     <- acc_loss + loss
        cnt          <- cnt + 1
      }
      cur_loss <- acc_loss / cnt
      cur_loss <- as.numeric(cur_loss)
      ppl_dc   <- round(exp(cur_loss), digits = 1)
    })
    ppl_dc
  },
  fit = function(data, test1, test2, optimizer, epoch, batch_size, normalize = TRUE, clip = 0, lr_damping_factor = 4, nonmono = 10){
    epochs       <- epoch
    anneal_lr    <- !missing(lr_damping_factor)
    best_epoch   <- 0
    best_val_ppl <- 1e9
    all_val_ppls <- c()
    losses       <- list()
    for(epoch in seq_len(epochs)){
      lossevolution   <- self$train_epoch(data = data, optimizer = optimizer, epoch = epoch, batch_size = batch_size, normalize = normalize, clip = clip)
      losses[[epoch]] <- lossevolution
      val_ppl         <- self$evaluate(test1, test2, batch_size = batch_size, normalize = normalize)
      if(val_ppl < best_val_ppl){
        best_epoch   <- epoch
        best_val_ppl <- val_ppl
        ## TODO save model
      }else{
        ## check whether to anneal lr
        lr <- optimizer$param_groups[[1]]$lr
        if(anneal_lr & lr > 1e-5 & (length(all_val_ppls) > nonmono) & val_ppl > min(tail(all_val_ppls, n = nonmono))){
          optimizer$param_groups[[1]]$lr <- optimizer$param_groups[[1]]$lr / lr_damping_factor
        }
      }
      all_val_ppls  <- append(all_val_ppls, val_ppl)
      lossevolution <- subset(lossevolution, batch_is_last == TRUE)
      cat(
        sprintf('Epoch: %03d/%03d, learning rate: %5f, KL_theta: %2f, Rec_loss: %2f, NELBO: %s',
                lossevolution$epoch, epochs, optimizer$param_groups[[1]][['lr']], lossevolution$kl_theta, lossevolution$loss, lossevolution$nelbo), sep = "\n")
    }
    losses <- do.call(rbind, losses)
    list(loss = losses, loss_test = all_val_ppls)
  }
)
get_batch <- function(tokens, counts, ind, vocab_size){
  ind        <- as.integer(ind)
  batch_size <- length(ind)
  data_batch <- torch_zeros(c(batch_size, vocab_size))
  tokens     <- tokens[ind]
  counts     <- counts[ind]
  for(i in seq_along(tokens)){
    tok <- tokens[[i]]
    cnt <- counts[[i]]
    data_batch[i, tok] <- as.numeric(cnt)
    #for(j in tok){
    #    data_batch[i, j] <- cnt[j]        
    #}
  }
  data_batch
}

get_activation = function(act) {
  switch(act, 
         tanh = nn_tanh(),
         relu = nn_relu(),
         softplus = nn_softplus(),
         rrelu = nn_rrelu(),
         leakyrelu = nn_leaky_relu(),
         elu = nn_elu(),
         selu = nn_selu(),
         glu = nn_glu())
}

# model$train_epoch(epoch = 1, data = traindata, batch_size = args$batch_size, normalize = args$normalize, clip = args$clip)
# model$evaluate(data1 = test1, data2 = test2, batch_size = args$batch_size, normalize = TRUE)
# model$train_epoch(epoch = 2, data = traindata, batch_size = args$batch_size, normalize = args$normalize, clip = args$clip)
# model$train_epoch(epoch = 3, data = traindata, batch_size = args$batch_size, normalize = args$normalize, clip = args$clip)
# model$train_epoch(epoch = 4, data = traindata, batch_size = args$batch_size, normalize = args$normalize, clip = args$clip)
# model$train_epoch(epoch = 5, data = traindata, batch_size = args$batch_size, normalize = args$normalize, clip = args$clip)

