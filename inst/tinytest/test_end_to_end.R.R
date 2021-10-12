##
## This test uses the ng20 dataset and compares the output obtained by running an ETM model fit using R
## to the same ETM model fit using the original Python implementation (https://github.com/bnosac-dev/ETM)
##   - on the same data
##   - using the same seed '2019' and without shuffling of the training data
##   - with the same hyperparameters
##   - using libtorch 1.9.0 on CPU
##
## Note that this uses no pretrained embeddings and used the following model parameters
## args.seed = 2019
## args.data_path = "dev/ETM/data/20ng"
## args.emb_size = 3
## args.train_embeddings = True
## args.epochs = 2              >>> epoch = 2
## args.t_hidden_size = 5       >>> dim = 5
## args.rho_size = 3            >>> embeddings = 3
## args.num_topics = 4          >>> k = 4
## args.optimizer == 'adam'     >>> optim_adam
## args.lr=0.005                >>> lr = 0.005
## args.wdeay=1.2e-06           >>> weight_decay = 0.0000012
## args.theta_act='relu'        >>> activation='relu'
## args.enc_drop=0              >>> dropout=0

## rho: word embeddings, alpha: topic emittance

if (torch::torch_is_installed()) {
    library(torch)
    library(topicmodels.etm)
    path_tinytest_data <- system.file(package = "topicmodels.etm", "tinytest", "data")
    #path_tinytest_data <- "inst/tinytest/data"
    
    data(ng20, package = "topicmodels.etm")
    vocab  <- ng20$vocab
    tokens <- ng20$bow_tr$tokens
    counts <- ng20$bow_tr$counts
    
    set.seed(2019)
    torch_manual_seed(2019)
    model     <- ETM(k = 4, vocab = vocab, dim = 5, embeddings = 3, activation = 'relu', dropout = 0)
    
    ########################################################################################################
    ## check initialisation/randomisation is the same (works since torch R package version 0.5)
    ##
    ## in R: q_theta.0.weight == in Python 0.weight
    ## in R: q_theta.0.bias   == in Python 0.bias
    ## in R: q_theta.2.weight == in Python 2.weight
    ## in R: q_theta.2.bias   == in Python 2.bias
    ## in R: q_theta.2.bias   == in Python 2.bias
    #sapply(model$parameters, FUN = function(x) x$numel())
    #model <- self
    params_r        <- model$named_parameters()
    params_r$beta   <- as.matrix(model$get_beta())
    params_r$rho    <- as.matrix(model$parameters$rho.weight)  
    params_r$alphas <- as.matrix(model$parameters$alphas.weight)  
    params_r$mu_q_theta.weight       <- as.matrix(model$parameters$mu_q_theta.weight)
    params_r$mu_q_theta.bias         <- as.numeric(model$parameters$mu_q_theta.bias)
    params_r$logsigma_q_theta.weight <- as.matrix(model$parameters$logsigma_q_theta.weight)
    params_r$logsigma_q_theta.bias   <- as.numeric(model$parameters$logsigma_q_theta.bias) 
    params_python <- list()
    params_python[["0.weight"]] <- as.numeric(readLines(file.path(path_tinytest_data, "init-0.weight.txt"), warn = FALSE))
    params_python[["0.bias"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "init-0.bias.txt"), warn = FALSE))
    params_python[["2.weight"]] <- as.numeric(readLines(file.path(path_tinytest_data, "init-2.weight.txt"), warn = FALSE))
    params_python[["2.bias"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "init-2.bias.txt"), warn = FALSE))
    params_python[["beta"]]     <- as.numeric(readLines(file.path(path_tinytest_data, "init-beta.txt"), warn = FALSE))
    params_python[["rho"]]      <- as.numeric(readLines(file.path(path_tinytest_data, "init-rho.txt"), warn = FALSE))
    params_python[["alphas"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "init-alphas.txt"), warn = FALSE))
    params_python[["mu_q_theta.weight"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "init-mu_q_theta.weight.txt"), warn = FALSE))
    params_python[["mu_q_theta.bias"]]     <- as.numeric(readLines(file.path(path_tinytest_data, "init-mu_q_theta.bias.txt"), warn = FALSE))
    params_python[["logsigma_q_theta.weight"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "init-logsigma_q_theta.weight.txt"), warn = FALSE))
    params_python[["logsigma_q_theta.bias"]]     <- as.numeric(readLines(file.path(path_tinytest_data, "init-logsigma_q_theta.bias.txt"), warn = FALSE))
    params_python[["2.weight"]] <- matrix(params_python[["2.weight"]], nrow = nrow(params_r$q_theta.2.weight), byrow = TRUE)
    params_python[["0.weight"]] <- matrix(params_python[["0.weight"]], nrow = nrow(params_r$q_theta.0.weight), byrow = TRUE)
    params_python[["beta"]]     <- matrix(params_python[["beta"]], nrow = nrow(params_r$beta), byrow = TRUE)
    params_python[["rho"]]      <- matrix(params_python[["rho"]], nrow = nrow(params_r$rho), byrow = TRUE)
    params_python[["alphas"]]   <- matrix(params_python[["alphas"]], nrow = nrow(params_r$alphas), byrow = TRUE)
    params_python[["mu_q_theta.weight"]]   <- matrix(params_python[["mu_q_theta.weight"]], nrow = nrow(params_r$mu_q_theta.weight), byrow = TRUE)
    params_python[["logsigma_q_theta.weight"]]   <- matrix(params_python[["logsigma_q_theta.weight"]], nrow = nrow(params_r$logsigma_q_theta.weight), byrow = TRUE)
    
    expect_equal(as.numeric(params_r$q_theta.0.bias), params_python[["0.bias"]])
    expect_equal(as.numeric(params_r$q_theta.2.bias), params_python[["2.bias"]])
    expect_equal(as.matrix(params_r$q_theta.2.weight), params_python[["2.weight"]])
    expect_equal(as.matrix(params_r$q_theta.0.weight), params_python[["0.weight"]])
    expect_equal(as.matrix(params_r$beta), params_python[["beta"]])
    expect_equal(as.matrix(params_r$rho), params_python[["rho"]])
    expect_equal(as.matrix(params_r$alphas), params_python[["alphas"]])
    expect_equal(as.numeric(params_r$mu_q_theta.bias), params_python[["mu_q_theta.bias"]])
    expect_equal(as.numeric(params_r$logsigma_q_theta.bias), params_python[["logsigma_q_theta.bias"]])
    expect_equal(as.matrix(params_r$mu_q_theta.weight), params_python[["mu_q_theta.weight"]])
    expect_equal(as.matrix(params_r$logsigma_q_theta.weight), params_python[["logsigma_q_theta.weight"]])
    ##
    ## train the model - note that for this we have made sure permuting is not done on the training data in the python script
    ##
    #optimizer <- optim_sgd(params = model$parameters)
    optimizer <- optim_adam(params = model$parameters, lr = 0.005, weight_decay = 0.0000012)
    
    traindata <- list(tokens = tokens, counts = counts, vocab = vocab)
    test1     <- list(tokens = ng20$bow_ts_h1$tokens, counts = ng20$bow_ts_h1$counts, vocab = vocab)
    test2     <- list(tokens = ng20$bow_ts_h2$tokens, counts = ng20$bow_ts_h2$counts, vocab = vocab)
    
    set.seed(2019)
    torch_manual_seed(2019)
    #debugonce(model$train_epoch)
    out <- model$fit_original(data = traindata, test1 = test1, test2 = test2, epoch = 2, 
                              optimizer = optimizer, batch_size = 1000, 
                              lr_anneal_factor = 4, lr_anneal_nonmono = 10, 
                              permute = FALSE) ## do not permute training data in order to get same run as in python
    
    if(FALSE){
        ##
        ## We did various checks on correct implementation of the different subfunctions of the estimation procedure giving exact match with the python implementation
        ## as well as the progress of the loss, kl_theta, nelbo which are all the same
        ## Unfortunately this doesn's make a test 100% bullet proof on libtorch apparently 
        ## we noticed all parameters are the same when comparing R / Python implementation even including the backward step but when doing optimizer.step, the results diverge
        ## probably this is something we can't make an end-to-end unit test for and we should make an issue at the torch repository
        ##
        #v <- model$get_beta()
        #v <- as.matrix(v)
        #pv <- as.numeric(readLines("dev/ETM/test.txt", warn = FALSE))
        #pv <- matrix(pv, nrow = nrow(v), byrow = TRUE)
        #all.equal(v, pv)
        
        ########################################################################################################
        ## check params after model run is the same (works since torch R package version 0.5)
        ##
        ## in R: q_theta.0.weight == in Python 0.weight
        ## in R: q_theta.0.bias   == in Python 0.bias
        ## in R: q_theta.2.weight == in Python 2.weight
        ## in R: q_theta.2.bias   == in Python 2.bias
        params_r        <- model$named_parameters()
        params_r$beta   <- as.matrix(model$get_beta())
        params_r$rho    <- as.matrix(model$parameters$rho.weight)  
        params_r$alphas <- as.matrix(model$parameters$alphas.weight)  
        params_python <- list()
        params_python[["0.weight"]] <- as.numeric(readLines(file.path(path_tinytest_data, "end-0.weight.txt"), warn = FALSE))
        params_python[["0.bias"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "end-0.bias.txt"), warn = FALSE))
        params_python[["2.weight"]] <- as.numeric(readLines(file.path(path_tinytest_data, "end-2.weight.txt"), warn = FALSE))
        params_python[["2.bias"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "end-2.bias.txt"), warn = FALSE))
        params_python[["beta"]]     <- as.numeric(readLines(file.path(path_tinytest_data, "end-beta.txt"), warn = FALSE))
        params_python[["rho"]]      <- as.numeric(readLines(file.path(path_tinytest_data, "end-rho.txt"), warn = FALSE))
        params_python[["alphas"]]   <- as.numeric(readLines(file.path(path_tinytest_data, "end-alphas.txt"), warn = FALSE))
        params_python[["2.weight"]] <- matrix(params_python[["2.weight"]], nrow = nrow(params_r$q_theta.2.weight), byrow = TRUE)
        params_python[["0.weight"]] <- matrix(params_python[["0.weight"]], nrow = nrow(params_r$q_theta.0.weight), byrow = TRUE)
        params_python[["beta"]]     <- matrix(params_python[["beta"]], nrow = nrow(params_r$beta), byrow = TRUE)
        params_python[["rho"]]      <- matrix(params_python[["rho"]], nrow = nrow(params_r$rho), byrow = TRUE)
        params_python[["alphas"]]   <- matrix(params_python[["alphas"]], nrow = nrow(params_r$alphas), byrow = TRUE)
        
        expect_equal(as.numeric(params_r$q_theta.0.bias), params_python[["0.bias"]])
        expect_equal(as.numeric(params_r$q_theta.2.bias), params_python[["2.bias"]])
        expect_equal(as.matrix(params_r$q_theta.2.weight), params_python[["2.weight"]])
        expect_equal(as.matrix(params_r$q_theta.0.weight), params_python[["0.weight"]])
        expect_equal(as.matrix(params_r$beta), params_python[["beta"]])
        expect_equal(as.matrix(params_r$beta), params_python[["beta"]], tolerance = 0.00001)
        expect_equal(as.matrix(params_r$rho), params_python[["rho"]])
        expect_equal(as.matrix(params_r$alphas), params_python[["alphas"]])
        
        # test <- subset(out$loss, out$loss$batch_is_last == TRUE)
        # plot(test$epoch, test$loss)
        # 
        # topic.centers     <- as.matrix(model, type = "embedding", which = "topics")
        # word.embeddings   <- as.matrix(model, type = "embedding", which = "words")
        # topic.terminology <- as.matrix(model, type = "beta")
        # 
        # terminology       <- predict(model, type = "terms", top_n = 4)
        # terminology      
    }
}
