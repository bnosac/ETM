# #' @title Utility functions to map sparse matrices to token count lists 
# #' @description Convert a sparse matrix to counts of tokens
# #' @param x a sparse Matrix
# #' @return 
# #' a list with elements tokens, counts and vocab with the content of \code{x}
# #' @export
# #' @examples
# #' library(udpipe)
# #' library(Matrix)
# #' data(brussels_reviews_anno, package = "udpipe")
# #' x  <- subset(brussels_reviews_anno, upos %in% "NOUN")
# #' x  <- document_term_frequencies(x, document = "doc_id", term = "lemma")
# #' x  <- document_term_matrix(x)
# #' tk <- as_tokencounts(x)
# #' str(tk)
# #' 
# #' ## Test to do the other way around: tokencounts to sparse matrix
# #' as_dtm <- function(tokens, counts, vocab){
# #'   nm  <- seq_len(length(tokens))
# #'   mat <- sparseMatrix(i = unlist(Map(nm, tokens, 
# #'                                      f = function(nm, key) rep(nm, length(key)))), 
# #'                       j = unlist(tokens, use.names = FALSE), 
# #'                       x = unlist(counts, use.names = FALSE))
# #'   colnames(mat) <- vocab
# #'   mat
# #' }
# #' x_back      <- as_dtm(tokens = tk$tokens, counts = tk$counts, vocab = tk$vocab)
# #' rownames(x) <- NULL
# #' all.equal(x, x_back)
as_tokencounts <- function(x){
  stopifnot(inherits(x, "dgCMatrix"))
  m             <- Matrix::summary(x)
  tokens        <- split(m$j, m$i)
  counts        <- split(m$x, m$i)
  names(tokens) <- NULL
  names(counts) <- NULL
  list(tokens = tokens, counts = counts, vocab = colnames(x))
}



as_dtm <- function(tokens, counts, vocab){
  nm  <- seq_len(length(tokens))
  mat <- sparseMatrix(i = unlist(Map(nm, tokens,
                                     f = function(nm, key) rep(nm, length(key)))),
                      j = unlist(tokens, use.names = FALSE),
                      x = unlist(counts, use.names = FALSE))
  colnames(mat) <- vocab
  mat
}
