suppressPackageStartupMessages({
  library(arrow); library(quanteda); library(stm)
  library(dplyr); library(readr)
})
say <- function(...) { message(sprintf(...)); flush.console() }
t0 <- Sys.time()
# --- repo root -------------------------------------------------------------
# Resolution order: TOPIC2IRT_ROOT env var -> this script's own location
# (Rscript) -> author fallback. Every path below is base_dir + a RELATIVE path.
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
base_dir <- Sys.getenv("TOPIC2IRT_ROOT", unset = "")
if (!nzchar(base_dir)) {
  base_dir <- if (length(.f) > 0) {
    normalizePath(file.path(dirname(.f[1]), ".."), winslash = "/", mustWork = FALSE)
  } else {
    "G:/My Drive/Papers/transfer_learning/topic2irt"
  }
}
# ---------------------------------------------------------------------------
here <- function(...) file.path(base_dir, ...)
el <- function(a) as.numeric(difftime(Sys.time(), a, units = "secs"))

.args <- commandArgs(trailingOnly = TRUE)
corpus <- if (length(.args) >= 1) tolower(.args[1]) else stop("usage: 40I_stm_fit.R [us|br]")

SEED   <- 14601
K      <- c(us = 108, br = 158)[[corpus]]
# One full EM pass over the Brazilian corpus costs about an hour (16,686 long
# documents at K=158), so the caps are set to what a fit can actually run:
# the topic-term rankings the comparison scores stabilize within the early
# iterations of a spectral start.
MAXIT  <- c(us = 50, br = 20)[[corpus]]
TOP_N  <- 25
# The US candidate factor makes the prevalence design high-dimensional (~3,450
# columns); the default pooled prior fails to converge there and stm's
# documented remedy is the sparse L1 prevalence prior (glmnet).
GAMMA_PRIOR <- if (corpus == "us") "L1" else "Pooled"

# ---- documents on the shared vocabulary ------------------------------------
vocab <- readLines(here(sprintf("data/irt/topic_quality_vocab_%s.txt", corpus)))
vocab <- vocab[nzchar(vocab)]
tok_df <- read_feather(here(sprintf("data/processed/wf_tokens_%s.feather", corpus)))
txt <- tok_df$tokens; names(txt) <- tok_df$doc_id
dfmat <- tokens(txt, what = "fastestword") |> dfm() |> dfm_match(features = vocab)
say("[%s] dfm: %d docs x %d feats (target vocab %d)", corpus, ndoc(dfmat),
    nfeat(dfmat), length(vocab))

# ---- covariates ------------------------------------------------------------
docs_id <- docnames(dfmat)
if (corpus == "us") {
  p <- strsplit(docs_id, "|", fixed = TRUE)
  meta <- tibble(doc_id = docs_id,
                 author = factor(vapply(p, `[`, character(1), 1)),
                 state  = factor(vapply(p, `[`, character(1), 2)),
                 party  = factor(vapply(p, `[`, character(1), 4)))
  prevalence <- ~ author + state + party
} else {
  map <- read_feather(here("data/br/platform_party_map.feather")) |>
    transmute(doc_id = as.character(platform_id),
              state = as.character(state), party = toupper(as.character(party)))
  meta <- tibble(doc_id = docs_id) |> left_join(map, by = "doc_id") |>
    mutate(state = factor(state), party = factor(party))
  stopifnot(!anyNA(meta$state), !anyNA(meta$party))
  prevalence <- ~ state + party
}
say("[%s] prevalence: %s", corpus, deparse(prevalence))

empty <- Matrix::rowSums(dfmat) == 0
if (any(empty)) {
  say("[%s] dropping %d documents empty on the shared vocabulary", corpus, sum(empty))
  dfmat <- dfmat[!empty, ]; meta <- meta[!empty, ]
}
stm_in <- convert(dfmat, to = "stm", docvars = as.data.frame(meta))

# ---- fit -------------------------------------------------------------------
tf <- Sys.time()
fit <- stm(documents = stm_in$documents, vocab = stm_in$vocab, K = K,
           prevalence = prevalence, data = stm_in$meta,
           init.type = "Spectral", seed = SEED, max.em.its = MAXIT,
           gamma.prior = GAMMA_PRIOR, verbose = TRUE, reportevery = 5)
secs <- el(tf)
its <- length(fit$convergence$bound)
conv <- isTRUE(fit$convergence$converged)
say("[%s] STM done: %d EM iterations, converged=%s, %.0fs", corpus, its, conv, secs)

# ---- top terms by word probability -----------------------------------------
beta <- exp(fit$beta$logbeta[[1]])          # K x V
vv <- stm_in$vocab
rows <- do.call(rbind, lapply(seq_len(K), function(t) {
  idx <- order(beta[t, ], decreasing = TRUE)[seq_len(TOP_N)]
  tibble(corpus = corpus, model = "stm", topic = t - 1L,
         rank = seq_len(TOP_N), term = vv[idx])
}))
write_csv(rows, here(sprintf("data/irt/topic_quality_top_stm_%s.csv", corpus)))

write_csv(tibble(
  corpus = corpus, model = "stm",
  implementation = sprintf("stm %s (R %s)", as.character(packageVersion("stm")),
                           paste(R.version$major, R.version$minor, sep = ".")),
  K = K, vocab = length(vv), docs = length(stm_in$documents),
  config = sprintf("prevalence %s; gamma.prior=%s; Spectral init; max.em.its=%d; seed=%d",
                   paste(deparse(prevalence), collapse = ""), GAMMA_PRIOR, MAXIT, SEED),
  iterations = its, converged = conv, seconds = round(secs, 1)),
  here(sprintf("data/irt/topic_quality_stm_meta_%s.csv", corpus)))

saveRDS(fit, here(sprintf("data/irt/stm_fit_%s.rds", corpus)))
say("[%s] wrote topic_quality_top_stm_%s.csv + meta + rds | TOTAL %.0fs",
    corpus, corpus, el(t0))
