suppressPackageStartupMessages({
  library(arrow); library(quanteda); library(keyATM)
  library(dplyr); library(readr)
})
say <- function(...) { message(format(Sys.time(), "%H:%M:%S "), sprintf(...)); flush.console() }
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
corpus <- if (length(.args) >= 1) tolower(.args[1]) else stop("usage: 40K_keyatm_fit.R [us|br] [full|ndocs] [budget_min] [max_iters]")
N_SUB  <- if (length(.args) >= 2 && tolower(.args[2]) != "full") as.integer(.args[2]) else NA_integer_
BUDGET <- if (length(.args) >= 3) as.numeric(.args[3]) * 60 else 110 * 60   # seconds
IT_MAX <- if (length(.args) >= 4) as.integer(.args[4]) else 1500            # keyATM default

SEED  <- 14601
K     <- c(us = 108, br = 158)[[corpus]]
TOP_N <- 25
SUFFIX <- if (!is.na(N_SUB)) "_test" else ""       # never overwrite real outputs from a test
RESUME <- here(sprintf("data/irt/katm_resume_%s%s.rds", corpus, SUFFIX))

# ---- documents on the shared vocabulary ------------------------------------
vocab <- readLines(here(sprintf("data/irt/topic_quality_vocab_%s.txt", corpus)))
vocab <- vocab[nzchar(vocab)]
tok_df <- read_feather(here(sprintf("data/processed/wf_tokens_%s.feather", corpus)))
txt <- tok_df$tokens; names(txt) <- tok_df$doc_id
if (!is.na(N_SUB)) {
  set.seed(SEED)
  txt <- txt[sort(sample(length(txt), N_SUB))]
  if (file.exists(RESUME)) file.remove(RESUME)     # tests always start fresh
  say("[%s] TEST MODE: %d-doc subsample, max %d iterations", corpus, N_SUB, IT_MAX)
}
dfmat <- tokens(txt, what = "fastestword") |> dfm() |> dfm_match(features = vocab)
empty <- Matrix::rowSums(dfmat) == 0
if (any(empty)) {
  say("[%s] dropping %d documents empty on the shared vocabulary", corpus, sum(empty))
  dfmat <- dfmat[!empty, ]
}
say("[%s] dfm: %d docs x %d feats, %.0f tokens (%.1f min)",
    corpus, ndoc(dfmat), nfeat(dfmat), sum(Matrix::rowSums(dfmat)), el(t0) / 60)
docs <- keyATM_read(texts = dfmat)
n_docs <- ndoc(dfmat); n_tokens <- sum(Matrix::rowSums(dfmat))
rm(tok_df, txt, dfmat); invisible(gc())

# ---- chunked fit against a wall-clock budget -------------------------------
if (file.exists(RESUME)) say("[%s] resuming from existing checkpoint %s", corpus, basename(RESUME))
run_chunk <- function(n_iter) {
  weightedLDA(docs = docs, model = "base", number_of_topics = K,
              options = list(seed = SEED, iterations = n_iter, resume = RESUME,
                             verbose = TRUE, llk_per = 25))
}

write_outputs <- function(fit, iters_done, secs) {
  saveRDS(fit, here(sprintf("data/irt/katm_fit_%s%s.rds", corpus, SUFFIX)))
  phi <- fit$phi                                   # K x V
  vv <- colnames(phi)
  rows <- do.call(rbind, lapply(seq_len(K), function(t) {
    idx <- order(phi[t, ], decreasing = TRUE)[seq_len(TOP_N)]
    tibble(corpus = corpus, model = "katm", topic = t - 1L,
           rank = seq_len(TOP_N), term = vv[idx])
  }))
  write_csv(rows, here(sprintf("data/irt/topic_quality_top_katm_%s%s.csv", corpus, SUFFIX)))
  write_csv(tibble(
    corpus = corpus, model = "katm",
    implementation = sprintf("keyATM %s weightedLDA (R %s)",
                             as.character(packageVersion("keyATM")),
                             paste(R.version$major, R.version$minor, sep = ".")),
    K = K, vocab = length(vv), docs = n_docs,
    config = sprintf("weighted LDA without keywords (information-theory weights), collapsed Gibbs; iterations=%d; seed=%d",
                     iters_done, SEED),
    iterations = iters_done, converged = NA, seconds = round(secs, 1)),
    here(sprintf("data/irt/topic_quality_katm_meta_%s%s.csv", corpus, SUFFIX)))
}

CHUNK1 <- 25L
iters_done <- 0L
fit_secs <- 0
repeat {
  left <- BUDGET - el(t0)
  chunk <- if (iters_done == 0L) CHUNK1 else {
    rate <- fit_secs / iters_done                  # secs per iteration so far
    feasible <- floor((left * 0.9) / rate)
    min(300L, max(25L, feasible), IT_MAX - iters_done)
  }
  if (iters_done >= IT_MAX) { say("[%s] reached max iterations (%d)", corpus, IT_MAX); break }
  if (iters_done > 0L && (chunk < 25L || chunk * (fit_secs / iters_done) > left)) {
    say("[%s] budget reached: %d iterations in %.1f min", corpus, iters_done, el(t0) / 60); break
  }
  tc <- Sys.time()
  fit <- run_chunk(as.integer(chunk))
  fit_secs <- fit_secs + el(tc)
  iters_done <- iters_done + as.integer(chunk)
  say("[%s] progress: %d/%d iterations, %.2f s/iter, %.1f min elapsed",
      corpus, iters_done, IT_MAX, fit_secs / iters_done, el(t0) / 60)
  write_outputs(fit, iters_done, fit_secs)
  say("[%s] checkpoint + outputs written at %d iterations", corpus, iters_done)
}

say("[%s] wrote topic_quality_top_katm_%s%s.csv + meta + rds | %d iterations | TOTAL %.0fs",
    corpus, corpus, SUFFIX, iters_done, el(t0))
