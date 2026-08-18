suppressPackageStartupMessages({
  library(arrow); library(quanteda); library(quanteda.textmodels)
  library(SnowballC); library(Matrix); library(dplyr); library(readr); library(stringi)
})
set.seed(14601)
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
WHICH <- if (length(.args) >= 1) tolower(.args[1]) else "both"
CORPORA <- if (WHICH == "both") c("us", "br") else WHICH

DISP        <- "quasipoisson"  # allow overdispersion (indifferent for point estimates)
COUNT_CAP   <- 50L             # winsorize per-document term counts (no feature dropped)
TOL         <- c(1e-6, 1e-6)   # convergence tolerance
TARGETS     <- c(Inf, 8000, 5000, 3000, 2000)  # OOV-fold cascade
RARE_FREQ   <- 20L             # corpus frequency at or below which a type folds to OOV
RARE_DOCF   <- 10L             # document frequency at or below which a type folds to OOV
SHORT_LEN   <- 3L              # non-acronym types this short or shorter fold to SHORT
CA_FLOOR    <- 0.1             # residual floor for the sparse CA solver (package default)

STEM_LANG <- c(us = "english", br = "portuguese")
deacc <- function(x) stri_trans_general(x, "Latin-ASCII")

# =========================== base DFM (cached) ==============================
build_dfm <- function(corpus, variant) {
  suf   <- if (variant == "unigram") "_uni" else ""
  cache <- here(sprintf("data/processed/wf_dfm_%s%s.rds", corpus, suf))
  tok_path <- here(sprintf("data/processed/wf_tokens_%s%s.feather", corpus, suf))
  # The cache is keyed to the document set, so a change of estimation sample
  # rebuilds it instead of silently scaling the wrong corpus. Every other
  # threshold below remains a cheap operation on the cached DFM.
  ids <- as.character(read_feather(tok_path, col_select = "doc_id")$doc_id)
  if (file.exists(cache)) {
    ck <- readRDS(cache)
    if (ndoc(ck$dfmat) == length(ids) &&
        identical(sort(docnames(ck$dfmat)), sort(ids))) {
      say("[%s/%s] cached DFM: %d docs x %d feats", corpus, variant,
          ndoc(ck$dfmat), nfeat(ck$dfmat))
      return(ck)
    }
    say("[%s/%s] cached DFM covers %d docs but the token stream has %d; rebuilding",
        corpus, variant, ndoc(ck$dfmat), length(ids))
  }
  tb <- Sys.time()
  acr <- toupper(readLines(here(sprintf("data/processed/wf_acronyms_%s.txt", corpus)),
                           warn = FALSE))
  acr <- acr[nzchar(acr)]

  tok_df <- read_feather(tok_path)
  txt <- tok_df$tokens; names(txt) <- tok_df$doc_id
  toks <- tokens(txt, what = "fastestword") |>
    tokens_tolower() |>
    tokens_remove(pattern = "^[^a-zà-ÿ]+$", valuetype = "regex") |>
    tokens_remove(pattern = "eos", valuetype = "fixed")

  # ---- stem + deaccent, with a readable stem -> surface-form map ------------
  d0    <- dfm(toks); types <- featnames(d0)
  tf    <- as.numeric(colSums(d0)); names(tf) <- types
  stem  <- deacc(SnowballC::wordStem(types, language = STEM_LANG[[corpus]]))
  toks  <- tokens_replace(toks, types, stem, valuetype = "fixed")
  label_map <- tibble(type = types, feature = stem, freq = tf[types]) |>
    group_by(feature) |> slice_max(freq, n = 1, with_ties = FALSE) |>
    transmute(feature, label = type) |> ungroup()
  say("[%s/%s] types: %d -> %d stems", corpus, variant, length(types), n_distinct(stem))

  # ---- short non-acronym types -> SHORT bucket ------------------------------
  short <- if ("SHORT" %in% featnames(dfm(toks))) "SHORT_TOK" else "SHORT"
  ft    <- featnames(dfm(toks))
  is_short_nonacr <- nchar(ft) <= SHORT_LEN & !(toupper(ft) %in% acr)
  toks  <- tokens_replace(toks, ft[is_short_nonacr],
                          rep(short, sum(is_short_nonacr)), valuetype = "fixed")
  say("[%s/%s] short non-acronym types folded to %s: %d", corpus, variant, short,
      sum(is_short_nonacr))

  # ---- rare types -> OOV bucket --------------------------------------------
  ds   <- dfm(toks); ff <- as.numeric(colSums(ds)); dfr <- docfreq(ds)
  rare <- featnames(ds)[ff <= RARE_FREQ | dfr <= RARE_DOCF]
  oov  <- if ("__oov__" %in% featnames(ds)) "__oov_tok__" else "__oov__"
  toks <- tokens_replace(toks, rare, rep(oov, length(rare)), valuetype = "fixed")

  ck <- list(dfmat = dfm(toks), label_map = label_map, oov = oov, short = short)
  saveRDS(ck, cache)
  say("[%s/%s] built DFM: %d docs x %d feats (%.1fs) -> cached", corpus, variant,
      ndoc(ck$dfmat), nfeat(ck$dfmat), el(tb))
  ck
}

# Fold the least-frequent tail into the OOV bucket to reach a target vocabulary.
# This is masking, not truncation: the folded mass is added to OOV, so document
# lengths are preserved and no word is silently dropped.
fold_to_target <- function(dm, N, oov, short) {
  if (nfeat(dm) <= N) return(dm)
  fr <- Matrix::colSums(dm)
  keepj <- order(fr, decreasing = TRUE)[seq_len(N)]
  special <- match(intersect(c(oov, short), featnames(dm)), featnames(dm))
  keepj <- sort(unique(c(keepj, special)))
  foldj <- setdiff(seq_len(nfeat(dm)), keepj)
  folded <- Matrix::rowSums(dm[, foldj, drop = FALSE])
  Mk <- dm[, keepj, drop = FALSE]
  oovc <- match(oov, colnames(Mk)); nz <- which(folded != 0)
  add <- Matrix::sparseMatrix(i = nz, j = rep(oovc, length(nz)), x = folded[nz],
                              dims = dim(Mk), dimnames = dimnames(Mk))
  quanteda::as.dfm(Mk + add)
}
winsor <- function(dm) { dm@x <- pmin(dm@x, COUNT_CAP); dm }

# =============================== the two fits ===============================
fit_wordfish <- function(dfmat, oov, short) {
  full_vocab <- nfeat(dfmat)
  cascade <- character(0)
  for (N in TARGETS) {
    dmt <- winsor(fold_to_target(dfmat, N, oov, short))
    tw  <- Sys.time()
    # quanteda warns rather than errors when its inner loop hits the iteration
    # limit, and still returns finite scores. Record that instead of losing it:
    # "finite" and "converged" are not the same claim.
    noconv <- FALSE
    wf <- withCallingHandlers(
      textmodel_wordfish(dmt, dir = c(1, 2), sparse = TRUE, svd_sparse = TRUE,
                         dispersion = DISP, tol = TOL),
      warning = function(w) {
        if (grepl("did not converge", conditionMessage(w), fixed = TRUE)) {
          noconv <<- TRUE
          invokeRestart("muffleWarning")
        }
      })
    s   <- el(tw)
    fin <- all(is.finite(wf$theta)) && stats::sd(wf$theta) > 0
    lab <- if (is.infinite(N)) "full" else as.character(N)
    say("  wordfish vocab=%s (%d feats, cap %d): finite=%d/%d %s%s | %.0fs",
        lab, nfeat(dmt), COUNT_CAP, sum(is.finite(wf$theta)), length(wf$theta),
        if (fin) "FINITE" else "NaN",
        if (fin && noconv) " (tolerance not reached)" else if (fin) " CONVERGED" else "", s)
    cascade <- c(cascade, sprintf("%s=%s", lab,
                                  if (!fin) "nan" else if (noconv) "finite_noconv" else "ok"))
    if (fin) {
      return(list(theta = as.numeric(wf$theta), beta = as.numeric(wf$beta),
                  psi = as.numeric(wf$psi), dfmat = dmt, seconds = s,
                  full_vocab = full_vocab, vocab = nfeat(dmt), converged = !noconv,
                  target = lab, cascade = paste(cascade, collapse = ";"),
                  cap = COUNT_CAP))
    }
  }
  say("  wordfish: NO configuration converged (%s)", paste(cascade, collapse = " "))
  NULL
}

fit_ca <- function(dfmat) {
  tw <- Sys.time()
  ca <- textmodel_ca(dfmat, nd = 1, sparse = TRUE, residual_floor = CA_FLOOR)
  s  <- el(tw)
  th <- as.numeric(ca$rowcoord[, 1])
  fin <- all(is.finite(th)) && stats::sd(th) > 0
  say("  CA vocab=full (%d feats, uncapped): finite=%d/%d %s | %.0fs",
      nfeat(dfmat), sum(is.finite(th)), length(th), if (fin) "CONVERGED" else "NaN", s)
  if (!fin) return(NULL)
  list(theta = th, beta = as.numeric(ca$colcoord[, 1]), psi = rep(NA_real_, nfeat(dfmat)),
       dfmat = dfmat, seconds = s, full_vocab = nfeat(dfmat), vocab = nfeat(dfmat),
       target = "full", cascade = "full=ok", cap = NA_integer_, converged = TRUE)
}

# ============================== external benchmarks =========================
externals_us <- function(docs) {
  sc <- read_csv(here("data/us/campaignview_with_scores.csv"),
                 col_types = cols(cd = col_character(), year = col_character(),
                                  .default = col_guess())) |>
    mutate(doc_id = paste(candidate_webname, state_postal, cd, cand_party, year, sep = "|")) |>
    distinct(doc_id, .keep_all = TRUE) |>
    select(doc_id, cfscore, nominate_dim1)
  tibble(doc_id = docs, party = vapply(strsplit(docs, "|", fixed = TRUE),
                                       function(p) p[4], character(1))) |>
    left_join(sc, by = "doc_id")
}

externals_br <- function(docs) {
  map <- read_feather(here("data/br/platform_party_map.feather")) |>
    transmute(doc_id = as.character(platform_id), party = toupper(as.character(party)))
  exp <- read_csv(here("data/raw/party_scores_bolognesi2023.csv"), show_col_types = FALSE) |>
    transmute(party = toupper(SG_PARTIDO), expert = as.numeric(party_mean_expert))
  tibble(doc_id = docs) |> left_join(map, by = "doc_id") |> left_join(exp, by = "party")
}

pcorr <- function(a, b) {
  m <- is.finite(a) & is.finite(b)
  if (sum(m) < 3) return(NA_real_)
  suppressWarnings(cor(a[m], b[m]))
}

party_level_r <- function(theta, party, expert) {
  d <- tibble(theta = theta, party = party, expert = expert) |>
    filter(!is.na(party), is.finite(expert)) |>
    group_by(party) |> summarise(mt = mean(theta), expert = first(expert), .groups = "drop")
  list(r = pcorr(d$mt, d$expert), n = nrow(d), means = d)
}

# ================================== driver ==================================
metrics <- list(); metas <- list()

for (corpus in CORPORA) {
  for (variant in c("unigram", "bigram")) {
    suf <- if (variant == "unigram") "_uni" else ""
    say("\n%s\n[%s / %s]\n%s", strrep("=", 70), toupper(corpus), variant, strrep("=", 70))
    ck <- build_dfm(corpus, variant)
    dfmat <- ck$dfmat
    docs <- docnames(dfmat)
    ext <- if (corpus == "us") externals_us(docs) else externals_br(docs)
    stopifnot(identical(ext$doc_id, docs))

    for (model in c("wordfish", "ca")) {
      fit <- if (model == "wordfish") fit_wordfish(dfmat, ck$oov, ck$short) else fit_ca(dfmat)
      if (is.null(fit)) next
      th <- fit$theta

      # Orient once, globally, against the primary benchmark of the corpus.
      ref <- if (corpus == "us") pcorr(th, ext$cfscore) else party_level_r(th, ext$party, ext$expert)$r
      sgn <- if (!is.na(ref) && ref < 0) -1 else 1
      th <- th * sgn
      fit$beta <- fit$beta * sgn

      name <- sprintf("%s (%s)", if (model == "wordfish") "Wordfish" else "CA",
                      if (variant == "unigram") "unigrams" else "with bigrams")
      row <- list(estimator = name, corpus = corpus, model = model, variant = variant)

      if (corpus == "us") {
        dem <- ext$party == "Democrat"; rep_ <- ext$party == "Republican"
        row$CFScore_overall  <- pcorr(th, ext$cfscore)
        row$CFScore_Dem      <- pcorr(th[dem], ext$cfscore[dem])
        row$CFScore_Rep      <- pcorr(th[rep_], ext$cfscore[rep_])
        row$NOMINATE_overall <- pcorr(th, ext$nominate_dim1)
        row$NOMINATE_Dem     <- pcorr(th[dem], ext$nominate_dim1[dem])
        row$NOMINATE_Rep     <- pcorr(th[rep_], ext$nominate_dim1[rep_])
        say("  -> %s | CF %.3f (D %.3f, R %.3f) | NOM %.3f (D %.3f, R %.3f)",
            name, row$CFScore_overall, row$CFScore_Dem, row$CFScore_Rep,
            row$NOMINATE_overall, row$NOMINATE_Dem, row$NOMINATE_Rep)
      } else {
        pl <- party_level_r(th, ext$party, ext$expert)
        row$BR_overall <- pl$r
        say("  -> %s | BR party-level r = %.3f over %d parties", name, pl$r, pl$n)
        if (model == "wordfish") {
          # kept for the standalone BR Wordfish report (code/19, code/20)
          pl$means |> rename(mt = mt) |> arrange(expert) |>
            mutate(n_doc = vapply(party, function(p) sum(ext$party == p, na.rm = TRUE), integer(1))) |>
            select(party, mt, n_doc, expert) |>
            write_csv(here(sprintf("data/irt/wordfish_party_means_br%s.csv", suf)))
          writeLines(c(
            sprintf("ndoc=%d", ndoc(fit$dfmat)), sprintf("nfeat=%d", fit$vocab),
            sprintf("n_bigram_feats=%d", sum(grepl("_", featnames(fit$dfmat)))),
            sprintf("count_cap=%d", COUNT_CAP), sprintf("dispersion=%s", DISP),
            sprintf("tol=%g", TOL[1]), sprintf("full_vocab=%d", fit$full_vocab),
            sprintf("target_vocab=%s", fit$target), sprintf("cascade=%s", fit$cascade),
            sprintf("wordfish_fit_seconds=%.1f", fit$seconds),
            sprintf("party_level_r=%.4f", pl$r), sprintf("n_parties=%d", pl$n),
            sprintf("variant=%s", variant)),
            here(sprintf("data/irt/wordfish_meta_br%s.txt", suf)))
        }
      }
      metrics[[length(metrics) + 1]] <- row

      # ---- persist the fit so no report or table ever needs a re-fit --------
      write_csv(tibble(doc_id = docnames(fit$dfmat), theta = th),
                here(sprintf("data/irt/%s_theta_%s%s.csv", model, corpus, suf)))
      tibble(feature = featnames(fit$dfmat), beta = fit$beta, psi = fit$psi) |>
        left_join(ck$label_map, by = "feature") |>
        mutate(label = coalesce(label, feature)) |>
        select(feature, label, beta, psi) |>
        write_csv(here(sprintf("data/irt/%s_features_%s%s.csv", model, corpus, suf)))

      metas[[length(metas) + 1]] <- tibble(
        estimator = name, corpus = corpus, model = model, variant = variant,
        ndoc = ndoc(fit$dfmat), full_vocab = fit$full_vocab, vocab = fit$vocab,
        n_bigram_feats = sum(grepl("_", featnames(fit$dfmat))),
        count_cap = fit$cap, target_vocab = fit$target, cascade = fit$cascade,
        tolerance_reached = fit$converged,
        dispersion = if (model == "wordfish") DISP else NA_character_,
        tol = if (model == "wordfish") TOL[1] else NA_real_,
        residual_floor = if (model == "ca") CA_FLOOR else NA_real_,
        fit_seconds = fit$seconds)
    }
  }
}

# ================================== output ==================================
COLS <- c("CFScore_overall", "CFScore_Dem", "CFScore_Rep",
          "NOMINATE_overall", "NOMINATE_Dem", "NOMINATE_Rep", "BR_overall")
mt <- bind_rows(lapply(metrics, function(r) {
  for (c in COLS) if (is.null(r[[c]])) r[[c]] <- NA_real_
  as_tibble(r)
}))
ORDER <- c("CA (unigrams)", "CA (with bigrams)",
           "Wordfish (unigrams)", "Wordfish (with bigrams)")
# One row per estimator: the US fit supplies the US columns, the BR fit the BR column.
wide <- mt |> group_by(estimator) |>
  summarise(across(all_of(COLS), ~ if (all(is.na(.x))) NA_real_ else .x[which(!is.na(.x))[1]]),
            .groups = "drop") |>
  mutate(estimator = factor(estimator, levels = ORDER)) |> arrange(estimator)

# Running one corpus at a time must not discard the other corpus's rows: merge
# with whatever is already on disk, replacing only the corpora just fitted.
merge_keep <- function(path, new, key_cols) {
  if (file.exists(path)) {
    old <- suppressMessages(read_csv(path, show_col_types = FALSE))
    if (all(key_cols %in% names(old))) {
      old <- old |> filter(!(corpus %in% CORPORA))
      new <- bind_rows(old, new)
    }
  }
  new
}

meta_all <- merge_keep(here("data/irt/wordfish_ca_meta.csv"),
                       bind_rows(metas), c("corpus", "estimator")) |>
  arrange(corpus, variant, model)
write_csv(meta_all, here("data/irt/wordfish_ca_meta.csv"))

# The long metrics file keeps one row per corpus x model x variant; the wide
# table is a convenience view. The paper's correlations are NOT read from here:
# stage 12 recomputes them from the persisted scores so every row of the model
# comparison table shares one benchmark and one set of documents.
metrics_all <- merge_keep(here("data/irt/wordfish_ca_metrics_long.csv"),
                          mt, c("corpus", "estimator")) |>
  arrange(corpus, variant, model)
write_csv(metrics_all, here("data/irt/wordfish_ca_metrics_long.csv"))
OUTP <- here("data/irt/wordfish_ca_metrics.csv")
write_csv(wide, OUTP)

say("\n%s\nMODEL COMPARISON ROWS (Pearson r with the external benchmarks)\n%s",
    strrep("=", 96), strrep("=", 96))
print(as.data.frame(wide), digits = 3)
say("\nwrote %s", OUTP)
say("wrote %s", here("data/irt/wordfish_ca_meta.csv"))
say("TOTAL %.1fs", el(t0))
