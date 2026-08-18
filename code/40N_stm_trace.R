suppressPackageStartupMessages(library(readr))
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

rows <- do.call(rbind, lapply(c("us", "br"), function(corpus) {
  f <- readRDS(here(sprintf("data/irt/stm_fit_%s.rds", corpus)))
  b <- f$convergence$bound
  n <- nrow(f$theta)
  message(sprintf("[%s] %d EM iterations, %d docs", corpus, length(b), n))
  data.frame(model = "stm", corpus = corpus, iteration = seq_along(b),
             objective = "bound_per_doc", value = b / n)
}))
write_csv(rows, here("data/irt/topic_quality_trace_stm.csv"))
message("wrote topic_quality_trace_stm.csv")
