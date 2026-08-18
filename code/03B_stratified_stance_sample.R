rm(list = ls())
suppressMessages(pacman::p_load(tidyverse, arrow, readr))

# ---- Config ----
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
dir_proc <- file.path(base_dir, "data", "processed")

N_PER <- 1000         # target per (language x dimension x stance) cell
SEED  <- 20260618     # reproducibility

pool <- arrow::read_feather(file.path(dir_proc, "sentence_training_pool.feather"))

# ---- Drop incomplete fragments -------------------------------------------
# Keep a sentence only if its first non-space character is an UPPERCASE letter
# or a non-letter symbol (bullet, dash, quote, digit, parenthesis). Strings
# whose first letter is lowercase are mid-sentence continuations produced by
# over-segmentation ("taking advantage of...", "strengthens...", "end."), which
# are uninformative fragments. "\\p{Ll}" = any Unicode lowercase letter.
n_before <- nrow(pool)
pool <- pool |>
  mutate(.c1 = stringr::str_sub(stringr::str_trim(text), 1, 1)) |>
  filter(nzchar(.c1), !stringr::str_detect(.c1, "\\p{Ll}")) |>
  select(-.c1)
cat(sprintf("Fragment filter: kept %s of %s (%.1f%%); dropped %s lowercase-initial.\n",
            format(nrow(pool), big.mark = ","), format(n_before, big.mark = ","),
            100 * nrow(pool) / n_before,
            format(n_before - nrow(pool), big.mark = ",")))

# ---- Stratified draw (caps at cell size when < N_PER) ----
set.seed(SEED)
samp <- pool |>
  arrange(language, dimension, stance) |>        # fixed group order -> reproducible RNG
  group_by(language, dimension, stance) |>
  slice_sample(n = N_PER) |>                      # returns all rows if cell < N_PER
  ungroup() |>
  slice_sample(prop = 1)                          # shuffle (same seeded stream)

# ---- Stable id + empty GPT label columns ----
samp <- samp |>
  mutate(
    sid           = sprintf("s%06d", row_number()),
    gpt_dimension = NA_character_,
    gpt_stance    = NA_character_
  ) |>
  select(sid, manifesto_id, party, partyname, countryname, country_group,
         language, date, cmp_code, dimension, stance, dim_stance, n_words,
         text, text_en, gpt_dimension, gpt_stance)

# ---- Save (UTF-8 BOM: opens cleanly in Excel and pandas) ----
out_csv <- file.path(dir_proc, "stratified_stance_sample.csv")
readr::write_excel_csv(samp, out_csv)

# ---- Report ----
cat(sprintf("\nSEED = %d | N_PER = %d\n", SEED, N_PER))
cat(sprintf("Rows: %s | Languages: %d\n",
            format(nrow(samp), big.mark = ","), n_distinct(samp$language)))
cat("\nPer language:\n");      samp |> count(language, sort = TRUE) |> print(n = Inf)
cat("\nPer dimension-stance:\n"); samp |> count(dimension, stance) |> print(n = Inf)
cat("\nCells below N_PER:\n")
samp |> count(language, dimension, stance) |> filter(n < N_PER) |>
  arrange(n) |> print(n = Inf)
cat(sprintf("\nSaved: %s\n", out_csv))
