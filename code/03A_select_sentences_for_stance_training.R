rm(list = ls())

pacman::p_load(
  manifestoR, tidyverse, tidylog,
  stringi, stringr, arrow, readxl, countrycode
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

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
dir_raw    <- file.path(base_dir, "data", "raw")
dir_proc   <- file.path(base_dir, "data", "processed")
dir_report <- file.path(base_dir, "data", "quick_report")
dir_misc   <- file.path(base_dir, "misc")
dir.create(dir_proc,   recursive = TRUE, showWarnings = FALSE)
dir.create(dir_report, recursive = TRUE, showWarnings = FALSE)

mp_setapikey(file.path(dir_misc, "manifesto_apikey.txt"))

# Persistent manifestoR cache (avoids re-hitting the API on re-runs)
cache_file <- file.path(dir_proc, "manifesto_cache.RData")
if (file.exists(cache_file)) {
  mp_load_cache(file = cache_file)
  cat(sprintf("Loaded manifestoR cache: %s\n", cache_file))
}

# --- Tunable selection parameters --------------------------------
MIN_WORDS   <- 8                          # drop stubs shorter than this
TERM_REGEX  <- "[.!?;:…][\"')”’\\]]?$"    # ends a sentence (incl. ; and :)
# -----------------------------------------------------------------

europe   <- c("Sweden", "Spain", "Germany", "United Kingdom", "Portugal",
              "Italy", "France")
americas <- c("Canada", "United States", "Mexico", "Brazil", "Argentina")
sample_countries <- sort(c(europe, americas))

# ============================================================
# 2. LOAD MANIFESTO DATA & CODEBOOK
# ============================================================

mpds <- mp_maindataset()

country_table <- mpds |>
  select(party, partyname, partyabbrev, country, countryname, parfam) |>
  distinct(party, .keep_all = TRUE)

# ============================================================
# 3. CMP -> DIMENSION / STANCE MAPPING (from xlsx)
#    Keep only non-CEE codes mapped to Economy / Society (Other = none; the
#    inference label space is {Econ,Soc} x {L,R,N} -- no Others class).
#    Codes flagged "Do not use" (e.g. parent 201, split into 201.1/201.2) are
#    intentionally dropped: a bare 3-digit 201 conflates Freedom (Right) and
#    Human Rights (Left) and cannot be poled.
# ============================================================

cmp_mapping_raw <- read_xlsx(file.path(dir_raw, "mapping_cmp_2d_stance.xlsx"))

norm_code <- function(x) sub("\\.0$", "", trimws(as.character(x)))   # 601.0 -> "601"

cmp_mapping <- cmp_mapping_raw |>
  filter(type != "cee",
         dimension %in% c("Economy", "Society"),
         stance %in% c("Left", "Right", "Neutral")) |>   # drop "Do not use" codes (201/303/305...)
  transmute(code_key = norm_code(code), dimension, stance)

cat(sprintf("Mapped codes kept (Economy/Society x L/R/N): %d\n", nrow(cmp_mapping)))

# ============================================================
# 4. DOWNLOAD CORPUS (bilingual; keep ORIGINAL-language text)
# ============================================================

cat("\nDownloading corpus from CMP API...\n")

sample_list <- mpds |>
  filter(countryname %in% sample_countries) |>
  select(party, date) |>
  as.data.frame()

corpus <- mp_corpus_df_bilingual(sample_list)

# Persist whatever was newly downloaded
mp_save_cache(file = cache_file)
cat(sprintf("Saved manifestoR cache: %s\n", cache_file))

# Keep coded manifestos with real original text
corpus <- corpus |>
  filter(annotations == TRUE, !is.na(text), nchar(trimws(text)) > 0)

cat(sprintf("Downloaded: %s quasi-sentences / %s manifestos\n",
            format(nrow(corpus), big.mark = ","),
            format(n_distinct(corpus$manifesto_id), big.mark = ",")))

# ============================================================
# 5. SENTENCE-COMPLETENESS FLAGS  (computed on FULL ordered corpus)
# ============================================================

# Preserve document order, then work within each manifesto.
corpus <- corpus |> mutate(.row = row_number())
if ("pos" %in% names(corpus)) corpus <- corpus |> arrange(manifesto_id, pos, .row)

corpus <- corpus |>
  mutate(text_o = str_squish(text)) |>
  group_by(manifesto_id) |>
  mutate(
    ends_term  = str_detect(text_o, TERM_REGEX),
    unit_break = ends_term | cmp_code == "H",          # headers are boundaries too
    prev_break = lag(unit_break, default = TRUE)        # first unit: treat prev as boundary
  ) |>
  ungroup() |>
  mutate(
    n_words          = str_count(text_o, "\\S+"),
    is_full_sentence = ends_term & prev_break,          # STRICT: starts and ends a sentence
    is_sentence_final = ends_term                       # RELAXED: only ends one
  )

# ============================================================
# 6. FILTER TO MAPPED FULL SENTENCES
# ============================================================

corpus <- corpus |>
  mutate(code_key = norm_code(cmp_code)) |>
  left_join(cmp_mapping, by = "code_key") |>
  left_join(country_table, by = "party") |>
  mutate(country_group = if_else(countryname %in% europe, "Europe", "Americas"))

# Dropped-code diagnostic: among full sentences >= MIN_WORDS, which real CMP
# content codes are NOT in the Econ/Society map (so get dropped)? Surfaces the
# bare-201 cost and per000/boilerplate volume.
dropped_codes <- corpus |>
  filter(is_full_sentence, n_words >= MIN_WORDS, is.na(dimension),
         !cmp_code %in% c("H", "0", "000", NA)) |>
  count(cmp_code, sort = TRUE)
cat("\n================ DROPPED content codes (unmapped, full sentences) ================\n")
print(head(dropped_codes, 15))
cat(sprintf("  total dropped full sentences (unmapped, excl. headers/0): %s\n",
            format(sum(dropped_codes$n), big.mark = ",")))

pool <- corpus |>
  filter(!is.na(dimension),                 # mapped to Economy/Society
         is_full_sentence,
         n_words >= MIN_WORDS) |>
  mutate(dim_stance = paste(dimension, stance, sep = "/"))

# Yield diagnostics --------------------------------------------------
mapped_all     <- corpus |> filter(!is.na(dimension))
relaxed_pool_n <- mapped_all |> filter(is_sentence_final, n_words >= MIN_WORDS) |> nrow()
cat(sprintf(
  "\nYield among MAPPED quasi-sentences (%s total):\n  strict full sentences : %s (%.1f%%)\n  relaxed (ends-only)   : %s (%.1f%%)  <- extra units a looser rule adds\n",
  format(nrow(mapped_all), big.mark = ","),
  format(nrow(pool), big.mark = ","), 100 * nrow(pool) / nrow(mapped_all),
  format(relaxed_pool_n, big.mark = ","), 100 * relaxed_pool_n / nrow(mapped_all)))

# ============================================================
# 7. DISTRIBUTIONS  (for composing a stratified sample)
# ============================================================

cat("\n================ Selected length distribution (words) ================\n")
print(summary(pool$n_words))

cat("\n================ Per LANGUAGE ================\n")
dist_language <- pool |> count(language, sort = TRUE)
print(dist_language, n = Inf)

cat("\n================ Per COUNTRY ================\n")
dist_country <- pool |> count(countryname, country_group, sort = TRUE)
print(dist_country, n = Inf)

cat("\n================ Per DIMENSION-STANCE ================\n")
dist_dim_stance <- pool |> count(dimension, stance, sort = TRUE)
print(dist_dim_stance, n = Inf)

# Bonus crosstab (country x dim-stance) to help stratify
xtab_country_ds <- pool |> count(countryname, dim_stance) |>
  pivot_wider(names_from = dim_stance, values_from = n, values_fill = 0)

# ============================================================
# 8. SAVE
# ============================================================

pool_out <- pool |>
  select(manifesto_id, party, partyname, partyabbrev, countryname, country_group,
         language, date, cmp_code, code_key, dimension, stance, dim_stance,
         n_words, text, text_en)

arrow::write_feather(pool_out, file.path(dir_proc, "sentence_training_pool.feather"))
write_csv(dist_language,   file.path(dir_report, "pool_dist_language.csv"))
write_csv(dist_country,    file.path(dir_report, "pool_dist_country.csv"))
write_csv(dist_dim_stance, file.path(dir_report, "pool_dist_dim_stance.csv"))
write_csv(xtab_country_ds, file.path(dir_report, "pool_xtab_country_dimstance.csv"))

cat(sprintf("\n=== Done ===\n  Pool sentences: %s\n  Manifestos: %s | Countries: %s | Languages: %s\n  Saved: %s\n",
            format(nrow(pool_out), big.mark = ","),
            n_distinct(pool_out$manifesto_id),
            n_distinct(pool_out$countryname),
            n_distinct(pool_out$language),
            file.path(dir_proc, "sentence_training_pool.feather")))
