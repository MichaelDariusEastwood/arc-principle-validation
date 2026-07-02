#!/usr/bin/env Rscript
# ==============================================================================
# INDEPENDENT R CROSS-VALIDATION OF PYTHON SCIPY CURVE-FITTING RESULTS
# ==============================================================================
#
# Purpose:
#   Verify that the Python scipy best-model selections for the 25 empirical
#   curve-fit domains are not optimiser artifacts by reproducing the fits
#   independently in R using nls() / lm().
#
# Method:
#   For each domain, fit all 7 candidate models (power_law, exponential,
#   logistic, hill, saturation_exp, michaelis_menten, hyperbolic_decay),
#   compute AICc, and select the best model. Compare to the Python results.
#
# Threshold for cross-validation success: >= 23/25 best-model agreements.
#
# Michael Darius Eastwood | March 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(minpack.lm)  # nlsLM for robust Levenberg-Marquardt
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  # fallback for Rscript invocation
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  if (length(script_path) > 0) {
    script_dir <- dirname(script_path)
  } else {
    script_dir <- "."
  }
}

repo_root    <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
manifest_path <- file.path(repo_root, "data", "canonical_50_domain_manifest.json")
python_results_path <- file.path(repo_root, "results", "results_50_domain_validation.json")
output_path  <- file.path(repo_root, "results", "cross_validation_R_results.json")

cat("===========================================================================\n")
cat("  R CROSS-VALIDATION OF PYTHON CURVE-FIT RESULTS\n")
cat("===========================================================================\n")
cat(sprintf("  Manifest:        %s\n", manifest_path))
cat(sprintf("  Python results:  %s\n", python_results_path))
cat(sprintf("  Output:          %s\n", output_path))
cat("\n")

# ---------------------------------------------------------------------------
# Load manifest and Python results
# ---------------------------------------------------------------------------
manifest <- fromJSON(manifest_path, simplifyVector = FALSE)
domains  <- manifest$domains

python_json <- fromJSON(python_results_path, simplifyVector = FALSE)
python_results <- python_json$results

# Build lookup: domain_id -> python best_model
python_best <- list()
for (r in python_results) {
  if (!is.null(r$best_model)) {
    python_best[[as.character(r$domain_id)]] <- r$best_model
  }
}

# ---------------------------------------------------------------------------
# Family labels (must match Python exactly)
# ---------------------------------------------------------------------------
family_for <- function(model_name) {
  map <- list(
    power_law        = "power_law",
    exponential      = "exponential",
    saturation_exp   = "bounded",
    michaelis_menten = "bounded",
    logistic         = "bounded",
    hill             = "bounded",
    hyperbolic_decay = "bounded"
  )
  map[[model_name]]
}

# ---------------------------------------------------------------------------
# AICc computation (must match Python formula exactly)
# ---------------------------------------------------------------------------
compute_aicc <- function(n, k, rss) {
  if (n <= k + 1 || !is.finite(rss) || rss <= 0) return(Inf)
  rss_term <- max(rss / n, 1e-12)
  aic <- n * log(rss_term) + 2 * k
  aicc <- aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
  return(aicc)
}

# ---------------------------------------------------------------------------
# Saturation guard (must match Python)
# ---------------------------------------------------------------------------
has_meaningful_saturation <- function(y, asymptote, threshold = 0.5) {
  if (!is.finite(asymptote) || asymptote <= 0) return(FALSE)
  return((max(y) / asymptote) >= threshold)
}

# ---------------------------------------------------------------------------
# Safe NLS wrapper using Levenberg-Marquardt
# ---------------------------------------------------------------------------
safe_nls <- function(formula, data, start, lower = NULL, upper = NULL,
                     maxiter = 5000) {
  tryCatch({
    if (!is.null(lower) && !is.null(upper)) {
      fit <- nlsLM(formula, data = data, start = start,
                    lower = lower, upper = upper,
                    control = nls.lm.control(maxiter = maxiter))
    } else {
      fit <- nlsLM(formula, data = data, start = start,
                    control = nls.lm.control(maxiter = maxiter))
    }
    return(fit)
  }, error = function(e) {
    return(NULL)
  })
}

# ---------------------------------------------------------------------------
# Model fitters (each returns list with name, family, valid, rss, aicc, params)
# ---------------------------------------------------------------------------

fit_power_law <- function(x, y) {
  name <- "power_law"
  fam  <- "power_law"
  mask <- (x > 0) & (y > 0)
  if (sum(mask) < 3) return(list(name=name, family=fam, valid=FALSE))

  lx <- log(x[mask])
  ly <- log(y[mask])
  fit <- lm(ly ~ lx)
  a <- exp(coef(fit)[1])
  b <- coef(fit)[2]
  y_pred <- a * x^b
  if (any(!is.finite(y_pred))) return(list(name=name, family=fam, valid=FALSE))
  rss <- sum((y - y_pred)^2)
  aicc <- compute_aicc(length(x), 2, rss)
  list(name=name, family=fam, valid=TRUE, rss=rss, aicc=aicc,
       params=list(a=unname(a), b=unname(b)))
}

fit_exponential <- function(x, y) {
  name <- "exponential"
  fam  <- "exponential"
  mask <- y > 0
  if (sum(mask) < 3) return(list(name=name, family=fam, valid=FALSE))

  ly <- log(y[mask])
  fit <- lm(ly ~ x[mask])
  a <- exp(coef(fit)[1])
  b <- coef(fit)[2]
  y_pred <- a * exp(b * x)
  if (any(!is.finite(y_pred))) return(list(name=name, family=fam, valid=FALSE))
  rss <- sum((y - y_pred)^2)
  aicc <- compute_aicc(length(x), 2, rss)
  list(name=name, family=fam, valid=TRUE, rss=rss, aicc=aicc,
       params=list(a=unname(a), b=unname(b)))
}

fit_saturation_exp <- function(x, y) {
  name <- "saturation_exp"
  fam  <- "bounded"

  y_max0 <- max(max(y) * 1.05, 1e-6)
  k0     <- max(1.0 / (mean(abs(x)) + 1e-6), 1e-6)

  fit <- safe_nls(y ~ y_max * (1 - exp(-k * x)),
                  data = data.frame(x=x, y=y),
                  start = list(y_max = y_max0, k = k0),
                  lower = c(0, 0),
                  upper = c(Inf, Inf))
  if (is.null(fit)) return(list(name=name, family=fam, valid=FALSE))

  p <- coef(fit)
  y_pred <- predict(fit)
  if (!has_meaningful_saturation(y, p["y_max"]))
    return(list(name=name, family=fam, valid=FALSE))

  rss <- sum(residuals(fit)^2)
  aicc <- compute_aicc(length(x), 2, rss)
  list(name=name, family=fam, valid=TRUE, rss=rss, aicc=aicc,
       params=list(y_max=unname(p["y_max"]), k=unname(p["k"])))
}

fit_michaelis_menten <- function(x, y) {
  name <- "michaelis_menten"
  fam  <- "bounded"

  L0 <- max(max(y) * 1.05, 1e-6)
  K0 <- max(median(abs(x)), 1e-6)

  fit <- safe_nls(y ~ L * x / (K + x),
                  data = data.frame(x=x, y=y),
                  start = list(L = L0, K = K0),
                  lower = c(0, 0),
                  upper = c(Inf, Inf))
  if (is.null(fit)) return(list(name=name, family=fam, valid=FALSE))

  p <- coef(fit)
  if (!has_meaningful_saturation(y, p["L"]))
    return(list(name=name, family=fam, valid=FALSE))

  rss <- sum(residuals(fit)^2)
  aicc <- compute_aicc(length(x), 2, rss)
  list(name=name, family=fam, valid=TRUE, rss=rss, aicc=aicc,
       params=list(L=unname(p["L"]), K=unname(p["K"])))
}

fit_logistic <- function(x, y) {
  name <- "logistic"
  fam  <- "bounded"

  K0  <- max(max(y) * 1.05, 1e-6)
  r0  <- 0.1
  x0_init <- median(x)

  fit <- safe_nls(y ~ K / (1 + exp(-r * (x - x0))),
                  data = data.frame(x=x, y=y),
                  start = list(K = K0, r = r0, x0 = x0_init),
                  lower = c(0, 0, -Inf),
                  upper = c(Inf, Inf, Inf))
  if (is.null(fit)) return(list(name=name, family=fam, valid=FALSE))

  p <- coef(fit)
  if (!has_meaningful_saturation(y, p["K"]))
    return(list(name=name, family=fam, valid=FALSE))

  rss <- sum(residuals(fit)^2)
  aicc <- compute_aicc(length(x), 3, rss)
  list(name=name, family=fam, valid=TRUE, rss=rss, aicc=aicc,
       params=list(K=unname(p["K"]), r=unname(p["r"]), x0=unname(p["x0"])))
}

fit_hill <- function(x, y) {
  name <- "hill"
  fam  <- "bounded"

  safe_x <- pmax(x, 0)
  y_max0 <- max(max(y) * 1.05, 1e-6)
  K0     <- max(median(abs(safe_x)), 1e-6)
  n0     <- 2.0

  fit <- safe_nls(y ~ y_max * safe_x^n / (K^n + safe_x^n),
                  data = data.frame(safe_x=safe_x, y=y),
                  start = list(y_max = y_max0, K = K0, n = n0),
                  lower = c(0, 0, 0.1),
                  upper = c(Inf, Inf, 20))
  if (is.null(fit)) return(list(name=name, family=fam, valid=FALSE))

  p <- coef(fit)
  if (!has_meaningful_saturation(y, p["y_max"]))
    return(list(name=name, family=fam, valid=FALSE))

  rss <- sum(residuals(fit)^2)
  aicc <- compute_aicc(length(x), 3, rss)
  list(name=name, family=fam, valid=TRUE, rss=rss, aicc=aicc,
       params=list(y_max=unname(p["y_max"]), K=unname(p["K"]), n=unname(p["n"])))
}

fit_hyperbolic_decay <- function(x, y) {
  name <- "hyperbolic_decay"
  fam  <- "bounded"

  positive_x <- x[x >= 0]
  a0 <- max(max(y) * (min(positive_x) + 1), 1e-6)
  b0 <- 1.0

  fit <- safe_nls(y ~ a / (b + x),
                  data = data.frame(x=x, y=y),
                  start = list(a = a0, b = b0),
                  lower = c(0, 0),
                  upper = c(Inf, Inf))
  if (is.null(fit)) return(list(name=name, family=fam, valid=FALSE))

  p <- coef(fit)
  rss <- sum(residuals(fit)^2)
  aicc <- compute_aicc(length(x), 2, rss)
  list(name=name, family=fam, valid=TRUE, rss=rss, aicc=aicc,
       params=list(a=unname(p["a"]), b=unname(p["b"])))
}

# ---------------------------------------------------------------------------
# Evaluate one empirical domain
# ---------------------------------------------------------------------------
evaluate_domain <- function(domain) {
  x <- as.numeric(domain$dataset$x)
  y <- as.numeric(domain$dataset$y)
  n <- length(x)

  # Decide which fitters to run (match Python: hyperbolic only if predicted)
  allow_hyp <- identical(domain$predicted_model, "hyperbolic_decay") ||
               isTRUE(domain$allow_hyperbolic)

  fits <- list(
    fit_power_law(x, y),
    fit_exponential(x, y),
    fit_saturation_exp(x, y),
    fit_michaelis_menten(x, y),
    fit_logistic(x, y),
    fit_hill(x, y)
  )
  if (allow_hyp) {
    fits <- c(fits, list(fit_hyperbolic_decay(x, y)))
  }

  # Keep only valid fits
  valid_fits <- Filter(function(f) isTRUE(f$valid), fits)

  if (length(valid_fits) == 0) {
    return(list(
      domain_id = domain$id,
      name      = domain$name,
      status    = "no_valid_fits",
      r_best_model  = NA_character_,
      r_best_family = NA_character_,
      r_best_aicc   = NA_real_,
      all_fits  = list()
    ))
  }

  # Sort by AICc (ascending)
  aiccs <- sapply(valid_fits, function(f) f$aicc)
  ord   <- order(aiccs)
  ranked <- valid_fits[ord]

  best <- ranked[[1]]

  # Build summary of all valid fits for output
  fit_summaries <- lapply(ranked, function(f) {
    list(name = f$name, family = f$family, aicc = f$aicc, rss = f$rss)
  })

  list(
    domain_id     = domain$id,
    name          = domain$name,
    n_points      = n,
    r_best_model  = best$name,
    r_best_family = best$family,
    r_best_aicc   = best$aicc,
    r_best_params = best$params,
    all_fits      = fit_summaries
  )
}

# ---------------------------------------------------------------------------
# Run cross-validation on the 25 empirical_curve_fit domains (IDs 1-25)
# ---------------------------------------------------------------------------
empirical_domains <- Filter(
  function(d) identical(d$evidence_tier, "empirical_curve_fit"),
  domains
)

cat(sprintf("  Evaluating %d empirical curve-fit domains ...\n\n", length(empirical_domains)))

results <- list()
for (domain in empirical_domains) {
  cat(sprintf("  [%2d] %-45s", domain$id, domain$name))
  res <- evaluate_domain(domain)
  results <- c(results, list(res))

  py_best <- python_best[[as.character(domain$id)]]
  match_flag <- identical(res$r_best_model, py_best)
  cat(sprintf("  R: %-18s  Py: %-18s  %s\n",
              ifelse(is.na(res$r_best_model), "FAILED", res$r_best_model),
              ifelse(is.null(py_best), "???", py_best),
              ifelse(match_flag, "AGREE", "DISAGREE")))
}

# ---------------------------------------------------------------------------
# Tally agreements
# ---------------------------------------------------------------------------
n_domains <- length(results)
agreements <- 0
disagreements <- list()

for (res in results) {
  py_best <- python_best[[as.character(res$domain_id)]]
  if (identical(res$r_best_model, py_best)) {
    agreements <- agreements + 1
  } else {
    disagreements <- c(disagreements, list(list(
      domain_id    = res$domain_id,
      name         = res$name,
      r_best       = res$r_best_model,
      python_best  = py_best
    )))
  }
}

cat("\n")
cat("===========================================================================\n")
cat("  CROSS-VALIDATION SUMMARY\n")
cat("===========================================================================\n")
cat(sprintf("  Agreements:    %d / %d\n", agreements, n_domains))
cat(sprintf("  Disagreements: %d / %d\n", n_domains - agreements, n_domains))
cat("\n")

if (length(disagreements) > 0) {
  cat("  Disagreement details:\n")
  for (d in disagreements) {
    cat(sprintf("    ID %2d  %-45s  R: %-18s  Python: %-18s\n",
                d$domain_id, d$name, d$r_best, d$python_best))
  }
  cat("\n")
}

threshold <- 23
if (agreements >= threshold) {
  verdict <- sprintf("PASS: %d/%d agreements meets the >= %d threshold. Python fits are independently validated.",
                     agreements, n_domains, threshold)
} else {
  verdict <- sprintf("FAIL: %d/%d agreements does not meet the >= %d threshold. Investigate disagreements.",
                     agreements, n_domains, threshold)
}
cat(sprintf("  Verdict: %s\n", verdict))
cat("\n")

# ---------------------------------------------------------------------------
# Also check family-level agreement (which is what the primary endpoint uses)
# ---------------------------------------------------------------------------
family_agreements <- 0
for (res in results) {
  py_best <- python_best[[as.character(res$domain_id)]]
  if (!is.null(py_best) && !is.na(res$r_best_family)) {
    py_family <- family_for(py_best)
    if (identical(res$r_best_family, py_family)) {
      family_agreements <- family_agreements + 1
    }
  }
}
cat(sprintf("  Family-level agreements: %d / %d\n", family_agreements, n_domains))
cat("\n")

# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------
output <- list(
  metadata = list(
    description = "Independent R cross-validation of Python scipy curve-fitting results",
    date = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    R_version = R.version.string,
    threshold = threshold
  ),
  summary = list(
    n_domains = n_domains,
    model_agreements = agreements,
    family_agreements = family_agreements,
    verdict = verdict
  ),
  disagreements = disagreements,
  domain_results = results
)

json_out <- toJSON(output, auto_unbox = TRUE, pretty = TRUE, digits = 10)
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
writeLines(json_out, output_path)
cat(sprintf("  Results written to: %s\n", output_path))
cat("===========================================================================\n")
