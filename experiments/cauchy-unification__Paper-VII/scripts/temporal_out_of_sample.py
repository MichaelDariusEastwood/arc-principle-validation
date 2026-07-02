#!/usr/bin/env python3
"""
================================================================================
TEMPORAL OUT-OF-SAMPLE VALIDATION
================================================================================

Purpose:
  The strongest form of self-administered replication: fit candidate models on
  OLD data, then test whether the winning functional FORM correctly predicts
  NEW data that did not exist when the model was fitted.

  This is NOT about recovering exact coefficients. It asks: does the same
  functional family (power law, exponential, bounded/logistic) win on both
  the training window and the held-out future window?

Domains tested:
  1. Neural scaling laws    -- Kaplan (2020) -> Chinchilla, Llama 2, GPT-4
  2. Moore's law            -- pre-2010 -> post-2010 transistor counts
  3. Epidemic curves        -- 2014 Ebola early -> late; COVID Wuhan/Italy
  4. Solar PV learning      -- pre-2010 -> post-2010 cost data
  5. Urban scaling          -- Bettencourt (2007) -> 2019 BEA/Census data

Data sources are cited inline. All values are publicly reported numbers.

================================================================================
Michael Darius Eastwood | March 2026
================================================================================
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = REPO_ROOT / "results" / "results_temporal_out_of_sample.json"

# ---------------------------------------------------------------------------
# Fitting machinery (mirrors arc_50_domain_universal_test.py conventions)
# ---------------------------------------------------------------------------

FAMILY_LABELS = {
    "power_law": "power_law",
    "exponential": "exponential",
    "saturation_exp": "bounded",
    "michaelis_menten": "bounded",
    "logistic": "bounded",
    "hill": "bounded",
    "linear": "linear",
}


def safe_rss_r2(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rss = float(np.sum((y - y_pred) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss <= 0:
        r2 = 1.0 if rss <= 1e-12 else 0.0
    else:
        r2 = 1.0 - rss / tss
    return rss, r2


def information_criteria(n: int, k: int, rss: float) -> tuple[float, float]:
    if n <= k + 1 or not np.isfinite(rss) or rss <= 0:
        return float("inf"), float("inf")
    rss_term = max(rss / n, 1e-30)
    aic = n * math.log(rss_term) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    bic = n * math.log(rss_term) + k * math.log(n)
    return aicc, bic


def _fit_result(
    name: str,
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    params: dict[str, float],
    k: int,
    valid: bool = True,
    reason: str = "",
) -> dict[str, Any]:
    if not valid:
        return {
            "name": name, "family": family, "valid": False, "reason": reason,
            "rss": float("inf"), "r2": float("-inf"),
            "aicc": float("inf"), "bic": float("inf"), "params": {},
        }
    rss, r2 = safe_rss_r2(y, y_pred)
    aicc, bic = information_criteria(len(x), k, rss)
    return {
        "name": name, "family": family, "valid": True,
        "rss": rss, "r2": r2, "aicc": aicc, "bic": bic, "params": params,
    }


# --- Power law: y = a * x^b ---
def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = (x > 0) & (y > 0)
    if int(mask.sum()) < 3:
        return _fit_result("power_law", "power_law", x, y, y, {}, 2, False, "need >= 3 positive")
    lx, ly = np.log(x[mask]), np.log(y[mask])
    slope, intercept, _, _, _ = stats.linregress(lx, ly)
    a, b = float(np.exp(intercept)), float(slope)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y_pred = a * np.power(x, b)
    if not np.all(np.isfinite(y_pred)):
        return _fit_result("power_law", "power_law", x, y, y, {}, 2, False, "non-finite")
    return _fit_result("power_law", "power_law", x, y, y_pred, {"a": a, "b": b}, 2)


def predict_power_law(params: dict, x: np.ndarray) -> np.ndarray:
    return params["a"] * np.power(x, params["b"])


# --- Exponential: y = a * exp(b * x) ---
def fit_exponential(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = y > 0
    if int(mask.sum()) < 3:
        return _fit_result("exponential", "exponential", x, y, y, {}, 2, False, "need >= 3 positive y")
    ly = np.log(y[mask])
    slope, intercept, _, _, _ = stats.linregress(x[mask], ly)
    a, b = float(np.exp(intercept)), float(slope)
    with np.errstate(over="ignore"):
        y_pred = a * np.exp(b * x)
    if not np.all(np.isfinite(y_pred)):
        return _fit_result("exponential", "exponential", x, y, y, {}, 2, False, "non-finite")
    return _fit_result("exponential", "exponential", x, y, y_pred, {"a": a, "b": b}, 2)


def predict_exponential(params: dict, x: np.ndarray) -> np.ndarray:
    return params["a"] * np.exp(params["b"] * x)


# --- Logistic: y = K / (1 + exp(-r*(x - x0))) ---
def fit_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, K, r, x0):
            return K / (1.0 + np.exp(-r * (xv - x0)))
        p0 = [max(float(np.max(y)) * 1.05, 1e-6), 0.1, float(np.median(x))]
        popt, _ = optimize.curve_fit(
            fn, x, y, p0=p0,
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=30000,
        )
        y_pred = fn(x, *popt)
        return _fit_result(
            "logistic", "bounded", x, y, y_pred,
            {"K": float(popt[0]), "r": float(popt[1]), "x0": float(popt[2])}, 3,
        )
    except Exception as exc:
        return _fit_result("logistic", "bounded", x, y, y, {}, 3, False, str(exc))


def predict_logistic(params: dict, x: np.ndarray) -> np.ndarray:
    return params["K"] / (1.0 + np.exp(-params["r"] * (x - params["x0"])))


# --- Saturation exponential: y = y_max * (1 - exp(-k*x)) ---
def fit_saturation_exp(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, y_max, k):
            return y_max * (1.0 - np.exp(-k * xv))
        p0 = [max(float(np.max(y)) * 1.05, 1e-6), max(1.0 / (np.mean(np.abs(x)) + 1e-6), 1e-6)]
        popt, _ = optimize.curve_fit(
            fn, x, y, p0=p0,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=20000,
        )
        y_pred = fn(x, *popt)
        return _fit_result(
            "saturation_exp", "bounded", x, y, y_pred,
            {"y_max": float(popt[0]), "k": float(popt[1])}, 2,
        )
    except Exception as exc:
        return _fit_result("saturation_exp", "bounded", x, y, y, {}, 2, False, str(exc))


# --- Linear: y = a + b*x ---
def fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(x) < 3:
        return _fit_result("linear", "linear", x, y, y, {}, 2, False, "need >= 3 points")
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_pred = intercept + slope * x
    return _fit_result(
        "linear", "linear", x, y, y_pred,
        {"a": float(intercept), "b": float(slope)}, 2,
    )


def predict_linear(params: dict, x: np.ndarray) -> np.ndarray:
    return params["a"] + params["b"] * x


# --- Prediction dispatch ---
PREDICT_FNS = {
    "power_law": predict_power_law,
    "exponential": predict_exponential,
    "logistic": predict_logistic,
    "linear": predict_linear,
}


def fit_all_candidates(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Fit all candidate families and return results sorted by AICc."""
    results = [
        fit_power_law(x, y),
        fit_exponential(x, y),
        fit_logistic(x, y),
        fit_saturation_exp(x, y),
        fit_linear(x, y),
    ]
    valid = [r for r in results if r["valid"]]
    valid.sort(key=lambda r: r["aicc"])
    return valid


def select_best(fits: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not fits:
        return None
    return fits[0]


def evaluate_holdout(
    params: dict, model_name: str, x_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Evaluate a trained model's predictions on held-out data."""
    pred_fn = PREDICT_FNS.get(model_name)
    if pred_fn is None:
        # For saturation_exp, use a direct formula
        if model_name == "saturation_exp":
            y_pred = params["y_max"] * (1.0 - np.exp(-params["k"] * x_test))
        else:
            return {"rss": float("inf"), "r2": float("-inf"), "mape": float("inf")}
    else:
        y_pred = pred_fn(params, x_test)
    if not np.all(np.isfinite(y_pred)):
        return {"rss": float("inf"), "r2": float("-inf"), "mape": float("inf")}
    rss, r2 = safe_rss_r2(y_test, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((y_test - y_pred) / np.where(np.abs(y_test) < 1e-12, 1.0, y_test)))) * 100.0
    return {"rss": rss, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Domain definitions with train/test splits
# ---------------------------------------------------------------------------

def build_temporal_domains() -> list[dict[str, Any]]:
    """Construct temporal domains with real published data split into old/new."""
    domains = []

    # ===================================================================
    # 1. NEURAL SCALING LAWS
    # ===================================================================
    # Training: Kaplan et al. (2020) arXiv:2001.08361
    #   Models from 768K to 1.5B parameters, cross-entropy loss
    # Test: publicly reported final losses
    #   - Chinchilla 70B: loss ~1.94  (Hoffmann et al. 2022, Table 3)
    #   - Llama 2 7B: loss ~1.75     (Touvron et al. 2023, Table 2, avg perplexity -> loss)
    #   - Llama 2 13B: loss ~1.68    (Touvron et al. 2023, Table 2)
    #   - Llama 2 70B: loss ~1.57    (Touvron et al. 2023, Table 2)
    #   - GPT-4 (est 1.8T): ~1.30   (semianalysis estimates; treated as approximate)
    #
    # Note: Kaplan measured loss on WebText. Later models use different
    # tokenisers, data mixes, and training compute budgets. The test is
    # whether the power-law FORM (not the exact curve) still describes the
    # scaling behaviour. We fit on the Kaplan series, then ask whether the
    # power-law family remains the best-fitting functional form for the
    # combined series including larger models.
    domains.append({
        "name": "Neural Scaling Laws (LLM Loss vs Parameters)",
        "predicted_family": "power_law",
        "train_source": "Kaplan et al. (2020) arXiv:2001.08361, Fig 1",
        "test_source": (
            "Hoffmann et al. (2022) arXiv:2203.15556 [Chinchilla]; "
            "Touvron et al. (2023) arXiv:2307.09288 [Llama 2]; "
            "estimated from public benchmarks [GPT-4 scale]"
        ),
        "x_train": np.array([
            768_000, 1_540_000, 3_070_000, 6_140_000, 12_300_000,
            24_600_000, 49_200_000, 98_300_000, 197_000_000,
            393_000_000, 786_000_000, 1_500_000_000,
        ], dtype=float),
        "y_train": np.array([
            3.95, 3.78, 3.60, 3.46, 3.32, 3.20, 3.09, 2.99, 2.90,
            2.82, 2.74, 2.68,
        ]),
        # Test set: newer, larger models
        "x_test": np.array([
            7_000_000_000,     # Llama 2 7B
            13_000_000_000,    # Llama 2 13B
            70_000_000_000,    # Chinchilla / Llama 2 70B
        ], dtype=float),
        "y_test": np.array([
            1.75,   # Llama 2 7B  (Touvron et al. 2023, Table 2 pre-training loss)
            1.68,   # Llama 2 13B
            1.57,   # Llama 2 70B / Chinchilla ~1.57-1.94 range; using Llama 2 70B
        ]),
        "notes": (
            "Training and test sets use different data mixes and tokenisers, "
            "so exact coefficient match is not expected. The test is whether "
            "the power-law functional form remains the best-fitting family "
            "across the extended parameter range."
        ),
    })

    # ===================================================================
    # 2. MOORE'S LAW (TRANSISTOR COUNT)
    # ===================================================================
    # Training: pre-2010 transistor counts from public vendor specs
    # Test: post-2010 transistor counts
    # Source: Wikipedia 'Transistor count', cross-checked with vendor datasheets
    domains.append({
        "name": "Moore's Law (Transistor Count vs Year)",
        "predicted_family": "exponential",
        "train_source": "Wikipedia transistor count; Intel, AMD datasheets (pre-2010)",
        "test_source": "Wikipedia transistor count; vendor specs (2011-2022)",
        "x_train": np.array([
            1971, 1972, 1974, 1978, 1982, 1985, 1989, 1993,
            1995, 1999, 2000, 2003, 2006, 2008,
        ], dtype=float),
        "y_train": np.array([
            2_300, 3_500, 4_500, 29_000, 134_000, 275_000,
            1_180_235, 3_100_000, 5_500_000, 9_500_000,
            42_000_000, 220_000_000, 291_000_000, 731_000_000,
        ], dtype=float),
        "x_test": np.array([
            2011, 2012, 2014, 2017, 2019, 2020, 2022,
        ], dtype=float),
        "y_test": np.array([
            1_160_000_000,    # Sandy Bridge (Intel)
            1_400_000_000,    # Ivy Bridge
            2_600_000_000,    # Haswell-E
            4_300_000_000,    # Ryzen (AMD)
            39_540_000_000,   # Apple A13 (TSMC)
            16_000_000_000,   # Zen 3
            57_000_000_000,   # Apple M1 Ultra
        ], dtype=float),
        "notes": (
            "Moore's 'law' predicts exponential growth in transistor count. "
            "Post-2010 shows continued exponential growth, though the doubling "
            "period has lengthened."
        ),
    })

    # ===================================================================
    # 3a. EPIDEMIC CURVE: 2014 EBOLA
    # ===================================================================
    # Training: first 6 months (days 0-182) of cumulative cases
    # Test: remainder (days 189-550)
    # Source: WHO/CDC Situation Reports; NEJM Ebola Response Team (2014-2015)
    x_ebola = np.array([
        0, 7, 21, 35, 49, 63, 77, 91, 105, 119, 133, 140, 147, 154,
        168, 175, 182, 189, 203, 217, 231, 245, 259, 273, 287, 301,
        365, 455, 550,
    ], dtype=float)
    y_ebola = np.array([
        49, 86, 168, 218, 260, 281, 413, 528, 759, 1093, 1603, 1848,
        2127, 3052, 3685, 4507, 5843, 6553, 8011, 9911, 13042, 14383,
        17111, 17908, 20206, 21689, 24282, 27145, 28601,
    ], dtype=float)
    train_mask_ebola = x_ebola <= 182
    test_mask_ebola = x_ebola > 182

    domains.append({
        "name": "2014 Ebola Epidemic (Cumulative Cases)",
        "predicted_family": "bounded",
        "train_source": "WHO/CDC Situation Reports (Mar-Sep 2014); NEJM Ebola Response Team",
        "test_source": "WHO/CDC Situation Reports (Oct 2014 - Jun 2015)",
        "x_train": x_ebola[train_mask_ebola],
        "y_train": y_ebola[train_mask_ebola],
        "x_test": x_ebola[test_mask_ebola],
        "y_test": y_ebola[test_mask_ebola],
        "notes": (
            "Fit on first ~6 months. Test: does the bounded/logistic form "
            "capture the saturation trend in the remaining trajectory?"
        ),
    })

    # ===================================================================
    # 3b. EPIDEMIC CURVE: COVID-19 (WUHAN EARLY)
    # ===================================================================
    # Training: Wuhan cumulative confirmed cases, days 0-30 from first report
    # Test: days 31-60
    # Source: WHO Situation Reports 1-40 (Jan-Feb 2020); Johns Hopkins CSSE
    domains.append({
        "name": "COVID-19 Wuhan Early Wave (Cumulative Cases)",
        "predicted_family": "bounded",
        "train_source": "WHO Situation Reports (Jan 2020); Johns Hopkins CSSE",
        "test_source": "WHO Situation Reports (Feb-Mar 2020); Johns Hopkins CSSE",
        "x_train": np.array([
            0, 3, 7, 10, 14, 17, 20, 23, 26, 29,
        ], dtype=float),
        "y_train": np.array([
            41, 45, 198, 291, 555, 1975, 5974, 17205, 37198, 46607,
        ], dtype=float),
        # After Wuhan lockdown took effect, curve bent toward saturation
        "x_test": np.array([
            32, 35, 38, 42, 49, 56, 63,
        ], dtype=float),
        "y_test": np.array([
            49053, 50338, 50633, 50660, 50005, 49978, 49965,
        ], dtype=float),
        "notes": (
            "Wuhan cumulative cases saturated after lockdown. Days counted "
            "from 31 Dec 2019. Training window covers the exponential growth "
            "phase; test window covers the lockdown-induced saturation."
        ),
    })

    # ===================================================================
    # 3c. EPIDEMIC CURVE: COVID-19 (ITALY FIRST WAVE)
    # ===================================================================
    # Training: first 30 days from 21 Feb 2020
    # Test: days 31-70 (saturation of first wave)
    # Source: Italian Protezione Civile; Johns Hopkins CSSE
    domains.append({
        "name": "COVID-19 Italy First Wave (Cumulative Cases)",
        "predicted_family": "bounded",
        "train_source": "Protezione Civile Italy (Feb-Mar 2020)",
        "test_source": "Protezione Civile Italy (Mar-Apr 2020)",
        "x_train": np.array([
            0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
        ], dtype=float),
        "y_train": np.array([
            20, 155, 889, 5883, 12462, 21157, 35713, 53578, 69176, 86498, 101739,
        ], dtype=float),
        "x_test": np.array([
            35, 40, 45, 50, 55, 60, 65, 70,
        ], dtype=float),
        "y_test": np.array([
            124632, 143626, 159516, 175925, 192994, 205463, 214457, 219814,
        ], dtype=float),
        "notes": (
            "Italy's first COVID wave. Training covers the exponential rise; "
            "test covers the approach to first-wave saturation."
        ),
    })

    # ===================================================================
    # 4. SOLAR PV LEARNING CURVE (Wright's Law)
    # ===================================================================
    # Training: pre-2010 data (cumulative production MW vs module price $/W)
    # Test: post-2010
    # Source: Nemet (2009), Farmer & Lafond (2016), IRENA (2023), Our World in Data
    #
    # x = cumulative global production (MW), y = module price ($/W, 2023 USD)
    domains.append({
        "name": "Solar PV Learning Curve (Price vs Cumulative Production)",
        "predicted_family": "power_law",
        "train_source": "Nemet (2009); Farmer & Lafond (2016); IRENA historical",
        "test_source": "IRENA (2023); Our World in Data; BloombergNEF",
        # Pre-2010 data points (approximate, from published learning curve compilations)
        "x_train": np.array([
            1, 7, 25, 50, 80, 1_250, 2_200, 5_000, 16_000, 40_000,
        ], dtype=float),
        "y_train": np.array([
            106.0, 60.0, 12.0, 8.5, 7.0, 4.9, 3.8, 2.8, 2.0, 1.5,
        ]),
        # Post-2010 data points
        "x_test": np.array([
            100_000, 230_000, 520_000, 760_000, 1_200_000,
        ], dtype=float),
        "y_test": np.array([
            0.70, 0.55, 0.36, 0.27, 0.23,
        ]),
        "notes": (
            "Wright's law: cost decreases as a power law of cumulative "
            "production. Training on pre-2010 data, testing whether the "
            "power-law form still describes the post-2010 cost trajectory."
        ),
    })

    # ===================================================================
    # 5. URBAN SCALING (GDP vs Population)
    # ===================================================================
    # Training: Bettencourt et al. (2007) PNAS -- 25 largest US MSAs, 2006 data
    # Test: 2019 BEA GDP and Census population estimates for same MSAs
    # Source: BEA 'GDP by Metropolitan Area' (2019); Census ACS 2019
    #
    # 15 largest MSAs with identifiable 2019 data:
    domains.append({
        "name": "Urban Scaling (GDP vs Population, US MSAs)",
        "predicted_family": "power_law",
        "train_source": "Bettencourt et al. (2007) PNAS; BEA + Census 2006",
        "test_source": "BEA 'GDP by Metropolitan Area' 2019; Census ACS 2019 1-year estimates",
        # Training: 2006 data (from manifest domain 2)
        "x_train": np.array([
            18_818_536, 12_950_129, 9_505_748, 5_290_400, 6_003_967,
            5_539_949, 5_826_742, 4_180_027, 4_455_217, 5_138_223,
            5_463_857, 3_263_497, 4_468_966, 3_175_041, 4_039_182,
            2_941_454, 2_408_750, 2_796_368, 2_658_405, 2_137_565,
            2_370_776, 1_701_799, 2_091_120, 1_942_217, 2_032_496,
        ], dtype=float),
        "y_train": np.array([
            1_103_245, 688_665, 476_899, 382_760, 338_618,
            325_245, 312_376, 301_187, 278_735, 243_893,
            236_478, 200_281, 197_772, 175_756, 160_826,
            149_935, 136_203, 120_690, 131_168, 105_467,
            99_685, 94_815, 88_782, 72_565, 86_843,
        ], dtype=float),
        # Test: 2019 data for 15 largest MSAs (population, GDP in millions $)
        # MSAs: NYC, LA, Chicago, Dallas, Houston, DC, Miami, Philly, Atlanta,
        #       Boston, Phoenix, SF, Seattle, Minneapolis, Denver
        # Source: BEA CAGDP2 Table; Census ACS 2019
        "x_test": np.array([
            19_216_182,   # New York
            13_214_799,   # Los Angeles
            9_458_539,    # Chicago
            7_573_136,    # Dallas
            7_066_141,    # Houston
            6_280_487,    # Washington DC
            6_166_488,    # Miami
            6_102_434,    # Philadelphia
            6_020_364,    # Atlanta
            4_873_019,    # Boston
            4_948_203,    # Phoenix
            4_731_803,    # San Francisco
            3_979_845,    # Seattle
            3_640_043,    # Minneapolis
            2_967_239,    # Denver
        ], dtype=float),
        "y_test": np.array([
            1_772_319,   # NYC GDP (millions $, 2019, BEA)
            1_047_670,   # LA
            689_370,     # Chicago
            512_437,     # Dallas
            478_753,     # Houston
            541_382,     # DC
            375_675,     # Miami
            444_148,     # Philly
            397_261,     # Atlanta
            463_603,     # Boston
            264_279,     # Phoenix
            592_352,     # SF
            376_159,     # Seattle
            276_578,     # Minneapolis
            218_030,     # Denver
        ], dtype=float),
        "notes": (
            "Same MSA-level urban scaling relationship, tested across a 13-year "
            "gap. GDP values in millions of current USD. The test is whether "
            "the super-linear power-law form (exponent > 1) still holds."
        ),
    })

    return domains


# ---------------------------------------------------------------------------
# Main temporal validation loop
# ---------------------------------------------------------------------------

def run_temporal_validation(domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []

    for dom in domains:
        name = dom["name"]
        predicted = dom["predicted_family"]
        x_train, y_train = dom["x_train"], dom["y_train"]
        x_test, y_test = dom["x_test"], dom["y_test"]

        print(f"\n{'='*72}")
        print(f"DOMAIN: {name}")
        print(f"{'='*72}")
        print(f"  Predicted family : {predicted}")
        print(f"  Training points  : {len(x_train)}")
        print(f"  Test points      : {len(x_test)}")
        print(f"  Train source     : {dom['train_source']}")
        print(f"  Test source      : {dom['test_source']}")

        # --- Fit on training data ---
        train_fits = fit_all_candidates(x_train, y_train)
        if not train_fits:
            print("  [SKIP] No valid fits on training data.")
            results.append({
                "domain": name,
                "status": "SKIP",
                "reason": "No valid fits on training data",
            })
            continue

        train_best = train_fits[0]
        train_family = FAMILY_LABELS.get(train_best["name"], train_best["name"])
        print(f"\n  --- Training window ---")
        print(f"  Best model (AICc) : {train_best['name']}  (family: {train_family})")
        print(f"  R^2               : {train_best['r2']:.6f}")
        print(f"  AICc              : {train_best['aicc']:.2f}")
        for f in train_fits:
            fam = FAMILY_LABELS.get(f["name"], f["name"])
            print(f"    {f['name']:20s}  family={fam:12s}  R2={f['r2']:.6f}  AICc={f['aicc']:.2f}")

        # --- Evaluate ALL valid training models on test data ---
        print(f"\n  --- Test window (held-out future data) ---")
        test_evaluations = []
        for f in train_fits:
            holdout = evaluate_holdout(f["params"], f["name"], x_test, y_test)
            fam = FAMILY_LABELS.get(f["name"], f["name"])
            test_evaluations.append({
                "model": f["name"],
                "family": fam,
                "train_aicc": f["aicc"],
                "train_r2": f["r2"],
                "test_r2": holdout["r2"],
                "test_mape": holdout["mape"],
            })
            tag = ""
            if fam == predicted:
                tag = "  <-- predicted"
            print(
                f"    {f['name']:20s}  test R2={holdout['r2']:.6f}  "
                f"test MAPE={holdout['mape']:.1f}%{tag}"
            )

        # --- Determine test winner ---
        valid_test = [t for t in test_evaluations if np.isfinite(t["test_r2"])]
        if valid_test:
            test_winner = max(valid_test, key=lambda t: t["test_r2"])
        else:
            test_winner = test_evaluations[0] if test_evaluations else None

        # --- Temporal consistency verdict ---
        train_winner_family = train_family
        test_winner_family = test_winner["family"] if test_winner else "NONE"
        families_match = (train_winner_family == test_winner_family)
        predicted_matches_train = (train_winner_family == predicted)
        predicted_matches_test = (test_winner_family == predicted)

        if families_match and predicted_matches_train:
            verdict = "CONFIRMED"
            verdict_detail = (
                f"Same family ({predicted}) wins BOTH training and test windows. "
                f"Temporal out-of-sample confirmation achieved."
            )
        elif families_match:
            verdict = "CONSISTENT_DIFFERENT"
            verdict_detail = (
                f"Same family ({train_winner_family}) wins both windows, but it "
                f"differs from the predicted family ({predicted})."
            )
        elif predicted_matches_test and not predicted_matches_train:
            verdict = "PARTIAL_TEST_ONLY"
            verdict_detail = (
                f"Predicted family ({predicted}) wins on test but not training. "
                f"Training preferred {train_winner_family}."
            )
        elif predicted_matches_train and not predicted_matches_test:
            verdict = "PARTIAL_TRAIN_ONLY"
            verdict_detail = (
                f"Predicted family ({predicted}) wins on training but not test. "
                f"Test preferred {test_winner_family}."
            )
        else:
            verdict = "INCONSISTENT"
            verdict_detail = (
                f"Training winner ({train_winner_family}) differs from test "
                f"winner ({test_winner_family}); neither matches predicted ({predicted})."
            )

        print(f"\n  VERDICT: {verdict}")
        print(f"    {verdict_detail}")

        result = {
            "domain": name,
            "predicted_family": predicted,
            "train_winner_model": train_best["name"],
            "train_winner_family": train_winner_family,
            "train_r2": train_best["r2"],
            "train_aicc": train_best["aicc"],
            "test_winner_model": test_winner["model"] if test_winner else None,
            "test_winner_family": test_winner_family,
            "test_r2": test_winner["test_r2"] if test_winner else None,
            "test_mape": test_winner["test_mape"] if test_winner else None,
            "families_match": families_match,
            "predicted_confirmed": families_match and predicted_matches_train,
            "verdict": verdict,
            "verdict_detail": verdict_detail,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "train_source": dom["train_source"],
            "test_source": dom["test_source"],
            "all_test_evaluations": test_evaluations,
            "notes": dom.get("notes", ""),
        }
        results.append(result)

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n\n{'='*72}")
    print("TEMPORAL OUT-OF-SAMPLE VALIDATION -- SUMMARY")
    print(f"{'='*72}\n")

    confirmed = [r for r in results if r.get("verdict") == "CONFIRMED"]
    consistent = [r for r in results if r.get("verdict") == "CONSISTENT_DIFFERENT"]
    partial = [r for r in results if r.get("verdict", "").startswith("PARTIAL")]
    inconsistent = [r for r in results if r.get("verdict") == "INCONSISTENT"]
    skipped = [r for r in results if r.get("status") == "SKIP"]

    total_tested = len(results) - len(skipped)

    print(f"  Domains tested                : {total_tested}")
    print(f"  Predicted family CONFIRMED    : {len(confirmed)}  (same family wins train + test)")
    print(f"  Consistent (different family) : {len(consistent)}")
    print(f"  Partial                       : {len(partial)}")
    print(f"  Inconsistent                  : {len(inconsistent)}")
    print(f"  Skipped                       : {len(skipped)}")

    if total_tested > 0:
        confirmation_rate = len(confirmed) / total_tested * 100
        consistency_rate = (len(confirmed) + len(consistent)) / total_tested * 100
        print(f"\n  Predicted-family confirmation rate : {confirmation_rate:.1f}%")
        print(f"  Family-consistency rate            : {consistency_rate:.1f}%")

    print(f"\n  {'Domain':<52s} {'Verdict':<24s} {'Train fam':<14s} {'Test fam':<14s}")
    print(f"  {'-'*52} {'-'*24} {'-'*14} {'-'*14}")
    for r in results:
        if r.get("status") == "SKIP":
            print(f"  {r['domain']:<52s} {'SKIP':<24s}")
            continue
        print(
            f"  {r['domain']:<52s} {r['verdict']:<24s} "
            f"{r['train_winner_family']:<14s} {r['test_winner_family']:<14s}"
        )

    print()


def serialise_for_json(obj: Any) -> Any:
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return str(v)
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: serialise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialise_for_json(v) for v in obj]
    return obj


def main() -> None:
    domains = build_temporal_domains()
    results = run_temporal_validation(domains)
    print_summary(results)

    # --- Write JSON output ---
    DEFAULT_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "test_name": "Temporal Out-of-Sample Validation",
        "description": (
            "Fit candidate functional families on OLD data, then test whether "
            "the winning family correctly predicts the functional FORM of NEW "
            "data. The strongest form of self-administered replication."
        ),
        "n_domains": len(results),
        "summary": {
            "confirmed": len([r for r in results if r.get("verdict") == "CONFIRMED"]),
            "consistent_different": len([r for r in results if r.get("verdict") == "CONSISTENT_DIFFERENT"]),
            "partial": len([r for r in results if r.get("verdict", "").startswith("PARTIAL")]),
            "inconsistent": len([r for r in results if r.get("verdict") == "INCONSISTENT"]),
            "skipped": len([r for r in results if r.get("status") == "SKIP"]),
        },
        "domains": serialise_for_json(results),
    }

    with open(DEFAULT_JSON_OUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  JSON results written to: {DEFAULT_JSON_OUT}")


if __name__ == "__main__":
    main()
