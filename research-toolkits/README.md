# ARC Principle Research Toolkits

This folder contains experimental validation scripts for the ARC Principle papers.

## Paper I: Preliminary Evidence

**Published:** 17 January 2026

The original validation framework using publicly available data from OpenAI o1 and DeepSeek R1 technical reports.

## Paper II: Experimental Validation

**Published:** 22 January 2026

Direct experimental validation using DeepSeek R1 with visible reasoning tokens.

### Running the Experiment

```bash
# Clone the repository
git clone https://github.com/MichaelDariusEastwood/arc-principle-validation.git
cd arc-principle-validation

# Install dependencies
pip install -r requirements.txt

# Set your DeepSeek API key
export DEEPSEEK_API_KEY="your-api-key"

# Run the experiment
cd code
python arc_validation_deepseek.py
```

### Experiment Script

- `../code/arc_validation_deepseek.py` - Complete experiment script

### Raw Data

- `../data/arc_deepseek_results_20260121_175028.json` - Experimental results

### Figures

All 15 visualisations are in `../figures/`

## Citation

```bibtex
@article{eastwood2026arc,
  title={Eastwood's ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing},
  author={Eastwood, Michael Darius},
  year={2026},
  month={January},
  note={Paper II in the ARC Principle series}
}
```

## Replication Welcome

All contributions welcome, **including falsifications**.

---

**Copyright 2026 Michael Darius Eastwood**
