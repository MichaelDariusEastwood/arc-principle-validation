# How to read the Paper VI coupling tables

At each location listed in the field key, `null` represents the original file’s `NaN`. The two fields hold the Pearson correlation between mean ethical score and maths accuracy across pressure levels, and the associated p-value. A row with an unavailable coefficient supplies no estimate of the strength of coupling. The archived script assigned those rows the label “Partially coupled” through its default branch. The combined table repeats the affected entries from the two component files. [Generating code, lines 749 to 764](https://github.com/MichaelDariusEastwood/arc-principle-validation/blob/6b6f1c1d04f4caa1561404eb3ee4a3c912b1c403/experiments/honey-architecture__Paper-VI/scripts/eden_honey_tests.py#L749-L764) · [Composition of the combined table](https://github.com/MichaelDariusEastwood/arc-principle-validation/blob/6b6f1c1d04f4caa1561404eb3ee4a3c912b1c403/experiments/honey-architecture__Paper-VI/scripts/merge_honey_test_results.py#L32-L60).

The [field key](field-key.json) lists every changed position and its file fingerprints. To check the three tables, run the following command from this directory. The verifier opens each reader copy and reconstructs the archived bytes in memory.

    python3 verify.py
