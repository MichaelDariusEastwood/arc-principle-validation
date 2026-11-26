# Paper VII result files

Use the standard JSON tables to open the archived Paper VII results in a browser or a JSON parser. In domain 20, the original program wrote `Infinity` for the AICc and BIC scores of the Hill and logistic fits. The standard tables use `null` for those four unavailable scores in each file. [JSON number specification](https://www.rfc-editor.org/rfc/rfc8259#section-6)

| Table | Standard JSON | Dated original |
| --- | --- | --- |
| Full validation results | [Open the full table](paper-vii-results-standard.json) | [Original results file](results_50_domain_validation.json) |
| Miss analysis | [Open the misses table](paper-vii-miss-analysis-standard.json) | [Original misses file](empirical_miss_analysis.json) |

The [field key](paper-vii-result-format.json) lists each changed position, the file fingerprints, the dated source links and the current Rydberg citation. Each standard table can be converted back to the exact original bytes.

From this directory, the following command writes a separate copy for checking. It uses the [converter](paper_vii_results_view.py) and Python 3’s standard library.

```sh
python3 paper_vii_results_view.py --restore paper-vii-results-standard.json reconstructed-results.json
```
