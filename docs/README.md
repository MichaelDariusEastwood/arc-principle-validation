# Documentation

Supporting documents that are not the research itself. The papers are in `../papers/`,
the experiments in `../experiments/`, and the entry point is the repository `README.md`.

| File | What it is |
|------|------------|
| `open-core-reproducibility.md` | what can be reproduced from a clone alone, and what needs credentials and spend |
| `release-notes-v1.0.md` | what the first tagged release contained |
| `paper-pdf-sha256sums.txt` | SHA-256 of every paper PDF, so a downloaded file can be checked against the copy in this repository |

## Checking a PDF you downloaded

```bash
shasum -a 256 -c docs/paper-pdf-sha256sums.txt
```

Run from the repository root. A mismatch means the file you have is not the file this
repository ships, which may simply mean the paper has been revised since that checksum was
recorded.
