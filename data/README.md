# Data

This project uses the **2024 U.S. National Survey of Children's Health (NSCH) Topical Public Use File**.

The raw `.dta` file is **not included** in this public repository.

## Expected local path

After downloading the public-use file from the official NSCH source, place it here:

```text
data/raw/nsch_2024e_topical.dta
```

The analysis notebook expects that filename by default.

## Why the raw file is not committed

Keeping source data outside the repository:
- avoids duplicating an externally maintained public dataset;
- keeps the repository lightweight;
- preserves a clear separation between source data and project code;
- encourages users to consult the official NSCH documentation and terms.

The original NSCH variable list and frequency documentation should be consulted as the authoritative data dictionary.
