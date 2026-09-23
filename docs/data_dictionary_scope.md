# Data Dictionary Scope

The NSCH already provides an official variable list containing:
- variable names;
- question wording;
- allowed response values;
- skip logic; and
- survey coding information.

That official documentation is the authoritative source data dictionary.

For the capstone report, reproducing all 457 NSCH variables would be unnecessarily unwieldy. The project therefore documents:
1. the variables used to construct the outcome;
2. the 102 candidate predictors in Model A;
3. the domain-informed 38-feature set in Model B; and
4. the data-driven 44-feature set in Model C.

Exact feature lists are stored in `src/config.py`.

This approach keeps the project-specific documentation concise while preserving traceability to the official NSCH documentation.
