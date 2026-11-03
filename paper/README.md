# Manuscript draft

`main.tex` is an IEEE conference-format **draft**, not a submission-ready paper.
It contains an honest offline-fixture result and marks the real-model study as
not executed. Build with:

```bash
pdflatex main.tex && pdflatex main.tex
```

`results_table.tex` is generated from
`baseline_logs/research_smoke_full/analysis.json`. Regenerate it if results
change. Do not submit until: (1) the NQ/BEIR real-model matrix is run, (2)
detection precision/recall are non-zero or the negative result is explicitly
framed, (3) references are checked against final versions, and (4) all authors
review the claims. Adapted TrustRAG/RobustRAG baselines are not the published
algorithms.
