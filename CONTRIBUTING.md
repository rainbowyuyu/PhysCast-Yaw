# Contributing

Thank you for improving PhysCast-Yaw. Small, reproducible changes are welcome.

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests examples
python -m pytest -q
python -m build
physcast-yaw demo --quick --out outputs/smoke
```

Use a new output directory when changing an experiment. Keep the original result
available for comparison. The quick demo checks the pipeline; it is not the
reference benchmark.

## A useful pull request

Explain the problem, the changed behavior, and the checks you ran. Changes to
coordinates, units, data splits, objectives or uncertainty semantics need a matching
update to the algorithm/data documentation. Add regression tests for numerical or
control defects; examples include reflected layouts, wind-direction wraparound
and a controller falling back outside its training support.

## Data and results

Record the source, permission to redistribute, turbine specification, sensor
meaning and split unit for any new dataset. Keep measurements used for final
evaluation outside model selection. Report energy together with heading travel,
prediction error together with correlation, and interval coverage together with
width. Report all tested conditions, including ones where the baseline wins.

Do not submit credentials, personal records, proprietary turbine curves, or large
raw datasets. This repository contains original source and small synthetic result
tables. External packages and data retain their own licenses.
