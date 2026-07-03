# Reference

Curated, task-oriented reference pages, plus the fully auto-generated API reference.

```{toctree}
:maxdepth: 1

core
processing
analysis
utils
project
units
api/ramappy/index
```

## Curated pages

- [Pipeline Steps](steps.md): every registered pipeline step, grouped by category, generated
  from {py:class}`~ramappy.core.pipeline.PipelineStepRegistry`.
- [IO Formats](io-formats.md): every registered file format, generated from
  {py:class}`~ramappy.io.core.InputFormatRegistry`/{py:class}`~ramappy.io.core.OutputFormatRegistry`.
- [Core Data Model](core.md), [Processing Algorithms](processing.md),
  [Analysis Algorithms](analysis.md), [Utilities](utils.md), [Project Model](project.md),
  [Units](units.md): narrative tours of the corresponding `ramappy` subpackages.

## Full API reference

Every public module, class, and function in `ramappy`, generated directly from source and
docstrings by `sphinx-autoapi`: {doc}`ramappy package index <api/ramappy/index>`.
