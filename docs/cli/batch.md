# ramappy-batch

`ramappy` installs a console script, `ramappy-batch`, that runs a YAML-defined
[pipeline](../user-guide/building-pipelines.md) over multiple files (or an entire directory) in
parallel using `joblib`, and aggregates all output spectra into a single file.

## Examples

Process every file in a directory, writing a single aggregated CSV:

```bash
ramappy-batch ./raw_data/ ./processed_output/ -c pipeline_config.yaml
```

Process a specific set of files, using all but one CPU core, overwriting any existing output:

```bash
ramappy-batch sample_1.csv sample_2.csv sample_3.csv ./out/ \
    -c pipeline_config.yaml --parallel -2 --overwrite
```

Run sequentially (useful for debugging a pipeline before parallelizing):

```bash
ramappy-batch ./raw_data/ ./out/ -c pipeline_config.yaml --parallel 0
```

### Configuration file

`pipeline_config.yaml` must contain `input`, `pipeline`, and `output` keys. Note that this is a
different (batch-specific) YAML shape than {py:meth}`Pipeline.from_yaml() <ramappy.pipeline.pipeline.Pipeline.from_yaml>`:
the `pipeline` key here holds the same list of `{name, params}` steps described in
[Building Pipelines](../user-guide/building-pipelines.md), just nested inside the larger batch
document alongside the input reader and output writer configuration.

```yaml
input:
  format: csv               # any registered format, see IO Formats
  format_params:
    has_x_axis: true
    csv_format: wide

pipeline:
  - name: crop_spectral
    params: {roi_x: [600, 1800]}
  - name: correct_baseline
    params: {method: arpls, lambda_: 1.0e6}

output:
  format: parquet            # csv, parquet, or feather
  single_spectrum: true
  transpose: false
```

## Command line reference

```{argparse}
:module: ramappy.cli.batch
:func: build_parser
:prog: ramappy-batch
```
