import argparse
import copy
import re
import sys
from pathlib import Path

import polars as pl
import yaml
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ramappy.core.pipeline import ParamsOutputFile, ParamsRequireMask, PipelineStepRegistry
from ramappy.pipeline import process_pipeline
from ramappy.pipeline.configuration import BatchConfig, RuntimePipelineConfig


def inject_outpath(runtime_config: RuntimePipelineConfig, basepath) -> None:
    for i, (step, step_params) in enumerate(runtime_config.steps):
        if isinstance(step_params, ParamsOutputFile):
            if step_params.out_file is None:
                out_filename = f"{i}{step.step_name}"
                sep = ""
                if isinstance(step_params, ParamsRequireMask):
                    out_filename += f"_{step_params.mask}"
                if hasattr(step_params, "image_id"):
                    out_filename += f"_{step_params.image_id}"
                out_filename += ".png"
            else:
                out_filename = step_params.out_file.name
                sep = "_" if out_filename[0] != "." else ""
            runtime_config.steps[i] = (
                step,
                step_params.model_copy(update={"out_file": Path(f"{basepath}{sep}{out_filename}")}),
            )


def inject_masks(runtime_config: RuntimePipelineConfig, mask_dir, basename) -> None:
    mask_files = list(Path(mask_dir).glob(f"{basename}_mask_*.png"))
    pattern = re.compile(rf"{re.escape(basename)}_mask_(\w+)\.png$")
    mask_entries: list[tuple[Path, str]] = []
    for mf in mask_files:
        match = pattern.search(mf.name)
        if match is None:
            raise ValueError(
                f"Mask file '{mf}' does not match the expected '{basename}_mask_<name>.png' naming pattern"
            )
        mask_entries.append((mf, match.group(1)))

    mask_names = [name for _, name in mask_entries]
    if "substrate" in mask_names and "foreground" in mask_names:
        seg_steps = [i for i, (step, _) in enumerate(runtime_config.steps) if step.step_name == "cell_segmentation"]
        for i in reversed(seg_steps):
            del runtime_config.steps[i]

    import_mask_step = PipelineStepRegistry.get_step("import_mask")
    for mf, mask_id in mask_entries:
        params = import_mask_step.validate_params({"id": mask_id, "mask_path": str(mf)})
        runtime_config.steps.insert(0, (import_mask_step, params))


def process_single_file(filename, runtime_config: RuntimePipelineConfig, output, mask_dir=None):
    config_cur = copy.deepcopy(runtime_config)
    basename = Path(filename).stem
    if mask_dir is not None:
        inject_masks(config_cur, mask_dir, basename)
    inject_outpath(config_cur, output / basename)
    return process_pipeline(filename, config=config_cur)


def build_parser() -> argparse.ArgumentParser:
    """Build the `ramappy-batch` argument parser.

    Kept separate from :func:`main` so it can be introspected without side
    effects, both by tests and by ``sphinx-argparse`` when generating the CLI
    reference documentation.
    """
    parser = argparse.ArgumentParser(
        prog="ramappy-batch",
        description="Batch process files using a pipeline config (YAML) and aggregate output into a single file.",
    )
    parser.add_argument("src_files", type=str, nargs="+", help="Input files or a single directory")
    parser.add_argument("dest_filedir", type=str, help="Output directory")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="YAML pipeline config file")
    parser.add_argument(
        "-p",
        "--parallel",
        metavar="N",
        help="Number of parallel workers (-1 = all CPUs, 0 = serial)",
        type=int,
        default=-1,
    )
    parser.add_argument("-w", "--overwrite", help="Overwrite existing output file", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    with Path(args.config_file).open(encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not raw_config.get("pipeline"):
        raise ValueError("No steps defined in the pipeline")

    batch_config = BatchConfig.model_validate(raw_config)
    reader = batch_config.input.resolve_reader()
    # extracted before build_runtime_config(): RuntimeGeneralConfig only carries metadata_reader/inject, not this
    mask_dir = getattr(batch_config.general, "mask_dir", None)
    runtime_config = batch_config.build_runtime_config(reader=reader)

    if len(args.src_files) == 1 and Path(args.src_files[0]).is_dir():
        src_files = []
        for fe in reader.extensions:
            src_files.extend(Path(args.src_files[0]).glob(f"*.{fe}"))
    else:
        src_files = [Path(f) for f in args.src_files]

    output = Path(args.dest_filedir)
    output.mkdir(parents=True, exist_ok=True)
    output_cfg = runtime_config.output or {}
    out_file_format = output_cfg.pop("format", "csv")
    out_file_ext = {"feather": ".feather", "parquet": ".parquet"}.get(out_file_format, ".csv")
    out_filename = (
        output / f"{Path(args.config_file).stem}{out_file_ext}" if not output.suffix or output.is_dir() else output
    )
    output = out_filename.parent if output.suffix else output

    if out_filename.exists() and not args.overwrite:
        print("Output file already exists. Use -w to overwrite.")
        sys.exit(1)
    out_filename.unlink(missing_ok=True)

    n_jobs = args.parallel if args.parallel != 0 else None

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(f, runtime_config, output, mask_dir) for f in tqdm(src_files)
    )

    transpose = output_cfg.get("transpose", False)
    df = pl.concat(results, how="vertical" if transpose else "align")

    if out_file_format == "feather":
        df.write_ipc(out_filename, compression="zstd", compression_level=15)
    elif out_file_format == "parquet":
        df.write_parquet(out_filename, compression="zstd", compression_level=15)
    else:
        df.write_csv(out_filename, include_header=True)


if __name__ == "__main__":
    main()
