"""Generate curated Steps / IO Formats reference pages from the live registries.

These pages group entries from :class:`~ramappy.core.pipeline.PipelineStepRegistry`
and :class:`~ramappy.io.core.InputFormatRegistry` / :class:`~ramappy.io.core.OutputFormatRegistry`
into short, hand-organized tables that cross-reference the class/function pages
``autoapi`` already generates -- no duplicate ``autoclass``/``autofunction`` stubs
beyond linking straight into the existing API reference.

Both pages are (re)written on every build (``builder-inited``), so they always
reflect whatever steps/formats are currently registered in the source tree.
"""

from __future__ import annotations

import pathlib

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

STEP_SUBCATEGORIES = [
    "Processing / Baseline & Spectral",
    "Processing / Denoising",
    "Processing / Geometric",
    "Analysis / Decomposition",
    "Analysis / Clustering",
    "Analysis / Fitting",
    "Visualization",
    "Utility / Masks",
    "Utility / Export",
    "Utility / Images & Math",
    "Utility / External Spectra",
]

STEP_SUBCATEGORY_DESCRIPTIONS = {
    "Processing / Baseline & Spectral": "Steps that modify spectral data: baseline correction, cropping, resampling, normalization, and spectral subtraction.",
    "Processing / Denoising": "Steps for noise reduction: cosmic-ray removal, spectral smoothing, spatial smoothing, and SVD-based denoising.",
    "Processing / Geometric": "Steps that modify the spatial layout of the map without altering spectral data.",
    "Analysis / Decomposition": "Steps that decompose the dataset into latent components or endmembers.",
    "Analysis / Clustering": "Steps that group pixels by spectral similarity.",
    "Analysis / Fitting": "Steps that fit reference spectra to pixel data.",
    "Visualization": "Visualization steps generate images or composite visual representations.",
    "Utility / Masks": "Steps for creating, manipulating, and removing spatial masks.",
    "Utility / Export": "Steps for writing spectra, images, and masks to files.",
    "Utility / Images & Math": "Steps for managing images and applying arithmetic operations on spectral data.",
    "Utility / External Spectra": "Steps for importing, configuring, and removing external reference spectra attached to the map.",
}

STEP_FALLBACK_DESCRIPTIONS = {
    "export_spectrum": "Export a single aggregated spectrum (from a mask or image) to a file.",
    "export_spectra": "Export all pixel spectra in the map to a file using a registered output format.",
    "export_grouped_spectra": "Export all spectra in a mask or image group to individual files.",
    "export_spectra_from_image_group": "Export all image-attached spectra in an image group to a CSV file.",
    "export_spectra_from_mask_group": "Export the mean spectrum of each mask in a mask group to a CSV file.",
}


def _walk_schema(schema, found_classes):
    if isinstance(schema, dict):
        if "cls" in schema:
            found_classes.add(schema["cls"])
        for v in schema.values():
            _walk_schema(v, found_classes)
    elif isinstance(schema, list):
        for item in schema:
            _walk_schema(item, found_classes)


def _step_subcategory(step):
    from ramappy.core.pipeline import StepClass

    category = step.step_category
    module = step.func.__module__
    if category == StepClass.PROCESSING:
        if "denoising" in module or "cars" in module:
            return "Processing / Denoising"
        if "geometric" in module:
            return "Processing / Geometric"
        return "Processing / Baseline & Spectral"
    if category == StepClass.ANALYSIS:
        if "cluster" in module or "substrate" in module:
            return "Analysis / Clustering"
        if "spectral_fitting" in module:
            return "Analysis / Fitting"
        return "Analysis / Decomposition"
    if category == StepClass.VISUALIZATION:
        return "Visualization"
    if category == StepClass.UTILITY:
        if "exporting" in module or step.step_name.startswith("export_"):
            return "Utility / Export"
        if (
            "masks" in module
            or step.step_name.endswith(("_mask", "_masks_group"))
            or step.step_name.startswith("import_mask")
        ):
            return "Utility / Masks"
        if "external_spectra" in module or "external_spectrum" in step.step_name:
            return "Utility / External Spectra"
        return "Utility / Images & Math"
    return "Other"


def _step_models(step):
    from ramappy.core.pipeline import SingleModelParamsValidator, UnionParamsValidator

    val = step.params_validator
    classes = set()
    if isinstance(val, SingleModelParamsValidator):
        classes.add(val.model)
    elif isinstance(val, UnionParamsValidator):
        _walk_schema(val.adapter.core_schema, classes)

    models = [c for c in classes if isinstance(c, type) and issubclass(c, BaseModel)]
    return sorted(models, key=lambda m: m.__name__)


def generate_steps_doc(app) -> None:
    """Write the curated, registry-driven ``reference/steps.md`` page."""
    from ramappy.core.pipeline import PipelineStepRegistry
    from ramappy.pipeline.steps import register_all

    register_all()

    steps_by_subcat: dict[str, list] = {cat: [] for cat in STEP_SUBCATEGORIES}
    for step in PipelineStepRegistry.available_steps.values():
        # Skip steps contributed by external plugin packages (e.g., `ramappy-plugin-*`):
        # they live outside `autoapi_dirs`, so autoapi has no page to link to and
        # a generic `{func}`/`{class}` reference here would just be a dead link.
        if not step.func.__module__.startswith("ramappy."):
            continue
        subcat = _step_subcategory(step)
        steps_by_subcat.setdefault(subcat, []).append(step)

    for steps in steps_by_subcat.values():
        steps.sort(key=lambda s: s.step_name)

    content = ["# Pipeline Steps\n"]
    content.append(
        "This page documents all available pipeline steps in `ramappy`. Each step is registered in the global\n"
        "{py:class}`~ramappy.core.pipeline.PipelineStepRegistry` under a unique step name via the\n"
        "{py:func}`~ramappy.core.pipeline.pipeline_step` decorator.\n\n"
        "Steps are configured in YAML/JSON as a list of `{name, params}` objects. The `name` field must match\n"
        "a registered step name exactly. The `params` dict is validated against the step's Pydantic params model.\n\n"
        "For each step you will find:\n\n"
        "- **Parameters model** / the Pydantic model used to validate YAML/JSON configuration. Fields inherited\n"
        "  from mixin base classes ({py:class}`~ramappy.core.pipeline.ParamsAcceptMask`,\n"
        "  {py:class}`~ramappy.core.pipeline.ParamsAcceptRoIX`, {py:class}`~ramappy.core.pipeline.ParamsSetResultId`, etc.)\n"
        "  are shown alongside step-specific fields.\n"
        "- **Step function** / the Python function called by the step registry at runtime.\n\n"
        "---\n\n"
        "## Quick Reference\n"
    )

    for cat in STEP_SUBCATEGORIES:
        steps = steps_by_subcat.get(cat, [])
        if not steps:
            continue
        content.append(f"### {cat}\n")
        content.append("| Friendly Name | Step Name | Step Function | Params Model | Preview |")
        content.append("| :--- | :--- | :--- | :--- | :--- |")
        for step in steps:
            models = _step_models(step)
            model_links = " <br> ".join(f"{{class}}`~{m.__module__}.{m.__name__}`" for m in models)
            func_link = f"{{func}}`~{step.func.__module__}.{step.func.__name__}`"
            preview = "Yes" if step.supports_preview else "No"
            content.append(
                f"| **{step.friendly_name}** | `{step.step_name}` | {func_link} | {model_links} | {preview} |"
            )
        content.append("\n")

    content.append("---\n")
    content.append(
        "## Common Parameter Mixins\n\n"
        "Step parameter models inherit from one or more of the following base classes to gain standard fields:\n\n"
        "```{eval-rst}\n"
        ".. autopydantic_model:: ramappy.core.pipeline.ParamsAcceptMask\n"
        "   :no-index:\n"
        "   :inherited-members: BaseModel\n\n"
        ".. autopydantic_model:: ramappy.core.pipeline.ParamsAcceptRoIX\n"
        "   :no-index:\n"
        "   :inherited-members: BaseModel\n\n"
        ".. autopydantic_model:: ramappy.core.pipeline.ParamsRequireMask\n"
        "   :no-index:\n"
        "   :inherited-members: BaseModel\n\n"
        ".. autopydantic_model:: ramappy.core.pipeline.ParamsRequireRoIX\n"
        "   :no-index:\n"
        "   :inherited-members: BaseModel\n\n"
        ".. autopydantic_model:: ramappy.core.pipeline.ParamsSetResultId\n"
        "   :no-index:\n"
        "   :inherited-members: BaseModel\n\n"
        ".. autopydantic_model:: ramappy.core.pipeline.ParamsSeparateRegions\n"
        "   :no-index:\n"
        "   :inherited-members: BaseModel\n\n"
        ".. autopydantic_model:: ramappy.core.pipeline.ParamsOutputFile\n"
        "   :no-index:\n"
        "   :inherited-members: BaseModel\n"
        "```\n\n"
        "---\n"
    )

    for cat in STEP_SUBCATEGORIES:
        steps = steps_by_subcat.get(cat, [])
        if not steps:
            continue
        content.append(f"## {cat}\n")
        desc = STEP_SUBCATEGORY_DESCRIPTIONS.get(cat, "")
        if desc:
            content.append(f"{desc}\n")

        for step in steps:
            content.append(f"### `{step.step_name}`\n")

            doc = step.func.__doc__
            step_desc = None
            if doc:
                first_paragraph = doc.strip().split("\n\n")[0].strip().replace("\n", " ")
                first_paragraph = " ".join(first_paragraph.split())
                if not first_paragraph.startswith(("Parameters for", "Step parameters for")):
                    step_desc = first_paragraph
            step_desc = step_desc or STEP_FALLBACK_DESCRIPTIONS.get(step.step_name) or f"{step.friendly_name} step."
            content.append(f"{step_desc}\n")

            content.append("```{eval-rst}")
            for m in _step_models(step):
                content.append(f".. autopydantic_model:: {m.__module__}.{m.__name__}")
                content.append("   :no-index:")
                content.append("   :inherited-members: BaseModel\n")
            content.append(f".. autofunction:: {step.func.__module__}.{step.func.__name__}")
            content.append("   :no-index:")
            content.append("```\n")
            content.append("---")

    # A trailing "---" transition with nothing after it is invalid docutils.
    if content and content[-1] == "---":
        content.pop()

    out_path = pathlib.Path(app.srcdir) / "reference" / "steps.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(content) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# IO formats
# ---------------------------------------------------------------------------

# Short curated blurbs for formats that need more than the registry provides
# (features worth calling out, quirks, etc). Formats without an entry here
# still get a page, just with a generic one-liner.
FORMAT_BLURBS = {
    "csv": "Highly customizable parser/writer for CSV/TSV/ASCII files; supports both wide and long layouts.",
    "matlab": "Reads and writes standard MATLAB `.mat` files (supports versions < 7.3 and v7.3+).",
    "hdf5": "Simplified reader/writer for plain HDF5 files, storing the cube under `/data` and the spectral axis under `/x`.",
    "matlab_witec": "Specialized reader that automatically sniffs and parses MATLAB mat files exported from WITec Control software.",
    "witec": "Direct binary parser for WITec's proprietary `.wip`/`.wid` project files. Extracts spatial metadata, laser wavelength, and spectral axis calibration.",
    "renishaw_wdf": "Direct binary parser for Renishaw's `.wdf` format. Automatically aligns and crops the white-light camera image to the hyperspectral map.",
    "horiba5": "Binary parser for Horiba LabSpec 5 `.ngc` (cube) and `.ngs` (spectrum) files.",
    "hdf5_nexus": "Parser for NeXus-compliant HDF5 files. Auto-detects dimension ordering and imports white-light camera images.",
    "pickle": "Full Python serialization of `SpectralMap`/`Spectrum` objects, with optional `zstd` compression.",
    "zarr": "Chunked directory/zip-based store; the primary format used by the RamApp frontend/backend to persist projects.",
    "ownformat": (
        "RamApp's internal DataFrame-based interchange format for spectra/masks/images. "
        "Supports CSV, Apache Arrow Feather/IPC, and Apache Parquet (the latter two with `zstd` compression)."
    ),
}


def _model_link(model) -> str:
    return f"{{py:class}}`~{model.__module__}.{model.__name__}`"


def _format_row(name, in_fmt, out_fmt):
    fmt = in_fmt or out_fmt
    extensions = ", ".join(f"`.{e}`" for e in sorted(fmt.extensions))
    blurb = FORMAT_BLURBS.get(name, f"{fmt.friendly_name} format.")
    read_mark = "✅" if in_fmt is not None else ""
    write_mark = "✅" if out_fmt is not None else ""

    in_model = in_fmt.format_params_model if in_fmt is not None else None
    out_model = out_fmt.format_params_model if out_fmt is not None else None
    if in_model is not None and out_model is not None and in_model is out_model:
        # Same model used for both directions: show it once, spanning both columns.
        read_model_col = write_model_col = _model_link(in_model)
    else:
        read_model_col = _model_link(in_model) if in_model is not None else "—"
        write_model_col = _model_link(out_model) if out_model is not None else "—"

    return (
        f"* - `{name}`\n"
        f"  - {fmt.friendly_name}\n"
        f"  - {extensions}\n"
        f"  - {read_mark}\n"
        f"  - {write_mark}\n"
        f"  - {read_model_col}\n"
        f"  - {write_model_col}\n"
        f"  - {blurb}\n"
    )


def generate_io_doc(app) -> None:
    """Write the curated, registry-driven ``reference/io-formats.md`` page."""
    import ramappy.io  # noqa: F401  (import for side effect: registers built-in formats)
    from ramappy.io.core import InputFormatRegistry, OutputFormatRegistry, SpectrumType

    def _core_only(fmt):
        # Skip formats contributed by external plugin packages (e.g., `ramappy-plugin-*`):
        # they live outside `autoapi_dirs`, so autoapi has no page to link their params model to.
        return fmt if fmt is not None and fmt.format_params_model.__module__.startswith("ramappy.") else None

    # `feather`/`parquet` are pure output-format aliases of `ownformat` (same writer
    # function and params model, just the `format` field pre-locked) registered solely so
    # the CLI batch config's per-extension `output: format: <name>` shortcut (see
    # `ramappy-batch` / README "Config file format") can pick the right file extension.
    ownformat_aliases = {"feather", "parquet"}
    all_names = sorted((set(InputFormatRegistry.formats) | set(OutputFormatRegistry.formats)) - ownformat_aliases)

    content = ["# IO Formats\n"]
    content.append(
        "`ramappy` reads and writes hyperspectral datasets, spectra, masks, and images through a plugin-based\n"
        "IO system. Each format below is registered by name in\n"
        "{py:class}`~ramappy.io.core.InputFormatRegistry` and/or {py:class}`~ramappy.io.core.OutputFormatRegistry`,\n"
        "with a Pydantic model describing its parameters.\n\n"
        "The primary entrypoints for reading and writing files are\n"
        "{py:func}`ramappy.io.common.read_file` and {py:func}`ramappy.io.common.export_spectra`.\n\n"
        "---\n\n"
        "```{list-table}\n"
        ":header-rows: 1\n\n"
        "* - Format\n"
        "  - Name\n"
        "  - Extensions\n"
        "  - Read\n"
        "  - Write\n"
        "  - Read Params Model\n"
        "  - Write Params Model\n"
        "  - Notes\n"
    )
    for name in all_names:
        in_fmt = _core_only(InputFormatRegistry.formats.get(name))
        out_fmt = _core_only(OutputFormatRegistry.get_format(name, supported_type=SpectrumType.SPECTRAL_MAP))
        if in_fmt is None and out_fmt is None:
            continue
        content.append(_format_row(name, in_fmt, out_fmt))
    content.append("```\n")

    out_path = pathlib.Path(app.srcdir) / "reference" / "io-formats.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def generate(app) -> None:
    generate_steps_doc(app)
    generate_io_doc(app)


def setup(app):
    app.connect("builder-inited", generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
