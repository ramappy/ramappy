import pickle
from io import BytesIO

from ramappy.core import SpectralMap, Spectrum
from ramappy.io.core import IOParams, input_format, output_format


class PickleFormatParams(IOParams):
    """Parameters for reading/writing Pickle files."""

    compressed: bool = False
    """Whether to use Zstandard compression for the Pickle file. If `True`, the file will be compressed using Zstandard. If `False`, the file will be uncompressed."""


@input_format(
    "pickle",
    friendly_name="Pickle",
    extensions={
        "pickle",
    },
    format_params_model=PickleFormatParams,
)
def read_pickle(filepath_or_buffer, compressed=False):
    with open(filepath_or_buffer, "rb") as infile:
        if compressed:
            import zstandard as zstd  # type: ignore

            dctx = zstd.ZstdDecompressor()
            f = dctx.stream_reader(infile)
            return pickle.load(f)
        return pickle.load(infile)


@output_format(
    "pickle",
    friendly_name="Pickle",
    extensions={
        "pickle",
    },
    format_params_model=PickleFormatParams,
)
def write_pickle(spectral_map: SpectralMap | Spectrum, out_file=None, compress=False, compression_level=3):
    if out_file is None:
        return pickle.dumps(spectral_map, protocol=pickle.HIGHEST_PROTOCOL)
    if compress:
        import zstandard as zstd  # type: ignore

        cctx = zstd.ZstdCompressor(level=compression_level)
        with open(f"{out_file}.zst", "wb") as outfile:
            cctx.copy_stream(BytesIO(pickle.dumps(spectral_map, protocol=pickle.HIGHEST_PROTOCOL)), outfile)
    else:
        with open(out_file, "wb") as outfile:
            pickle.dump(spectral_map, outfile, protocol=pickle.HIGHEST_PROTOCOL)
