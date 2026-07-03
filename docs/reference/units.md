# Units API Reference

This section covers the unit representation and physical quantity tracking classes in `ramappy`.

```{note}
Units and quantities in `ramappy` are purely cosmetic metadata labels. They are used to annotate axes and label plots, but the library does not perform automatic physical unit conversions or dimensional analysis on the underlying data arrays.
```

For the full auto-generated API reference see {py:mod}`ramappy.units`.

---

## Supported Physical Units

The {py:class}`ramappy.units.Unit` enum defines the physical units supported by `ramappy`. The parser automatically normalizes common alternate spellings and abbreviations.

| Enum Member | Unit Symbol | Category | Recognized Alternate Spellings / Input |
| :--- | :---: | :--- | :--- |
| `MICROMETER` | `μm` | Spatial / Length | `um`, `micrometer`, `micron` |
| `NANOMETER` | `nm` | Spatial / Length | `nm`, `nanometer` |
| `MILLIMETER` | `mm` | Spatial / Length | `mm` |
| `PIXEL` | `px` | Spatial / Length | `px` |
| `CM_1` | `cm⁻¹` | Spectral / Energy | `1/cm`, `cm-1`, `cm^-1` |
| `ELECTRONVOLT` | `eV` | Spectral / Energy | `ev` |
| `MILLI_ELECTRONVOLT` | `meV` | Spectral / Energy | `mev` |
| `JOULE` | `J` | Spectral / Energy | `j` |
| `ARBITRARY_UNITS` | `a.u.` | Intensity / Data | `au`, `a.u` |
| `COUNTS` | `counts` | Intensity / Data | `counts` |
| `COUNTS_PER_SECOND` | `counts/s` | Intensity / Data | `counts/s` |
| `PERCENT` | `%` | Intensity / Data | `%` |
| `SECOND` | `s` | Time | `s` |
| `DEGREE` | `°` | Angular | `°` |
| `RADIANS` | `rad` | Angular | `rad` |
| `HERTZ` | `Hz` | Frequency | `hz` |
| `KELVIN` | `K` | Temperature | `k` |
| `UNDEFINED` | `""` | Fallback | Any unrecognized string |

---

## Supported Physical Quantities

The {py:class}`ramappy.units.Quantity` enum describes the physical dimension of a data axis or array.

| Enum Member | Quantity Value | Context / Category |
| :--- | :--- | :--- |
| `WAVENUMBER` | `"wavenumber"` | Spectral axis / Calibration |
| `RAMANSHIFT` | `"Raman shift"` | Spectral axis / Calibration |
| `WAVELENGTH` | `"wavelength"` | Spectral axis / Calibration |
| `ENERGY` | `"energy"` | Spectral axis / Calibration |
| `FREQUENCY` | `"frequency"` | Spectral axis / Calibration |
| `UNCALIBRATED` | `"uncalibrated"` | Spectral axis / Calibration |
| `INTENSITY` | `"intensity"` | Data / Intensity axis |
| `RELATIVE_INTENSITY` | `"relative intensity"` | Data / Intensity axis |
| `ABSORBANCE` | `"absorbance"` | Data / Intensity axis |
| `REFLECTANCE` | `"reflectance"` | Data / Intensity axis |
| `TRANSMITTANCE` | `"transmittance"` | Data / Intensity axis |
| `LENGTH` | `"length"` | Spatial axis / Map coordinates |
| `LATITUDE` | `"latitude"` | Spatial axis / Map coordinates |
| `LONGITUDE` | `"longitude"` | Spatial axis / Map coordinates |
| `TEMPERATURE` | `"temperature"` | Temperature tracking |
| `TIME` | `"time"` | Time tracking |
| `UNDEFINED` | `""` | Fallback quantity |
