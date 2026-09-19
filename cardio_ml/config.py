"""Experiment constants used in the engineering thesis."""

from __future__ import annotations

AAMI_CLASS_NAMES = ("N", "S", "V")

# Beat symbols grouped into the three AAMI classes used in the project.
AAMI_SYMBOL_TO_CLASS = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
}

# Records used in the intra-patient experiment.
INTRA_RECORDS = (
    "232",
    "209",
    "222",
    "201",
    "207",
    "118",
    "220",
    "223",
    "202",
    "234",
    "124",
    "213",
    "210",
    "208",
    "228",
)

# Standard DS1/DS2 patient-independent split.
DS1_RECORDS = (
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
)

DS2_RECORDS = (
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
)

FEATURE_NAMES = (
    "maximum",
    "minimum",
    "mean",
    "standard_deviation",
    "sum_absolute_differences",
    "spectrum_0_10_hz",
    "spectrum_10_40_hz",
    "spectrum_40_100_hz",
    "wavelet_energy_cA2",
    "wavelet_energy_cD2",
    "wavelet_energy_cD1",
    "previous_rr_seconds",
)
