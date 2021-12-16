import re

SERIES_REPR_PATT = re.compile(r"\nSeries: ", flags=re.MULTILINE)
FRAME_REPR_PATT = re.compile(r"\A(shape: \(\d+, \d+\)\n)", flags=re.MULTILINE)
