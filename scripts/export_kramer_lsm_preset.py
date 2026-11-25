#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.utils.kramer_lsm_spindle import export_kramer_lsm_json_preset


def main():
    parser = argparse.ArgumentParser(description="Convert a Kramer LSM .mat parameter file into a PyHFO JSON preset.")
    parser.add_argument("input", help="Input Kramer .mat parameter file")
    parser.add_argument("output", help="Output PyHFO JSON preset file")
    parser.add_argument("--name", default="", help="Human-readable preset name")
    parser.add_argument("--source", default="", help="Source/citation note")
    args = parser.parse_args()

    output_path = export_kramer_lsm_json_preset(
        args.input,
        args.output,
        name=args.name,
        source=args.source,
    )
    print(output_path)


if __name__ == "__main__":
    main()
