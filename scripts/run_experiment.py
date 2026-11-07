#!/usr/bin/env python
"""Run one experiment config.

    python scripts/run_experiment.py configs/beir_main.yaml
    python scripts/run_experiment.py configs/conteb.yaml --subsets esg
"""

import argparse
import logging

from seismos.experiment import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--subsets", nargs="+", help="override the config's subsets")
    parser.add_argument("--encoders", nargs="+", help="override the config's encoders")
    parser.add_argument("--output-dir", help="override the config's output_dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.subsets:
        cfg["subsets"] = args.subsets
    if args.encoders:
        cfg["encoders"] = args.encoders
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    path = run_experiment(cfg)
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
