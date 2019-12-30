"""
This script parses benchmark data from numpy files and makes summary tables.
"""

# System
import os
import re
import argparse

# Externals
import numpy as np
import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('results_dirs', nargs='*', help='Benchmark output directories')
    add_arg('-o', '--output-file', help='Text file to dump results to')
    add_arg('-v', '--verbose', action='store_true')
    return parser.parse_args()

def load_result(path, ranks=1, **kwargs):
    print(path)
    summary_files = [f for f in os.listdir(path) if f.startswith('summaries_')]
    assert (ranks == len(summary_files))
    train_rate, inference_rate = 0, 0
    for summary_file in summary_files:
        with np.load(os.path.join(path, summary_file)) as f:
            train_rate += f['train_rate'].mean()
            inference_rate += f['valid_rate'].mean()
    return dict(train_rate=train_rate, inference_rate=inference_rate,
                ranks=ranks, **kwargs)

def load_results(results_dirs):
    results = []

    for results_dir in results_dirs:
        print(results_dir)

        # Extract hardware, software, from path
        m = re.match('(.*)-(.*)-n(\d+)', os.path.basename(results_dir))
        hw, sw, ranks = m.group(1), m.group(2), int(m.group(3))

        # Use all subdirectories as models
        models = [m for m in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, m))]
        for model in models:
            model_subdir = os.path.join(results_dir, model)
            results.append(load_result(model_subdir, hardware=hw, version=sw,
                                       model=model, ranks=ranks))

    return pd.DataFrame(results)

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    results = load_results(args.results_dirs)
    print(results)

    if args.output_file is not None:
        print('Writing data to', args.output_file)
        results.to_csv(args.output_file, index=False, sep='\t')

if __name__ == '__main__':
    main()
