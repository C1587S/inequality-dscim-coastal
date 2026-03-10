#!/usr/bin/env python3
"""
Combine separate adaptation scenario Zarr files into a single file.

This script takes the 3 separate coastal-proj.zarr files (fixedadapt, globaladapt, optimal)
and combines them into a single Zarr file with an 'adapt_type' dimension, matching the
reference format at gs://impactlab-data/gcp/outputs/coastal/coastal_damages_v0.22.zarr

Usage:
    # Combine scenarios (saves to same folder as inputs by default)
    python combine_scenarios.py

    # Custom input/output paths
    python combine_scenarios.py --input-base gs://path/to/inputs --output gs://path/to/output.zarr

    # Force overwrite if output exists
    python combine_scenarios.py --force

    # Delete intermediate files after combining
    python combine_scenarios.py --cleanup
"""

import argparse
import xarray as xr
import logging
import gcsfs
from datetime import datetime

# Default paths - output goes in same folder as inputs
DEFAULT_INPUT_BASE = 'gs://impactlab-data/coastal/local-scc-model/results/formatted_regional_scc'

SCENARIOS = ['fixedadapt', 'globaladapt', 'optimal']


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def check_output_exists(path: str, fs: gcsfs.GCSFileSystem) -> bool:
    """Check if output path already exists."""
    path_no_prefix = path.replace('gs://', '')
    return fs.exists(path_no_prefix)


def main():
    parser = argparse.ArgumentParser(
        description='Combine adaptation scenario Zarr files into a single file'
    )
    parser.add_argument(
        '--input-base', type=str, default=DEFAULT_INPUT_BASE,
        help=f'Base path containing scenario folders (default: {DEFAULT_INPUT_BASE})'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for combined Zarr (default: {input-base}/coastal-proj-combined.zarr)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite output if it already exists'
    )
    parser.add_argument(
        '--cleanup', action='store_true',
        help='Delete intermediate scenario folders after combining'
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Default output is in same folder as inputs
    if args.output is None:
        args.output = f"{args.input_base}/coastal-proj-combined.zarr"

    logger.info("=" * 60)
    logger.info("COMBINING ADAPTATION SCENARIOS")
    logger.info("=" * 60)
    logger.info(f"Input base: {args.input_base}")
    logger.info(f"Output: {args.output}")
    logger.info("")

    # Initialize filesystem
    fs = gcsfs.GCSFileSystem()

    # Check if output already exists
    if check_output_exists(args.output, fs):
        if args.force:
            logger.warning(f"Output exists and --force specified. Will overwrite: {args.output}")
        else:
            logger.error(f"Output already exists: {args.output}")
            logger.error("Use --force to overwrite, or specify a different --output path")
            return 1

    # Check all input scenarios exist
    logger.info("Checking input scenarios exist...")
    for scenario in SCENARIOS:
        path = f"{args.input_base}/{scenario}/coastal-proj.zarr"
        path_no_prefix = path.replace('gs://', '')
        if not fs.exists(path_no_prefix):
            logger.error(f"Input not found: {path}")
            return 1
        logger.info(f"  Found: {scenario}/coastal-proj.zarr")
    logger.info("")

    # Load each scenario
    datasets = []
    for scenario in SCENARIOS:
        path = f"{args.input_base}/{scenario}/coastal-proj.zarr"
        logger.info(f"Loading {scenario} from {path}...")
        ds = xr.open_zarr(path)
        ds = ds.expand_dims(adapt_type=[scenario])
        datasets.append(ds)
        logger.info(f"  Shape: histclim {ds.histclim.shape}")

    # Combine along adapt_type dimension
    logger.info("")
    logger.info("Combining datasets along 'adapt_type' dimension...")
    combined = xr.concat(datasets, dim='adapt_type')

    # Reorder dimensions to match reference: (adapt_type, region, year, batch, slr, model, ssp)
    logger.info("Reordering dimensions...")
    combined = combined.transpose('adapt_type', 'region', 'year', 'batch', 'slr', 'model', 'ssp')

    # Add metadata
    combined.attrs = {
        'description': 'Probabilistic projected per-capita coastal damages as a function of GMSL, year, SSP, and growth model',
        'author': 'Climate Impact Lab',
        'created': datetime.now().isoformat(),
        'adapt_types': 'fixedadapt=fixed optimal adaptation, globaladapt=global optimal adaptation, optimal=locally optimal adaptation',
    }

    logger.info("")
    logger.info("Combined dataset summary:")
    logger.info(f"  Dimensions: {dict(combined.dims)}")
    logger.info(f"  Variables: {list(combined.data_vars)}")
    logger.info(f"  Total size: histclim {combined.histclim.shape}, delta {combined.delta.shape}")
    logger.info("")

    # Save
    logger.info(f"Saving to {args.output}...")
    combined.to_zarr(args.output, mode='w')

    logger.info("Done!")
    logger.info("")

    # Close datasets
    for ds in datasets:
        ds.close()

    # Cleanup intermediate files if requested
    if args.cleanup:
        logger.info("Cleaning up intermediate files...")
        for scenario in SCENARIOS:
            folder = f"{args.input_base}/{scenario}"
            folder_path = folder.replace('gs://', '')
            logger.info(f"  Deleting {folder}/...")
            try:
                fs.rm(folder_path, recursive=True)
                logger.info(f"    Deleted {scenario}/")
            except Exception as e:
                logger.warning(f"    Failed to delete {scenario}/: {e}")
        logger.info("Cleanup complete!")

    return 0


if __name__ == '__main__':
    exit(main())
