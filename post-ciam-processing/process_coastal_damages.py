#!/usr/bin/env python3
"""
Post-CIAM Coastal Damages Processing
=====================================

Processes pyCIAM coastal damage model outputs into an integration-ready dataset.

This script converts raw CIAM outputs (total damages per coastal segment) into
per-capita damages binned by GMSL, year, SSP, and growth model. It processes
regions in batches to avoid memory issues (~500MB per batch instead of 40GB+).

This is an adaptation/simplification of the original notebook:
https://gitlab.com/ClimateImpactLab/coastal/projects/glo-co/-/blob/master/code/pipeline/data_processing/pyCIAM/post-CIAM/process-projected-pyCIAM-damages.ipynb

Key differences from the original:
- Does NOT include VSL (Value of Statistical Life) adjustments
- Uses batch processing for memory efficiency
- Simplified for standalone execution

Runtime: approx. 3.5 hours for all 3 scenarios on:
    - Platform: https://notebooks.cilresearch.org/
    - CPU: Intel Xeon @ 2.20GHz (8 cores)
    - Memory: 50GB RAM

Dependencies:
    - numpy
    - pandas
    - xarray
    - dask
    - gcsfs
    - tqdm
    - config.py (local module)

Usage:
    # Test mode (50 regions)
    python process_coastal_damages.py --test

    # Full run (all regions, all scenarios)
    python process_coastal_damages.py

    # Specific scenario only
    python process_coastal_damages.py --scenario optimal
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import gcsfs
from tqdm import tqdm

from config import Settings


# =============================================================================
# CONFIGURATION
# =============================================================================

ADAPTATION_SCENARIOS = {
    'fixedadapt': {
        'input_path': 'gs://impactlab-data/coastal/local-scc-model/results/pyciam-fixedadapt-output.zarr',
        'description': 'Fixed adaptation - adaptation decisions fixed at optimal level',
    },
    'globaladapt': {
        'input_path': 'gs://impactlab-data/coastal/local-scc-model/results/pyciam-globaladapt-output.zarr',
        'description': 'Global adaptation - globally coordinated adaptation',
    },
    'optimal': {
        'input_path': 'gs://impactlab-data/coastal/local-scc-model/results/pyciam-output.zarr',
        'description': 'Optimal adaptation - locally optimal adaptation decisions',
    },
}

OUTPUT_BASE = 'gs://impactlab-data/coastal/local-scc-model/results/formatted_regional_scc'
OUTPUT_BASE_TEST = 'gs://impactlab-data/coastal/local-scc-model/results/formatted_regional_scc_test'

# Batch size for processing
BATCH_SIZE = 100


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Configure logging to both file and console."""
    if log_dir is None:
        log_dir = Path.cwd()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"post_ciam_{timestamp}.log"

    logger = logging.getLogger('post_ciam')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")

    return logger, log_file


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ir_mapping(fs: gcsfs.GCSFileSystem, path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load impact region mapping (GADMID -> HIERID).

    The pyCIAM outputs use GADMID format (e.g., 'IR_84155') while downstream
    integration uses HIERID format (e.g., 'USA.1.2.3'). This mapping converts
    between the two.
    """
    logger.info(f"Loading IR mapping from: {path}")
    with fs.open(str(path), 'rb') as f:
        ir_mapping = pd.read_parquet(f)
    logger.info(f"  Loaded {len(ir_mapping)} regions")
    return ir_mapping


def load_slr_bin_map(path: str, logger: logging.Logger):
    """
    Load SLR bin mapping data.

    This maps (year, batch, slr_bin) -> (scenario, sample) to link GMSL bins
    to the specific climate scenarios in pyCIAM outputs.
    """
    logger.info(f"Loading SLR bin map from: {path}")
    ds_slr = xr.open_zarr(str(path), chunks=None)
    slr_scenario = ds_slr.scenario.load()
    slr_sample = ds_slr.sample.load()
    slr_gmsl = ds_slr.gsl_msl05.load()
    slr_years = ds_slr.year.values
    logger.info(f"  SLR dims: {dict(ds_slr.sizes)}")
    return slr_scenario, slr_sample, slr_gmsl, slr_years, ds_slr


def load_population_data(ssps: tuple, fs: gcsfs.GCSFileSystem,
                         econvars_dir: str, logger: logging.Logger) -> xr.DataArray:
    """
    Load and combine population data for all SSPs.

    Population is used to convert total damages to per-capita damages.
    """
    logger.info("Loading population data...")
    pop_data = {}

    for ssp in ssps:
        gcs_path = f"{str(econvars_dir).replace('gs://', '')}/{ssp}.nc4"
        local_path = f"/tmp/{ssp}.nc4"

        if not os.path.exists(local_path):
            logger.info(f"  Downloading {ssp}.nc4...")
            fs.get(gcs_path, local_path)

        ds = xr.open_dataset(local_path)
        pop_data[ssp] = ds.pop

    pop_combined = xr.concat(
        [pop_data[ssp] for ssp in ssps],
        dim=pd.Index(ssps, name='ssp')
    )
    logger.info(f"  Population dims: {pop_combined.dims}")
    return pop_combined


# =============================================================================
# MAPPING FUNCTIONS
# =============================================================================

def build_slr_to_scen_mc_mapping(slr_scenario: xr.DataArray,
                                 slr_sample: xr.DataArray,
                                 scen_mc_to_idx: dict) -> tuple:
    """
    Build mapping from (year, batch, slr) to scen_mc index.

    The pyCIAM outputs are indexed by 'scen_mc' which combines scenario and
    sample (e.g., 'ssp585_wf_1e_15372'). This function builds lookup arrays
    to map from the (year, batch, slr) dimensions to the
    corresponding scen_mc indices.

    scen_mc formats:
        - Climate: '{scenario}_{sample}' e.g., 'ssp585_wf_1e_15372'
        - Histclim: 'ncc_ar6_{sample}' e.g., 'ncc_ar6_15372'
    """
    years = slr_scenario.year.values
    batches = slr_scenario.batch.values
    slr_bins = slr_scenario.slr.values

    n_years = len(years)
    n_batches = len(batches)
    n_slr = len(slr_bins)

    # Pre-allocate index arrays (-1 indicates no mapping)
    climate_idx = np.full((n_years, n_batches, n_slr), -1, dtype=np.int32)
    histclim_idx = np.full((n_years, n_batches, n_slr), -1, dtype=np.int32)

    for y_i in range(n_years):
        for b_i in range(n_batches):
            for s_i in range(n_slr):
                scenario_val = slr_scenario.values[y_i, b_i, s_i]
                sample_val = slr_sample.values[y_i, b_i, s_i]

                if pd.isna(scenario_val) or pd.isna(sample_val):
                    continue

                scenario = str(scenario_val)
                sample = int(sample_val)

                # Climate scenario: '{scenario}_{sample}'
                climate_scen_mc = f"{scenario}_{sample}"
                if climate_scen_mc in scen_mc_to_idx:
                    climate_idx[y_i, b_i, s_i] = scen_mc_to_idx[climate_scen_mc]

                # Historical climate (counterfactual): 'ncc_ar6_{sample}'
                histclim_scen_mc = f"ncc_ar6_{sample}"
                if histclim_scen_mc in scen_mc_to_idx:
                    histclim_idx[y_i, b_i, s_i] = scen_mc_to_idx[histclim_scen_mc]

    return climate_idx, histclim_idx


def ir_to_hierid(ir_values: np.ndarray, ir_mapping_df: pd.DataFrame) -> np.ndarray:
    """Convert IR_XXXXX format to hierid format."""
    gadmids = [int(ir.split('_')[1]) for ir in ir_values]
    hierids = ir_mapping_df.loc[gadmids, 'hierid'].values
    return hierids


# =============================================================================
# PROCESSING
# =============================================================================

def process_region(region_ir: str, region_hierid: str, ds_ciam: xr.Dataset,
                   pop_all: xr.DataArray, climate_idx: np.ndarray,
                   histclim_idx: np.ndarray, common_years: np.ndarray,
                   slr_years: np.ndarray, case: str = 'optimalfixed') -> tuple:
    """
    Process a single region's costs into target format.

    For each region, this function:
    1. Loads the raw costs from pyCIAM
    2. Maps costs to (year, batch, slr) dimensions using the index arrays
    3. Computes delta damages (climate - histclim)
    4. Converts to per-capita by dividing by population

    Returns:
        (histclim_out, delta_out): Arrays of shape (year, batch, slr, model, ssp)
    """
    # Load costs for this region
    costs = ds_ciam.costs.sel(
        impact_region=region_ir,
        case=case,
        year=common_years
    ).load()

    # Get population for this region
    pop = pop_all.sel(region=region_hierid, year=common_years).load()

    # Get year indices for SLR mapping
    year_idx = np.array([np.where(slr_years == y)[0][0] for y in common_years])

    # Initialize output arrays
    n_years = len(common_years)
    n_batches = climate_idx.shape[1]
    n_slr = climate_idx.shape[2]
    n_models = len(costs.iam)
    n_ssps = len(costs.ssp)

    histclim_out = np.full((n_years, n_batches, n_slr, n_models, n_ssps), np.nan, dtype=np.float32)
    delta_out = np.full((n_years, n_batches, n_slr, n_models, n_ssps), np.nan, dtype=np.float32)

    costs_values = costs.values
    pop_values = pop.values

    # Map costs to output dimensions
    for y_i in range(n_years):
        slr_y_i = year_idx[y_i]
        for b_i in range(n_batches):
            for s_i in range(n_slr):
                clim_idx = climate_idx[slr_y_i, b_i, s_i]
                hist_idx = histclim_idx[slr_y_i, b_i, s_i]

                if clim_idx >= 0 and hist_idx >= 0:
                    for ssp_i in range(n_ssps):
                        for model_i in range(n_models):
                            clim_cost = costs_values[y_i, clim_idx, ssp_i, model_i]
                            hist_cost = costs_values[y_i, hist_idx, ssp_i, model_i]
                            pop_val = pop_values[ssp_i, model_i, y_i]

                            if pop_val > 0:
                                histclim_out[y_i, b_i, s_i, model_i, ssp_i] = hist_cost / pop_val
                                delta_out[y_i, b_i, s_i, model_i, ssp_i] = (clim_cost - hist_cost) / pop_val

    return histclim_out, delta_out


def process_adaptation_scenario(scenario_name: str, input_path: str, output_base: str,
                                ir_mapping: pd.DataFrame, slr_scenario: xr.DataArray,
                                slr_sample: xr.DataArray, slr_gmsl: xr.DataArray,
                                slr_years: np.ndarray, ds_slr: xr.Dataset,
                                pop_all: xr.DataArray, ps: Settings,
                                n_test_regions: int, batch_size: int,
                                logger: logging.Logger) -> xr.Dataset:
    """
    Process a single adaptation scenario.

    This is the main processing function that:
    1. Initializes an empty Zarr store with the correct schema
    2. Processes regions in batches to avoid memory issues
    3. Writes each batch incrementally to Zarr
    4. Saves auxiliary GMSL data
    """

    logger.info("=" * 60)
    logger.info(f"Processing: {scenario_name}")
    logger.info("=" * 60)

    # Load pyCIAM outputs
    logger.info(f"Loading {input_path}...")
    ds_ciam = xr.open_zarr(input_path, chunks={'impact_region': 100})

    # Build scen_mc lookup
    scen_mc_values = ds_ciam.scen_mc.values
    scen_mc_to_idx = {sm: i for i, sm in enumerate(scen_mc_values)}

    # Build mapping from (year, batch, slr) to scen_mc indices
    logger.info("Building SLR to scen_mc mapping...")
    climate_idx, histclim_idx = build_slr_to_scen_mc_mapping(
        slr_scenario, slr_sample, scen_mc_to_idx
    )
    valid_climate = (climate_idx >= 0).sum()
    valid_histclim = (histclim_idx >= 0).sum()
    logger.info(f"  Climate coverage: {100 * valid_climate / climate_idx.size:.1f}%")
    logger.info(f"  Histclim coverage: {100 * valid_histclim / histclim_idx.size:.1f}%")

    # Get regions and convert to HIERID format
    ciam_regions = ds_ciam.impact_region.values
    hierid_regions = ir_to_hierid(ciam_regions, ir_mapping)

    # Filter to regions with population data
    pop_regions = set(pop_all.region.values)
    valid_pop_mask = [h in pop_regions for h in hierid_regions]
    ciam_regions = ciam_regions[valid_pop_mask]
    hierid_regions = hierid_regions[valid_pop_mask]

    # Limit regions for testing
    if n_test_regions:
        ciam_regions = ciam_regions[:n_test_regions]
        hierid_regions = hierid_regions[:n_test_regions]

    n_regions = len(ciam_regions)
    logger.info(f"Processing {n_regions} regions in batches of {batch_size}...")

    # Get common years between SLR data and CIAM outputs
    ciam_years = ds_ciam.year.values
    common_years = np.intersect1d(slr_years, ciam_years)
    common_years = common_years[
        (common_years >= ps.PROJ_YEAR_RANGE[0]) &
        (common_years <= ps.PROJ_YEAR_RANGE[1])
    ]

    # Dimensions
    n_years = len(common_years)
    n_batches = climate_idx.shape[1]
    n_slr = climate_idx.shape[2]
    n_models = len(ds_ciam.iam)
    n_ssps = len(ds_ciam.ssp)

    # -------------------------------------------------------------------------
    # 1. INITIALIZE ZARR STORE (schema only, no data)
    # -------------------------------------------------------------------------
    output_damages_path = f"{output_base}/{scenario_name}/coastal-proj.zarr"
    logger.info(f"Initializing Zarr store at: {output_damages_path}")

    # Define coordinates
    batch_names = [f'batch{i}' for i in range(n_batches)]
    model_names = ['IIASA GDP', 'OECD Env-Growth']
    ssp_names = list(ds_ciam.ssp.values)
    year_coords = common_years.astype(np.int64)
    slr_coords = ds_slr.slr.values

    # Define chunks
    region_chunk = min(batch_size, n_regions)
    year_chunk = min(10, n_years)
    dask_chunks = (region_chunk, year_chunk, n_batches, n_slr, n_models, n_ssps)

    # Create dask template (lazy array) to define schema
    dummy_histclim = da.full(
        (n_regions, n_years, n_batches, n_slr, n_models, n_ssps),
        np.nan, dtype=np.float32, chunks=dask_chunks
    )
    dummy_delta = da.full(
        (n_regions, n_years, n_batches, n_slr, n_models, n_ssps),
        np.nan, dtype=np.float32, chunks=dask_chunks
    )

    ds_template = xr.Dataset(
        {
            'histclim': (['region', 'year', 'batch', 'slr', 'model', 'ssp'], dummy_histclim),
            'delta': (['region', 'year', 'batch', 'slr', 'model', 'ssp'], dummy_delta),
        },
        coords={
            'region': hierid_regions,
            'year': year_coords,
            'batch': batch_names,
            'slr': slr_coords,
            'model': model_names,
            'ssp': ssp_names,
        },
        attrs={
            'description': ps.DESCRIPTION,
            'adaptation_scenario': scenario_name,
            'author': ps.AUTHOR,
            'contact': ps.CONTACT,
            'history': ps.HISTORY,
            'updated': pd.Timestamp.now(tz='US/Pacific').strftime('%c'),
            'units': 'USD per capita',
        }
    )

    # Initialize the store (write metadata only)
    ds_template.to_zarr(output_damages_path, mode='w', compute=False)
    logger.info("  Zarr store initialized (schema only)")

    # -------------------------------------------------------------------------
    # 2. BATCH PROCESSING WITH INCREMENTAL WRITES
    # -------------------------------------------------------------------------
    total_errors = 0
    total_non_nan = 0
    total_values = 0

    n_batches_total = (n_regions + batch_size - 1) // batch_size

    for batch_num, start_idx in enumerate(range(0, n_regions, batch_size)):
        end_idx = min(start_idx + batch_size, n_regions)
        current_slice = slice(start_idx, end_idx)
        current_batch_size = end_idx - start_idx

        # Get batch regions
        batch_ciam = ciam_regions[current_slice]
        batch_hierid = hierid_regions[current_slice]

        # Allocate batch arrays (~500 MB max)
        b_histclim = np.full(
            (current_batch_size, n_years, n_batches, n_slr, n_models, n_ssps),
            np.nan, dtype=np.float32
        )
        b_delta = np.full(
            (current_batch_size, n_years, n_batches, n_slr, n_models, n_ssps),
            np.nan, dtype=np.float32
        )

        # Process regions in this batch
        batch_errors = []
        for i, (region_ir, region_hierid) in enumerate(zip(batch_ciam, batch_hierid)):
            try:
                histclim_out, delta_out = process_region(
                    region_ir, region_hierid, ds_ciam, pop_all,
                    climate_idx, histclim_idx, common_years, slr_years
                )
                b_histclim[i] = histclim_out
                b_delta[i] = delta_out
            except Exception as e:
                batch_errors.append((region_ir, str(e)))

        if batch_errors:
            logger.warning(f"  Batch {batch_num + 1}/{n_batches_total}: {len(batch_errors)} errors")
            total_errors += len(batch_errors)

        # Track data quality
        total_non_nan += (~np.isnan(b_histclim)).sum()
        total_values += b_histclim.size

        # Create batch Dataset for incremental write
        ds_batch = xr.Dataset(
            {
                'histclim': (['region', 'year', 'batch', 'slr', 'model', 'ssp'], b_histclim),
                'delta': (['region', 'year', 'batch', 'slr', 'model', 'ssp'], b_delta),
            },
            coords={'region': batch_hierid}
        )

        # Write batch to Zarr using region parameter
        ds_batch.to_zarr(output_damages_path, region={'region': current_slice})

        # Progress update
        if (batch_num + 1) % 10 == 0 or batch_num == n_batches_total - 1:
            logger.info(f"  Batch {batch_num + 1}/{n_batches_total} complete "
                       f"({end_idx}/{n_regions} regions)")

    # Data quality summary
    logger.info(f"Data quality: {total_non_nan}/{total_values} "
               f"({100 * total_non_nan / total_values:.1f}%) non-NaN values")

    if total_errors > 0:
        logger.warning(f"Total errors processing {scenario_name}: {total_errors}")

    # -------------------------------------------------------------------------
    # 3. SAVE AUXILIARY GMSL DATA
    # -------------------------------------------------------------------------
    gmsl_out = slr_gmsl.sel(year=common_years).rename('gmsl')
    gmsl_out['batch'] = batch_names
    gmsl_ds = gmsl_out.to_dataset()
    gmsl_ds.attrs = {
        'description': f"GMSL associated with {scenario_name} damage projections",
        'units': ps.GMSL_UNITS,
        'reference_period': '1991-2009',
    }

    output_gmsl_path = f"{output_base}/{scenario_name}/coastal-proj-gmsl.zarr"
    logger.info(f"Saving GMSL to: {output_gmsl_path}")
    gmsl_ds.to_zarr(output_gmsl_path, mode='w')

    ds_ciam.close()

    logger.info(f"Completed {scenario_name}")

    return xr.open_zarr(output_damages_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process pyCIAM outputs into integration-ready format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run in test mode (50 regions by default)'
    )
    parser.add_argument(
        '--n-regions', type=int, default=50,
        help='Number of regions for test mode (default: 50)'
    )
    parser.add_argument(
        '--scenario', type=str, choices=['fixedadapt', 'globaladapt', 'optimal', 'all'],
        default='all', help='Which scenario to process (default: all)'
    )
    parser.add_argument(
        '--log-dir', type=str, default=None,
        help='Directory for log files (default: current directory)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='Batch size for processing (default: 100)'
    )

    args = parser.parse_args()

    batch_size = args.batch_size

    # Setup logging
    log_dir = Path(args.log_dir) if args.log_dir else Path.cwd()
    logger, log_file = setup_logging(log_dir)

    # Configuration
    n_test_regions = args.n_regions if args.test else None
    scenarios_to_process = (
        [args.scenario] if args.scenario != 'all'
        else list(ADAPTATION_SCENARIOS.keys())
    )

    # Determine output directory
    if args.output_dir:
        output_base = args.output_dir
    elif args.test:
        output_base = OUTPUT_BASE_TEST
    else:
        output_base = OUTPUT_BASE

    logger.info("=" * 60)
    logger.info("POST-CIAM COASTAL DAMAGES PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Mode: {'TEST' if args.test else 'FULL'}")
    if args.test:
        logger.info(f"Test regions: {n_test_regions}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Scenarios: {scenarios_to_process}")
    logger.info(f"Output base: {output_base}")
    logger.info("")

    # Initialize config and filesystem
    ps = Settings()
    fs = ps.FS

    # Load common data
    logger.info("Loading common reference data...")
    ir_mapping = load_ir_mapping(fs, ps.PATH_IR_MAPPING, logger)
    slr_scenario, slr_sample, slr_gmsl, slr_years, ds_slr = load_slr_bin_map(
        ps.PATH_SLR_BIN_MAP, logger
    )
    pop_all = load_population_data(ps.SSPS, fs, ps.DIR_ECONVARS, logger)

    # Process scenarios
    results = {}
    for scenario_name in scenarios_to_process:
        config = ADAPTATION_SCENARIOS[scenario_name]
        try:
            ds_out = process_adaptation_scenario(
                scenario_name=scenario_name,
                input_path=config['input_path'],
                output_base=output_base,
                ir_mapping=ir_mapping,
                slr_scenario=slr_scenario,
                slr_sample=slr_sample,
                slr_gmsl=slr_gmsl,
                slr_years=slr_years,
                ds_slr=ds_slr,
                pop_all=pop_all,
                ps=ps,
                n_test_regions=n_test_regions,
                batch_size=batch_size,
                logger=logger,
            )
            results[scenario_name] = 'SUCCESS'
        except Exception as e:
            logger.error(f"FAILED processing {scenario_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[scenario_name] = f'FAILED: {e}'

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Results:")
    for scenario, status in results.items():
        logger.info(f"  {scenario}: {status}")
    logger.info("")
    logger.info(f"Output location: {output_base}/")
    logger.info(f"Log file: {log_file}")

    failed = sum(1 for s in results.values() if s != 'SUCCESS')
    return failed


if __name__ == '__main__':
    sys.exit(main())
