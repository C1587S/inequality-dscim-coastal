"""
Configuration for post-CIAM processing.

This module replaces glo_co.settings.Settings with a standalone configuration
that can be used without the full glo-co infrastructure.

Usage:
    from config import Settings
    ps = Settings()
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from cloudpathlib import AnyPath


@dataclass
class Settings:
    """
    Configuration settings for post-CIAM processing.

    This class provides all the paths and parameters needed for the
    post-processing workflow. It can be instantiated with defaults or
    customized for different environments.

    Attributes
    ----------
    CLIP_BATCHES : int
        Number of Monte Carlo batches to use (15 in slr-bin-map).
    PROJ_YEAR_RANGE : tuple
        (start_year, end_year) for projections.
    SSPS : list
        List of SSP scenarios to process.
    """

    # Versions
    SLIIDERS_VERS: str = "v1.2"
    RES_VERS: str = "regional-v0.1"

    # Processing parameters
    # The slr-bin-map-path.zarr has 15 batches and 81 years (2020-2100)
    CLIP_BATCHES: int = 15
    PROJ_YEAR_RANGE: tuple = (2020, 2099)

    # SSPs available in pyCIAM output
    SSPS: tuple = ("SSP2", "SSP3", "SSP4")

    # Dataset metadata
    DESCRIPTION: str = "Probabilistic projected per-capita coastal damages as a function of GMSL, year, SSP, and growth model"
    HISTORY: str = "version 0.23: Updated GMSL projections (AR6). SSPs 2, 3, 4 only. Years 2020-2099."
    AUTHOR: str = "Sebastian Cadavid-Sanchez | Climate Impact Lab (CIL)"
    CONTACT: str = "cadavidsanchez@uchicago.edu"

    # pyCIAM specs
    SEG_CHUNKSIZE: int = 1

    # Filesystem - lazy loaded
    _fs: Optional[object] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize paths after dataclass creation."""
        # ===========================================
        # ROOT DIRECTORIES
        # ===========================================
        self.DIR_SCRATCH = AnyPath("gs://impactlab-data-scratch/dscim-coastal-tmp")
        self.DIR_COASTAL = AnyPath("gs://impactlab-data/coastal")
        self.DIR_PROJ = self.DIR_COASTAL / "local-scc-model"
        self.DIR_DATA_EXT = self.DIR_COASTAL / "data"
        self.DIR_INT_EXT = self.DIR_DATA_EXT / "int"
        self.DIR_RAW_EXT = self.DIR_DATA_EXT / "raw"
        self.DIR_DATA = self.DIR_PROJ / "data"
        self.DIR_RAW = self.DIR_DATA / "raw"
        self.DIR_INT = self.DIR_DATA / "int"
        self.DIR_RES = self.DIR_PROJ / "results"

        # Spatial data - impact regions
        self.DIR_SPATIAL = AnyPath("gs://impactlab-data/spatial/shapefiles/source/impactlab")
        self.DIR_WORLD_COMBO = self.DIR_SPATIAL / "world-combo-new"

        # ===========================================
        # SLIIDERS / CIAM INPUTS
        # ===========================================
        # SLIIDERS data with coastal segment and impact region coordinates
        # Provides: seg_ir, impact_region, iam coordinates
        # Dimensions: seg_ir (19714), ssp (3), iam (2), year (100)
        self.PATH_SLIIDERS = self.DIR_INT / "sliiders-ir.zarr"
        self.PATH_CIAM_INPUTS = self.PATH_SLIIDERS

        # ===========================================
        # pyCIAM OUTPUTS (MODEL RESULTS)
        # ===========================================
        # pyCIAM fixed adaptation output:
        #   - Variables: costs, optimal_case
        #   - Dimensions: impact_region (7430), case (2), year (95), scen_mc (14936), ssp (3), iam (2)
        #   - impact_region format: 'IR_84155'
        #   - case: ['noAdaptation', 'optimalfixed']
        #   - scen_mc: scenario + sample combinations (e.g., 'ncc_ar6_2', 'ssp245_ar6_100')
        self.PATH_CIAM_OUTPUTS = self.DIR_RES / "pyciam-fixedadapt-output.zarr"

        # Temporary processing path
        self.TMPPATH = self.DIR_SCRATCH / "postCIAM-tmp"

        # ===========================================
        # SLR / SEA LEVEL RISE DATA
        # ===========================================
        # SLR bin mapping - maps scenario/sample to GMSL bins
        # Variables: gsl_msl05 (GMSL centroid), sample, scenario
        # Dimensions: year (81), batch (15), slr (10)
        self.PATH_SLR_BIN_MAP = self.DIR_INT / "slr-bin-map-path.zarr"

        # Alias for compatibility
        self.BIN_MAP_PATH = self.PATH_SLR_BIN_MAP
        self.PATH_HAZARD_SLR_ESTIMATES_SYNTH_BINNED = self.PATH_SLR_BIN_MAP

        # GMSL units reference (hardcoded to 'mm' if not available)
        self.GMSL_UNITS = "mm"

        # ===========================================
        # IMPACT REGION MAPPING
        # ===========================================
        # Official parquet file mapping GADMID (integer) to HIERID (string)
        # Index: gadmid (e.g., 5, 21, 24, 28, 36...)
        # Columns: hierid (e.g., 'AFG.7.68'), color, ISO, geometry
        self.PATH_IR_MAPPING = self.DIR_INT_EXT / "exposure/impactregions/impact_regions_v0.1.parquet"

        # Alternative/legacy sources
        self.PATH_WORLD_COMBO_SHAPEFILE = self.DIR_WORLD_COMBO / "agglomerated-world-new.shp"
        self.PATH_WORLD_COMBO_DBF = self.DIR_WORLD_COMBO / "agglomerated-world-new.dbf"

        # ===========================================
        # ECONOMIC VARIABLES (VSL, POPULATION)
        # ===========================================
        # VSL/population data by SSP
        # Files: SSP1.nc4, SSP2.nc4, SSP3.nc4, SSP4.nc4, SSP5.nc4
        # Variables: pop, vsl_epa_scaled, vsl_epa_popavg, gdp, etc.
        # Dimensions: region (hierid format), model, year
        self.DIR_ECONVARS = AnyPath(
            "gs://impactlab-data/gcp/integration_sync/float32/dscim_input_data/econvars/vsl"
        )

        # ===========================================
        # OUTPUT PATHS
        # ===========================================
        self.DIR_OUTPUT = self.DIR_SCRATCH / "postCIAM"
        self.PATH_OUTPUT_DAMAGES = self.DIR_OUTPUT / "coastal-proj.zarr"
        self.PATH_OUTPUT_GMSL = self.DIR_OUTPUT / "coastal-proj-gmsl.zarr"

    @property
    def FS(self):
        """Lazy-loaded GCS filesystem."""
        if self._fs is None:
            import gcsfs
            self._fs = gcsfs.GCSFileSystem()
        return self._fs


def get_settings(**kwargs):
    """Get a Settings instance with optional overrides."""
    return Settings(**kwargs)
