"""Arabic-first Egyptian National ID document understanding pipeline."""

from .config import OCRConfig
from .pipeline import EgyptianIDPipeline, PipelineOutput
from .nid_validator import validate_national_id

__all__ = ["EgyptianIDPipeline", "OCRConfig", "PipelineOutput", "validate_national_id"]
__version__ = "0.1.0"
