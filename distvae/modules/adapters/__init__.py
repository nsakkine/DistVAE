# Export downsampling adapters
from .downsampling_adapters import (
    WanResampleDownAdapter,
    WanResidualDownBlockAdapter,
)

# Export upsampling adapters
from .upsampling_adapters import (
    Upsample2DAdapter,
    WanResampleAdapter,
    WanResidualUpBlockAdapter,
    WanUpBlockAdapter,
)

# Export other adapters
from .midblock_adapters import WanMidBlockAdapter
from .resnet_adapters import WanResidualBlockAdapter

__all__ = [
    # Downsampling
    "WanResampleDownAdapter",
    "WanResidualDownBlockAdapter",
    # Upsampling
    "Upsample2DAdapter",
    "WanResampleAdapter",
    "WanResidualUpBlockAdapter",
    "WanUpBlockAdapter",
    # Other
    "WanMidBlockAdapter",
    "WanResidualBlockAdapter",
]
