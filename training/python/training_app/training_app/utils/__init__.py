from typing import TYPE_CHECKING
import sys
from .import_utils import(
    _LazyModule
)


_import_structure = {
    # "auto_factory": ["get_values"]
}

if TYPE_CHECKING:
    pass
else:
    sys.modules[__name__] = _LazyModule(
        name=__name__, 
        module_file=globals()["__file__"],
        import_structure=_import_structure,
        module_spec=__spec__
    )