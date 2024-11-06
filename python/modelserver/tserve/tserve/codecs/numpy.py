import numpy as np


def to_np_dtype(dtype):
    dtype_map = {
        "BOOL": bool,
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
        "FP16": np.float16,
        "FP32": np.float32,
        "FP64": np.float64,
        "BYTES": np.object_
    }
    return dtype_map.get(dtype, None)


def from_np_dtype(np_dtype):
    if np_dtype == bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None