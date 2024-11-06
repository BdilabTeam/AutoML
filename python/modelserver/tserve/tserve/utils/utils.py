import os
import sys
from typing import Dict, Union
import numpy as np
import pandas as pd
from tserve.protocol.infer_type import InferRequest, InferResponse
from cloudevents.conversion import to_binary, to_structured
from cloudevents.http import CloudEvent


def cpu_count():
    """
    Get the available CPU count for this system.
    Takes the minimum value from the following locations:
        - Total system cpus available on the host.
        - CPU Affinity (if set)
        - Cgroups limit (if set)
    """
    count = os.cpu_count()

    # Check CPU affinity if available
    # try:
    #     affinity_count = len(psutil.Process().cpu_affinity())
    #     if affinity_count > 0:
    #         count = min(count, affinity_count)
    # except Exception:
    #     pass

    # Check cgroups if available
    if sys.platform == "linux":
        try:
            with open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us") as f:
                quota = int(f.read())
            with open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us") as f:
                period = int(f.read())
            cgroups_count = int(quota / period)
            if cgroups_count > 0:
                count = min(count, cgroups_count)
        except Exception:
            pass

    return count


def is_structured_cloudevent(body: Dict) -> bool:
    """Returns True if the JSON request body resembles a structured CloudEvent"""
    return "time" in body \
           and "type" in body \
           and "source" in body \
           and "id" in body \
           and "specversion" in body \
           and "data" in body

def create_response_cloudevent(model_name: str, response: Dict, req_attributes: Dict,
                               binary_event=False) -> tuple:
    ce_attributes = {}

    if os.getenv("CE_MERGE", "false").lower() == "true":
        if binary_event:
            ce_attributes = req_attributes
            if "datacontenttype" in ce_attributes:  # Optional field so must check
                del ce_attributes["datacontenttype"]
        else:
            ce_attributes = req_attributes

        # Remove these fields so we generate new ones
        del ce_attributes["id"]
        del ce_attributes["time"]

    ce_attributes["type"] = os.getenv("CE_TYPE", "io.tserve.inference.response")
    ce_attributes["source"] = os.getenv("CE_SOURCE", f"io.tserve.inference.{model_name}")

    event = CloudEvent(ce_attributes, response)

    if binary_event:
        event_headers, event_body = to_binary(event)
    else:
        event_headers, event_body = to_structured(event)

    return event_headers, event_body

def get_predict_input(payload: Union[Dict, InferRequest]) -> Union[np.ndarray, pd.DataFrame]:
    if isinstance(payload, Dict):
        instances = payload["inputs"] if "inputs" in payload else payload["instances"]
        if len(instances) == 0:
            return np.array(instances)
        if isinstance(instances[0], Dict):
            dfs = []
            for input in instances:
                dfs.append(pd.DataFrame(input))
            inputs = pd.concat(dfs, axis=0)
            return inputs
        else:
            return np.array(instances)


def get_predict_response(payload: Union[Dict, InferRequest], result: Union[np.ndarray, pd.DataFrame],
                         model_name: str) -> Union[Dict, InferResponse]:
    if isinstance(payload, Dict):
        infer_outputs = result
        if isinstance(result, pd.DataFrame):
            infer_outputs = []
            for label, row in result.iterrows():
                infer_outputs.append(row.to_dict())
        elif isinstance(result, np.ndarray):
            infer_outputs = result.tolist()
        return {"predictions": infer_outputs}
    

def strtobool(val: str) -> bool:
    """Convert a string representation of truth to True or False.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Adapted from deprecated `distutils`
    https://github.com/python/cpython/blob/3.11/Lib/distutils/util.py
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))