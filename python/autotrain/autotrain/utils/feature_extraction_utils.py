from typing import Any, Callable


class BaseFeatureExtractorOutput(object):
    pass

class BaseFeatureExtractor(object):
    def extract(
        self,
        inputs: Any,
        trainer: Callable,
        **kwargs
    ):
        raise NotImplementedError