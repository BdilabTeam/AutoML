import os
import inspect
from typing import no_type_check, Optional, Union
from pydantic import Field, BaseSettings as _BaseSettings

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPT_TEMPLATE_FILE_PATH = os.path.join(PARENT_DIR, 'resources', 'prompt-templates', 'model-selection-prompt-v1.json')
MODEL_METADATA_FILE_PATH = os.path.join(PARENT_DIR, 'resources', 'automl-models-metadata.jsonl')
ENV_FILE_PATH = os.path.join(PARENT_DIR, '.env')

class BaseSettings(_BaseSettings):
    @no_type_check
    def __setattr__(self, name, value):
        """
        Patch __setattr__ to be able to use property setters.
        From:
            https://github.com/pydantic/pydantic/issues/1577#issuecomment-790506164
        """
        try:
            super().__setattr__(name, value)
        except ValueError as e:
            setters = inspect.getmembers(
                self.__class__,
                predicate=lambda x: isinstance(x, property) and x.fset is not None,
            )
            for setter_name, func in setters:
                if setter_name == name:
                    object.__setattr__(self, name, value)
                    break
            else:
                raise e

    def dict(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().dict(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )

    def json(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().json(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )


class ModelSelectionSettings(BaseSettings):
    prompt_template_file_path: str = Field(
        default=PROMPT_TEMPLATE_FILE_PATH,
        description="Model selection prompts template file path"
    )
    model_metadata_file_path: str = Field(
        default=MODEL_METADATA_FILE_PATH,
        description="Model metadata file path"
    )