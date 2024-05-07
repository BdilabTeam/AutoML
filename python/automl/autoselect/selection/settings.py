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

class LLMSettings(BaseSettings):
    """LLM config"""
    llm_name_or_path: Optional[str] = Field(
        default="gpt-3.5-turbo-instruct",
        description="The name or path of the llm"
    )
    openai_api_key: str = Field(
        default=None,
        description="Automatically inferred from env var `OPENAI_API_KEY` if not provided."
    )
    openai_base_url: str = Field(
        default=None,
        description="Base URL path for API requests, leave blank if not using a proxy or service emulator."
    )
    env_file_path: str = Field(
        default=ENV_FILE_PATH,
        description="Path to 'environment variables' file"
    )

class ModelSelectionLLMSettings(LLMSettings):
    llm_max_tokens: Optional[int] = Field(
        default=4097,
        description="Maximum number of tokens that llm can handle"
    )
    logit_bias: Union[int, float] = Field(
        default=None,
        description="Logit bias is used to bias the output in machine learning and deep learning models to improve the performance of the model."
    )
    temperature: Optional[float] = Field(
        default=0,
        description="Used to adjust the degree of randomness of generated text."
    )

class OutputFixingLLMSettings(LLMSettings):
    llm_max_tokens: Optional[int] = Field(
        default=4097,
        description="Maximum number of tokens that llm can handle"
    )
    temperature: Optional[float] = Field(
        default=0,
        description="Used to adjust the degree of randomness of generated text."
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