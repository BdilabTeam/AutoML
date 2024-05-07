from typing import Optional
from dotenv import load_dotenv

from langchain_openai import OpenAI, ChatOpenAI
from .settings import LLMSettings, ModelSelectionLLMSettings, OutputFixingLLMSettings

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMFactory():
    def __init__(self):
        raise EnvironmentError(
            "LLMFactory is designed not to be instantiated"
            "You can use the `LLMFactory.from_openai(LLMSettings)` method or others."
        )
    
    @classmethod
    def from_openai(cls, llm_settings: LLMSettings = LLMSettings()):
        """Get the model selection llm"""
        values = dict()
        openai_params = dict()
        if llm_settings.env_file_path and load_dotenv(llm_settings.env_file_path, verbose=True):
            return ChatOpenAI(**openai_params)
        else:
            if not llm_settings.openai_api_key:
                raise ValueError(f"Did not find the 'api_key', you must set one.")
            else:
                values["openai_api_key"] = llm_settings.openai_api_key
            if llm_settings.base_url:
                values["openai_base_url"] = llm_settings.openai_base_url
            openai_params = {
                "api_key": values.get("openai_api_key", None),
                "base_url": values.get("openai_base_url", None),
            }
        return ChatOpenAI(**openai_params)
    
    @classmethod
    def get_model_selection_llm(cls, llm_settings: ModelSelectionLLMSettings = ModelSelectionLLMSettings()):
        return cls.from_openai(llm_settings=llm_settings)
    
    @classmethod
    def get_output_fixing_llm(cls, llm_settings: OutputFixingLLMSettings = OutputFixingLLMSettings()):
        return cls.from_openai(llm_settings=llm_settings)