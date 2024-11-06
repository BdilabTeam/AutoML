import json
import ast
from typing import List, Optional
from collections import defaultdict
from pathlib import Path

from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.llms.base import BaseLLM
from pydantic import BaseModel, Field

from ..utils.logging import get_logger
from .settings import ModelSelectionSettings

logger = get_logger()

class Model(BaseModel):
    id: str = Field(description="ID of the model")
    reason: str = Field(description="Reason for selecting this model")

def _read_models_metadata(model_metadata_file_path: str):
    """Reads the metadata of all models from the local models cache file."""
    if not Path(model_metadata_file_path).exists:
        raise ValueError(f"File path {model_metadata_file_path} does not exist.")
    with open(model_metadata_file_path) as f:
        models = [json.loads(line) for line in f]
    models_map = defaultdict(list)
    for model in models:
        models_map[model["task"]].append(model)
    return models_map

def _get_top_k_models(
    task: str, 
    top_k: int, 
    max_description_length: int,
    model_metadata_file_path: str
):
    """Returns the best k available models for a given task, sorted by number of likes."""
    MODELS_MAP = _read_models_metadata(model_metadata_file_path=model_metadata_file_path)
    candidates = MODELS_MAP[task][: top_k * 2]
    logger.info(f"Task: {task}; All candidate models: {[c['id'] for c in candidates]}")
    available_models = candidates

    top_k_available_models = available_models[:top_k]
    if not top_k_available_models:
        raise Exception(f"No available models for task: {task}")
    logger.info(
        f"Task: {task}; Top {top_k} available models: {[c['id'] for c in top_k_available_models]}"
    )
    
    top_k_models_info = [
        {
            "id": model["id"],
            "likes": model.get("likes"),
            "description": model.get("description", "")[:max_description_length],
            "tags": model.get("meta").get("tags") if model.get("meta") else None,
        }
        for model in top_k_available_models
    ]
    return top_k_models_info

class ModelSelection():
    def __init__(self, settings: ModelSelectionSettings) -> None:
        if not Path(settings.prompt_template_file_path).exists:
            raise ValueError(f"Prompt template file {settings.prompt_template_file_path} does not exist.")
        self._prompt_template_file_path = settings.prompt_template_file_path
        
        if not Path(settings.model_metadata_file_path).exists:
            raise ValueError(f"Model metadata file {settings.model_metadata_file_path} does not exist.")
        self._model_metadata_file_path = settings.model_metadata_file_path
        
    async def aselect_model(
        self,
        user_input: str,
        task: str,
        model_nums: int,
        model_selection_llm: BaseLLM,
        output_fixing_llm: Optional[BaseLLM] = None,
        top_k: int = 10,
        description_length: int = 300
    ) -> List[Model]:
        
        logger.info(f"Starting model selection for task: {task}")
        top_k_models = _get_top_k_models(task, top_k, description_length, model_metadata_file_path=self._model_metadata_file_path)
        
        prompt_template = load_prompt(self._prompt_template_file_path)

        llm_chain = LLMChain(prompt=prompt_template, llm=model_selection_llm, verbose=True)
        
        models_str = json.dumps(top_k_models).replace('"', "'")
        output = await llm_chain.apredict(
            user_input=user_input, task=task, models=models_str, model_nums=model_nums, stop=["<im_end>"]
        )
        logger.info(f"Model selection raw output: {output}")
        
        # Method 1: Using the 'OutputFixingParser' and the 'PydanticOutputParser'
        if output_fixing_llm:
            parser = PydanticOutputParser(pydantic_object=Model)
            fixing_parser = OutputFixingParser.from_llm(
                llm=output_fixing_llm, 
                parser=parser, 
            )
            model = fixing_parser.parse(output)
            models = [model]
        else:
            # Method 2: BaseModel.parse_obj(obj)
            output = ast.literal_eval(output)
            if not isinstance(output, list):
                raise ValueError("Expect the llm output to be list")
            models = []
            for item in output:
                model = Model.parse_obj(item)
                models.append(model)
        
        logger.info(f"Output after parsing llm content: {models}")
        return models


    def select_model(
        self,
        user_input: str,
        task: str,
        model_selection_llm : BaseLLM,
        output_fixing_llm: Optional[BaseLLM] = None,
        model_nums: int = 1,
        top_k: int = 10,
        description_length: int = 300
    ) -> List[Model]:
        
        logger.info(f"Starting model selection for task: {task}")
        top_k_models = _get_top_k_models(task, top_k, description_length, model_metadata_file_path=self._model_metadata_file_path)
        
        prompt_template = load_prompt(self._prompt_template_file_path)

        llm_chain = LLMChain(prompt=prompt_template, llm=model_selection_llm, verbose=True)
        models_str = json.dumps(top_k_models).replace('"', "'")
        output = llm_chain.predict(
            user_input=user_input, models=models_str, model_nums=model_nums, task = task,stop=["<im_end>"]
        )
        logger.info(f"Model selection raw output: {output}")

        # Method 1: Using the 'OutputFixingParser' and the 'PydanticOutputParser'
        if output_fixing_llm:
            parser = PydanticOutputParser(pydantic_object=Model)
            fixing_parser = OutputFixingParser.from_llm(
                llm=output_fixing_llm, 
                parser=parser, 
            )
            model = fixing_parser.parse(output)
            models = [model]
        else:
            # Method 2: BaseModel.parse_obj(obj)
            output = ast.literal_eval(output)
            if not isinstance(output, list):
                raise ValueError("Expect the llm output to be list")
            models = []
            for item in output:
                model = Model.parse_obj(item)
                models.append(model)
            
        logger.info(f"Output after parsing llm content: {models}")
        return models