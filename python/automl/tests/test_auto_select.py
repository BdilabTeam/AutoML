from automl.autoselect import (
    LLMFactory, 
    ModelSelectionLLMSettings,
    OutputFixingLLMSettings,
    ModelSelection, 
    ModelSelectionSettings,
)
import pytest

ENV_FILE_PATH = "/Users/treasures_y/Documents/code/HG/AutoML/python/autoselect/autoselect/.env"
PROMPT_TEMPLATE_FILE_PATH = "/Users/treasures_y/Documents/code/HG/AutoML/python/autoselect/autoselect/resources/prompt-templates/model-selection-prompt-v1.json"
MODEL_METADATA_FILE_PATH = "/Users/treasures_y/Documents/code/HG/AutoML/python/autoselect/autoselect/resources/huggingface-models-metadata.jsonl"

class TestModelSelection:
    @pytest.fixture(scope="class")
    def model_selection(self):
        model_selection_settings = ModelSelectionSettings(
            prompt_template_file_path=PROMPT_TEMPLATE_FILE_PATH,
            model_metadata_file_path=MODEL_METADATA_FILE_PATH
        )
        model_selection = ModelSelection(settings=model_selection_settings)
        return model_selection
    
    def test_select_model(self, model_selection: ModelSelection):
        model_selection_llm_settings = ModelSelectionLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0.5
        )
        model_selection_llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
        
        output_fixing_llm_settings = OutputFixingLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0
        )
        output_fixing_llm = LLMFactory.get_output_fixing_llm(llm_settings=output_fixing_llm_settings)
        
        models = model_selection.select_model(
            user_input="I want a image classification model",
            task="image-classification",
            model_selection_llm=model_selection_llm,
            output_fixing_llm=output_fixing_llm,
            top_k=5,
            description_length=200
        )
        assert models[0].id is not None
        assert models[0].reason is not None