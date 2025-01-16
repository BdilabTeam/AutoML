import pytest
import os
from autoselect.selection import LLMFactory, LLMSettings, ModelSelectionLLMSettings, OutputFixingLLMSettings
from autoselect.selection import ModelSelection
from autoselect.selection.settings import ModelSelectionSettings

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_FILE_PATH = os.path.join(PARENT_DIR, 'autoselect', '.env')
PROMPT_TEMPLATE_FILE_PATH = os.path.join(PARENT_DIR, 'autoselect', 'resources', 'prompt-templates', 'model-selection-prompt-v1.json')
MODEL_METADATA_FILE_PATH = os.path.join(PARENT_DIR, 'autoselect', 'resources', 'huggingface-models-metadata.jsonl')

class TestLLMFactory:
    @pytest.fixture(scope="class")
    def llm_settings(self):
        return LLMSettings(
            env_file_path=ENV_FILE_PATH
        )
    
    @pytest.fixture(scope="class")
    def model_selection_llm_settings(self):
        return ModelSelectionLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0.5
        )
    
    @pytest.fixture(scope="class")
    def output_fixing_llm_settings(self):
        return OutputFixingLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0.5
        )
    
    def test_from_openai(self, llm_settings):
        llm = LLMFactory.from_openai(llm_settings=llm_settings)
        res = llm.invoke("what is your name ?")
        assert res is not None
    
    def test_get_model_selection_llm(self, model_selection_llm_settings):
        llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
        res = llm.invoke("how are you ?")
        assert res is not None
        
    def test_get_output_fixing_llm(self, output_fixing_llm_settings):
        llm = LLMFactory.get_output_fixing_llm(llm_settings=output_fixing_llm_settings)
        res = llm.invoke("Can you tell me a joke ?")
        assert res is not None
        
@pytest.mark.asyncio
class TestModelSelection:
    @pytest.fixture(scope="class")
    def model_selection(self):
        model_selection_settings = ModelSelectionSettings(
            prompt_template_file_path=PROMPT_TEMPLATE_FILE_PATH,
            model_metadata_file_path=MODEL_METADATA_FILE_PATH
        )
        model_selection = ModelSelection(settings=model_selection_settings)
        return model_selection
    
    def test_select_model_v1(self, model_selection: ModelSelection):
        """Without output fixing llm"""
        model_selection_llm_settings = ModelSelectionLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0.5
        )
        model_selection_llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
        
        models = model_selection.select_model(
            user_input="yoloyoloyolo",
            task="image-classification",
            model_selection_llm=model_selection_llm,
            top_k=5,
            model_nums=2,
            description_length=100
        )
        assert models[0] is not None
    
    def test_select_model_v2(self, model_selection: ModelSelection):
        """With output fixing llm"""
        model_selection_llm_settings = ModelSelectionLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0.5
        )
        model_selection_llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
        
        output_fixing_llm_settings = OutputFixingLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0
        )
        output_fixing_llm = LLMFactory.get_output_fixing_llm(output_fixing_llm_settings)
        
        models = model_selection.select_model(
            user_input="I want a image classification model",
            task="image-classification",
            model_selection_llm=model_selection_llm,
            output_fixing_llm=output_fixing_llm,
            top_k=5,
            model_nums=2,
            description_length=100
        )
        assert models[0] is not None
    
    async def test_aselect_model_v1(self, model_selection: ModelSelection):
        """Without output fixing llm"""
        model_selection_llm_settings = ModelSelectionLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0.5
        )
        model_selection_llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
        
        models = await model_selection.aselect_model(
            user_input="I want a image classification model",
            task="image-classification",
            model_selection_llm=model_selection_llm,
            top_k=5,
            model_nums=2,
            description_length=100
        )
        assert models[0] is not None
    
    async def test_aselect_model_v2(self, model_selection: ModelSelection):
        """With output fixing llm"""
        model_selection_llm_settings = ModelSelectionLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0.5
        )
        model_selection_llm = LLMFactory.get_model_selection_llm(llm_settings=model_selection_llm_settings)
        
        output_fixing_llm_settings = OutputFixingLLMSettings(
            env_file_path=ENV_FILE_PATH,
            temperature=0
        )
        output_fixing_llm = LLMFactory.get_output_fixing_llm(output_fixing_llm_settings)
        
        models = await model_selection.aselect_model(
            user_input="I want a image classification model",
            task="image-classification",
            model_selection_llm=model_selection_llm,
            output_fixing_llm=output_fixing_llm,
            top_k=5,
            model_nums=2,
            description_length=100
        )
        assert models[0] is not None