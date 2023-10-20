from enum import Enum

from collections import OrderedDict


# MODEL_NAMES_MAPPINGS = OrderedDict(
#     [
#         # Add full (and cased) model names here
#         ("bert", "BERT")
#     ]
# )


class ImageClassificationModels(Enum):
    """涵盖图像分类任务相关模型"""
    Biet = "Biet"


class TextGenerationModels(Enum):
    """涵盖文本分类任务相关模型"""
    Bert = "Bert"
    
    
class TaskType(Enum):
    image_classification = ImageClassificationModels
    text_generation = TextGenerationModels


if __name__ == "__main__":
    models = []
    for item in TaskType:
        print(item.value.__members__)