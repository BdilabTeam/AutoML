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
    Vit = "google/vit-base-patch16-224-in21k"
    Resnet = "microsoft/resnet-50"
    Timm = "timm/mobilenetv3_large_100.ra_in1k"


class TextGenerationModels(Enum):
    """涵盖文本分类任务相关模型"""
    Bert = "Bert"
    Xlnet_base = "xlnet-base-cased"
    Xlnet_large = "xlnet-large-cased"
    Xlnet_chinese = "hfl/chinese-xlnet-base"

class TranslationModels(Enum):
    T5_small = "t5-small",
    T5_base = "t5-base"
    Opus_mt_zh_en = "Helsinki-NLP/opus-mt-zh-en"


class AudiClassification(Enum):
    Wav2vec2_Base = "facebook/wav2vec2-base"
    Hubert_Base = "SZTAKI-HLT/hubert-base-cc"
    Xlsr_Wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

class QuestionAnswering(Enum):
    Bert_base = "bert-base-uncased"
    Distilbert_base_uncased_distilled_squad = "distilbert-base-uncased-distilled-squad"
    Roberta_base_squad2 = "deepset/roberta-base-squad2"

class TextClassification(Enum):
    Bert_base = "bert-base-uncased"
    Distilbert_base_uncased_finetuned_sst_2_english = "distilbert-base-uncased-finetuned-sst-2-english"
    Twitter_roberta_base_sentiment = "cardiffnlp/twitter-roberta-base-sentiment"
    Roberta_base_go_emotions = "SamLowe/roberta-base-go_emotions"

class TaskType(Enum):
    image_classification = ImageClassificationModels
    text_generation = TextGenerationModels






if __name__ == "__main__":
    models = []
    for item in TaskType:
        print(item.value.__members__)