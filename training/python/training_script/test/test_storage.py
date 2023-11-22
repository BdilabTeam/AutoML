from huggingface_training_script.utils.storage import Storage

storage = Storage()
storage.download("http://124.70.188.119:32090/automl/pretrained-models/image_classification.zip", out_dir="/Users/treasures_y/Documents/code/bdilab/AutoML/training/python/training_script/test/output")