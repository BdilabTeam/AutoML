import sys
sys.path.append("/Users/treasures_y/Documents/code/HG/AutoML/python/autotrain")

from autotrain.trainings.utils import AutoArgumentParser
from autotrain.trainings.utils import TrainingArguments

def main():
    args = {
        "task_type": "structured_data_regression",
        "model_type": "densenet"
    }
    parser = AutoArgumentParser(TrainingArguments)
    train_args, = parser.parse_dict(args=args)
    print(train_args)
    print(f"task_type: {train_args.task_type}")

if __name__=="__main__":
    main()