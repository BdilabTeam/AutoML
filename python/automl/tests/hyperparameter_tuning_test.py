import sys
sys.path.append("/Users/treasures/Desktop/AutoML/python/automl")
from automl import AutoModelWithAK, AutoConfig

if __name__=="__main__":
    densenet_config = AutoConfig.from_model_type("densenet")
    DenseNetForStructredDataClassification = AutoModelWithAK.from_class_name(densenet_config.model_class_name)
    auto_model = DenseNetForStructredDataClassification(densenet_config)
    output = auto_model(
        inputs="/Users/treasures/Desktop/AutoML/python/automl/automl/datasets/train.csv", 
        output_metrics=True, 
        output_best_hyperparameters=True,
        output_search_space_summary=True,
        output_results_summary=True,
        output_model_summary=True
    )
    print(f"{'*'*15}_Metrics:\n{output.metrics}")
    print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
    print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")
    print(f"{'*'*15}_Train Results Summary:\n{output.results_summary}")
    print(f"{'*'*15}_Model Summary:\n{output.model_summary}")