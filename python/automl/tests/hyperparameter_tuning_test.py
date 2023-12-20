import sys
sys.path.append("/Users/treasures/AllProjects/Projects/Git/Bdilab/AutoML/python/automl")
from automl import AutoModelWithAK, AutoConfig

if __name__=="__main__":
    densenet_config = AutoConfig.from_model_type("densenet")

    Trainer = AutoModelWithAK.from_class_name(densenet_config.model_class_name)
    trainer = Trainer(densenet_config)
    
    output = trainer(
        inputs="/Users/treasures/AllProjects/Projects/Git/Bdilab/AutoML/python/automl/automl/datasets/train.csv", 
        return_summary_dict=True
    )
    
    print(f"{'*'*15}_Metrics:\n{output.metrics}")
    print(f"{'*'*15}_Best Hyperparameters:\n{output.best_hyperparameters}")
    print(f"{'*'*15}_Search Space Summary:\n{output.search_space_summary}")
    print(f"{'*'*15}_Train Results Summary:\n{output.results_summary}")
    print(f"{'*'*15}_Model Summary:\n{output.model_summary}")
    