import sys
sys.path.append("/Users/treasures/Desktop/AutoML/python/automl")

from automl.operators import KubeConfigSettings, TFJobSettings, TFJobOrchestrator

def main():
    kube_config_file="/Users/treasures_y/Documents/code/HG/AutoML/python/automl/automl/utils/config"
    kube_settings = KubeConfigSettings(kube_config_file=kube_config_file)
    tfjob_orchestrator = TFJobOrchestrator(settings=kube_settings)
    
    tfjob_settings = TFJobSettings()
    tfjob = tfjob_orchestrator.construct_kubeflow_v1_tfjob(settings=tfjob_settings)
    
    print(f"tfjob: {tfjob}")

if __name__=="__main__":
    main()