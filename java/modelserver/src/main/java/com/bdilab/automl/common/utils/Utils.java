package com.bdilab.automl.common.utils;

public class Utils {
    public static final String NAMESPACE = "zauto";
    public static final String BASE_IMAGE = "registry.cn-hangzhou.aliyuncs.com/treasures/automl-autokeras-server:latest";
    public static final String BEST_MODEL_FOLDER_NAME = "best_model";
    public static final String TP_PROJECT_NAME = "output";
    public static final String METADATA_DIR_IN_CONTAINER = "/metadata";
    public static final String PVC_NAME = "automl-metadata-pvc";

    public static String getExperimentWorkspaceDirInContainer(String experimentName) {
        return String.join("/", METADATA_DIR_IN_CONTAINER, experimentName);
    }
    public static String generateHost(String ksvcName) {
        return String.format("%s.%s.example.com", ksvcName, NAMESPACE);
    }
    public static String getBestModelDirInContainer(String experimentName) {
        return  String.join("/", getExperimentWorkspaceDirInContainer(experimentName), TP_PROJECT_NAME, BEST_MODEL_FOLDER_NAME);
    }
}
