package com.bdilab.automl.common.utils;

import com.bdilab.automl.common.exception.InternalServerErrorException;
import org.springframework.lang.Nullable;

import java.util.*;

public class Utils {
    public static final String NAMESPACE = "zauto";
    public static final String BASE_IMAGE = "registry.cn-hangzhou.aliyuncs.com/treasures/automl-autokeras-server:latest";
    public static final String BASE_MODEL_SERVER_NAME = "model-server";
    public static final String BEST_MODEL_FOLDER_NAME = "best_model";
    public static final String METADATA_DIR_IN_CONTAINER = "/metadata";
    public static final String PVC_NAME = "automl-metadata-pvc";

    public static String generateHost(String ksvcName) {
        return String.format("%s.%s.example.com", ksvcName, NAMESPACE);
    }

    public static String getModelServerName(Integer experimentId) {
        return String.join("-", BASE_MODEL_SERVER_NAME, String.valueOf(experimentId));
    }

    public static String getBestModelStorageDir(String workspaceDir, String experimentName) {
        return  String.join("/", workspaceDir, experimentName, BEST_MODEL_FOLDER_NAME);
    }
    public static String getBestModelDirInContainer(Integer experimentId, String experimentName) {
        return  String.join("/", METADATA_DIR_IN_CONTAINER, String.valueOf(experimentId), experimentName, BEST_MODEL_FOLDER_NAME);
    }
}
