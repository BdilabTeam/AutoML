package com.bdilab.automl.common.utils;

public class HuggingFaceServerUtils {
    public static final String HUGGINGFACE_SERVER_IMAGE = "registry.cn-hangzhou.aliyuncs.com/treasures/huggingface-server:v0.0.1";

    public static final String HUGGINGFACE_SERVER_VOLUME_MOUNT_NAME = "model-dir";

    public static final String HUGGINGFACE_SERVER_VOLUME_MOUNT_PATH = "/treasures/model";

    public static final String HUGGINGFACE_SERVER_PVC_NAME = "automl-pvc";
}
