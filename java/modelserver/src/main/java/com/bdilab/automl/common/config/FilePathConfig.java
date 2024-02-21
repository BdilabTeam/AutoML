package com.bdilab.automl.common.config;

import lombok.Data;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
@Data
public class FilePathConfig {
    @Value("${kubernetes.config.path}")
    private String kubernetesConfigPath;
}
