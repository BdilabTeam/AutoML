package com.bdilab.automl.common.config;

import lombok.Data;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
@Data
public class PathConfig {
    @Value("${kubernetes.config.path}")
    private String kubernetesConfigPath;

    @Value("${grafana.url}")
    private String grafanaUrl;
}
