package com.bdilab.automl.common.utils;

import com.bdilab.automl.common.config.FilePathConfig;
import io.fabric8.knative.client.DefaultKnativeClient;
import io.fabric8.kubernetes.client.Config;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.Resource;
import java.io.FileInputStream;

@Component
@Slf4j
public class CloudNativeClientHelper {

    public static KubernetesClient kubernetesClient = null;
    public static DefaultKnativeClient knativeClient = null;

    @Resource
    private FilePathConfig filePathConfig;

    @PostConstruct
    public void init() {
        String kubeConfigPath = filePathConfig.getKubernetesConfigPath();
        if (kubeConfigPath.isEmpty()) {
            kubernetesClient = new KubernetesClientBuilder().withConfig(Config.autoConfigure(null)).build();
        } else {
            try {
                kubernetesClient = new KubernetesClientBuilder().withConfig(new FileInputStream(kubeConfigPath)).build();
            } catch (Exception e) {
                log.error(e.getMessage());
            }
        }
        knativeClient = new DefaultKnativeClient(kubernetesClient);
    }
}
