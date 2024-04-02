package com.bdilab.automl.common.config;

import io.fabric8.knative.client.DefaultKnativeClient;
import io.fabric8.kubernetes.client.Config;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;

import java.io.FileInputStream;
import java.io.FileReader;

@Configuration
@Slf4j
public class CloudNativeClientConfig {
    @Value("${kubernetes.config.path}")
    private String kubeConfigPath;

    @Bean
    public KubernetesClient kubernetesClient() {
        if (!StringUtils.isEmpty(kubeConfigPath)) {
            try {
                return new KubernetesClientBuilder().withConfig(new FileInputStream(kubeConfigPath)).build();
            } catch (Exception e) {
                log.error(e.getMessage());
            }
        }
        return new KubernetesClientBuilder().withConfig(Config.autoConfigure(null)).build();
    }

    @Bean
    public DefaultKnativeClient defaultKnativeClient(KubernetesClient kubernetesClient) {
        return new DefaultKnativeClient(kubernetesClient);
    }


    @Bean
    public CoreV1Api coreV1Api() throws Exception {
        ApiClient client;
        if (kubeConfigPath.isEmpty()) {
            client = io.kubernetes.client.util.Config.defaultClient();
        } else {
            // loading the out-of-cluster config, a kubeconfig from file-system
            client = ClientBuilder.kubeconfig(KubeConfig.loadKubeConfig(new FileReader(kubeConfigPath))).build();
        }
        io.kubernetes.client.openapi.Configuration.setDefaultApiClient(client);
        return new CoreV1Api(client);
    }

}

