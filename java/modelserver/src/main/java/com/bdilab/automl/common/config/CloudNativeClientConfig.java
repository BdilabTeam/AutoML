package com.bdilab.automl.common.config;

import com.bdilab.automl.common.exception.HttpServerErrorException;
import io.cloudevents.CloudEvent;
import io.cloudevents.core.provider.EventFormatProvider;
import io.cloudevents.jackson.JsonFormat;
import io.cloudevents.spring.http.CloudEventHttpUtils;
import io.fabric8.knative.client.DefaultKnativeClient;
import io.fabric8.kubernetes.client.Config;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.Nullable;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestTemplate;

import java.io.FileInputStream;
import java.util.HashMap;

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

}

