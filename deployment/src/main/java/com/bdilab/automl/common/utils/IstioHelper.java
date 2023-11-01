package com.bdilab.automl.common.utils;

import io.fabric8.kubernetes.api.model.PodList;
import io.fabric8.kubernetes.api.model.Service;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;
import javax.annotation.PostConstruct;

@Component
@DependsOn({"cloudNativeClientHelper"})
@Slf4j
public class IstioHelper {

    @Value("${istio.ingressgateway.port}")
    private String STATIC_INGRESS_GATEWAY_PORT;

    public static String INGRESS_GATEWAY_HOST;

    public static String INGRESS_GATEWAY_PORT;

    @PostConstruct
    public void init () {
        getIstioConfig();
    }

    public void getIstioConfig() {
        log.info("获取Istio配置");
        try {
            // 获取 INGRESS_HOST
            PodList podList = CloudNativeClientHelper.kubernetesClient.pods().inNamespace("istio-system").withLabelSelector("istio=ingressgateway").list();
            INGRESS_GATEWAY_HOST = podList.getItems().get(0).getStatus().getHostIP();
            log.info("istio-ingress-host:" + INGRESS_GATEWAY_HOST);
            // 获取 INGRESS_PORT
            Service istioService = CloudNativeClientHelper.kubernetesClient.services().inNamespace("istio-system").withName("istio-ingressgateway").get();
            istioService.getSpec().getPorts().forEach(
                    (port) -> {
                        if ("http2".equals(port.getName())) {
                            INGRESS_GATEWAY_PORT = String.valueOf(port.getNodePort());
                        }
                    }
            );
            log.info("istio-ingress-port:" + INGRESS_GATEWAY_PORT);
        } catch (Exception e) {
            log.info("动态获取Istio入口网关配置失败，启用静态配置");
            INGRESS_GATEWAY_PORT = STATIC_INGRESS_GATEWAY_PORT;
        }
    }
}

