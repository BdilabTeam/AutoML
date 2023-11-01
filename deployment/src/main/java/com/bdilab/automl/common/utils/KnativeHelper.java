package com.bdilab.automl.common.utils;

import com.bdilab.automl.common.exception.InternalServerErrorException;
import io.fabric8.knative.serving.v1.*;
import io.fabric8.knative.serving.v1.Service;
import io.fabric8.knative.serving.v1.ServiceBuilder;
import io.fabric8.knative.serving.v1.ServiceSpec;
import io.fabric8.knative.serving.v1.ServiceSpecBuilder;
import io.fabric8.kubernetes.api.model.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.DependsOn;
import org.springframework.lang.Nullable;
import org.springframework.stereotype.Component;
import java.util.*;

@Component
@DependsOn({"cloudNativeClientHelper"})
@Slf4j
public class KnativeHelper {

    public static final String KNATIVE_SERVING_API_VERSION = "serving.knative.dev/v1";

    public static final String KNATIVE_SERVING_KIND = "Service";

    public static final String NAMESPACE = "zauto";

    public static final String IMAGE_PULL_POLICY = "IfNotPresent";

    private static final Integer TIMEOUT_SECONDS = 10;

    private static final Integer POLLING_INTERVAL = 2;

    public static String generateHost(String ksvcName) {
        return String.format("%s.%s.example.com", ksvcName, NAMESPACE);
    }

    private static Service createKsvc(Service kService) throws InternalServerErrorException {
        Service service = CloudNativeClientHelper.knativeClient.services().resource(kService).create();
        waitKsvcReady(kService.getMetadata().getNamespace(), kService.getMetadata().getName(), null, null);
        return service;
    }

    private static Boolean isKsvcReady(String namespace, String name) {
        // TODO 通过获取资源Status判断服务状态是否就绪
        return CloudNativeClientHelper.knativeClient.services().inNamespace(namespace).withName(name).isReady();
    }

    private static void waitKsvcReady(String namespace, String ksvcName, @Nullable Integer timeoutSeconds, @Nullable Integer pollingInterval) throws InternalServerErrorException {
        if (null == timeoutSeconds) {
            timeoutSeconds = TIMEOUT_SECONDS * 1000;
        }
        if (null == pollingInterval) {
            pollingInterval = POLLING_INTERVAL * 1000;
        }
        for (int i = 0; i < timeoutSeconds / pollingInterval; i++) {
            try {
                Thread.sleep(pollingInterval);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (isKsvcReady(namespace, ksvcName)) {
                return;
            }
        }
        throw new InternalServerErrorException(new HashMap<String, Object>(){
            {
                put("ErrorInfo", "Timeout to start the KService");
            }
        });
    }

    private static Service generateService(ObjectMeta objectMeta, ServiceSpec serviceSpec) {
        return new ServiceBuilder()
                .withApiVersion(KNATIVE_SERVING_API_VERSION)
                .withKind(KNATIVE_SERVING_KIND)
                .withMetadata(objectMeta)
                .withSpec(serviceSpec)
                .build();
    }

    private static ObjectMeta generateObjectMeta(String name, String namespace, @Nullable Map<String, String> annotations, @Nullable Map<String, String> labels) {
        return new ObjectMetaBuilder()
                .withName(name)
                .withNamespace(namespace)
                .withAnnotations(annotations)
                .withLabels(labels)
                .build();
    }

    private static ServiceSpec generateServiceSpec(RevisionTemplateSpec templateSpec, @Nullable List<TrafficTarget> trafficTargets) {
        return new ServiceSpecBuilder()
                .withTemplate(templateSpec)
                .withTraffic(trafficTargets)
                .build();
    }

    private static RevisionTemplateSpec generateRevisionTemplateSpec(ObjectMeta objectMeta, RevisionSpec revisionSpec) {
        return new RevisionTemplateSpecBuilder()
                .withMetadata(objectMeta)
                .withSpec(revisionSpec)
                .build();
    }

    private static RevisionSpec generateRevisionSpec(Container container, @Nullable List<Volume> volumes, @Nullable String nodeName, @Nullable List<LocalObjectReference> imagePullSecrets) {
        return new RevisionSpecBuilder()
                .withContainers(container)
                .withVolumes(volumes)
                .withNodeName(nodeName)
                .withImagePullSecrets(imagePullSecrets)
                .build();
    }

    private static Container generateContainer(String containerName, String imageFullName, String imagePullPolicy, @Nullable List<String> args, @Nullable List<VolumeMount> volumeMounts) {
        return new ContainerBuilder()
                .withName(containerName)
                .withImage(imageFullName)
                .withImagePullPolicy(imagePullPolicy)
                .withArgs(args)
                .withVolumeMounts(volumeMounts)
                .build();
    }

    private static VolumeMount generateVolumeMount(String volumeMountName, String mountPath, @Nullable Boolean readOnly) {
        return new VolumeMountBuilder()
                .withName(volumeMountName)
                .withMountPath(mountPath)
                .withReadOnly(readOnly)
                .build();
    }

    private static Volume generateHostPathVolume(String volumeName, String hostPath) {
        return new VolumeBuilder()
                .withName(volumeName)
                .withNewHostPath()
                .withPath(hostPath)
                .endHostPath()
                .build();
    }

    private static LocalObjectReference generateLocalObjectReference(String name) {
        return new LocalObjectReferenceBuilder()
                .withName(name)
                .build();
    }
}
