package com.bdilab.automl.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.bdilab.automl.common.exception.InternalServerErrorException;
import com.bdilab.automl.common.utils.*;
import com.bdilab.automl.common.websocket.LogCallback;
import com.bdilab.automl.common.websocket.WebSocketOutputStream;
import com.bdilab.automl.dto.InferenceServiceInfo;
import com.bdilab.automl.mapper.ExperimentMapper;
import com.bdilab.automl.model.Experiment;
import com.bdilab.automl.service.ExperimentService;
import io.cloudevents.CloudEvent;
import io.cloudevents.core.v1.CloudEventBuilder;
import io.fabric8.knative.client.DefaultKnativeClient;
import io.fabric8.knative.internal.pkg.apis.Condition;
import io.fabric8.knative.serving.v1.ServiceList;
import io.fabric8.knative.serving.v1.ServiceSpec;
import io.fabric8.knative.serving.v1.ServiceSpecBuilder;
import io.fabric8.kubernetes.api.model.*;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.kubernetes.client.PodLogs;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.apis.LogsApi;
import io.kubernetes.client.openapi.models.V1Pod;
import io.kubernetes.client.openapi.models.V1PodList;
import io.kubernetes.client.util.Streams;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpMethod;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import javax.annotation.Resource;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.concurrent.TimeUnit;


@Service
@Slf4j
public class ExperimentServiceImpl implements ExperimentService {
    @Resource
    private ExperimentMapper experimentMapper;
    @Resource
    private DefaultKnativeClient defaultKnativeClient;
    @Resource
    private CoreV1Api coreV1Api;
//    @Resource
//    private KubernetesClient kubernetesClient;

    @Value("${server.ip}")
    private String serverIp;

    @Value("${server.port}")
    private String serverPort;

    @Override
    @Transactional
    public void deploy(String experimentName, String endpointName) {
        // TODO endpointName-random
        // Check whether the experiment exists
        Experiment experiment = experimentMapper.selectOne(new QueryWrapper<Experiment>().lambda().eq(Experiment::getExperimentName, experimentName));
        if (null == experiment) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("Name:%s for experiment does not exist.", experimentName)));
        }
        // Check whether the endpoint is already in use
        if (defaultKnativeClient.services().inNamespace(Utils.NAMESPACE).withName(endpointName).get() != null) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("The endpoint name %s is already in use, please change it", endpointName)));
        }

        io.fabric8.knative.serving.v1.Service service = new io.fabric8.knative.serving.v1.Service();
        service.setApiVersion("serving.knative.dev/v1");
        service.setKind("Service");
        ObjectMeta objectMeta = new ObjectMetaBuilder()
                .withName(endpointName)
                .withNamespace(Utils.NAMESPACE)
                .build();
        service.setMetadata(objectMeta);
        Map<String, String> annotations = new HashMap<String, String>() {
            {
                put("autoscaling.knative.dev/minScale", "0");
                put("automl/experimentName", experimentName);
                put("automl/taskType", experiment.getTaskType());
                put("automl/modelType", experiment.getModelType());
            }
        };
        Map<String, String> labels = new HashMap<String, String>() {
            {
                put("webhooks.knative.dev/exclude", "true");
            }
        };
//        Map<String, String> nodeSelector = new HashMap<String, String>() {
//            {
//                put("kubernetes.io/hostname", "node1");
//            }
//        };
        List<String> commands = new ArrayList<String>() {
            {
                add("python");
                add("-m");
                add(String.format("%s", experiment.getModelType().equals("yolov8")? "yolo_server": "autokeras_server"));
                add(String.format("--model_name=%s", endpointName));
                add(String.format("--model_dir=%s", experiment.getModelType().equals("yolov8")? Utils.getYOLOBestModelDirInContainer(experimentName): Utils.getBestModelDirInContainer(experimentName)));
            }
        };
        // 构造volumeMount
        VolumeMount volumeMount = new VolumeMountBuilder()
                .withName("model-dir")
                .withMountPath(Utils.METADATA_DIR_IN_CONTAINER)
                .withReadOnly()
                .build();
        // 构造container
        Container container1 = new ContainerBuilder()
                .withName(endpointName)
                .withImage(String.format("%s", experiment.getModelType().equals("yolov8")? Utils.YOLO_IMAGE: Utils.AUTOKERAS_IMAGE))
                .withImagePullPolicy("IfNotPresent")
                .withCommand(commands)
                .withVolumeMounts(volumeMount)
                .build();
//        Container container2 = new ContainerBuilder()
//                .withName("queue-proxy")
//                .withImage("registry.cn-hangzhou.aliyuncs.com/knative-releases/knative.dev/serving/cmd/queue:latest")
//                .withImagePullPolicy("IfNotPresent")
//                .build();
        // 构造volume
        Volume volume = new VolumeBuilder()
                .withName("model-dir")
                .withPersistentVolumeClaim(new PersistentVolumeClaimVolumeSource(Utils.PVC_NAME, true))
                .build();
        // spec
        ServiceSpec spec = new ServiceSpecBuilder()
                .withNewTemplate()
                .withNewMetadata()
                .withAnnotations(annotations)
                .withLabels(labels)
                .endMetadata()
                .withNewSpec()
//                .withNodeSelector(nodeSelector)
                .withContainers(container1)
                .withImagePullSecrets(new LocalObjectReference("registry-aliyun"))
                .withVolumes(volume).endSpec()
                .endTemplate()
                .build();

        service.setSpec(spec);

        log.info("Creating the inference endpoint.");
        try {
            log.info(service.toString());
            io.fabric8.knative.serving.v1.Service created_service = defaultKnativeClient.services().create(service);
            log.info("Create Successfully");
            defaultKnativeClient.services().resource(created_service).waitUntilReady(10, TimeUnit.SECONDS);
        } catch (Exception e) {
            defaultKnativeClient.services().resource(service).delete();
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("推理服务部署失败, 具体原因: %s", e)));
        }
    }

    @Override
    @Transactional
    public void undeploy(String endpointName) {
        io.fabric8.knative.serving.v1.Service service = defaultKnativeClient.services().inNamespace(Utils.NAMESPACE).withName(endpointName).get();
        if (service == null) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("Inference endpoint with name %s does not exist.", endpointName)));
        }
        try {
            List<StatusDetails> statusDetails = defaultKnativeClient.services().resource(service).delete();
            log.info(statusDetails.toString());
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("Failed to delete the endpoint %s, for a specific reason:", endpointName, e)));
        }
    }

    @Override
    public String infer(String endpointName, List<Object> instances) {
        // Check if the endpoint exists
        io.fabric8.knative.serving.v1.Service service = defaultKnativeClient.services().inNamespace(Utils.NAMESPACE).withName(endpointName).get();
        if (service == null) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("Inference endpoint with name %s does not exist.", endpointName)));
        }
        log.info("Parsing the data.");
        if (null == instances || instances.isEmpty()) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData("推理数据不能为空"));
        }
        JSONObject var1 = new JSONObject();
        var1.put("instances", instances);
        String jsonFormatInstances = var1.toJSONString();
        log.info(jsonFormatInstances);

        log.info("Sending the inference request.");
        String host = Utils.generateHost(endpointName);
        // 使用推理服务，并获取推理结果
        String url = "http://" + serverIp + ":" + IstioUtils.INGRESS_GATEWAY_PORT + "/v2/models/" + endpointName + "/infer";
        CloudEvent event = new CloudEventBuilder()
                .withId(UUID.randomUUID().toString())
                .withSource(URI.create("http://automl.deployment.com"))
                .withType("com.deployment.automl.inference.request")
                .withTime(OffsetDateTime.now())
                .withData("application/json", jsonFormatInstances.getBytes(StandardCharsets.UTF_8))
                .build();
        return CloudEventUtils.sendBinaryCloudEvent(event, url, HttpMethod.POST, host, null);
    }

    @Override
    public List<InferenceServiceInfo> serviceOverview() {
        ServiceList serviceList = defaultKnativeClient.services().inNamespace(Utils.NAMESPACE).list();
        List<InferenceServiceInfo> inferenceServiceInfoList = new ArrayList<>();
        List<io.fabric8.knative.serving.v1.Service> items = serviceList.getItems();
        for (io.fabric8.knative.serving.v1.Service item : items) {
            InferenceServiceInfo inferenceServiceInfo = new InferenceServiceInfo();
            inferenceServiceInfo.setName(item.getMetadata().getName());
            String creationTimestamp = item.getMetadata().getCreationTimestamp();
            Instant instant = Instant.parse(creationTimestamp);
            LocalDateTime creationTime = LocalDateTime.ofInstant(instant, ZoneOffset.UTC);
            inferenceServiceInfo.setCreateTime(creationTime);
//            inferenceServiceInfo.setTrafficPercent(item.getSpec().getTraffic().get(0).getPercent());
            inferenceServiceInfo.setTrafficPercent(100L);
            String status = "Ready";
            List<Condition> conditions = item.getStatus().getConditions();
            for (int i = conditions.size() - 1; i >= 0; i--) {
                if (!"True".equals(conditions.get(i).getStatus())) {
                    status = "NotReady";
                }
            }
            inferenceServiceInfo.setStatus(status);
            inferenceServiceInfo.setUrl(item.getStatus().getUrl() + "/v2/models/" + item.getMetadata().getName() + "/infer");
            inferenceServiceInfo.setExperimentName(item.getSpec().getTemplate().getMetadata().getAnnotations().getOrDefault("automl/experimentName", "Unknown"));
            inferenceServiceInfo.setTaskWithModel(
                    item.getSpec().getTemplate().getMetadata().getAnnotations().getOrDefault("automl/taskType", "Unknown") +
                    "/" +
                    item.getSpec().getTemplate().getMetadata().getAnnotations().getOrDefault("automl/modelType", "Unknown")
            );
            inferenceServiceInfoList.add(inferenceServiceInfo);
        }
        return inferenceServiceInfoList;
    }

    @Override
    public void logs(WebSocketSession session, String endpointName) throws Exception {
        log.info(String.format("获取%s端点日志", endpointName));
        // 获取 Pod
        V1PodList podList;
        try {
            podList = coreV1Api.listNamespacedPod(Utils.NAMESPACE, null, null, null, null, null, null, null, null, null, false);
        } catch (ApiException e) {
            throw new Exception(String.format("获取pod列表失败，具体原因：%s", e.getMessage()));
        }
        String podName = null;
        String phase = "";
        for (V1Pod pod : podList.getItems()) {
            phase = pod.getStatus().getPhase();
            // 排除一些不正常的pod
            if (phase.isEmpty() || "Terminating".equals(phase)  || ("Failed".equals(phase)  && "UnexpectedAdmissionError".equals(pod.getStatus().getReason()))) {
                continue;
            }
            String serviceName = pod.getMetadata().getLabels().get("serving.knative.dev/service");
            if (serviceName != null && serviceName.equals(endpointName)) {
                podName = pod.getMetadata().getName();
                log.info("Pod name:" + podName);
                break;
            }
        }

        // 获取 Pod 历史日志
        coreV1Api.readNamespacedPodLogAsync(podName, Utils.NAMESPACE, endpointName, null, null, null, null, null, null, 50, true, new LogCallback(session));
//        // 获取pod实时日志
//        coreV1Api.readNamespacedPodLogAsync(podName, Utils.NAMESPACE, endpointName, true, null, null, null, false, null, null, true, new LogCallback(session));
//        log.info("follow done");

//        log.info(String.format("获取%s端点日志", endpointName));
//        // 获取 Pod 名称
//        V1PodList podList;
//        try {
//            podList = coreV1Api.listNamespacedPod(Utils.NAMESPACE, null, null, null, null, null, null, null, null, null, false);
//        } catch (ApiException e) {
//            throw new Exception(String.format("获取pod列表失败，具体原因：%s", e.getMessage()));
//        }
//        V1Pod selectedPod = null;
//        for (V1Pod pod : podList.getItems()) {
//            String serviceName = pod.getMetadata().getLabels().get("serving.knative.dev/service");
//            if (serviceName != null && serviceName.equals(endpointName)) {
//                selectedPod = pod;
//                break;
//            }
//        }
//        if (selectedPod == null) {
//            log.info("未查询到关联pod");
//        }
//        PodLogs podLogs = new PodLogs();
//        InputStream is = podLogs.streamNamespacedPodLog(selectedPod);
//        log.info(IOUtils.toString(is, "UTF-8"));
//        while (session.isOpen()) {
//            if (is.available() != 0) {
//                session.sendMessage(new TextMessage(IOUtils.toString(is, "UTF-8")));
//                is.reset();
//            }
//        }
//        log.info(String.format("Session %s 关闭", session.getId()));
    }
}

