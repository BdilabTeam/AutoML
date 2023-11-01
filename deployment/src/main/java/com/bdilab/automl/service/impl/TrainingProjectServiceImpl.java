package com.bdilab.automl.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.bdilab.automl.common.exception.InternalServerErrorException;
import com.bdilab.automl.common.utils.*;
import com.bdilab.automl.mapper.TrainingProjectMapper;
import com.bdilab.automl.model.TrainingProject;
import com.bdilab.automl.service.TrainingProjectService;
import io.cloudevents.CloudEvent;
import io.cloudevents.core.v1.CloudEventBuilder;
import io.fabric8.knative.serving.v1.ServiceSpec;
import io.fabric8.knative.serving.v1.ServiceSpecBuilder;
import io.fabric8.kubernetes.api.model.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpMethod;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import javax.annotation.Resource;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.OffsetDateTime;
import java.util.*;


@Service
@Slf4j
public class TrainingProjectServiceImpl implements TrainingProjectService {
    @Resource
    private TrainingProjectMapper trainingProjectMapper;

    @Resource
    private HttpClientHelper httpClientHelper;

    @Value("${server.ip}")
    private String serverIp;

    @Value("${server.port}")
    private String serverPort;

    @Override
    @Transactional
    public void deployment(Integer id) {
        io.fabric8.knative.serving.v1.Service service = new io.fabric8.knative.serving.v1.Service();
        service.setApiVersion(KnativeHelper.KNATIVE_SERVING_API_VERSION);
        service.setKind(KnativeHelper.KNATIVE_SERVING_KIND);
        TrainingProject trainingProject = trainingProjectMapper.selectById(id);
        if (null == trainingProject) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("ID为%d的训练项目不存在", id)));
        }
        // metadata
        Integer trainingProjectId = trainingProject.getId();
        String trainingProjectName = trainingProject.getName();
        String ksvcName = String.join("-", trainingProjectName, String.valueOf(trainingProjectId));
        ObjectMeta objectMeta = new ObjectMetaBuilder()
                .withName(ksvcName)
                .withNamespace(KnativeHelper.NAMESPACE)
                .build();
        service.setMetadata(objectMeta);

        Map<String, String> annotations = new HashMap() {
            {
                put("autoscaling.knative.dev/minScale", "1");
            }
        };

        Map<String, String> nodeSelector = new HashMap() {
            {
                put("kubernetes.io/hostname", "node1");
            }
        };

        String containerName = trainingProject.getTaskType();
        if (StringUtils.isEmpty(containerName)) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("ID为%d的训练项目任务类型为空", id)));
        }

        // 构造container启动command
        String modelName = trainingProject.getTaskType();
        String modelNameOrPath = trainingProject.getModelNameOrPath();
        if (StringUtils.isEmpty(modelNameOrPath)) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("ID为%d的训练项目模型路径为空", id)));
        }
        List<String> commands = new ArrayList<String>() {
            {
                add("python");
                add("-m");
                add("image_classification_server");
                add(String.format("--model_name=%s", modelName));
                add(String.format("--model_dir=%s", HuggingfaceServerUtils.HUGGINGFACE_SERVER_VOLUME_MOUNT_PATH));
                // TODO task、pretrained_model、pretrained_tokenizer、batch_size、framework
            }
        };
        // 构造volumeMount
        VolumeMount volumeMount = new VolumeMountBuilder()
                .withName(HuggingfaceServerUtils.HUGGINGFACE_SERVER_VOLUME_MOUNT_NAME)
                .withMountPath(HuggingfaceServerUtils.HUGGINGFACE_SERVER_VOLUME_MOUNT_PATH)
                .withReadOnly()
                .build();

        // 构造container
        Container container = new ContainerBuilder()
                .withName(containerName)
                .withImage(HuggingfaceServerUtils.HUGGINGFACE_SERVER_IMAGE)
                .withImagePullPolicy(KnativeHelper.IMAGE_PULL_POLICY)
                .withCommand(commands)
                .withVolumeMounts(volumeMount)
                .build();

        // 构造volume
        Volume volume = new VolumeBuilder()
                .withName(HuggingfaceServerUtils.HUGGINGFACE_SERVER_VOLUME_MOUNT_NAME)
                .withNewPersistentVolumeClaim(HuggingfaceServerUtils.HUGGINGFACE_SERVER_PVC_NAME, true)
                .build();

        // spec
        ServiceSpec spec = new ServiceSpecBuilder()
                .withNewTemplate()
                .withNewMetadata()
                .withAnnotations(annotations)
                .endMetadata()
                .withNewSpec()
                .withNodeSelector(nodeSelector)
                .withContainers(container)
                .withVolumes(volume).endSpec()
                .endTemplate()
                .build();

        service.setSpec(spec);

        try {
            io.fabric8.knative.serving.v1.Service created_service = CloudNativeClientHelper.knativeClient.services().create(service);
            log.info(created_service.toString());
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("模型部署失败, 具体原因: %s", e)));
        }

        String host = KnativeHelper.generateHost(ksvcName);
        trainingProject.setHost(host);
        try {
            trainingProjectMapper.updateById(trainingProject);
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("数据库表更新失败, 具体原因: %s", e)));
        }
    }

    @Override
    @Transactional
    public void undeploy(Integer id) {
        TrainingProject trainingProject = trainingProjectMapper.selectById(id);
        if (null == trainingProject) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("ID为%d的训练项目不存在", id)));
        }
        Integer trainingProjectId = trainingProject.getId();
        String trainingProjectName = trainingProject.getName();
        String ksvcName = String.join("-", trainingProjectName, String.valueOf(trainingProjectId));
        try {
            CloudNativeClientHelper.knativeClient.services().inNamespace(KnativeHelper.NAMESPACE).withName(ksvcName).delete();
            // TODO 检查已删除ksvc状态，确保删除成功
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("删除ksvc失败, 具体原因:", e)));
        }
        trainingProject.setHost(null);
        try {
            trainingProjectMapper.updateById(trainingProject);
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("更新数据库失败, 具体原因:", e)));
        }
    }

    @Override
    public String infer(Integer id, List<Object> instances) {
        if (null == instances || instances.size() == 0) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData("推理数据不能为空"));
        }
        JSONObject var1 = new JSONObject();
        var1.put("instances", instances);
        String jsonFormatInstances = var1.toJSONString();

        TrainingProject trainingProject = trainingProjectMapper.selectById(id);
        if (null == trainingProject) {
            throw new InternalServerErrorException(HttpResponseUtil.generateExceptionResponseData(String.format("ID为%d的训练项目不存在", id)));
        }
        String modelName = trainingProject.getTaskType();
        String host = trainingProject.getHost();
        // 使用推理服务，并获取推理结果
        String url = "http://" + serverIp + ":" + IstioHelper.INGRESS_GATEWAY_PORT + "/v2/models/" + modelName + "/infer";
        CloudEvent event = new CloudEventBuilder()
                .withId(id + "-" + UUID.randomUUID())
                .withSource(URI.create("http://automl.deployment.com"))
                .withType("com.deployment.automl.inference.request")
                .withTime(OffsetDateTime.now())
                .withData("application/json", jsonFormatInstances.getBytes(StandardCharsets.UTF_8))
                .build();
        String res = httpClientHelper.sendBinaryCloudEvent(event, url, HttpMethod.POST, host, null);
        return res;
    }

}
