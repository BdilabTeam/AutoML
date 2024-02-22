package com.bdilab.automl.controller;

import com.bdilab.automl.common.exception.InternalServerErrorException;
import com.bdilab.automl.common.response.HttpResponse;
import com.bdilab.automl.common.utils.HttpResponseUtils;
import com.bdilab.automl.service.impl.ExperimentServiceImpl;
import com.bdilab.automl.vo.EndpointInfoVO;
import com.bdilab.automl.vo.InferenceDataVO;
import com.bdilab.automl.vo.ServiceInfoVo;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import javax.validation.Valid;
import java.lang.reflect.Field;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/automl")
public class ExperimentController {
    @Resource
    private ExperimentServiceImpl experimentService;

    @Resource
    private ObjectMapper objectMapper;

    @PostMapping("/deploy")
    @ApiOperation(value = "部署推理服务", notes = "将当前实验输出的预训练模型部署为推理端点")
    public HttpResponse deploy(@Valid @RequestBody EndpointInfoVO endpointInfoVO) {
        experimentService.deploy(endpointInfoVO.getExperimentName(), endpointInfoVO.getEndpointName());
        return new HttpResponse(HttpResponseUtils.generateSuccessResponseData("部署成功"));
    }

    @DeleteMapping("/undeploy/{endpointName}")
    @ApiOperation(value = "下线模型服务", notes = "下线已部署的模型推理服务")
    public HttpResponse undeploy(@PathVariable String endpointName) {
        experimentService.undeploy(endpointName);
        return new HttpResponse(HttpResponseUtils.generateSuccessResponseData("删除部署成功"));
    }

    @PostMapping("/infer")
    @ApiOperation(value = "推理", notes = "执行模型推理")
    public HttpResponse infer(@Valid @RequestBody InferenceDataVO inferenceData) {

        String inferenceResult = experimentService.infer(inferenceData.getEndpointName(), inferenceData.getInstances());
        try {
            Map<String, Object> data = objectMapper.readValue(inferenceResult, Map.class);
            return new HttpResponse(data);
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData("Inference result format is incorrect."));
        }
    }

    @GetMapping("/ServiceInfomation")
    @ApiOperation(value = "服务信息", notes = "获取Knative服务信息")
    public HttpResponse ServiceInfo(){
        Map<String, Object> serviceInfoMap = new LinkedHashMap<>();
        List<Map<String, Object>> serviceInfoMapList = new LinkedList<>();
        List<ServiceInfoVo> serviceInfoVoList = experimentService.ServiceInfo();
        for (ServiceInfoVo serviceInfoVo : serviceInfoVoList) {
            serviceInfoMap.put("name", serviceInfoVo.getName());
            serviceInfoMap.put("image", serviceInfoVo.getImage());
            serviceInfoMap.put("trafficPercent", serviceInfoVo.getTrafficPercent());
            serviceInfoMap.put("lastReadyTime", serviceInfoVo.getLastReadyTime());
            serviceInfoMap.put("url", serviceInfoVo.getUrl());
            serviceInfoMap.put("LastReadyVisionName", serviceInfoVo.getLastReadyRevision());
            serviceInfoMap.put("LastCreatedVisionName", serviceInfoVo.getLastCreatedRevision());
            serviceInfoMap.put("node", serviceInfoVo.getNodeSelect());
            serviceInfoMap.put("modificationCount", serviceInfoVo.getModificationCount());

            serviceInfoMapList.add(serviceInfoMap);
        }

        Map<String, Object> resultMap = new LinkedHashMap<>();
        resultMap.put("data", serviceInfoMapList);

        return new HttpResponse(resultMap);
    }
}
