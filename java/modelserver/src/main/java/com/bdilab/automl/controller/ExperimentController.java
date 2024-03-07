package com.bdilab.automl.controller;

import com.bdilab.automl.common.exception.InternalServerErrorException;
import com.bdilab.automl.common.response.HttpResponse;
import com.bdilab.automl.common.utils.HttpResponseUtils;
import com.bdilab.automl.dto.InferenceServiceInfo;
import com.bdilab.automl.service.impl.ExperimentServiceImpl;
import com.bdilab.automl.vo.EndpointInfoVO;
import com.bdilab.automl.vo.InferenceDataVO;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import javax.validation.Valid;
import java.util.*;

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

    @DeleteMapping("/undeploy/{endpoint-name}")
    @ApiOperation(value = "下线模型服务", notes = "下线已部署的模型推理服务")
    @ApiImplicitParam(name = "endpoint-name", value = "端点名称", required = true, example = "test")
    public HttpResponse undeploy(@PathVariable("endpoint-name") String endpointName) {
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

    @GetMapping("/inference-service/overview")
    @ApiOperation(value = "服务信息", notes = "获取Knative服务信息")
    public HttpResponse ServiceInfo(){
        List<InferenceServiceInfo> inferenceServiceInfoList;
        try {
            inferenceServiceInfoList = experimentService.serviceOverview();
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("Failed to get the overview of the services, for a specific reason: %s", e)));
        }

        Map<String, Object> res = new HashMap<>();
        res.put("services-overview", inferenceServiceInfoList);
        return new HttpResponse(res);
    }
}
