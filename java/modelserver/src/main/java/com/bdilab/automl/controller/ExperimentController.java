package com.bdilab.automl.controller;

import com.bdilab.automl.common.exception.InternalServerErrorException;
import com.bdilab.automl.common.response.HttpResponse;
import com.bdilab.automl.common.utils.HttpResponseUtils;
import com.bdilab.automl.dto.InferenceServiceInfo;
import com.bdilab.automl.service.impl.ExperimentServiceImpl;
import com.bdilab.automl.vo.EndpointInfoVO;
import com.bdilab.automl.vo.InferenceDataVO;
import com.bdilab.automl.vo.InferenceFolderVO;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import lombok.extern.slf4j.Slf4j;
import org.apache.ibatis.javassist.bytecode.ByteArray;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.types.TFloat32;

import javax.annotation.Resource;
import javax.validation.Valid;
import java.util.*;

import static com.bdilab.automl.common.utils.Utils.FileToData;

@Slf4j
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

    @PostMapping("/infer/tensor")
    @ApiOperation(value = "推理", notes = "使用'预转换张量'进行推理")
    public HttpResponse infer(@Valid @RequestBody InferenceDataVO inferenceData) {

        String inferenceResult = experimentService.infer(inferenceData.getEndpointName(), inferenceData.getInstances());
        try {
            Map<String, Object> data = objectMapper.readValue(inferenceResult, Map.class);
            return new HttpResponse(data);
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData("Inference result format is incorrect."));
        }
    }

    @PostMapping("/infer/file")
    @ApiOperation(value = "推理", notes = "使用文件/文件夹进行推理")
    public HttpResponse inferFolder(@Valid @ModelAttribute InferenceFolderVO inferenceFolder) {
        List<Object> allData = new ArrayList<>();
        for (MultipartFile file : inferenceFolder.getFiles()) {
            String fileName = file.getOriginalFilename().toLowerCase();
            if (fileName.endsWith(".csv")) {
                allData = FileToData(file);
                break;
            }
            List<Object> data = FileToData(file);
            allData.add(data);
        }
        log.info("Converted data：" + allData.toString());
        String inferenceResult = experimentService.infer(inferenceFolder.getEndpointName(), allData);
        try {
            Map<String, Object> data = objectMapper.readValue(inferenceResult, Map.class);
            return new HttpResponse(data);
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData("Inference result format is incorrect."));
        }
    }

    @GetMapping("/inference-service/overview")
    @ApiOperation(value = "服务信息", notes = "获取Knative服务信息")
    public HttpResponse serviceInfo(){
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
