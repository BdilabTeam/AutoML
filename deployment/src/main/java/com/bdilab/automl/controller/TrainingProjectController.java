package com.bdilab.automl.controller;

import com.bdilab.automl.common.response.HttpResponse;
import com.bdilab.automl.common.utils.HttpResponseUtil;
import com.bdilab.automl.service.impl.TrainingProjectServiceImpl;
import com.bdilab.automl.vo.InferenceDataVO;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import javax.validation.Valid;
import java.util.HashMap;

@RestController
@RequestMapping("/api/v1/automl")
public class TrainingProjectController {
    @Resource
    private TrainingProjectServiceImpl trainingProjectService;

    @PostMapping("/deploy/{id}")
    @ApiOperation(value = "部署模型", notes = "将预训练模型部署为推理端点")
    @ApiImplicitParam(name = "id", value = "训练项目ID", required = true, example = "1")
    public HttpResponse deploy(@PathVariable Integer id) {
        trainingProjectService.deployment(id);
        return new HttpResponse(HttpResponseUtil.generateSuccessResponseData("部署成功"));
    }

    @DeleteMapping("/undeploy/{id}")
    @ApiOperation(value = "删除部署", notes = "删除已部署的模型推理服务")
    @ApiImplicitParam(name = "id", value = "训练项目ID", required = true, example = "1")
    public HttpResponse undeploy(@PathVariable Integer id) {
        trainingProjectService.undeploy(id);
        return new HttpResponse(HttpResponseUtil.generateSuccessResponseData("删除部署成功"));
    }

    @PostMapping("/infer")
    @ApiOperation(value = "推理", notes = "执行模型推理")
    public HttpResponse infer(@Valid @RequestBody InferenceDataVO inferenceData) {

        String inferenceResult = trainingProjectService.infer(inferenceData.getId(), inferenceData.getInstances());
        return new HttpResponse(new HashMap<String, Object>() {
            {
                put("Inference result", inferenceResult);
            }
        });
    }
}
