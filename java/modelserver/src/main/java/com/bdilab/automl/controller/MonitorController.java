package com.bdilab.automl.controller;

import com.bdilab.automl.common.exception.InternalServerErrorException;
import com.bdilab.automl.common.response.HttpResponse;
import com.bdilab.automl.common.utils.HttpResponseUtils;
import com.bdilab.automl.common.utils.Utils;
import com.bdilab.automl.dto.prometheus.MetricsCharts;
import com.bdilab.automl.service.impl.MonitorServiceImpl;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.annotation.Resource;
import java.util.HashMap;

@RestController
@RequestMapping("/api/v1/automl")
public class MonitorController {
    @Resource
    private MonitorServiceImpl monitorService;

    @ApiOperation("获取'资源'监控指标")
    @GetMapping("/resource/{serviceName}/metrics")
    @ApiImplicitParam(name = "serviceName", value = "服务名称", example = "test")
    public HttpResponse getResourceMetrics(@PathVariable String serviceName){
        try {
            MetricsCharts metricsCharts = monitorService.getResourceUsageInfo(Utils.NAMESPACE, serviceName);
            return new HttpResponse(new HashMap<String, Object>(){
                {
                    put("metrics", metricsCharts);
                }
            });
        } catch (Exception e) {
            throw new InternalServerErrorException(HttpResponseUtils.generateExceptionResponseData(String.format("获取监控指标异常, 具体原因: %s", e)));
        }
    }

    @ApiOperation("获取Grafana访问URL")
    @GetMapping("/grafana/url")
    public HttpResponse getGrafanaUrl() {
        String grafanaUrl = monitorService.getGrafanaUrl();
        return new HttpResponse(new HashMap<String, Object>(){
            {
                put("grafanaUrl", grafanaUrl);
            }
        });
    }
}
