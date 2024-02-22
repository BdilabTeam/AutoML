package com.bdilab.automl.service;

import com.bdilab.automl.vo.ServiceInfoVo;

import java.util.List;
import java.util.Map;

public interface ExperimentService {
    void deploy(String experimentName, String endpointName);
    void undeploy(String endpointName);
    String infer(String endpointName, List<Object> instances);
    List<ServiceInfoVo> ServiceInfo();
}
