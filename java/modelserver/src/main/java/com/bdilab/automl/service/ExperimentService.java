package com.bdilab.automl.service;

import com.bdilab.automl.dto.InferenceServiceInfo;
import org.springframework.web.socket.WebSocketSession;

import java.util.List;

public interface ExperimentService {
    void deploy(String experimentName, String endpointName);
    void undeploy(String endpointName);
    String infer(String endpointName, List<Object> instances);
    List<InferenceServiceInfo> serviceOverview();
    void logs(WebSocketSession session, String endpointName) throws Exception;
}
