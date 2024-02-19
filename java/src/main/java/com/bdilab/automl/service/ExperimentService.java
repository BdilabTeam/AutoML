package com.bdilab.automl.service;

import java.util.List;

public interface ExperimentService {
    void deployment(Integer experimentId);
    void undeploy(Integer experimentId);
    String infer(Integer experimentId, List<Object> instances);
}
