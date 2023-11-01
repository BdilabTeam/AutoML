package com.bdilab.automl.service;

import java.util.List;

public interface TrainingProjectService {
    void deployment(Integer id);
    void undeploy(Integer id);
    String infer(Integer id, List<Object> instances);
}
