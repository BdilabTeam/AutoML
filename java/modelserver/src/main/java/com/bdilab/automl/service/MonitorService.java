package com.bdilab.automl.service;

import com.bdilab.automl.dto.prometheus.MetricsInfo;

public interface MonitorService {
    MetricsInfo getResourceUsageInfo(String namespace, String serviceName) throws Exception;
}
