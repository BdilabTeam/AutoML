package com.bdilab.automl.service;

import com.bdilab.automl.dto.prometheus.MetricsCharts;

public interface MonitorService {
    MetricsCharts getResourceUsageInfo(String namespace, String serviceName) throws Exception;

    String getGrafanaUrl();
}
