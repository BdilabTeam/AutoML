package com.bdilab.automl.common.config;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
@Slf4j
@Data
@Configuration
public class MonitorConfig {
    @Value("${prometheus.monitor.pod.cpu}")
    private String cpuMetricStatement;
    @Value("${prometheus.monitor.pod.memory}")
    private String memoryMetricStatement;
}
