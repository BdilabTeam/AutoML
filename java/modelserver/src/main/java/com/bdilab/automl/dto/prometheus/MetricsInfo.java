package com.bdilab.automl.dto.prometheus;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MetricsInfo {
    private Values cpuUsage;
    private Values memoryRss;
}
