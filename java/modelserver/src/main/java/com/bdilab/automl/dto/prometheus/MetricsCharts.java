package com.bdilab.automl.dto.prometheus;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MetricsCharts {
    private Chart cpuUsage;
    private Chart memoryRss;
}
