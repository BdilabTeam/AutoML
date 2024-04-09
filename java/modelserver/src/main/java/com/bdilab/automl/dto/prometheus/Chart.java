package com.bdilab.automl.dto.prometheus;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Chart {
    private List<Object> xAxis;
    private List<Object> yAxis;
}
