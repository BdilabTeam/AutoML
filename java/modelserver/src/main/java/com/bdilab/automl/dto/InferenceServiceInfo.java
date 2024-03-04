package com.bdilab.automl.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.validation.constraints.NotNull;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class InferenceServiceInfo {
    @NotNull
    private String name;
    @NotNull
    private Long trafficPercent;
    @NotNull
    private String status;
    @NotNull
    private String readyTime;
    @NotNull
    private String url;
    @NotNull
    private String experimentName;
    @NotNull
    private String taskWithModel;
}
