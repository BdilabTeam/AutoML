package com.bdilab.automl.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.validation.constraints.NotNull;
import java.time.LocalDateTime;

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
    @JsonFormat(pattern="yyyy-MM-dd HH:mm:ss", timezone="GMT+8")
    private LocalDateTime createTime;
    @NotNull
    private String url;
    @NotNull
    private String experimentName;
    @NotNull
    private String taskWithModel;
}
