package com.bdilab.automl.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.lang.reflect.Field;
import java.util.List;

import javax.validation.constraints.NotNull;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ServiceInfoVo {
    @NotNull
    private String name;
    @NotNull
    private String image;
    @NotNull
    private Long trafficPercent;
    @NotNull
    private String lastReadyTime;
    @NotNull
    private String url;
    @NotNull
    private String lastCreatedRevision;
    @NotNull
    private String lastReadyRevision;
    @NotNull
    private String nodeSelect;
    @NotNull
    private Long modificationCount;
}
