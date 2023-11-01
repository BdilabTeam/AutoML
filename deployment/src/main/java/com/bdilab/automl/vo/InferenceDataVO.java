package com.bdilab.automl.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.validation.constraints.NotNull;
import java.util.List;


@Data
@NoArgsConstructor
@AllArgsConstructor
public class InferenceDataVO {
    @NotNull
    private Integer id;
    @NotNull
    private List<Object> instances;       // 请求体json格式数据
}
