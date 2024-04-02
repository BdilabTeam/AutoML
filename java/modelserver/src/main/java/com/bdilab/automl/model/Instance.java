package com.bdilab.automl.model;

import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Instance {
    @ApiModelProperty("数据类型")
    private String dataType;
    @ApiModelProperty("数据体")
    private Object data;
}
