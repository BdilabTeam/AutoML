package com.bdilab.automl.vo;

import com.bdilab.automl.model.Instance;
import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.checkerframework.checker.units.qual.A;

import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Pattern;
import java.util.List;


@Data
@NoArgsConstructor
@AllArgsConstructor
public class InferenceDataVO {
    @NotNull
    @ApiModelProperty(value = "端点名称", required = true, example = "test")
    @Pattern(regexp="^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", message="包含不超过 63 个字符, 由小写字母、数字或 \"-\" 组成\n, 以字母或数字开头和结尾")
    private String endpointName;

    @NotNull
    @NotEmpty
    @ApiModelProperty(value = "推理数据", example = "[[0.92,0.4,0.17,0.97,0.6,0.31,0.38,0.25,0.96,0.79,707,0.3,3.13,842]]")
    private List<Object> instances;
}
