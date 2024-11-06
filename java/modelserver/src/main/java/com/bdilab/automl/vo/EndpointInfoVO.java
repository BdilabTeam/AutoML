package com.bdilab.automl.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.context.annotation.Description;

import javax.validation.constraints.NotNull;
import javax.validation.constraints.Pattern;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class EndpointInfoVO {
    @NotNull
    @ApiModelProperty(value = "实验名称", required = true, example = "test")
    @Pattern(regexp="^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", message="包含不超过 63 个字符, 由小写字母、数字或 \"-\" 组成\n, 以字母或数字开头和结尾")
    String experimentName;

    @NotNull
    @ApiModelProperty(value = "端点名称", required = true, example = "test")
    @Pattern(regexp="^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", message="包含不超过 63 个字符, 由小写字母、数字或 \"-\" 组成\n, 以字母或数字开头和结尾")
    String endpointName;
}
