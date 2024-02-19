package com.bdilab.automl.common.response;

import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.Map;

@Data
@AllArgsConstructor
public class HttpResponse {
    @ApiModelProperty(value = "状态码", required = true)
    private int code;
    @ApiModelProperty(value = "状态信息", required = true)
    private String message;
    @ApiModelProperty(value = "响应数据", required = true)
    private Object data;

    public HttpResponse(Map<String, Object> data) {
        this.code = 200;
        this.message = "请求成功";
        this.data = data;
    }
}
