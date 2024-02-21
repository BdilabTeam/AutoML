package com.bdilab.automl.common.exception;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;
import org.springframework.http.HttpStatus;

@ToString
@Getter
@AllArgsConstructor
public enum ErrorCode {
    BAD_REQUEST(400, HttpStatus.BAD_REQUEST, "Bad Request"),
    INTERNAL_SERVER_ERROR(500, HttpStatus.INTERNAL_SERVER_ERROR, "内部服务错误"),
    REQUEST_FAILED(502, HttpStatus.BAD_GATEWAY, "无法获取上游服务的有效响应"),
    RESOURCE_NOT_FOUND(1001, HttpStatus.NOT_FOUND, "未找到该资源"),
    REQUEST_VALIDATION_FAILED(1002, HttpStatus.BAD_REQUEST, "请求数据格式验证失败");

    private final int code;

    private final HttpStatus status;

    private final String message;
}
