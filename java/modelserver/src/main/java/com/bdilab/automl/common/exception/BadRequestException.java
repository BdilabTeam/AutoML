package com.bdilab.automl.common.exception;

import java.util.Map;

public class BadRequestException extends BaseException{
    public BadRequestException(Map<String, Object> data) {
        super(ErrorCode.BAD_REQUEST, data);
    }
}
