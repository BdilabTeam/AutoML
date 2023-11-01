package com.bdilab.automl.common.exception;

import java.util.Map;

public class InternalServerErrorException extends BaseException{
    public InternalServerErrorException(Map<String, Object> data) {
        super(ErrorCode.INTERNAL_SERVER_ERROR, data);
    }
}
