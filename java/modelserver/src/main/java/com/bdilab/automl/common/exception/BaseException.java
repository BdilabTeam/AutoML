package com.bdilab.automl.common.exception;

import lombok.Getter;

@Getter
public abstract class BaseException extends RuntimeException {
    private final ErrorCode error;
    private final Object data;

    public BaseException(ErrorCode error, Object data) {
        super(error.getMessage());
        this.error = error;
        this.data = data;
    }

    protected BaseException(ErrorCode error, Object data, Throwable cause) {
        super(error.getMessage(), cause);
        this.error = error;
        this.data = data;
    }
}
