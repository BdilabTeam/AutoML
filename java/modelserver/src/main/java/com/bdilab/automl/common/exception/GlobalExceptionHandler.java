package com.bdilab.automl.common.exception;

import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import javax.servlet.http.HttpServletRequest;
import java.util.HashMap;

@ControllerAdvice()
@ResponseBody
public class GlobalExceptionHandler {
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Object> handleValidationException(MethodArgumentNotValidException ex, HttpServletRequest request) {
        BadRequestException badRequestException = new BadRequestException(new HashMap<String, Object>() {
            {
                put("Validation Error", ex.getMessage());
            }
        });
        ErrorResponse representation = new ErrorResponse(badRequestException, request.getRequestURI());
        return new ResponseEntity<>(representation, new HttpHeaders(), badRequestException.getError().getStatus());
    }

    @ExceptionHandler(BaseException.class)
    public ResponseEntity<?> handleAppException(BaseException ex, HttpServletRequest request) {
        ErrorResponse representation = new ErrorResponse(ex, request.getRequestURI());
        return new ResponseEntity<>(representation, new HttpHeaders(), ex.getError().getStatus());
    }
}

