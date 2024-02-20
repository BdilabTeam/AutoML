package com.bdilab.automl.common.exception;

import org.junit.Test;

import java.util.HashMap;

public class TestException {
    @Test
    public void t() {
            ex();
    }

    public void ex() throws InternalServerErrorException {
        throw new InternalServerErrorException(new HashMap<String, Object>() {
            {
                put("ErrorInfo", "haha");
            }
        });
    }
}
