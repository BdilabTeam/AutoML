package com.bdilab.automl;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan({"com.bdilab.automl.mapper"})
public class DeploymentApplication {
    public static void main(String[] args) {
        SpringApplication.run(DeploymentApplication.class, args);
    }
}
