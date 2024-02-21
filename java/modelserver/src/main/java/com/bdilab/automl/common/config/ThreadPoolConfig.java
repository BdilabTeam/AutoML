package com.bdilab.automl.common.config;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Slf4j
@Configuration
@EnableAsync
public class ThreadPoolConfig {
    @Bean(destroyMethod = "shutdown")
    public ThreadPoolExecutor threadPoolExecutor () {
        return new ThreadPoolExecutor(3, 10, 60, TimeUnit.SECONDS,
                new LinkedBlockingQueue<Runnable>(10000),
                new ThreadFactoryBuilder().setNameFormat("default-executor-%d").build(),
                (r, executor) -> log.error("system pool is full! "));
    }
}
