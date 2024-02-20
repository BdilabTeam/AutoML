package com.bdilab.automl.common.config;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.converter.json.Jackson2ObjectMapperBuilder;

@Configuration
public class JacksonConfig {
    @Bean()
    public ObjectMapper objectMapper(Jackson2ObjectMapperBuilder builder) {
        return builder
                .build()
                .enable(SerializationFeature.INDENT_OUTPUT) // 输出格式化
                .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS)    // 禁止将日期写为时间戳
                .configure(JsonParser.Feature.ALLOW_NON_NUMERIC_NUMBERS, true)  // 解析包含 NaN、Infinity 和 -Infinity 这些非标准数字值的 JSON
                .configure(JsonParser.Feature.ALLOW_MISSING_VALUES, true)   // 允许 JSON 数据中的数组元素缺失值
                .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);    // 忽略JSON数据中的未知字段
    }
}
