package com.bdilab.automl.common.utils;

import com.bdilab.automl.common.exception.HttpServerErrorException;
import io.cloudevents.CloudEvent;
import io.cloudevents.core.provider.EventFormatProvider;
import io.cloudevents.jackson.JsonFormat;
import io.cloudevents.spring.http.CloudEventHttpUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.Nullable;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;


@Component
@Slf4j
public class CloudEventUtils {
    private static RestTemplate restTemplate;

    @Autowired
    public CloudEventUtils(@Qualifier("restTemplate") RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * 发送结构化CloudEvent
     * 结构化CE，CE中的属性、数据均作为请求体中的部分。
     */
    public static String sendStructuredCloudEvent(CloudEvent event, String url, HttpMethod method, @Nullable String virtualHost, @Nullable String cookie) throws Exception {
        // 设置 HTTP 请求头
        HttpHeaders headers = new HttpHeaders();
        // 发送结构化CE，需要将请求头中的Content-Type设置为application/cloudevents+json，以表示此CloudEvent是结构化CE
        headers.setContentType(new MediaType("application", "cloudevents+json"));
        if (null != virtualHost) {
            headers.set("Host", virtualHost);
        }
        if (null != cookie) {
            headers.add("Cookie", cookie);
        }
        // 序列化CE，并将序列化后的数据作为请求体。
        byte[] jsonEvent = EventFormatProvider.getInstance().resolveFormat(JsonFormat.CONTENT_TYPE).serialize(event);
        org.springframework.http.HttpEntity<byte[]> entity = new org.springframework.http.HttpEntity<>(jsonEvent, headers);
        ResponseEntity<String> response = restTemplate.exchange(url, method, entity, String.class);
        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new HttpServerErrorException(
                    new HashMap<String, Object>(){
                        {
                            put("ErrorInfo", response.toString());
                        }
                    });
        }
        return response.getBody();
    }

    /**
     * 发送二进制CE
     * 二进制CE，CE中的属性作为请求头中的属性，CE中的数据作为请求体。
     */
    public static String sendBinaryCloudEvent(CloudEvent event, String url, HttpMethod method, @Nullable String virtualHost, @Nullable String cookie) {
        // 设置 HTTP 请求头
        HttpHeaders headers= CloudEventHttpUtils.toHttp(event);
        if (null != virtualHost) {
            headers.set("Host", virtualHost);
        }
        if (null != cookie) {
            headers.add("Cookie", cookie);
        }
        org.springframework.http.HttpEntity<byte[]> entity = new org.springframework.http.HttpEntity<>(event.getData().toBytes(), headers);
        ResponseEntity<String> response = restTemplate.exchange(url, method, entity, String.class);
        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new HttpServerErrorException(
                    new HashMap<String, Object>(){
                        {
                            put("ErrorInfo", response.toString());
                        }
                    });
        }
        return response.getBody();
    }
}
