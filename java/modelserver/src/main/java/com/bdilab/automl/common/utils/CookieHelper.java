package com.bdilab.automl.common.utils;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.DependsOn;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;


@Component
@DependsOn("istioUtils")
@Slf4j
//@EnableScheduling
public class CookieHelper {
    @Value("${server.ip}")
    private String serverHost;

    public static String cookie;

    /**
     * PostConstruct注解让项目启动时获取一次cookie
     * cron表达式表示每小时获取一次
     */
    @PostConstruct
    public void init() {
        getCookie();
    }

//    @Scheduled(cron = "0 0 0/1 * * ?")
    public void getCookie() {
        try {
            String rootUrl = String.join("", "http://", serverHost, ":", IstioUtils.INGRESS_GATEWAY_PORT);
            String firstUrl = getRedirectUrl(rootUrl);
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
            MultiValueMap<String, String> requestBody = new LinkedMultiValueMap<>();
            requestBody.add("login", "user@example.com");
            requestBody.add("password", "12341234");
            org.springframework.http.HttpEntity request = new org.springframework.http.HttpEntity(requestBody, (MultiValueMap) headers);
            RestTemplate restTemplate = new RestTemplate();
            restTemplate.setRequestFactory(new NoRedirectClientHttpRequestFactory());
            ResponseEntity<String> responseEntity = restTemplate.postForEntity(firstUrl, request, String.class, new Object[0]);
            String secondUrl = String.valueOf(responseEntity.getHeaders().getLocation());
            HttpHeaders secondHeaders = new HttpHeaders();
            List<MediaType> mediaTypes = new ArrayList<>();
            secondHeaders.setAccept(mediaTypes);
            responseEntity = restTemplate.getForEntity(rootUrl + secondUrl, String.class, new Object[0]);
            String thirdUrl = String.valueOf(responseEntity.getHeaders().getLocation());
            responseEntity = restTemplate.getForEntity(rootUrl + thirdUrl, String.class, new Object[0]);
            cookie = responseEntity.getHeaders().get("Set-Cookie").get(0);
            log.info("cookie:" + cookie);
        } catch (Exception e) {
            log.error(String.format("获取cookie失败, 具体原因: %s", e.toString()));
        }

    }

    private String getRedirectUrl(String url) {
        String RedirectUrl = null;
        try {
            URL Url = new URL(url);
            HttpURLConnection conn = (HttpURLConnection) Url.openConnection();
            RedirectUrl = conn.getURL().toString();
        } catch (Exception e) {
            log.error(e.getMessage());
        }
        return RedirectUrl;
    }
}
//设置restTemplate不跳转
class NoRedirectClientHttpRequestFactory extends SimpleClientHttpRequestFactory {
    @Override
    protected void prepareConnection(HttpURLConnection connection, String httpMethod) throws IOException {
        super.prepareConnection(connection, httpMethod);
        connection.setInstanceFollowRedirects(false);
    }
}