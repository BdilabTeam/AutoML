package com.bdilab.automl.common.config;

import com.bdilab.automl.common.websocket.LogWebSocketHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

import javax.annotation.Resource;

@Configuration
@EnableWebSocket
@CrossOrigin("/api/v1/automl/inference-service/logs")
public class WebSocketConfig implements WebSocketConfigurer {
    @Resource
    private LogWebSocketHandler logWebSocketHandler;
    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(logWebSocketHandler, "/api/v1/automl/inference-service/logs");
    }
}