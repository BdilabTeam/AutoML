package com.bdilab.automl.common.websocket;

import com.bdilab.automl.service.impl.ExperimentServiceImpl;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import javax.annotation.Resource;

@Component
@Slf4j
public class LogWebSocketHandler extends TextWebSocketHandler {
    @Resource
    private ExperimentServiceImpl experimentService;

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        // 处理收到的消息
        String payload = message.getPayload();
        // 假设直接将消息返回
        session.sendMessage(new TextMessage("收到消息：" + payload));
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        // 获取连接URI中的参数
        String query = session.getUri().getQuery(); // 这里假设参数直接在URI中以查询字符串的形式传递，例如 ws://localhost:8080/my-websocket?param=value
        System.out.println("连接参数：" + query);
        String endpointName = query.split("=")[1];
        // 在这里处理参数，例如记录日志或执行特定逻辑
        try {
            experimentService.logs(session, endpointName);
        } catch (Exception e) {
            log.error(String.format("Failed to call experimentService.logs(), for a specific reason: %s", e));
        }
    }
}
