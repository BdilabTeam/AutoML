package com.bdilab.automl.common.websocket;

import io.kubernetes.client.openapi.ApiCallback;
import io.kubernetes.client.openapi.ApiException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import java.util.List;
import java.util.Map;

@Slf4j
public class LogCallback implements ApiCallback<String> {
    private final WebSocketSession session;

    public LogCallback(WebSocketSession session) {
        this.session = session;
    }

    @Override
    public void onSuccess(String s, int i, Map<String, List<String>> map) {
        try {
            session.sendMessage(new TextMessage(s));
        } catch (Exception e) {
            log.error("Error sending WebSocket message: " + e.getMessage());
        }
    }
    @Override
    public void onFailure(ApiException e, int i, Map<String, List<String>> map) {
        log.error("LogCallback failed.");
    }
    @Override
    public void onUploadProgress(long bytesWritten, long contentLength, boolean done) {
        // 处理上传进度
    }
    @Override
    public void onDownloadProgress(long bytesRead, long contentLength, boolean done) {
        // 处理下载进度
    }

}
