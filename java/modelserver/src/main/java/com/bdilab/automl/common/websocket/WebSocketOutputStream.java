package com.bdilab.automl.common.websocket;

import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class WebSocketOutputStream extends OutputStream {
    private WebSocketSession webSocketSession;

    public WebSocketOutputStream(WebSocketSession webSocketSession) {
        this.webSocketSession = webSocketSession;
    }

    @Override
    public void write(int b) throws IOException {

    }

    @Override
    public void write(byte b[], int off, int len) throws IOException {
        if (b == null || b.length == 0 || len == 0 || ((off < 0) || (off > b.length) || (len < 0) ||
                ((off + len) > b.length) || ((off + len) < 0))) {
            return;
        }
        String response = new String(b, off, len, StandardCharsets.UTF_8);
        webSocketSession.sendMessage(new TextMessage(response));
    }
}
