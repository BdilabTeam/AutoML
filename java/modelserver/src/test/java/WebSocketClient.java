import com.google.gson.Gson;

import javax.websocket.*;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

@ClientEndpoint
public class WebSocketClient {

    private static CountDownLatch latch;
    private static Session session;

    public static void main(String[] args) throws Exception {
        try {
            latch = new CountDownLatch(1);
            WebSocketContainer container = ContainerProvider.getWebSocketContainer();
//            String endpointName = "image";
//            String uri = "ws://124.70.188.119:32081/api/v1/automl/inference-service/logs?endpointName=" + endpointName;
//            String uri = "ws://124.70.188.119:32081/api/v1/automl/inference-service/logs?endpointName=image";
//            String uri = "ws://124.70.188.119:32081/api/v1/automl/inference-service/logs?endpointName=test";
//            String uri = "ws://60.204.186.96:31185/api/v1/experiment/job/logs?experiment_job_name=test";
//            container.connectToServer(WebSocketClient.class, new URI(uri));

            String uri = "ws://127.0.0.1:7860/api/v1/chat/31f3c023-ff16-4a97-8b6e-2fc3ed3865e0?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2MTA0YWVjZC03NjU3LTRmNzUtOWI4YS05YjJmMjc3MTNhNjAiLCJleHAiOjE3NDc5MDM0NjN9.WMib_ehZ7pu9qczo1ypVEv7q3zlJg8fxgpdvdvyS33o";
            session = container.connectToServer(WebSocketClient.class, new URI(uri));
            latch.await(120, TimeUnit.SECONDS); // 等待连接建立，超时时间为 5 秒
            if (session != null && session.isOpen()) {

            }
        } catch (URISyntaxException | InterruptedException ex) {
            Logger.getLogger(WebSocketClient.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @OnOpen
    public void onOpen(Session session) throws IOException {
        WebSocketClient.session = session;
        System.out.println("WebSocket connection established.");
        latch.countDown(); // 连接建立成功，释放 latch
    }

    @OnMessage
    public void onMessage(String message) {
        System.out.println("Received message from server: " + message);
    }

    @OnClose
    public void onClose(Session session, CloseReason reason) {
        System.out.println("WebSocket connection closed: " + reason);
    }
}