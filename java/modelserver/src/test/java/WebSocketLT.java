import javax.websocket.*;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

@ClientEndpoint
public class WebSocketLT {

    private static CountDownLatch latch;
    private static Session session;

    public static void main(String[] args) throws Exception {
        try {
            latch = new CountDownLatch(1);
            WebSocketContainer container = ContainerProvider.getWebSocketContainer();
//            String uri = "ws://127.0.0.1:7860/api/v1/chat/9a901515-92c5-4dac-becd-772dc9637404?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyYTdkODk2MC1hMDM2LTQ0YTUtYjE5Ny1jYjdlZjVhZGUwY2YiLCJleHAiOjE3MTcxNjc5MDl9.9XDuwZhN-viZ_fDHD2-gPUxroIOWWK-bzeDcK3Zq3XE";
//            String uri = "ws://172.22.102.61:7860/api/v1/chat/85c767f3-3f85-4c7d-8faa-73ce6dfa2469?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyN2JlYmE3NS1kMjY5LTQwMWQtYjRlOS1kMWQzOWEwNzczNjAiLCJleHAiOjE3MTgwOTY3MzZ9.OWH64i10YkpwZ2CQfPyyfS5mS__hVCJbGD79zGuy_S0";
            String uri = "ws://10.8.0.6:7860/api/v1/chat/98f57b82-668d-4e5a-a4de-c919002e7caf?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjZjY1OTk4Ni02MWNmLTQ5ODctOWQ4YS01NTI1ZGNhNGM0OTQiLCJleHAiOjE3MTg4NzExNTJ9.WjN-GVeEWKPXJuhHF2bXnnae0Oxei2jYDCMYvdht7JM";
            session = container.connectToServer(WebSocketLT.class, new URI(uri));
            latch.await(120, TimeUnit.SECONDS); // 等待连接建立，超时时间为 5 秒

            try {
                // 让线程睡眠3秒（3000毫秒）
                Thread.sleep(30000);
            } catch (InterruptedException e) {
                // 捕获异常，如果线程在睡眠过程中被中断
                e.printStackTrace();
            }
        } catch (URISyntaxException | InterruptedException ex) {
            Logger.getLogger(WebSocketLT.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @OnOpen
    public void onOpen(Session session) throws IOException {
        WebSocketLT.session = session;
        System.out.println("WebSocket connection established.");

        // 消息数据结构 {"inputs": {"input": ""}}
        String text3 = "{\n" +
                "        \"inputs\": {\n" +
                "            \"input\": \"介绍太阳黑子\"\n" +
                "        }\n" +
                "    }";
//        演练主题：安全攻防、类型登录凭据(用户名和密码)、发送者：zz、接收者：zz
//        String text3 = "{\n" +
//                "    \"inputs\": {\n" +
//                "        \"input\": \"安全攻防演练\",\n" +
//                "        \"chat_history\": \"[\\\"介绍普罗米休斯\\\"]\"\n" +
//                "    }\n" +
//                "}";
        session.getBasicRemote().sendText(text3);
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