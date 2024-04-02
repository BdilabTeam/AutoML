import com.bdilab.automl.dto.prometheus.Values;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;


import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class TestAll {
    @org.junit.Test
    public void string2Json() throws JsonProcessingException {
        String jsonStr = "{\"status\":\"success\",\"data\":{\"resultType\":\"matrix\",\"result\":[{\"metric\":{\"__name__\":\"node_namespace_pod_container:container_memory_rss\",\"container\":\"test\",\"endpoint\":\"https-metrics\",\"id\":\"/kubepods/burstable/pod21dccd4c-3dc6-4440-a390-4e27f31a701a/6857be177fe77975da47fec29f1f6b80d5bafff979366b65078419052b1d5a74\",\"image\":\"sha256:c77e0b20c132a81274d2bd96f091747b32bacd9e488ed0de137ea96286c3287b\",\"instance\":\"192.168.0.209:10250\",\"job\":\"kubelet\",\"metrics_path\":\"/metrics/cadvisor\",\"name\":\"k8s_test_test-00001-deployment-86b7bc79fb-mgfsd_zauto_21dccd4c-3dc6-4440-a390-4e27f31a701a_0\",\"namespace\":\"zauto\",\"node\":\"node1\",\"pod\":\"test-00001-deployment-86b7bc79fb-mgfsd\",\"service\":\"kubelet\"},\"values\":[[1711898510.476,\"269864960\"],[1711898540.476,\"269864960\"],[1711898570.476,\"269864960\"],[1711898600.476,\"269864960\"],[1711898630.476,\"269864960\"],[1711898660.476,\"269864960\"],[1711898690.476,\"269864960\"],[1711898720.476,\"269864960\"],[1711898750.476,\"269864960\"],[1711898780.476,\"269864960\"],[1711898810.476,\"269864960\"],[1711898840.476,\"269864960\"],[1711898870.476,\"269864960\"],[1711898900.476,\"269864960\"],[1711898930.476,\"269864960\"],[1711898960.476,\"269864960\"],[1711898990.476,\"269864960\"],[1711899020.476,\"269864960\"],[1711899050.476,\"269864960\"],[1711899080.476,\"269864960\"],[1711899110.476,\"269864960\"],[1711899140.476,\"269864960\"],[1711899170.476,\"269864960\"],[1711899200.476,\"269864960\"],[1711899230.476,\"269864960\"],[1711899260.476,\"269864960\"],[1711899290.476,\"269864960\"],[1711899320.476,\"269864960\"],[1711899350.476,\"269864960\"],[1711899380.476,\"269864960\"]]}]}}";
        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode rootNode = objectMapper.readTree(jsonStr);
        // 获取 result 字段的值
        JsonNode resultNode = rootNode.get("data").get("result");
        // 遍历 result 数组中的元素
        for (JsonNode node : resultNode) {
            // 获取 values 字段的值
            JsonNode valuesNode = node.get("values");
            Values values = new Values();
            for (JsonNode valueNode : valuesNode) {
                ArrayList<Object> list = new ArrayList<>();
                List value = objectMapper.treeToValue(valueNode, List.class);
                Double timestampDouble = (Double) value.get(0);
                long timestampLong = timestampDouble.longValue();
                Instant instant = Instant.ofEpochSecond(timestampLong);
                LocalDateTime dateTime = LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
                String formattedDateTime = dateTime.format(formatter);
                list.add(formattedDateTime);
                // 转换内存单位 byte -> Mi
                int memoryRssMi = Integer.parseInt(value.get(1).toString()) / (1024 * 1024);
                String cpu = "0.0015787246333350898";
                double cpuUsageM = Double.parseDouble(cpu) * 1000;
                System.out.println(cpuUsageM);
            }
        }
    }
    @org.junit.Test
    public void timestamp2Datetime() {
        long timestamp = 1711903526;
        // 将事件戳转换为Instant对象
        Instant instant = Instant.ofEpochSecond(timestamp);
        // 将Instant对象转换为LocalDateTime对象
        LocalDateTime dateTime = LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
        // 定义日期时间格式
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        // 将LocalDateTime对象格式化为标准时间字符串
        String formattedDateTime = dateTime.format(formatter);
        // 输出转换后的标准时间
        System.out.println("标准时间: " + formattedDateTime);
    }
    @org.junit.Test
    public void time(){
        LocalDateTime end = LocalDateTime.ofInstant(Instant.now(), ZoneId.systemDefault());
        LocalDateTime start = end.minusMinutes(15);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
        System.out.println("Start: " + formatter.format(start));
        System.out.println("End: " + end);
    }
    @Test
    public void imageConvertToTensor() throws Exception {
        String imagePath = "/Users/treasures_y/Documents/code/HG/AutoML/python/autotrain/autotrain/datasets/image-classification/angular_leaf_spot/angular_leaf_spot_val.0.jpg";
        BufferedImage image = ImageIO.read(new File(imagePath));
        // 图像预处理：将像素值缩放到 [0, 1] 范围内
        ByteBuffer buffer = normalizeImage(image);
        // 创建张量（Tensor）
        int width = image.getWidth();
        int height = image.getHeight();
        long[] shape = new long[]{1, 256, 256, 3}; // 图像的形状（batch size, height, width, channels）
//        Tensor tensor = Tensor.of(Float.class, Shape.of(shape), buffer); // 创建张量
        // 打印张量的形状和数据类型
//        System.out.println("Tensor shape: " + tensor.shape().toString());
    }
    private static ByteBuffer normalizeImage(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 3 * Float.BYTES);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                float red = ((rgb >> 16) & 0xFF) / 255.0f; // Red
                float green = ((rgb >> 8) & 0xFF) / 255.0f; // Green
                float blue = (rgb & 0xFF) / 255.0f; // Blue
                buffer.putFloat(red);
                buffer.putFloat(green);
                buffer.putFloat(blue);
            }
        }

        buffer.flip(); // 重置缓冲区的位置
        return buffer;
    }
}
