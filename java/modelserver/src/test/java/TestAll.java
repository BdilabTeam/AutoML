import com.bdilab.automl.dto.prometheus.Values;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.NdArraySequence;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.types.TUint8;


import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

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

    @Test
    public void timestamp() {
        // 获取当前时间戳
        long currentTimestamp = Instant.now().toEpochMilli();
        System.out.println("当前时间戳: " + currentTimestamp);

        // 获取过去半天（12小时）的时间戳
        long halfDayAgoTimestamp = LocalDateTime.now()
                .minusHours(2)
                .atZone(ZoneId.systemDefault())
                .toInstant()
                .toEpochMilli();
        System.out.println("过去半天时间戳: " + halfDayAgoTimestamp);
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
        String imageFile = "D:\\word\\OneDrive\\Desktop\\image-classification\\angular_leaf_spot\\angular_leaf_spot_val.0.jpg";
        // 读取图像文件
        BufferedImage image = ImageIO.read(new File(imageFile));

        // 调整图像大小到256x256
        Image tmp = image.getScaledInstance(256, 256, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(256, 256, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        // 将调整大小后的图像转换为Tensor
        byte[] imageBytes = new byte[256 * 256 * 3];
        int index = 0;
        for (int y = 0; y < 256; y++) {
            for (int x = 0; x < 256; x++) {
                int pixel = resizedImage.getRGB(x, y);
                imageBytes[index++] = (byte) ((pixel >> 16) & 0xff); // Red
                imageBytes[index++] = (byte) ((pixel >> 8) & 0xff);  // Green
                imageBytes[index++] = (byte) (pixel & 0xff);         // Blue
            }
        }

        // 创建Tensor
        ByteNdArray byteNdArray = NdArrays.wrap(Shape.of(256, 256, 3), DataBuffers.of(imageBytes));

        // 从ByteNdArray中提取数据到Java多维数组
        byte[][][] javaArray = new byte[256][256][3];

        // 填充Java多维数组
        for (int h = 0; h < byteNdArray.shape().size(0); h++) { // Height dimension
            for (int w = 0; w < byteNdArray.shape().size(1); w++) { // Width dimension
                for (int c = 0; c < byteNdArray.shape().size(2); c++) { // Channels dimension
                    javaArray[h][w][c] = byteNdArray.getByte(h, w, c);
                }
            }
        }
        System.out.println(javaArray.length);
        System.out.println(javaArray[0].length);
        System.out.println(javaArray[0][0].length);


//        Tensor tensor = TUint8.tensorOf(byteNdArray);
//        System.out.println(byteNdArray.size());
//        System.out.println(byteNdArray.shape());
//        NdArraySequence<ByteNdArray> scalars = byteNdArray.scalars();
//        scalars.forEach(new Consumer<ByteNdArray>() {
//            @Override
//            public void accept(ByteNdArray byteNdArray) {
//                System.out.println(byteNdArray.getByte());
//            }
//        });
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
