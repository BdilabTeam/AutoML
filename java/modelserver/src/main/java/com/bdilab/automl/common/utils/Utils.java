package com.bdilab.automl.common.utils;

import org.springframework.web.multipart.MultipartFile;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Utils {
    public static final String NAMESPACE = "zauto";
    public static final String AUTOKERAS_IMAGE = "registry.cn-hangzhou.aliyuncs.com/treasures/automl-autokeras-server:latest";
    public static final String YOLO_IMAGE = "registry.cn-hangzhou.aliyuncs.com/treasures/yolo-server:0.0.1";
    public static final String BEST_MODEL_FOLDER_NAME = "best_model";
    public static final String TP_PROJECT_NAME = "output";
    public static final String METADATA_DIR_IN_CONTAINER = "/metadata";
    public static final String PVC_NAME = "automl-metadata-pvc";

    public static String getExperimentWorkspaceDirInContainer(String experimentName) {
        return String.join("/", METADATA_DIR_IN_CONTAINER, experimentName);
    }
    public static String generateHost(String ksvcName) {
        return String.format("%s.%s.example.com", ksvcName, NAMESPACE);
    }
    public static String getBestModelDirInContainer(String experimentName) {
        return  String.join("/", getExperimentWorkspaceDirInContainer(experimentName), TP_PROJECT_NAME, BEST_MODEL_FOLDER_NAME);
    }

    public static String getYOLOBestModelDirInContainer(String experimentName) {
        return  String.join("/", getExperimentWorkspaceDirInContainer(experimentName), TP_PROJECT_NAME, "weights", "best.pt");
    }

    //文件的数据转换成可以推理的格式
    public static List<Object> FileToData(MultipartFile file){
        List<Object> data = null;

        try {
            String fileName = file.getOriginalFilename().toLowerCase();
            if (fileName.endsWith(".csv")) {
                data = csvConvertToTensor(file);
            }
            else  {
                data = imageConvertToTensor(file);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return data;
    }

    //csv文件转换成数组格式
    public static List<Object> csvConvertToTensor(MultipartFile csvFile) {
        String line = "";
        List<Object> allData = new ArrayList<>(); // 用于存储所有数据的大列表

        try (BufferedReader br = new BufferedReader(new InputStreamReader(csvFile.getInputStream()))) {
            br.readLine(); // 跳过第一行（列名）

            while ((line = br.readLine()) != null) {
                String[] values = line.split(","); // 假设CSV是用逗号分隔的
                List<Double> numberList = new ArrayList<>();

                for (String value : values) {
                    try {
                        double number = Double.parseDouble(value.trim()); // 将字符串转换为double
                        numberList.add(number); // 添加到当前行的列表中
                    } catch (NumberFormatException e) {
                        System.out.println("无法将字符串转换为数字: " + value);
                        // 处理不是数字的情况或决定如何处理异常
                    }
                }
                allData.add(numberList); // 将当前行添加到大列表中
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return allData;
    }

    public static List<Object> imageConvertToTensor(MultipartFile imageFile) throws Exception{
        // 读取图像文件
        BufferedImage image = ImageIO.read(imageFile.getInputStream());

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

        List<Object> outerList = new ArrayList<>();
        for (byte[][] innerArray1 : javaArray) {
            List<Object> innerList1 = new ArrayList<>();
            for (byte[] innerArray2 : innerArray1) {
                List<Byte> innerList2 = new ArrayList<>();
                for (byte b : innerArray2) {
                    innerList2.add(b);
                }
                innerList1.add(innerList2);
            }
            outerList.add(innerList1);
        }
        return outerList;
    }
}
