package com.bdilab.automl.service.impl;

import com.bdilab.automl.common.config.MonitorConfig;
import com.bdilab.automl.common.config.PathConfig;
import com.bdilab.automl.dto.prometheus.Chart;
import com.bdilab.automl.dto.prometheus.MetricsCharts;
import com.bdilab.automl.dto.prometheus.Values;
import com.bdilab.automl.service.MonitorService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import javax.annotation.Resource;
import java.net.URI;
import java.net.URLEncoder;
import java.text.DecimalFormat;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.function.Consumer;


@Service
@Slf4j
public class MonitorServiceImpl implements MonitorService {
    private static ExecutorService executorService = Executors.newFixedThreadPool(10);
    @Resource
    private  MonitorConfig monitorConfig;
    @Resource
    private RestTemplate restTemplate;
    @Resource
    private ObjectMapper objectMapper;
    @Resource
    private PathConfig pathConfig;

    @Value("${prometheus.server.ip}")
    private String prometheusServerIp;

    @Value("${prometheus.server.port}")
    private Integer prometheusServerPort;

    private String queryPrometheus(String queryStatement){
        try{
            LocalDateTime end = LocalDateTime.ofInstant(Instant.now(), ZoneId.systemDefault());
            LocalDateTime start = end.minusMinutes(30);
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
            String url = "http://" + prometheusServerIp + ":" + prometheusServerPort + "/api/v1/query_range?query=" + URLEncoder.encode(queryStatement,"UTF-8").replaceAll("\\+", "%20") + "&start=" + formatter.format(start) + "&end=" + formatter.format(end) + "&step=3";
            URI uri = URI.create(url);
            ResponseEntity<String> prometheusResponse = restTemplate.getForEntity(uri, String.class);
            return prometheusResponse.getBody();
        } catch (Exception e){
            log.error(String.format("请求Prometheus获取指标异常, 具体原因: %s", e));
        }
        return null;
    }

    private Values getValues(String jsonString, String metricType) throws JsonProcessingException {
        JsonNode rootNode = objectMapper.readTree(jsonString);
        // 获取 result 字段的值
        JsonNode resultNode = rootNode.get("data").get("result");

        ArrayList<List<Object>> values = new ArrayList<>();
        // 遍历 result 数组中的元素
        for (JsonNode node : resultNode) {
            // 默认只有一组
            // 获取 values 字段的值
            JsonNode valuesNode = node.get("values");
            for (JsonNode valueNode : valuesNode) {
                ArrayList<Object> valueList = new ArrayList<>();
                List value = objectMapper.treeToValue(valueNode, List.class);
                // 转换时间戳格式
                Double timestampDouble = (Double) value.get(0);
                long timestampLong = timestampDouble.longValue();
                Instant instant = Instant.ofEpochSecond(timestampLong);
                LocalDateTime dateTime = LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
                String formattedDateTime = dateTime.format(formatter);
                valueList.add(formattedDateTime);
                // 转换指标值格式
                if (metricType.equals("cpuUsage")){
                    // 1m = 1/1000CPU
                    double cpuUsageM = Double.parseDouble(value.get(1).toString()) * 1000;
                    // 创建 DecimalFormat 对象，并设置保留四位小数的格式
                    DecimalFormat df = new DecimalFormat("#.####");
                    // 格式化数字
                    String formattedCpuUsage = df.format(cpuUsageM);
                    log.info(formattedCpuUsage);
                    valueList.add(formattedCpuUsage);
                } else if (metricType.equals("memoryRss")) {
                    // 转换内存单位 byte -> Mi
                    int memoryRssMi = Integer.parseInt(value.get(1).toString()) / (1024 * 1024);
                    valueList.add(memoryRssMi);
                } else {
                    log.error(String.format("不支持指标类型：%s", metricType));
                }
                values.add(valueList);
            }
            return new Values(values);
        }

        // TODO 构造兜底数据
        for (int i = 0; i < 50; i++) {
            ArrayList<Object> valueList = new ArrayList<>();
            long currentTimestamp = Instant.now().toEpochMilli();
            Instant instant = Instant.ofEpochSecond(currentTimestamp);
            LocalDateTime dateTime = LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String formattedDateTime = dateTime.format(formatter);
            valueList.add(formattedDateTime);

            // 转换指标值格式
            if (metricType.equals("cpuUsage")){
                // 生成一个0.0到8.0之间的随机double数
                double cpuUsageM = Math.random() * 4;
                // 创建 DecimalFormat 对象，并设置保留四位小数的格式
                DecimalFormat df = new DecimalFormat("#.####");
                // 格式化数字
                String formattedCpuUsage = df.format(cpuUsageM);
                valueList.add(formattedCpuUsage);
            } else if (metricType.equals("memoryRss")) {
                Random random = new Random();
                int min = 200;
                int max = 300;
                int memoryRssMi = random.nextInt(max - min + 1) + min;
                valueList.add(memoryRssMi);
            } else {
                log.error(String.format("不支持指标类型：%s", metricType));
            }
            values.add(valueList);
        }
        return new Values(values);
    }

    private Chart getChart(Values values) {
        Chart chart = new Chart();
        ArrayList<Object> xAxis = new ArrayList<>();
        ArrayList<Object> yAxis = new ArrayList<>();
        values.getValue().forEach(objects -> {
            xAxis.add(objects.get(0));
            yAxis.add(objects.get(1));
        });
        chart.setXAxis(xAxis);
        chart.setYAxis(yAxis);
        return chart;
    }

    @Override
    public MetricsCharts getResourceUsageInfo(String namespace, String serviceName) throws Exception {
        CompletableFuture<String> cpuUsageInfoFuture = CompletableFuture.supplyAsync(
                () -> queryPrometheus(String.format(monitorConfig.getCpuMetricStatement(), namespace, serviceName, serviceName)),
                executorService
        );
        CompletableFuture<String> memoryRssInfoFuture = CompletableFuture.supplyAsync(
                () -> queryPrometheus(String.format(monitorConfig.getMemoryMetricStatement(), namespace, serviceName, serviceName)),
                executorService
        );
        MetricsCharts metricsCharts = new MetricsCharts();
        try {
            String cpuUsageInfoJsonString = cpuUsageInfoFuture.get(10, TimeUnit.SECONDS);
            Values cpuUsage = getValues(cpuUsageInfoJsonString, "cpuUsage");
            log.info(String.format("CPU Usage Metrics: %s", cpuUsageInfoJsonString));
            Chart cpuUsageChart = getChart(cpuUsage);

            String memoryRssInfoJsonString = memoryRssInfoFuture.get(10, TimeUnit.SECONDS);
            Values memoryRss = getValues(memoryRssInfoJsonString, "memoryRss");
            Chart memoryRssChart = getChart(memoryRss);

            metricsCharts.setCpuUsage(cpuUsageChart);
            metricsCharts.setMemoryRss(memoryRssChart);
        } catch (Exception e) {
            metricsCharts.setCpuUsage(new Chart());
            metricsCharts.setMemoryRss(new Chart());
            log.error(e.getMessage());
        }
//        Chart cpuUsageChart = getChart(cpuUsage);
//
//        String memoryRssInfoJsonString = memoryRssInfoFuture.get(10, TimeUnit.SECONDS);
//        Values memoryRss = getValues(memoryRssInfoJsonString, "memoryRss");
//        Chart memoryRssChart = getChart(memoryRss);
//
//        metricsCharts.setCpuUsage(cpuUsageChart);
//        metricsCharts.setMemoryRss(memoryRssChart);
        return metricsCharts;
    }

    @Override
    public String getGrafanaUrl() {
        return pathConfig.getGrafanaUrl();
    }
}
