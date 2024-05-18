window.g = {
  title: "自动化机器学习平台",
  baseUrl: "",
  // HWY
  wsUrl: "ws://60.204.186.96:31185/api/v1/experiment/job/logs",
  wsUrl1: "ws://124.70.188.119:32081/api/v1/automl/inference-service/logs", 
  //SG
  // wsUrl: "ws://10.8.104.110:31185/api/v1/experiment/job/logs", 
  // wsUrl1: "ws://10.8.104.100:32081/api/v1/automl/inference-service/logs", 
  heartInterval: 30000,
  isRem: true,
  openCancelRequest: true,
  maxRecieveCount: 5000, // 调试接收区最大条数
  isDev: true, // 是否为开发平台
  intervalTime: 5000,
};
