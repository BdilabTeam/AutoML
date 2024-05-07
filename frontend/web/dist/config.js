window.g = {
  title: "垃圾分类系统",
  baseUrl: "",
  // wsUrl: "ws://60.204.186.96:31185/api/v1/experiment/job/logs",
  //实验websocket地址
  wsUrl: "ws://60.204.186.96:31185/api/v1/experiment/job/logs", 
  // 推理websocket地址
  wsUrl1: "ws://124.70.188.119:32081/api/v1/automl/inference-service/logs", 
  heartInterval: 30000,
  isRem: true,
  openCancelRequest: true,
  maxRecieveCount: 5000, // 调试接收区最大条数
  isDev: true, // 是否为开发平台
  intervalTime: 5000,
};
