const reconnectInterval = 5000; // 重连时间间隔
const maxReconnectCount = 3; // 重连时间间隔
let lockReconnect = false; // 防止重复连接

const heartCheck = {
  timeout: 30000, // 毫秒
  timeoutObj: null,
  reset: function () {
    clearInterval(this.timeoutObj);
    return this;
  },
  start: function (wsClass, heartObj) {
    const self = this;
    let count = 0;
    // console.log('ws-device', deviceObj.projectID, deviceObj.deviceID)
    wsClass.send(heartObj);
    if (self.timeoutObj) clearInterval(self.timeoutObj);
    self.timeoutObj = setInterval(() => {
      // if (count < 3) {
      if (wsClass.socket.readyState === 1) {
        wsClass.send(heartObj);

        console.info(`HeartBeat第${count + 1}次`);
      }
      count++;
      // } else {
      //   clearInterval(this.timeoutObj)
      //   count = 0
      //   if (ws.readyState === 0 && ws.readyState === 1) {
      //     ws.close()
      //   }
      // }
    }, self.timeout);
    console.info("ws-start-intervalID");
  },
};

class WebSocketClass {
  /**
   *
   * @param {path:string, onmessage:function} params
   */
  constructor(params) {
    //只有params这个参数必须卸载constructor方法里，其他的实例属性可以写在外面
    // 比如 socket = null
    this.socket = null;
    this.params = params;

    this.reconnectCount = 0; //websocket重连次数
    this.i = 0; //发送信息次数
    if ("WebSocket" in window) {
      this.init(params);
    } else {
      console.warn("error", "浏览器不支持，请换浏览器再试");
    }
  }

  init(params) {
    console.log("init方法开始");
    if (params.path) {
      this.path = params.path;
    } else {
      throw new Error("参数socket服务器地址path必须存在");
    }

    this.socket = new WebSocket(this.path);

    this.socket.onopen = () => {
      console.log("连接开启");
      this.reconnectCount = 0;
      heartCheck.timeout = this.params.heartInterval || 30000;
      heartCheck.reset().start(this, this.params.userInfo);
    };
    this.socket.onclose = (event) => {
      console.log(
        "file: WebsocketClass.js ~ line 41 ~ WebSocketClass ~ init ~ event",
        event
      );
      console.log("连接关闭");
      clearInterval(heartCheck.timeoutObj);
      console.log("ws-clear-intervalID");

      if (![3000].some((item) => item === event.code)) {
        this.reconnect();
      }
    };
    this.socket.onerror = () => {
      // clearInterval(heartCheck.timeoutObj)
      console.log("连接错误");
    };

    this.socket.onmessage = params.onmessage || this.getMessage;
  }

  getMessage(msg) {
    console.log("收到的消息", msg);
    return msg;
  }

  send(data) {
    let s = null;
    try {
      if (this.socket.readyState == 1) {
        this.i = 0;
        clearTimeout(s);
        this.socket.send(JSON.stringify(data));
      } else {
        if (this.i <= 10) {
          console.log(this.socket.readyState);
          ++this.i;
          // this.send(data)
          s = setTimeout(() => {
            this.send(data);
          }, 100 * this.i);
        } else {
          this.i = 0;
          clearTimeout(s);
        }
      }
    } catch (e) {
      console.warn("send", e);
    }
  }

  close() {
    console.log("调用关闭");
    clearTimeout(this.reconnectTimeout);
    this.socket.close(3000, "手动关闭");
  }

  reconnect() {
    //1.开始重连
    if (lockReconnect) return;

    lockReconnect = true;

    // 没连接上会一直重连，设置延迟避免请求过多
    if (this.reconnectCount < maxReconnectCount) {
      this.reconnectTimeout = setTimeout(() => {
        console.info(`正在重连第${this.reconnectCount + 1}次`);
        this.reconnectCount++;
        lockReconnect = false;
        this.init(this.params);
      }, reconnectInterval); // 这里设置重连间隔(ms)
    }
  }
}
