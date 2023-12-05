import logging

class MyLogger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        formatter = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(formatter) #设置屏幕上显示的格式

        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)

        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(fh) #把对象加到logger里

if __name__ == '__main__':
    mylog = MyLogger('main.log',level='debug')
    mylog.logger.debug('debug')
    mylog.logger.info('info')
    mylog.logger.warning('警告')
    mylog.logger.error('报错')
    mylog.logger.critical('严重')
