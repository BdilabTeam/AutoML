#!/user/bin/env python
# -*- coding: UTF-8 -*-

"""
@author: liu wenjing
@create: 2021/11/3 20:06
"""
import paramiko

ip = '192.168.1.77'  # 服务器ip
port = 22  # 端口号
username = "root"  # 用户名
password = "123456"  # 密码


def uploadfiletoserver(local, remote):  # 上传文件到服务器.local是要上传文件的本地路径；remote是上传到服务器的路径
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port, username, password)

    sftp = ssh.open_sftp()
    sftp.put(local, remote)
    return remote

def downloadfilefromserver(local, remote):  # 下载文件到服务器.local是接受下载文件的本地路径；remote是下载文件在服务器的路径
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port, username, password)

    sftp = ssh.open_sftp()
    sftp.get(remote, local)
    return local

# def main():


# if __name__ == '__main__':
#     local = './DRF-test-20211103190123.json'
#     remote = '/27T/lwj/DRF-test-20211103190123.json'
#     uploadfiletoserver(local, remote)
