#!/usr/bin/env bash

set -x
op=$1

# 获取当前脚本所在目录
DIR_PATH="$(cd "$(dirname "$0")"; pwd -P)"

# 后端镜像信息
BACKEND_IMAGE_FULL="registry.cn-hangzhou.aliyuncs.com/treasures/automl-deployment:v0.0.1"
BACKEND_IMAGE_NAME="registry.cn-hangzhou.aliyuncs.com/treasures/automl-deployment"
BACKEND_IMAGE_TAG="v0.0.1"
BACKEND_DOCKERFILE="${DIR_PATH}/Dockerfile"
BACKEND_CONTEXT="${DIR_PATH}"

# 后端jar文件路径
BACKEND_JAR="${DIR_PATH}/deployment-0.0.1-SNAPSHOT.jar"

# 部署yaml文件
AUTOML_DEPLOY_YAML="${DIR_PATH}/deployment-app.yaml"

# pv-pvc
AUTOML_PV_PVC_YAML="${DIR_PATH}/automl-pv-pvc.yaml"

is_file_exist() {
    # 此方法用于检验文件是否存在，若不存在，返回1; 存在，正常退出，返回0.
    # params:
    #   file_path: 文件路径
    if [ -z "$1" ]; then
        echo "need a param {file_path}"
        return 1
    fi

    file_path=$1
    if [ ! -f "${file_path}" ]; then
        echo "${file_path} does not exist."
        return 1
    fi
}

is_dir_exist() {
    # 此方法用于检验目录是否存在，若不存在，返回1; 存在，正常退出，返回0.
    # params:
    #   dir_path: 目录路径
    if [ -z "$1" ]; then
        echo "need a param {dir_path}"
        return 1
    fi

    dir_path=$1
    if [ ! -d "${dir_path}" ]; then
        echo "${dir_path} does not exist"
        return 1
    fi
}

is_image_exist() {
    # 此方法用于检验镜像是否存在，若不存在，返回1; 存在，正常退出，返回0.
    # params:
    #   image_full: 镜像全称
    if [ -z "$1" ]; then
        echo "need a param {image_full}"
        return 1
    fi

    image_full=$1
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q ${image_full}
    if [ $? -ne 0 ]; then
        echo "${image_full} dose not exist"
        return 1
    fi
}

op_yaml() {
    # 此方法用于操作.yaml后缀文件, 需要 2 个参数。
    # params:
    #   op_type: 操作类型。例如apply、delete等
    #   yaml_path: yaml文件路径。
    if [ -z "$1" ]; then
        echo "need a param {op_type}"
        return 1
    fi

    if [ -z "$2" ]; then
        echo "need a param {yaml_path}"
        return 1
    fi

    op_type=$1
    yaml_path=$2

    is_file_exist ${yaml_path}
    if [ $? -ne 0 ]; then
        return 1
    fi

    case ${op_type} in
        apply)
            echo "applying ${yaml_path}"
            kubectl apply -f ${yaml_path}
            if [ $? -ne 0 ]; then
                echo "fail to apply ${yaml_path}"
                return 1
            fi
            echo "apply ${yaml_path} succseefully!"
            ;;
        delete)
            echo "deleting ${yaml_path}"
            kubectl delete -f ${yaml_path}
            if [ $? -ne 0 ]; then
                echo "fail to delete ${yaml_path}"
                return 1
            fi
            echo "delete ${yaml_path} successfully!"
            ;;
        *)  # 后续扩展operations
            echo "${op_type} is not support"
            ;;
    esac
}

op_iamge() {
    # 此方法用于操作docker镜像，需要四个参数。
    # params:
    #   op_type: 操作类型。例如build、rmi等;
    #   image_full: 镜像全称（镜像name：镜像tag）;
    #   dokerfile_path: dockerfile路径;
    #   context_path: docker build命令构建上下文路径。
    if [ -z "$1" ]; then
        echo "need a param {op_type}"
        return 1
    fi

    if [ -z "$2" ]; then
        echo "need a param {image_full}"
        return 1
    fi

    context_path="."
    if [ ! -z "$4" ]; then
        context_path=$4
    fi

    op_type=$1
    image_full=$2
    # 从字符串的结束处删除最短匹配的子串, 此处删除镜像tag。例如image:tag -> image
    # image_name=${image_full%:*}

    case ${op_type} in
        build)
            is_image_exist ${image_full}
            if [ $? -eq 0 ]; then
                return 1
                # 镜像存在，可以考虑删除原镜像
                # docker rmi ${image_full}
            fi

            if [ -z "$3" ]; then
                echo "need a param {dokerfile_path}"
                return 1
            fi
            dockerfile_path=$3
            is_file_exist ${dockerfile_path}
            if [ $? -ne 0 ]; then
                return 1
            fi

            echo "building ${image_full}"
            docker build -t "${image_full}" -f "${dockerfile_path}" "${context_path}"
            if [ $? -ne 0 ]; then
                echo "faid to build ${image_full}"
                return 1
            fi
            echo "build successfully!"
            ;;
        rmi)
            echo "deleting ${image_full}"
            is_image_exist ${image_full}
            if [ $? -ne 0 ]; then
                return 1
            fi
            docker rmi ${image_full}
            if [ $? -ne 0 ]; then
                echo "fail to rmi ${image_full}"
                return 1
            fi
            echo "delete ${image_full} successfully!"
            ;;
        *)  # 后续扩展operations
            echo "${op_type} is not support"
            ;;
    esac
}


# 上线Automl-deployment
up_automl_deployment() {
    # 构建后端服务镜像
    op_iamge build ${BACKEND_IMAGE_FULL} ${BACKEND_DOCKERFILE}  ${BACKEND_CONTEXT}
    # apply 后端部署yaml文件
    op_yaml apply ${AUTOML_DEPLOY_YAML}
    # apply pv-pvc
    op_yaml apply ${AUTOML_PV_PVC_YAML}
}

# 下线AI平台
down_automl_deployment() {
    # delete pv-pvc
#    op_yaml delete ${AUTOML_PV_PVC_YAML}
    # delete 后端部署yaml文件
    op_yaml delete ${AUTOML_DEPLOY_YAML}
    sleep 2
    # 删除后端服务镜像
    op_iamge rmi ${BACKEND_IMAGE_FULL}
}

case "$op" in
up)
    echo "Deploying AutoML-deployment server"
    up_automl_deployment
    ;;
down)
    echo "Closing AutoML-deployment server"
    down_automl_deployment
    ;;
*)
    echo "${op} dees not support"
    ;;
esac
