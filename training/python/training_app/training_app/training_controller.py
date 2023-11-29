from fastapi import (
    FastAPI,
    Body,
    Path,
    Cookie,
    Header,
    UploadFile,
    File,
    Form,
    status,
    Depends,
)

from fastapi.exceptions import (
    HTTPException,
    RequestValidationError
)

from fastapi.responses import (
    Response,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
    FileResponse
)
import uvicorn

from typing import Union, Set, List, Dict, Any
from pathlib import Path

import schemas

from database import (
    SessionLocal,
    Base,
    # AsyncSessionLocal
)
from sqlalchemy.orm import Session
import crud
from utils.task_models import TaskType

from training_operator_client import (
    TrainingOperatorClient
)

import subprocess

import json
from dataclasses import asdict

from utils import (
    logging
)

import shutil

logger = logging.get_logger(__name__)

# to_client = TrainingOperatorClient(config_file="/root/workspace/YJX/Env/yjx/app/public/config")
to_client = TrainingOperatorClient(config_file=f"{Path().cwd().__str__()}/public/config")


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()


def generate_training_project_base_path(training_project_id: int) -> str:
    return f"{Path().cwd().__str__()}/public/training_project_meta/{training_project_id}"


def generate_training_project_output_path(training_project_id: int) -> str:
    return f"{Path().cwd().__str__()}/public/training_project_meta/{training_project_id}/output"


def generate_training_project_data_path(training_project_id: int) -> str:
    return f"{Path().cwd().__str__()}/public/training_project_meta/{training_project_id}/data"

def generate_training_project_model_path(training_project_id: int) -> str:
    return f"{Path().cwd().__str__()}/public/training_project_meta/{training_project_id}/model"

# def generate_training_project_base_path(training_project_id: int) -> str:
#    return f"{Path().cwd().__str__()}/meta/training_project/{training_project_id}"


# def generate_training_project_output_path(training_project_id: int) -> str:
#    return f"{Path().cwd().__str__()}/meta/training_project/{training_project_id}/output"


# def generate_training_project_data_path(training_project_id: int) -> str:
#    return f"{Path().cwd().__str__()}/meta/training_project/{training_project_id}/data"


@app.post("/projects/add", description="创建项目, 返回项目ID")
def create_training_project(
        training_project: schemas.TrainingProjectCreate = Body(),
        db: Session = Depends(get_db)
) -> Any:
    if training_project.is_automatic:
        """模型选择"""
        pass
    else:
        # 获取模型列表
        # models = []
        # if training_project.selected_model not in models:
        # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"暂不支持训练{training_project.selected_model}模型")
        # 信息入库
        training_project = crud.create_training_project(db=db, training_project=training_project)
        # 生成存放训练项目元数据的目录
        # 训练项目根路径, 一个项目对应一个路径
        traininig_project_base_path = Path(generate_training_project_base_path(training_project_id=training_project.id))
        traininig_project_base_path.mkdir(exist_ok=True)
        # 训练结果存放路径
        traininig_project_output_path = Path(
            generate_training_project_output_path(training_project_id=training_project.id))
        traininig_project_output_path.mkdir(exist_ok=True)

        #创建模型存储路径
        traininig_project_model_path = Path(generate_training_project_model_path(training_project_id=training_project.id))
        traininig_project_model_path.mkdir(exist_ok=True)

        # 创建数据存储路径
        traininig_project_data_path = Path(
            generate_training_project_data_path(training_project_id=training_project.id))
        traininig_project_data_path.mkdir(exist_ok=True)

        return JSONResponse(content=json.dumps(asdict(training_project)))


@app.put("/project/update/{training_project_id}", description="修改项目, 返回后的项目")
def update_training_project(
        training_project_id: int = Path(),
        training_project: schemas.TrainingProjectCreate = Body(),
        db: Session = Depends(get_db)
) -> Any:
    try:
        training_project_old = crud.get_training_project(db=db, training_project_id=training_project_id)
        # training_project_old = db.merge(training_project_old)
        # db.refresh(training_project_old)
        training_project_update = crud.update_training_project(db=db, training_project=training_project,
                                                               db_training_project_update=training_project_old)
        return JSONResponse(content=json.dumps(asdict(training_project_update)))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"修改训练项目元数据失败, 具体原因:{e.args}")

@app.get("/project/get/{training_project_id}", description="获取项目元数据")
def update_training_project(
        training_project_id: int = Path(),
        db: Session = Depends(get_db)
) -> Any:
    try:
        training_project = crud.get_training_project(db=db, training_project_id=training_project_id)
        return JSONResponse(content=json.dumps(asdict(training_project)))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"获取训练项目元数据失败, 具体原因:{e.args}")

@app.delete("/project/delete/{training_project_id}", description="删除项目")
def delete_training_project(
        training_project_id: int = Path(),
        db: Session = Depends(get_db)
) -> Any:
    try:
        training_project = crud.delete_training_project(db=db, training_project_id=training_project_id)
        return JSONResponse(content=json.dumps(asdict(training_project)))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"删除训练项目元数据失败, 具体原因:{e.args}")

@app.post("/projects/run/{training_project_id}", description="开启训练")
def start_training(
        training_project_id: int = Path(),
        db: Session = Depends(get_db)
):
    # 获取训练项目元数据
    try:
        training_project = crud.get_training_project(db=db, training_project_id=training_project_id)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"获取训练项目元数据, 具体原因:{e.args}")

    training_project_name = training_project.name
    training_project_task_type = training_project.task_type
    model_name_or_path = training_project.model_name_or_path
    data_name_or_path = training_project.data_name_or_path
    # 打包image / 根据model/task type映射预打包image
    # try:
    #     subprocess.run(["sh", "op.sh", "up_training"], check=True, capture_output=True, text=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error: {e}")
    try:
        to_client.create_training_job(
            training_project_name=training_project_name,
            training_project_task_type = training_project_task_type,
            model_path=model_name_or_path,
            data_path=data_name_or_path,
            output_path=generate_training_project_output_path(training_project_id=training_project.id),
            # image_full="treasures/training:latest"
            image_full="registry.cn-hangzhou.aliyuncs.com/treasures/training-script-env:v0.0.3"
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"开启训练项目失败, 具体原因: {e.args}")
    else:
        return JSONResponse(content={"training_project_id": training_project.id})


@app.get("/projects/conditions/{training_project_id}", description="获取Job状态")
def get_training_job_conditions(
        training_project_id: int,
        db: Session = Depends(get_db)
):
    # 获取训练项目元数据
    try:
        training_project = crud.get_training_project(db=db, training_project_id=training_project_id)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"获取训练项目元数据失败, 具体原因:{e.args}")

    try:
        job_conditions = to_client.get_training_job_conditions(name=training_project.name, namespace="zauto")
        return JSONResponse(content={"job_conditions": job_conditions})
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"获取Training Job Conditions失败, 具体原因: {e.args}")


@app.get("/models", description="点击任务类型，响应相应类型任务的模型列表")
def get_models_by_task_type(task_type: str):
    # task = task_type + "Models"
    return  TaskType.task_type


@app.post("/data/{training_project_id}", description="上传文件夹数据或CSV文件")
def upload_data(
        training_project_id: int = Path(),
        # files: List[UploadFile] = File(...),
        files: List[UploadFile] = Form(...),
        db: Session = Depends(get_db)
):
    # 创建存放数据的目录
    data_path = Path(generate_training_project_data_path(training_project_id=training_project_id))
    data_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        path_parts = Path(file.filename).parts

        # Check the depth of the folder. The length should be less than or equal to 2.
        if len(path_parts) > 3:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' exceeds maximum folder depth of 2.")

        file_path = Path(data_path, *path_parts)
        file_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # 根据fk_project_id获取Projcet实例, 填充Project实例的data_file字段
    try:
        training_project = crud.get_training_project(db=db, training_project_id=training_project_id)
        training_project.data_name_or_path = data_path
        db.commit()
        return JSONResponse(content={"training_project_id": training_project_id})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"修改训练项目元数据失败, 具体原因:{e.args}")


@app.post("/deploy/{training_project_id}")
def deploy_model(
        training_project_id: int = Path(),
        db: Session = Depends(get_db)
):
    try:
        training_project = crud.get_training_project(db=db, training_project_id=training_project_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"修改训练项目元数据失败, 具体原因:{e.args}")
    output_path = generate_training_project_output_path(training_project_id=training_project.id)
    # 部署模型


if __name__ == "__main__":
    #  uvicorn.run("training_controller:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run("training_controller:app", host="0.0.0.0", port=32081, reload=True)
