from typing import Union, Any

from ultralytics import YOLO

from .configuration_yolov8 import YoloV8TrainerConfig
from ...utils import TaskType
from ...utils.trainer_utils import BaseTrainer, Trial, TrialsTracker, BestModelTracker, TrainerTracker, ConfigTracker


class AKBaseTrainerTracker(TrainerTracker):
    pass
    
class YoloV8MainTrainer:
    def __init__(
        self,
        config: YoloV8TrainerConfig,
        **kwargs
    ):
        # Step1: 实例化Model
        auto_model_params = {}

        if config.task_type == TaskType.IMAGE_CLASSIFICATION.value:
            auto_model_params["task"] = "classify"
        else:
            raise ValueError(f"`Task type` must be `{TaskType.IMAGE_CLASSIFICATION.value}`")
        
        if not (pretrained_model := config.pretrained_model):
            raise ValueError("Invalid model path.")
        auto_model_params["model"] = pretrained_model
        
        self._auto_model = YOLO(**auto_model_params)
        
        # Step2: 准备train()参数
        self._train_params = {
            "data": None,  # 这将在__call__方法中设置
            "epochs": config.epochs,
            "imgsz": config.imgsz,
            "batch": config.batch,
            "save": config.save,
            "save_period": config.save_period,
            "cache": config.cache,
            "device": config.device,
            "workers": config.workers,
            "project": config.tp_directory,
            "name": config.tp_project_name,
            "exist_ok": config.exist_ok,
            "pretrained": config.pretrained,
            "optimizer": config.optimizer,
            "verbose": config.verbose,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "single_cls": config.single_cls,
            "rect": config.rect,
            "cos_lr": config.cos_lr,
            "close_mosaic": config.close_mosaic,
            "resume": config.resume,
            "amp": config.amp,
            "fraction": config.fraction,
            "profile": config.profile,
            "freeze": config.freeze,
            "multi_scale": config.multi_scale,
            "overlap_mask": config.overlap_mask,
            "mask_ratio": config.mask_ratio,
            "dropout": config.dropout,
            "val": config.val,
            "split": config.split,
            "save_json": config.save_json,
            "save_hybrid": config.save_hybrid,
            "conf": config.conf,
            "iou": config.iou,
            "max_det": config.max_det,
            "half": config.half,
            "dnn": config.dnn,
            "plots": config.plots,
            "save_txt": config.save_txt,
            "save_conf": config.save_conf,
            "save_crop": config.save_crop,
            "show_labels": config.show_labels,
            "show_conf": config.show_conf,
            "show_boxes": config.show_boxes,
            "line_width": config.line_width,
            "agnostic_nms": config.agnostic_nms,
            "retina_masks": config.retina_masks,
            "box": config.boxes,
            "cls": config.cls,
            "dfl": config.dfl,
            "pose": config.pose,
            "kobj": config.kobj,
            "label_smoothing": config.label_smoothing,
            "nbs": config.nbs,
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            "degrees": config.degrees,
            "translate": config.translate,
            "scale": config.scale,
            "shear": config.shear,
            "perspective": config.perspective,
            "flipud": config.flipud,
            "fliplr": config.fliplr,
            "mosaic": config.mosaic,
            "mixup": config.mixup,
            "copy_paste": config.copy_paste,
            "auto_augment": config.auto_augment,
            "erasing": config.erasing,
            "crop_fraction": config.crop_fraction,
            "lr0": config.lr0,
            "lrf": config.lrf,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "warmup_epochs": config.warmup_epochs,
            "warmup_momentum": config.warmup_momentum,
            "warmup_bias_lr": config.warmup_bias_lr,
            "iterations": config.iterations
        }

        self._config = config
        
    def __call__(
        self,
        inputs: str,
        **kwargs
    ) -> AKBaseTrainerTracker:
        # 设置数据路径
        self._train_params["data"] = inputs

        # 执行训练
        # results = self._auto_model.train(**self._train_params)
        results = self._auto_model.tune(use_ray=bool, **self._train_params)

        # 处理训练结果
        best_model_tracker = BestModelTracker(
            history={
                'accuracy': list(list(results._experiment_analysis.get_best_trial(metric="metrics/accuracy_top1", mode="max").run_metadata.metric_n_steps.get('metrics/accuracy_top1').values())[-1]),
                'val_accuracy': list(list(results._experiment_analysis.get_best_trial(metric="metrics/accuracy_top1", mode="max").run_metadata.metric_n_steps.get('metrics/accuracy_top1').values())[-1]),
                'loss': list(list(results._experiment_analysis.get_best_trial(metric="metrics/accuracy_top1", mode="max").run_metadata.metric_n_steps.get('val/loss').values())[-1]),
                'val_loss': list(list(results._experiment_analysis.get_best_trial(metric="metrics/accuracy_top1", mode="max").run_metadata.metric_n_steps.get('val/loss').values())[-1])
            },
            hyperparameters=results.get_best_result(metric="metrics/accuracy_top1", mode="max").metrics.get('config'),
            model_graph_path=None  # YOLOv8 不提供模型图
        )

        # YOLOv8不使用trials，所以我们创建一个空的TrialsTracker
        trials_tracker = TrialsTracker(trials=[
            Trial(
                trial_id=trial.metrics.get('trial_id'),
                hyperparameters=trial.metrics.get('config'),
                metrics=trial.metrics,
                score=trial.metrics.get('metrics/accuracy_top1'),
                best_step=1,
                status='COMPLETED' if trial.metrics.get('done') else 'FAILED',
                model_graph_path=None,
                message=None
            ) for trial in results._results
        ])
        
        config_tracker = ConfigTracker(label2ids={}, id2labels={})

        return AKBaseTrainerTracker(
            best_model_tracker=best_model_tracker,
            trials_tracker=trials_tracker,
            config_tracker=config_tracker
        )

class YoloV8ForImageClassificationTrainer(BaseTrainer):
    def __init__(self, config: YoloV8TrainerConfig, **kwargs) -> None:
        if config.task_type != TaskType.IMAGE_CLASSIFICATION.value:
            raise ValueError(f"Task type '{config.task_type}' mismatch, expected task type is '{TaskType.IMAGE_CLASSIFICATION.value}'")
        super().__init__(config=config)

        self.trainer = YoloV8MainTrainer(config=config)
        
    def train(self, inputs: str, *args: Any, **kwds: Any):
        if not self.trainer:
            raise ValueError("No trainer is available")
        trainer_tracker = self.trainer(inputs=inputs)
        self.save_summary(trainer_tracker)