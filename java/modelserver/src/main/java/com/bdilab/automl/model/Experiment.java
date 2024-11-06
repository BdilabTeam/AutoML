package com.bdilab.automl.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@ApiModel(description = "实验元数据")
@TableName("experiment")
public class Experiment {
    @ApiModelProperty(value = "表唯一ID, 自动自增")
    @TableId(type = IdType.AUTO)
    private Integer id;

    @ApiModelProperty(value = "实验名称")
    private String experimentName;

    @ApiModelProperty("任务类型")
    private String taskType;

    @ApiModelProperty("任务描述")
    private String taskDesc;

    @ApiModelProperty("模型类型")
    private String modelType;

    @ApiModelProperty("当前实验工作区目录")
    private String workspaceDir;
}
