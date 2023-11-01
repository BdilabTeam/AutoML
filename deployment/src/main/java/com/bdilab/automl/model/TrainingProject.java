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
@ApiModel(description = "训练项目信息表")
@TableName("training_project")
public class TrainingProject {
    @ApiModelProperty(value = "表唯一ID, 自动自增")
    @TableId(type = IdType.AUTO)
    private Integer id;

    @ApiModelProperty(value = "训练项目名称")
    private String name;

    @ApiModelProperty("任务类型")
    private String taskType;

    @ApiModelProperty("是否开启自动化训练, 包括自动选择模型等")
    private Boolean isAutomatic;

    @ApiModelProperty("预模型存储路径")
    private String modelNameOrPath;

    @ApiModelProperty("数据存储路径")
    private String dataNameOrPath;

    @ApiModelProperty("主机头")
    private String host;
}
