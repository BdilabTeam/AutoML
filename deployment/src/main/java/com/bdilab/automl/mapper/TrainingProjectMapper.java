package com.bdilab.automl.mapper;

import com.baomidou.dynamic.datasource.annotation.DS;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.bdilab.automl.model.TrainingProject;

@DS("mysql")
public interface TrainingProjectMapper extends BaseMapper<TrainingProject> {
}
