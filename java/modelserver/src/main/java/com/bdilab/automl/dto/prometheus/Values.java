package com.bdilab.automl.dto.prometheus;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Values {
    List<List<Object>> value;
}
