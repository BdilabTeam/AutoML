package com.bdilab.automl.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.web.multipart.MultipartFile;

import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Pattern;


@Data
@NoArgsConstructor
@AllArgsConstructor
public class InferenceFolderVO {
    @NotNull
    @Pattern(regexp="^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", message="包含不超过 63 个字符, 由小写字母、数字或 \"-\" 组成\n, 以字母或数字开头和结尾")
    private String endpointName;

    @NotNull
    @NotEmpty
    private MultipartFile[] files;
}

