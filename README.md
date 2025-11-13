# LogicAnomaly: 面向图像逻辑异常的建模与检测方法研究

<div align="center">
    <img src="imgs/architecture.png" alt="项目架构图" width="80%">
</div>

## 项目简介

LogicAnomaly是一个基于深度学习的工业图像异常检测框架，专注于识别和定位工业产品中的逻辑异常。本项目基于SimpleNet架构，通过判别器网络学习正常样本的嵌入特征分布，能够有效检测出测试样本中的异常区域，适用于MVTec AD等工业异常检测数据集。

## 特性

- **高效的特征提取**：利用WideResNet50等预训练骨干网络提取图像深层特征
- **基于对抗学习的异常检测**：通过判别器区分正常样本和噪声样本的特征分布
- **多层特征融合**：同时利用多个网络层的特征以捕捉不同尺度的异常信息
- **像素级异常定位**：能够生成像素级的异常热力图，精确定位产品缺陷位置
- **支持多种工业产品**：适用于MVTec AD中的15类不同工业产品

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 支持 (推荐使用GPU加速)
- scikit-learn
- OpenCV
- tqdm
- pandas

## 数据集

本项目主要使用以下数据集进行测试和评估：

- **MVTec AD**：包含15个工业产品类别的异常检测数据集，每个类别包含正常样本和多种异常样本
- **BTAD**（可选）：另一个工业异常检测数据集
- **SDD/SDD2**（可选）：特殊场景异常检测数据集

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练和评估

使用提供的脚本一键运行训练和评估流程：

```bash
bash run.sh
```

该脚本将在MVTec AD数据集的所有15个类别上训练并评估模型。

### 自定义配置

您可以修改`run.sh`中的参数来自定义训练过程：

```bash
python3 main.py \
  --gpu <gpu_id> \
  --seed <random_seed> \
  --log_group <log_group_name> \
  --log_project <log_project_name> \
  --results_path <results_directory> \
  net \
  -b <backbone_name> \
  -le <layer_name1> \
  -le <layer_name2> \
  --pretrain_embed_dimension <dim> \
  --meta_epochs <epochs> \
  --gan_epochs <gan_epochs> \
  dataset \
  --batch_size <batch_size> \
  --resize <resize_size> \
  --imagesize <image_size> \
  -d <product_class1> -d <product_class2> ... \
  mvtec <data_path>
```

## 模型架构

该项目采用基于SimpleNet的架构，主要包括以下组件：

1. **特征提取器**：使用预训练的WideResNet50提取多层次特征
2. **特征预处理**：将不同层的特征统一维度并融合
3. **特征投影**：可选的特征降维/变换模块
4. **判别器**：学习区分正常样本和带噪声样本的分布
5. **异常评分**：根据判别器输出生成图像级和像素级异常分数

## 评估指标

本项目使用以下指标评估异常检测性能：

- **图像级AUROC**: 评估模型在图像级别的异常检测能力
- **像素级AUROC**: 评估模型在像素级别的异常定位能力
- **PRO分数**: Per-Region Overlap评分，评估异常区域的检测准确性

## 实验结果

项目在MVTec AD数据集上的表现达到了较好的水平。模型能够有效检测多种工业产品中的异常，包括纹理异常、结构异常和逻辑异常。

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{your-paper-reference,
  title={面向图像逻辑异常的建模与检测方法研究},
  author={Shi-Wei Zhou},
  year={2025}
}
```

## 参考文献

1. SimpleNet: A Simple Network for Image Anomaly Detection and Localization (CVPR 2023)
2. PatchCore: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022)
3. MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection (CVPR 2019)

## 许可证

本项目基于MIT许可证开源。
