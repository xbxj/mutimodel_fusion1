# mutimodel_fusion1

# 实验五 多模态情感分析

给定配对的文本和图像，预测对应的情感标签：positive, neutral, negative。

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- numpy==1.23.4
- numpy==1.20.1
- pandas==1.2.4
- Pillow==9.2.0
- Pillow==8.2.0
- Pillow==10.0.0
- scikit_learn==0.24.1
- torch==2.0.0+cu117
- torchvision==0.15.1+cu117
- transformers==4.30.2

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- data # 训练所用的所有图片以及文本数据
|-- main.py # 主函数，包括数据处理、模型搭建以及训练预测
|-- train.txt # 数据的guid和对应的情感标签。
|-- test_without_label.txt # 数据的guid和空的情感标签。
|-- result # 3种情况的生成结果
|-- result_picture # 3种情况的预测过程截图
|-- requirements.txt # 
|-- README.md #
    
```

## Run code
1. 直接运行
```
python main.py
```
默认使用图像和文本数据训练；
2. 运行
```
python main.py --option 0
```
只使用图像数据训练，运行
```
python main.py --option 1
```
只使用文本数据训练。
