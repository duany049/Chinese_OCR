# Chinese_OCR
### 项目结构：
1、ChineseRecognition.py
<br/>
使用12层卷积层神经网络
<br/>
2、ChineseRecognition_sample.py
<br/>
使用5层卷积层的神经网络
<br/>
3、GenerateWords.py
<br/>
生成字体图像的文件
<br/>
4、Segmentaion.py
<br/>
对待识别图像，进行单字切割的类
<br/>
5、chinese_labels
<br/>
字符集
<br/>
6、chinese_fonts
<br/>
存放字体的目录
<br/>
7、predict
<br/>
存放待预测图片和预测结果的目录
<br/>

### 运行项目步骤：
1、生成循环及测试数据
```
python GenerateWords.py --out_dir ./dataset --font_dir ./chinese_fonts
```
2、训练模型
使用含12层卷积层的神经网络来训练
```
python ChineseRecognition.py --mode=train
```
使用含5层卷积层的神经网络来训练
```
python ChineseRecognition_sample.py --mode=train
```
3、测试模型
```
python ChineseRecognition_sample.py --mode=test
```
4、预测模型
```
python ChineseRecognition_sample.py --mode=predict --predict_dir=./predict --to_predict_img=toPredict.png --predict_result=predict.result
```
### 项目预测的流程：
1、待预测图片toPredict.png是一篇文章的截图
<br/>
2、执行预测模式，先对进待预测图片进行单字切割，结果存成一个list数组，list数组中每个元素为待预测图片中一行文字的list
<br/>
3、对list中每个汉字进行识别，并且把结果输出到predict.result文本中
<br/>
### 在测试集上的正确率：
top 1 accuracy 0.999 top 5 accuracy 0.999

更多深度学习、机器学习、统计学习的内容可以观看我的博客
[段逍遥的博客](https://blog.csdn.net/u011070767)
