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
<br/>
python GenerateWords.py --out_dir ./dataset --font_dir ./chinese_fonts --width 64 --height 64 --margin 4 --rotate 30 --rotate_step 1
<br/>
2、训练模型
<br/>
3、测试模型
<br/>
4、预测模型
<br/>
