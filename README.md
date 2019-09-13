# RM19windmillDemo
## RoboMaster2019风车能量机关识别示例代码

使用方法：

下载后用qtcreator打开项目工程，编译运行即可。

默认是识别红色的风车如果要识别蓝色的风车把`#define RED`注释即可

```cpp
#define USE_CAMERA 用于使用摄像头
#define SAVE_VIDEO 保存视频
#define LEAF_IMG   保存扇叶
```

如果要保存扇叶图片需要事先建好图片要保存的文件夹否则不会保存