# HMETS

    一个简单的头戴式眼动追踪系统

## 目录结构

    HMETS                               # 头戴式眼动追踪系统
    ├── main.py                         # 主程序
    ├── lib                             # 函数
    │   ├── gaze_estimate.py            # 视线估计
    │   ├── gaze_estimate_polynomial.py # 视线估计-多项式
    │   ├── pupil_detect.py             # 瞳孔检测
    │   └── tracker_ui.py               # UI
    ├── data                            # 数据
    │   ├── displacement_parameter.npz  # 位移矩阵
    │   ├── mapping_matrix.npz          # 视线估计映射矩阵
    │   └── mapping_parameter.npy       # 视线估计映射参数
    ├── README.md                       # 自述文件
    └── LICENSE                         # 许可证

## 安装依赖

Windows:

```sh
pip install -r requirements.txt
```

## 使用指南

### 运行main.py

    运行前，先将main.py中的变量capA和capB修改为对应的摄像头编号

    capA是对眼相机，capB是视野相机

### 操作指南

    空格   # 暂停/继续采样
    c     # 切换标定模式
    f     # 切换帧率显示
    1     # 使用视线估计映射矩阵（默认）
    2     # 使用视线估计多项式参数
    3     # 同时使用视线估计映射矩阵和多项式参数

#### 标定模式

    回车        # 计算新的视线估计映射矩阵和多项式参数
    单击鼠标左键 # 增加标记
    鼠标左键拖拽 # 移动标记


标定要求

    需要9个瞳孔位置和对应的视线落点

滑动条可调节 

    1. 眼部区域的宽
    2. 眼部区域的高
    3. 过滤阀值
    4. 膨胀系数
    5. 腐蚀系数

## 许可证

    GPLv3 © ZTH
