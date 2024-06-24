from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, QComboBox, QLabel, QSpinBox, QFileDialog,QTextEdit
from PyQt5.QtGui import QFont
from PaintBoard import PaintBoard
import numpy as np
# from model_core import LeNet5Custom
from model_core_win import LeNet5Custom


class MainWidget(QWidget):
    def __init__(self, Parent=None):
        super().__init__(Parent)

        # 先初始化数据，再初始化界面
        self.__InitData()
        self.__InitView()


    def __InitData(self):
        self.__paintBoard = PaintBoard(self)
        self.__colorList = QColor.colorNames() # 获取颜色列表(字符串类型)


    def __InitView(self):
        self.setFixedSize(896, 672)
        self.setWindowTitle("Handwritten Digit Recognizer")

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)

        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear) # 将按键按下信号与画板清空函数相关联
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_predict = QPushButton("人工智能预测")
        self.__btn_predict.setParent(self)  # 设置父对象为本界面
        self.__btn_predict.clicked.connect(lambda:self.predict())
        sub_layout.addWidget(self.__btn_predict)

        self.__text_out = QTextEdit(self)
        self.__text_out.setParent(self)
        self.__text_out.setObjectName("预测结果为：")
        self.__text_out.setReadOnly(True)
        text_out_Font = QFont()
        text_out_Font.setPointSize(20)
        self.__text_out.setFont(text_out_Font)
        sub_layout.addWidget(self.__text_out)

        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("保存作品")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__btn_Load = QPushButton("加载作品")
        self.__btn_Load.setParent(self)
        self.__btn_Load.clicked.connect(self.on_btn_Load_Clicked)
        sub_layout.addWidget(self.__btn_Load)

        self.__cbtn_Eraser = QCheckBox("  使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(50) # 规定粗细数值范围
        self.__spinBox_penThickness.setMinimum(1)
        self.__spinBox_penThickness.setValue(20)  # 默认粗细
        self.__spinBox_penThickness.setSingleStep(1)  # 最小变化值
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)  # 用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(self.on_PenColorChange)  # 关联下拉列表的当前索引变更信号与函数 on_PenColorChange
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)  # 将子布局加入主布局


    def __fillColorList(self, comboBox):
        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)


    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)


    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)


    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])


    def on_btn_Load_Clicked(self):
        loadPath = QFileDialog.getOpenFileName(self, 'Load Your Paint', '.\\', '*.png')
        print(loadPath)
        if loadPath[0] == "":
            print("Load cancel")
            return
        pixmap = QPixmap(loadPath[0])
        self.__paintBoard.SetContentFromQPixmap(pixmap)


    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式


    def Quit(self):
        self.close()


    def predict(self):
        savePath = './image_temp/test.png'
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)

        model = LeNet5Custom(dropout_rate=0.0)
        model.load_model('./model/lenet5_model_best_20240621_010754.h5')
        # lenet5_model_best_20240621_010754.h5 is best

        predictions = model.predict(savePath)
        print(predictions)
        predicted_class = np.argmax(predictions[0])

        # 假设 self.__text_out 是一个文本输出控件
        self.__text_out.setText(str(predicted_class))
        print(f'Predictor: {predicted_class}')
