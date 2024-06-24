from MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication
import sys


def main():
    app = QApplication(sys.argv)

    # 新建主界面
    mainWidget = MainWidget()

    # 显示主界面
    mainWidget.show()

    # 消息循环
    exit(app.exec_())


if __name__ == '__main__':
    print("Start")
    main()