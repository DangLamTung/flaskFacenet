from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap
class CustomWidget(QtWidgets.QWidget):
    def __init__(self, user, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.user = user
        self.setGeometry(QtCore.QRect(0, 0, 211, 41))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.UserImage = QtWidgets.QLabel(self)
        self.UserImage.setPixmap(QPixmap("./img.jpg"))
        # self.UserImage.setText("This is an Image")
        # self.UserImage.setObjectName("UserImage")
        self.horizontalLayout.addWidget(self.UserImage)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.UserName = QtWidgets.QLabel(self)
        self.UserName.setText(user)
        self.UserName.setObjectName("UserName")
        self.verticalLayout.addWidget(self.UserName)
        self.UserStatus = QtWidgets.QLabel(self)
        self.UserStatus.setText("I am available")
        self.UserStatus.setObjectName("UserStatus")
        self.verticalLayout.addWidget(self.UserStatus)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    listWidget = QtWidgets.QListWidget()
    item = QtWidgets.QListWidgetItem(listWidget)
    item_widget = CustomWidget("adagfs")
    listWidget.addItem(item)
    listWidget.setItemWidget(item, item_widget)
    item.setSizeHint(item_widget.sizeHint())
    listWidget.show()
    sys.exit(app.exec_())