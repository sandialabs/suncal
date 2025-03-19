''' Styled buttons and button groups '''
from PyQt6 import QtWidgets, QtCore, QtGui

from .. import icons
from .. import gui_styles


class RoundButton(QtWidgets.QToolButton):
    ''' Round button '''
    # CSS stylesheet for nice round buttons
    _style_light = '''QToolButton {border: 1px solid #8f8f91; border-radius: 8px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #f6f7fa, stop: 1 #dadbde);}

                     QToolButton:pressed {border: 2px solid #8f8f91; border-radius: 8px; border-width: 2px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #dadbde, stop: 1 #f6f7fa);}

                     QToolButton:checked {border: 2px solid #8f8f91; border-radius: 8px; border-width: 1px;
                     background-color: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0, stop: 0 #dadbde, stop: 1 #7c7c7c);}
                     '''
    _style_dark = '''QToolButton {border: 1px solid #333333; border-radius: 8px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #666666, stop: 1 #444444);}

                     QToolButton:pressed {border: 2px solid #333333; border-radius: 8px; border-width: 2px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #444444, stop: 1 #666666);}

                     QToolButton:checked {border: 2px solid #333333; border-radius: 8px; border-width: 1px;
                     background-color: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0, stop: 0 #444444, stop: 1 #888888);}
                     '''

    def __init__(self, text=''):
        super().__init__(text=text)
        self.settheme()
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setFixedSize(18, 18)
        gui_styles.darkmode_signal().connect(self.settheme)

    def settheme(self):
        ''' Set style sheet '''
        if gui_styles.isdark():
            self.setStyleSheet(self._style_dark)
        else:
            self.setStyleSheet(self._style_light)


class PlusButton(RoundButton):
    ''' Round button with + icon '''
    def __init__(self):
        super().__init__()
        self.updateicon()
        gui_styles.darkmode_signal().connect(self.updateicon)

    def updateicon(self):
        self.setIcon(icons.icon('add'))


class MinusButton(RoundButton):
    ''' Round button with - icon '''
    def __init__(self):
        super().__init__()
        self.updateicon()
        gui_styles.darkmode_signal().connect(self.updateicon)

    def updateicon(self):
        self.setIcon(icons.icon('remove'))


class PlusMinusButton(QtWidgets.QWidget):
    ''' Widget containing plus/minus RoundButtons

        Args:
            label: Text to display before the buttons
            stretch: Add a Stretch after the buttons
    '''
    plusclicked = QtCore.pyqtSignal()
    minusclicked = QtCore.pyqtSignal()

    def __init__(self, label: str = '', stretch: bool = True, parent=None):
        super().__init__(parent)
        self.btnplus = PlusButton()
        self.btnminus = MinusButton()
        layout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(label)
        font = QtGui.QFont('Arial', 14)
        self.label.setFont(font)
        layout.addWidget(self.label)
        layout.addWidget(self.btnplus)
        layout.addWidget(self.btnminus)
        if stretch:
            layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.btnplus.clicked.connect(self.plusclicked)
        self.btnminus.clicked.connect(self.minusclicked)


class ToolButton(QtWidgets.QToolButton):
    ''' Button for selecting project components. ToolButton with text and icon. '''
    def __init__(self, text: str = '', iconname: str = None, parent=None):
        super().__init__(parent=parent)
        self.setIconSize(QtCore.QSize(32, 32))
        self.setFixedSize(84, 84)
        self.setText(text)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.iconname = iconname
        self.updateicon()
        gui_styles.darkmode_signal().connect(self.updateicon)

    def updateicon(self):
        if self.iconname:
            icon = icons.icon(self.iconname)
            self.setIcon(icon)


class SmallToolButton(ToolButton):
    ''' A smaller tool icon button with no text '''
    def __init__(self, iconname: str = None, parent=None):
        super().__init__(text='', iconname=iconname, parent=parent)
        self.setIconSize(QtCore.QSize(24, 24))
        self.setFixedSize(44, 44)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)


class RightButton(SmallToolButton):
    ''' Button with right arrow icon '''
    def __init__(self, parent=None):
        super().__init__('right', parent=parent)
        self.setFixedSize(30, 30)
        self.setToolTip('Next')


class LeftButton(SmallToolButton):
    ''' Button with right arrow icon '''
    def __init__(self, parent=None):
        super().__init__('left', parent=parent)
        self.setFixedSize(30, 30)
        self.setToolTip('Back')
