''' Stack Widget with animation '''

from PyQt6 import QtWidgets, QtCore


class SlidingStackedWidget(QtWidgets.QStackedWidget):
    ''' Animated Stack Widget
        adapted from:
        https://github.com/ThePBone/SlidingStackedWidget
    '''
    def __init__(self, parent=None):
        super(SlidingStackedWidget, self).__init__(parent)
        self.m_direction = QtCore.Qt.Orientation.Horizontal
        self.m_speed = 500
        self.m_animationType = QtCore.QEasingCurve.Type.InCurve
        self.m_now = 0
        self.m_next = 0
        self.m_wrap = False
        self.m_pnow = QtCore.QPoint(0, 0)
        self.m_active = False

    def setDirection(self, direction):
        self.m_direction = direction

    def setSpeed(self, speed):
        self.m_speed = speed

    def setAnimation(self, animation_type):
        self.m_animationType = animation_type

    def setWrap(self, wrap):
        self.m_wrap = wrap

    def slideInLeft(self, idx):
        self.slideInWgt(self.widget(idx), 'left')

    def slideInRight(self, idx):
        self.slideInWgt(self.widget(idx), 'right')

    def slideInWgt(self, new_widget, direction='left'):
        if self.m_active:
            return

        self.m_active = True

        _now = self.currentIndex()
        _next = self.indexOf(new_widget)

        if _now == _next:
            self.m_active = False
            return

        offset_X, offset_Y = self.frameRect().width(), self.frameRect().height()
        self.widget(_next).setGeometry(self.frameRect())

        if not self.m_direction == QtCore.Qt.Orientation.Horizontal:
            if direction == 'left':
                offset_X, offset_Y = 0, -offset_Y
            else:
                offset_X = 0
        else:
            if direction == 'left':
                offset_X, offset_Y = -offset_X, 0
            else:
                offset_Y = 0

        page_next = self.widget(_next).pos()
        pnow = self.widget(_now).pos()
        self.m_pnow = pnow

        offset = QtCore.QPoint(offset_X, offset_Y)
        self.widget(_next).move(page_next - offset)
        self.widget(_next).show()
        self.widget(_next).raise_()

        anim_group = QtCore.QParallelAnimationGroup(self)
        anim_group.finished.connect(self.animationDoneSlot)

        for index, start, end in zip(
            (_now, _next), (pnow, page_next - offset), (pnow + offset, page_next)
        ):
            animation = QtCore.QPropertyAnimation(self.widget(index), b'pos')
            animation.setEasingCurve(self.m_animationType)
            animation.setDuration(self.m_speed)
            animation.setStartValue(start)
            animation.setEndValue(end)
            anim_group.addAnimation(animation)

        self.m_next = _next
        self.m_now = _now
        self.m_active = True
        anim_group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    @QtCore.pyqtSlot()
    def animationDoneSlot(self):
        self.setCurrentIndex(self.m_next)
        self.widget(self.m_now).hide()
        self.widget(self.m_now).move(self.m_pnow)
        self.m_active = False
