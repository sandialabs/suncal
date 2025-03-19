''' Tree/Table delegates '''
from PyQt6 import QtCore, QtGui, QtWidgets

from . import gui_math
from .gui_common import BlockedSignals


class SuncalDelegate(QtWidgets.QStyledItemDelegate):
    ''' Super-class delegate with all the user roles '''
    ROLE_ENTERED = QtCore.Qt.ItemDataRole.UserRole + 1    # Original, user-entered data
    ROLE_MATH_DISABLE = ROLE_ENTERED + 1
    ROLE_DISABLE = ROLE_MATH_DISABLE + 1  # Boolean, column editing is disabled
    ROLE_BUTTON_RECT = ROLE_DISABLE + 1
    ROLE_TRIGGER = ROLE_BUTTON_RECT + 1

    # MQA-specific roles
    ROLE_QUANTITY = QtCore.Qt.ItemDataRole.UserRole + 100
    ROLE_COMPONENT = ROLE_QUANTITY + 1
    ROLE_TRUEEOPR = ROLE_COMPONENT + 1
    ROLE_EQUIP_MODE = ROLE_BUTTON_RECT + 1
    ROLE_TYPEA = ROLE_EQUIP_MODE + 1
    ROLE_TYPEB = ROLE_TYPEA + 1
    ROLE_TOLERANCE = ROLE_TYPEB + 1
    ROLE_COEFF = ROLE_TOLERANCE + 1
    ROLE_PREDICT = ROLE_COEFF + 1

    def updateEditorGeometry(self, editor, option, index):
        ''' Set editor widget to same size as table/tree cell '''
        editor.setGeometry(option.rect)

    def set_displayrole(self, index, text):
        ''' Set the text to display without triggering an itemChange signal '''
        with BlockedSignals(index.model()):
            index.model().setData(index, text, QtCore.Qt.ItemDataRole.DisplayRole)

    def emit_change(self, index):
        ''' Hackish way to trigger an itemChanged signal by using
            setData and ROLE_TRIGGER. setData() does not emit itemChanged
            if the data is the same object!
        '''
        val = index.model().data(index, self.ROLE_TRIGGER)
        if val is None:
            val = False
        val = not val
        index.model().setData(index, val, self.ROLE_TRIGGER)


class PopupDelegate(SuncalDelegate):
    ''' Delegate with popup dialog for editor '''
    def updateEditorGeometry(self, editor, option, index):
        topleft = editor.parent().mapToGlobal(editor.parent().rect().topLeft())
        x = topleft.x() + option.rect.x()
        y = topleft.y() + option.rect.y() + editor.rect().y()
        editor.setGeometry(x, y, editor.width(), editor.height())


class NoEditDelegate(SuncalDelegate):
    ''' Delegate where editing is always disabled '''
    def createEditor(self, parent, option, index):
        ''' Don't show any editor '''
        return None


class EditDelegate(SuncalDelegate):
    ''' Delegate with editing enabled by default, but can be disabled by role '''
    def createEditor(self, parent, option, index):
        ''' Show editor if enabled '''
        disable = index.model().data(index, self.ROLE_DISABLE)
        if disable:
            return None
        return super().createEditor(parent, option, index)


class DropdownDelegate(SuncalDelegate):
    ''' Delegate with a dropdown menu on right side '''
    def dropdown_menu(self, event, mode, option, index):
        menu = QtWidgets.QMenu()
        menu.exec(event.globalPosition().toPoint())

    def editorEvent(self, event, model, option, index):
        ''' Call the dropdown menu if clicked on the arrow '''
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            buttonrect = index.model().data(index, self.ROLE_BUTTON_RECT)
            if buttonrect and buttonrect.contains(event.pos()):
                self.dropdown_menu(event, model, option, index)
                return True
            return False
        return False

    def paint(self, painter, option, index):
        ''' Draw the cell with dropdown arrow '''
        super().paint(painter, option, index)
        bsize = 15
        painter.save()
        font = QtGui.QFont(option.font)
        font.setPixelSize(13)
        painter.setFont(font)
        buttonrect = QtCore.QRect(
            option.rect.right() - bsize,
            option.rect.bottom() - option.rect.height()//2 - bsize//2,
            bsize, bsize
        )
        with BlockedSignals(index.model()):
            index.model().setData(index, buttonrect, self.ROLE_BUTTON_RECT)
        painter.drawText(buttonrect, QtCore.Qt.AlignmentFlag.AlignCenter, '▾')
        painter.restore()


class LatexDelegate(SuncalDelegate):
    ''' Delegate class assigned to editable table/tree items for displaying
        expressions rendered as math. This overrides the editor so that the original
        user-entered (not calculated) expression is displayed for editing.
        Math expressions are rendered as graphics instead of text when not in edit mode.
    '''
    def setEditorData(self, editor, index):
        ''' Restore user-entered text when editing starts '''
        if not index.model().data(index, self.ROLE_MATH_DISABLE):
            text = index.model().data(index, self.ROLE_ENTERED)
            if text is None:
                text = index.model().data(index, QtCore.Qt.ItemDataRole.DisplayRole)
            editor.setText(text)
        else:
            super().setEditorData(editor, index)

    def updateEditorGeometry(self, editor, option, index):
        ''' Put the editor widget in the original location, (instead of
            default behavior of shifting to compensate for pixmap). '''
        if not index.model().data(index, self.ROLE_MATH_DISABLE):
            editor.setGeometry(option.rect)
        else:
            super().updateEditorGeometry(editor, option, index)

    def setModelData(self, editor, model, index):
        ''' Save user-entered text to restore in edit mode later '''
        if not index.model().data(index, self.ROLE_MATH_DISABLE):
            model.blockSignals(True)  # Only signal on one setData
            olddata = index.model().data(index, self.ROLE_ENTERED)
            if editor.text() != olddata:  # Don't refresh unless necessary
                model.setData(index, editor.text(), self.ROLE_ENTERED)    # Save for later
                px = gui_math.pixmap_from_latex(editor.text())
                model.blockSignals(False)
                model.setData(index, px, QtCore.Qt.ItemDataRole.DecorationRole)
            model.blockSignals(False)
        else:
            super().setModelData(editor, model, index)

    def createEditor(self, parent, option, index):
        ''' Show the editor if enabled '''
        disable = index.model().data(index, self.ROLE_DISABLE)
        if disable:
            return None
        return super().createEditor(parent, option, index)
