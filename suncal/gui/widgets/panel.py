''' Widget Panel '''
from contextlib import suppress
from PyQt6 import QtWidgets, QtCore

from .buttons import PlusMinusButton


class WidgetPanel(QtWidgets.QTreeWidget):
    ''' Tree widget for expanding/collapsing other widgets '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setHeaderHidden(True)
        self.setVerticalScrollMode(QtWidgets.QTreeView.ScrollMode.ScrollPerPixel)
        self.setColumnCount(1)
        self.itemExpanded.connect(self.wasexpanded)
        self.itemCollapsed.connect(self.wasexpanded)

    def add_widget(self, name, widget, buttons=False):
        ''' Add a widget to the tree at the end '''
        idx = self.invisibleRootItem().childCount()
        return self.insert_widget(name, widget, idx, buttons=buttons)

    def expand(self, name):
        ''' Expand the widget with the given name '''
        item = self.findItems(name, QtCore.Qt.MatchFlag.MatchExactly, 0)
        with suppress(IndexError):
            self.expandItem(item[0])
            self.wasexpanded(item[0])

        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if self.itemWidget(item, 0) and self.itemWidget(item, 0).label.text() == name:
                self.expandItem(item)
                self.wasexpanded(item)
                break

    def hide(self, name, hide=True):
        ''' Show or hide an item '''
        item = self.findItems(name, QtCore.Qt.MatchFlag.MatchExactly, 0)
        with suppress(IndexError):
            item[0].setHidden(hide)
            self.wasexpanded(item[0])

    def insert_widget(self, name, widget, idx, buttons=False):
        ''' Insert a widget into the tree '''
        item = QtWidgets.QTreeWidgetItem()
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)  # Enable, but not selectable/editable
        self.insertTopLevelItem(idx, item)
        if buttons:
            bwidget = PlusMinusButton(name)
            bwidget.btnplus.setVisible(False)
            bwidget.btnminus.setVisible(False)
            self.setItemWidget(item, 0, bwidget)
        else:
            bwidget = None
            item.setText(0, name)

        witem = QtWidgets.QTreeWidgetItem()
        witem.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        item.addChild(witem)
        self.setItemWidget(witem, 0, widget)
        return item, bwidget

    def fixSize(self):
        ''' Adjust the size of tree rows to fit the widget sizes '''
        self.scheduleDelayedItemsLayout()

    def wasexpanded(self, item):
        ''' Show/hide buttons when item is expanded '''
        buttons = self.itemWidget(item, 0)
        if buttons:
            with suppress(AttributeError):
                buttons.btnplus.setVisible(item.isExpanded())
                buttons.btnminus.setVisible(item.isExpanded())
