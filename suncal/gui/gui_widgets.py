''' Common widgets for user interface '''

from contextlib import suppress
from collections import ChainMap
import numpy as np
from dateutil.parser import parse
import matplotlib.dates as mdates
import markdown

from PyQt5 import QtWidgets, QtCore, QtGui

from . import gui_common
from .. import report
from .. import distributions
from .. import css


# Custom data roles for tree/table widgets
ROLE_ORIGDATA = QtCore.Qt.UserRole + 1    # Original, user-entered data
ROLE_HISTDATA = ROLE_ORIGDATA + 1         # Data for histogram distribution (tuple of (hist, edges))
ROLE_VARIABLE = ROLE_ORIGDATA + 2         # InputVar object for this row
ROLE_UNCERT = ROLE_ORIGDATA + 3           # InputUncert object for this row
ROLE_TOPITEM = ROLE_ORIGDATA + 4          # Top row of Uncertainty item in the table


def centerWindow(window, w, h):
    ''' Set window geometry so it appears centered in the window
        If window size is too big, maximize it
    '''
    desktopsize = QtWidgets.QDesktopWidget().availableGeometry()
    window.setGeometry(desktopsize.width()//2 - w//2, desktopsize.height()//2 - h//2, w, h)  # Center window on screen
    if h >= desktopsize.height() or w >= desktopsize.width():
        window.showMaximized()


class TreeButton(QtWidgets.QToolButton):
    ''' Round button for use in a tree widget '''
    # CSS stylesheet for nice round buttons
    buttonstyle = '''QToolButton {border: 1px solid #8f8f91; border-radius: 8px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #f6f7fa, stop: 1 #dadbde);}

                     QToolButton:pressed {border: 2px solid #8f8f91; border-radius: 8px; border-width: 2px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #dadbde, stop: 1 #f6f7fa);}

                     QToolButton:checked {border: 2px solid #8f8f91; border-radius: 8px; border-width: 1px;
                     background-color: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0, stop: 0 #dadbde, stop: 1 #7c7c7c);}
                     '''

    def __init__(self, text):
        super(TreeButton, self).__init__(text=text)
        self.setStyleSheet(self.buttonstyle)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedSize(18, 18)


class PlusMinusButton(QtWidgets.QWidget):
    plusclicked = QtCore.pyqtSignal()
    minusclicked = QtCore.pyqtSignal()

    ''' Widget containing plus/minus buttons '''
    def __init__(self, label='', parent=None):
        super().__init__(parent)
        self.btnplus = TreeButton('+')
        self.btnminus = TreeButton(gui_common.CHR_ENDASH)
        layout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(label)
        font = QtGui.QFont('Arial', 14)
        self.label.setFont(font)
        layout.addWidget(self.label)
        layout.addWidget(self.btnplus)
        layout.addWidget(self.btnminus)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.btnplus.clicked.connect(self.plusclicked)
        self.btnminus.clicked.connect(self.minusclicked)


class WidgetPanel(QtWidgets.QTreeWidget):
    ''' Tree widget for expanding/collapsing other widgets '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setHeaderHidden(True)
        self.setVerticalScrollMode(QtWidgets.QTreeView.ScrollPerPixel)
        self.setColumnCount(1)
        self.itemExpanded.connect(self.wasexpanded)
        self.itemCollapsed.connect(self.wasexpanded)

    def add_widget(self, name, widget, buttons=False):
        ''' Add a widget to the tree at the end '''
        idx = self.invisibleRootItem().childCount()
        return self.insert_widget(name, widget, idx, buttons=buttons)

    def expand(self, name):
        ''' Expand the widget with the given name '''
        item = self.findItems(name, QtCore.Qt.MatchExactly, 0)
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
        item = self.findItems(name, QtCore.Qt.MatchExactly, 0)
        with suppress(IndexError):
            item[0].setHidden(hide)
            self.wasexpanded(item[0])

    def insert_widget(self, name, widget, idx, buttons=False):
        ''' Insert a widget into the tree '''
        item = QtWidgets.QTreeWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)  # Enable, but not selectable/editable
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
        witem.setFlags(QtCore.Qt.ItemIsEnabled)
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


class ReadOnlyTableItem(QtWidgets.QTableWidgetItem):
    ''' Table Widget Item with read-only properties '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)


class EditableTableItem(QtWidgets.QTableWidgetItem):
    ''' Table Widget Item with editable flags '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GroupBoxWidget(QtWidgets.QGroupBox):
    ''' GroupBox set by widget instead of layout '''
    def __init__(self, widget, title, parent=None):
        super().__init__(title)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)


# TableWidget and TreeWidget item.data() roles
class LatexDelegate(QtWidgets.QStyledItemDelegate):
    ''' Delegate class assigned to editable table/tree items. This overrides the editor so that the original
        user-entered (not calculated) expression is displayed for editing. Math expressions are rendered
        as graphics instead of text when not in edit mode.
    '''
    def __init__(self):
        super().__init__()

    def setEditorData(self, editor, index):
        ''' Restore user-entered text when editing starts '''
        text = index.model().data(index, ROLE_ORIGDATA)
        if text is None:
            text = index.model().data(index, QtCore.Qt.DisplayRole)
        editor.setText(text)

    def updateEditorGeometry(self, editor, option, index):
        ''' Put the editor widget in the original location, (instead of
            default behavior of shifting to compensate for pixmap). '''
        editor.setGeometry(option.rect)

    def setModelData(self, editor, model, index):
        ''' Save user-entered text to restore in edit mode later '''
        model.blockSignals(True)  # Only signal on one setData
        olddata = index.model().data(index, ROLE_ORIGDATA)
        if editor.text() != olddata:  # Don't refresh unless necessary
            model.setData(index, editor.text(), ROLE_ORIGDATA)    # Save for later
            px = QtGui.QPixmap()
            ratio = QtWidgets.QApplication.instance().devicePixelRatio()
            px.loadFromData(report.Math(editor.text()).svg_buf(fontsize=16*ratio).read())
            px.setDevicePixelRatio(ratio)
            model.blockSignals(False)
            model.setData(index, px, QtCore.Qt.DecorationRole)
        model.blockSignals(False)


class EditableHeaderView(QtWidgets.QHeaderView):
    ''' Table Header that is user-editable by double-clicking.

        Credit: http://www.qtcentre.org/threads/12835-How-to-edit-Horizontal-Header-Item-in-QTableWidget
        Adapted for QT5.
    '''
    headeredited = QtCore.pyqtSignal()

    def __init__(self, orientation, floatonly=False, parent=None):
        super().__init__(orientation, parent)
        self.floatonly = floatonly
        self.line = QtWidgets.QLineEdit(parent=self.viewport())
        self.line.setAlignment(QtCore.Qt.AlignTop)
        self.line.setHidden(True)
        self.line.blockSignals(True)
        self.sectionedit = 0
        self.sectionDoubleClicked.connect(self.editHeader)
        self.line.editingFinished.connect(self.doneEditing)

    def doneEditing(self):
        self.line.blockSignals(True)
        self.line.setHidden(True)
        if self.floatonly:
            try:
                value = float(self.line.text())
            except ValueError:
                value = '---'
        else:
            value = self.line.text()
        self.model().setHeaderData(self.sectionedit, QtCore.Qt.Horizontal, str(value), QtCore.Qt.EditRole)
        self.line.setText('')
        self.setCurrentIndex(QtCore.QModelIndex())
        self.headeredited.emit()

    def editHeader(self, section):
        edit_geometry = self.line.geometry()
        edit_geometry.setWidth(self.sectionSize(section))
        edit_geometry.moveLeft(self.sectionViewportPosition(section))
        self.line.setGeometry(edit_geometry)
        self.line.setText(str(self.model().headerData(section, QtCore.Qt.Horizontal)))
        self.line.setHidden(False)
        self.line.blockSignals(False)
        self.line.setFocus()
        self.line.selectAll()
        self.sectionedit = section


class FloatTableWidget(QtWidgets.QTableWidget):
    ''' Widget for entering a table of floats

        Parameters
        ----------
        movebyrows: bool
            When done editing, move the selected cell to the next row (True)
            or the next column (False).
        headeredit: string or None
            Editable header. If None, no editing. string options are 'str' or 'float'
            to restrict header values to strings or floats.
        paste_multicol: bool
            Allow pasting multiple columns (and inserting columns as necessary)
    '''
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, movebyrows=False, headeredit=None, xstrings=False, paste_multicol=True, parent=None):
        super().__init__(parent=parent)
        self.movebyrows = movebyrows
        self.paste_multicol = paste_multicol
        self.maxrows = None
        self.maxcols = None
        self.setRowCount(1)
        self.setColumnCount(0)
        if headeredit is not None:
            assert headeredit in ['str', 'float']
            self.setHorizontalHeader(EditableHeaderView(orientation=QtCore.Qt.Horizontal, floatonly=(headeredit == 'float')))
            self.horizontalHeader().headeredited.connect(self.valueChanged)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+v'), self).activated.connect(self._paste)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+c'), self).activated.connect(self._copy)
        self.cellChanged.connect(self._itemchanged)

    def clear(self):
        ''' Clear table, but not header '''
        self.setRowCount(0)
        self.setRowCount(1)

    def _paste(self):
        ''' Handle pasting data into table '''
        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        startrow = self.currentRow()
        startcol = self.currentColumn()
        clipboard = QtWidgets.QApplication.instance().clipboard().text()
        rowlist = clipboard.split('\n')
        if self.maxrows is not None:
            rowlist = rowlist[:self.maxrows]
        if self.maxcols is not None:
            rowlist = ['\t'.join(r.split()[:self.maxcols-startcol]) for r in rowlist]

        j = 0
        for i, row in enumerate(rowlist):
            collist = row.split()
            if self.paste_multicol:
                for j, st in enumerate(collist):
                    try:
                        val = float(st)
                    except ValueError:
                        if st.lower() in ['pass', 'fail', 'true', 'false', 'yes', 'no', 'n/a', 'none', 'null']:
                            val = st
                        else:
                            try:
                                parse(st)
                                val = st
                            except (ValueError, OverflowError):
                                val = '-'

                    if self.rowCount() <= startrow+i:
                        self.setRowCount(startrow+i+1)
                    if self.columnCount() <= startcol+j:
                        self.setColumnCount(startcol+j+1)
                    self.setItem(startrow+i, startcol+j, QtWidgets.QTableWidgetItem(str(val)))
            else:
                try:
                    val = float(st)
                except ValueError:
                    if st.lower() in ['pass', 'fail', 'true', 'false', 'yes', 'no', 'n/a', 'none', 'null']:
                        val = st
                    else:
                        try:
                            parse(st)
                            val = st
                        except (ValueError, OverflowError):
                            val = '-'
                j = 0
                self.setItem(startrow+i, startcol, QtWidgets.QTableWidgetItem(str(val)))
        self.clearSelection()
        self.setCurrentCell(startrow+i, startcol+j)
        if self.maxrows is None or startrow+i+1 < self.maxrows:
            self.insertRow(startrow+i+1)  # Blank row at end
        self.blockSignals(signalstate)
        self.valueChanged.emit()

    def _copy(self):
        ''' Copy selected cells to clipboard '''
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.clear(mode=clipboard.Clipboard)
        ranges = self.selectedRanges()
        if len(ranges) < 1:
            return
        text = ''
        for rng in ranges:
            top = rng.topRow()
            bot = rng.bottomRow()
            lft = rng.leftColumn()
            rgt = rng.rightColumn()
            rows = []
            for row in range(top, bot+1):
                cols = []
                for col in range(lft, rgt+1):
                    item = self.item(row, col)
                    cols.append(item.text() if item else '')
                rows.append('\t'.join(cols))
        text = '\n'.join(rows)
        clipboard.setText(text, mode=clipboard.Clipboard)

    def _insertrow(self):
        ''' Insert a blank row in the table '''
        if self.maxrows is None or self.rowCount() < self.maxrows:
            self.insertRow(max(0, self.currentRow()))
            self.valueChanged.emit()

    def _removerow(self):
        ''' Remove row from table '''
        self.removeRow(self.currentRow())
        self.valueChanged.emit()

    def keyPressEvent(self, event):
        ''' Key was pressed. Capture delete key to clear selected items '''
        items = self.selectedItems()
        if event.key() == QtCore.Qt.Key_Delete and len(items) > 0:
            signalstate = self.signalsBlocked()
            self.blockSignals(True)
            for item in items:
                item.setText('')
            self.blockSignals(signalstate)
            self.valueChanged.emit()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        actCopy = QtWidgets.QAction('Copy', self)
        actPaste = QtWidgets.QAction('Paste', self)
        actPaste.setEnabled(QtWidgets.QApplication.instance().clipboard().text() != '')
        actInsert = QtWidgets.QAction('Insert Row', self)
        actRemove = QtWidgets.QAction('Remove Row', self)
        menu.addAction(actCopy)
        menu.addAction(actPaste)
        menu.addSeparator()
        menu.addAction(actInsert)
        menu.addAction(actRemove)
        actPaste.triggered.connect(self._paste)
        actCopy.triggered.connect(self._copy)
        actInsert.triggered.connect(self._insertrow)
        actRemove.triggered.connect(self._removerow)
        menu.popup(QtGui.QCursor.pos())

    def _itemchanged(self, row, col):
        ''' Item was changed. Add new row and move selected cell as appropriate. '''
        item = self.item(row, col)
        if item and item.text() != '':
            try:
                float(item.text())
            except ValueError:
                if item.text().lower() not in ['pass', 'true', 'fail', 'false',  'yes', 'no', 'none', 'n/a', 'null']:
                    try:
                        parse(item.text()).toordinal()
                    except (ValueError, OverflowError):
                        item.setText('-')
                    try:
                        parse(item.text()).toordinal()
                    except (ValueError, OverflowError):
                        item.setText('-')

        if row == self.rowCount() - 1 and item is not None and item.text() != '':
            # Edited last row. Add a blank one
            if self.maxrows is None or row+1 < self.maxrows:
                self.insertRow(row+1)
                self.setRowHeight(row+1, self.rowHeight(row))

        # Move cursor to next row or column
        if self.movebyrows:
            self.setCurrentCell(row+1, col)
        elif col == self.columnCount() - 1:
            self.setCurrentCell(row+1, 0)
        else:
            self.setCurrentCell(row, col+1)
        self.valueChanged.emit()

    def has_dates(self, column=0):
        ''' Determine if the data has datetime in column '''
        hasdates = False
        for i in range(self.rowCount()):
            text = self.item(i, column).text() if self.item(i, column) else ''
            try:
                float(text)
            except ValueError:
                try:
                    mdates.date2num(parse(text))
                except (ValueError, OverflowError):
                    pass
                else:
                    hasdates = True
                    break
        return hasdates

    def get_column(self, column):
        ''' Get array of values for one column '''
        vals = []
        for i in range(self.rowCount()):
            text = self.item(i, column).text() if self.item(i, column) else ''
            try:
                vals.append(float(text))
            except ValueError:
                if text.lower() in ['p', 'pass', 't', 'true', 'y', 'yes']:
                    vals.append(1)
                elif text.lower() in ['f', 'fail', 'false', 'n', 'no']:
                    vals.append(0)
                else:
                    try:
                        vals.append(mdates.date2num(parse(text)))
                    except (ValueError, OverflowError):
                        vals.append(np.nan)
        return np.asarray(vals)

    def get_columntext(self, column):
        ''' Get array of string values in column '''
        vals = []
        for i in range(self.rowCount()):
            try:
                vals.append(self.item(i, column).text())
            except AttributeError:
                vals.append('')
        return vals

    def get_table(self):
        ''' Get 2D array of values for entire table '''
        vals = []
        for col in range(self.columnCount()):
            vals.append(self.get_column(col))
        try:
            tbl = np.vstack(vals)
            while tbl.shape[1] > 0 and all(~np.isfinite(tbl[:, -1])):
                # Strip blank rows
                tbl = tbl[:, :-1]
        except ValueError:
            # No rows
            tbl = np.array([[]])

        return tbl


class MarkdownTextEdit(QtWidgets.QTextEdit):
    ''' Text Edit widget with a Save option in context menu. '''
    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setReadOnly(True)
        self.zoomIn(1)
        self.rawhtml = ''
        self.md = None
        self.showmd = False
        self.sigfigs = gui_common.settings.getSigfigs()
        self.numformat = gui_common.settings.getNumformat()

    def contextMenuEvent(self, event):
        ''' Create custom context menu '''
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        menufigs = QtWidgets.QMenu('Significant Figures')
        for i in range(8):
            actfigs = menufigs.addAction(str(i+1))
            actfigs.triggered.connect(lambda x, n=i+1: self.setFigs(n))
            if self.sigfigs == i+1:
                actfigs.setCheckable(True)
                actfigs.setChecked(True)

        menuformat = QtWidgets.QMenu('Number Format')
        for i in ['Auto', 'Decimal', 'Scientific', 'Engineering', 'SI']:
            actformat = menuformat.addAction(i)
            actformat.triggered.connect(lambda x, f=i.lower(): self.setFmt(f))
            if self.numformat == i.lower():
                actformat.setCheckable(True)
                actformat.setChecked(True)

        menu.addMenu(menufigs)
        menu.addMenu(menuformat)
        menu.addSeparator()
        actmd = menu.addAction('Show markdown')
        actmd.setCheckable(True)
        actmd.setChecked(self.showmd)
        actsave = menu.addAction('Save page...')
        actsave.triggered.connect(self.savepage)
        actmd.triggered.connect(self.toggledisplay)
        menu.exec(event.globalPos())

    def setFigs(self, figs):
        self.sigfigs = figs
        self.setReport(self.rpt)

    def setFmt(self, fmt):
        self.numformat = fmt
        self.setReport(self.rpt)

    def setReport(self, rpt):
        ''' Set text to display in markdown format.

            Parameters
            ----------
            rpt: report.Report
                Report object to format and display as HTML.
        '''
        # Don't just self.setHtml to self.rpt.get_html(), since that won't properly scale for
        # hi-dpi displays. Unfortunately, need to recreate report.get_md here but using
        # the TextCursor of TextEdit to add the images.

        self.rpt = rpt
        args = ChainMap({'n': self.sigfigs, 'fmt': self.numformat}, gui_common.get_rptargs())
        document = self.document()
        cursor = self.textCursor()
        ratio = QtWidgets.QApplication.instance().devicePixelRatio()

        # Convert markdown to HTML, but leave [[xxx]] image tags
        s = self.rpt._s
        html = markdown.markdown(s, extensions=['markdown.extensions.tables'])
        document.clear()
        document.setDefaultStyleSheet(css.css)
        self.insertHtml(html)

        regex = QtCore.QRegularExpression(r'(\[\[(?:EQN|VAL|PLT|UNT)[0-9].*?\]\])')
        cursor = document.find(regex)
        while cursor is not None and cursor.selectedText():
            tag = cursor.selectedText()
            tagindex = int(tag[5:-2])  # strip [[XXX and closing ]]
            if 'EQN' in tag:
                eqn = self.rpt._eqns[tagindex]
                svg = eqn.svg_buf(fontsize=16*ratio).getvalue()
                px = QtGui.QPixmap()
                px.loadFromData(svg)
                px.setDevicePixelRatio(ratio)
                im = QtGui.QImage(px)
                cursor.removeSelectedText()
                cursor.insertImage(im)
            elif 'VAL' in tag:
                cursor.removeSelectedText()
                cursor.insertText(self.rpt._values[tagindex].string(**args))
            elif 'UNT' in tag:
                cursor.removeSelectedText()
                cursor.insertText(self.rpt._units[tagindex].string(**args))
            elif 'PLT' in tag:
                plt = self.rpt._plots[tagindex]
                svg = plt.svg_buf(scale=ratio).getvalue()
                px = QtGui.QPixmap()
                px.loadFromData(svg)
                px.setDevicePixelRatio(ratio)
                im = QtGui.QImage(px)
                cursor.removeSelectedText()
                cursor.insertImage(im)
            else:
                raise ValueError
            cursor = document.find(regex, cursor)

    def setHtml(self, html):
        ''' Override setHtml to save off raw html as QTextEdit reformats it, strips css, etc. '''
        self.rawhtml = html
        super().setHtml(html)
        self.repaint()

    def toggledisplay(self):
        self.showmd = not self.showmd
        if self.showmd:
            self.setPlainText(self.rpt.get_md(mathfmt='latex', n=self.sigfigs, fmt=self.numformat))
        else:
            self.setReport(self.rpt)

    def savepage(self):
        ''' Save text edit contents to file '''
        savereport(self.rpt, fmt=self.numformat, n=self.sigfigs)


def savereport(rpt, **kwargs):
    ''' Save Report object contents to file, prompting user for options and file name '''
    kargs = ChainMap(kwargs, gui_common.get_rptargs())
    dlg = SaveReportOptions()
    ok = dlg.exec_()
    if ok:
        setup = dlg.get_setup()
        fmt = setup.get('fmt', 'html')
        filter = {'html': 'HTML (*.html)', 'tex': 'LaTeX source (*.tex)', 'md': 'Markdown (*.md *.txt)', 'docx': 'Word DOCX (*.docx)',
                  'pdf': 'PDF (*.pdf)', 'odt': 'Open Document Text (*.odt)'}[fmt]

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='File to Save', filter=filter)
        if fname:
            if fmt == 'md':
                data = rpt.get_md(mathfmt='latex', figfmt=setup.get('image'), unicode=setup.get('unicode', True), **kargs)
                with open(fname, 'w') as f:
                    f.write(data)
                err = None

            elif fmt == 'tex':
                err = rpt.save_tex(fname, **kargs)

            elif fmt == 'html':
                err = rpt.save_html(fname, mathfmt=setup.get('mathfmt'), figfmt=setup.get('image'), **kargs)

            elif fmt == 'docx':
                err = rpt.save_docx(fname, **kargs)

            elif fmt == 'pdf':
                err = rpt.save_pdf(fname, **kargs)

            elif fmt == 'odt':
                err = rpt.save_odt(fname, **kargs)

            else:
                assert False

            if err:
                QtWidgets.QMessageBox.warning(None, 'Error saving report', 'Error saving report:\n\n{}'.format(err))


class SaveReportOptions(QtWidgets.QDialog):
    ''' Dialog for selecting save report options '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Save report options')
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setMinimumWidth(500)
        self.btnbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btnbox.rejected.connect(self.reject)
        self.btnbox.accepted.connect(self.accept)
        self.cmbFormat = QtWidgets.QComboBox()
        self.cmbFormat.addItems(['HTML', 'Markdown', 'LaTeX', 'PDF', 'Open Office ODT', 'Word DOCX'])

        self.cmbMath = QtWidgets.QComboBox()
        self.cmbMath.addItems(['Mathjax', 'Matplotlib'])
        self.cmbImage = QtWidgets.QComboBox()
        self.cmbImage.addItems(['SVG', 'PNG'])  # EPS?
        self.mjurl = QtWidgets.QLineEdit(report._mathjaxurl)
        self.chkUnicode = QtWidgets.QCheckBox('Allow Unicode')

        if not report.pandoc_path:
            self.cmbFormat.setItemText(2, 'LaTeX (requires Pandoc)')
            self.cmbFormat.setItemText(3, 'PDF (requires Pandoc and LaTeX)')
            self.cmbFormat.setItemText(4, 'Open Office ODT (requires Pandoc)')
            self.cmbFormat.setItemText(5, 'Word DOCX (requires Pandoc)')
            self.cmbFormat.model().item(2).setEnabled(False)
            self.cmbFormat.model().item(3).setEnabled(False)
            self.cmbFormat.model().item(4).setEnabled(False)
            self.cmbFormat.model().item(5).setEnabled(False)

        if not report.latex_path:
            self.cmbFormat.setItemText(3, 'PDF (requires Pandoc and LaTeX)')
            self.cmbFormat.model().item(3).setEnabled(False)

        self.cmbFormat.setCurrentIndex(['html', 'md', 'tex', 'pdf', 'odt', 'docx'].index(gui_common.settings.getRptFormat()))
        self.cmbImage.setCurrentIndex(['svg', 'png'].index(gui_common.settings.getRptImgFormat()))
        self.cmbMath.setCurrentIndex(['mathjax', 'mpl'].index(gui_common.settings.getRptMath()))
        self.mjurl.setText(gui_common.settings.getRptMJURL())
        self.chkUnicode.setChecked(gui_common.settings.getRptUnicode())

        self.lblMath = QtWidgets.QLabel('Math Renderer')
        self.lblImage = QtWidgets.QLabel('Image Format')
        self.lblMJ = QtWidgets.QLabel('MathJax URL')

        glayout = QtWidgets.QGridLayout()
        glayout.addWidget(QtWidgets.QLabel('File Format'), 0, 0)
        glayout.addWidget(self.cmbFormat, 0, 1)
        glayout.addWidget(self.lblImage, 1, 0)
        glayout.addWidget(self.cmbImage, 1, 1)
        glayout.addWidget(self.lblMath, 2, 0)
        glayout.addWidget(self.cmbMath, 2, 1)
        glayout.addWidget(self.lblMJ, 3, 0)
        glayout.addWidget(self.mjurl, 3, 1)
        glayout.addWidget(self.chkUnicode, 4, 1)
        glayout.setColumnMinimumWidth(1, 350)
        glayout.setColumnStretch(1, 10)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(glayout)
        layout.addWidget(self.btnbox)
        self.setLayout(layout)
        self.cmbMath.currentIndexChanged.connect(self.refresh)
        self.cmbFormat.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self):
        ''' Refresh controls, show/hide based on options '''
        html = self.cmbFormat.currentText() == 'HTML'
        md = self.cmbFormat.currentText() == 'Markdown'
        self.cmbMath.setVisible(html)
        self.cmbImage.setVisible(html or md)
        self.lblMath.setVisible(html)
        self.lblImage.setVisible(html or md)
        self.chkUnicode.setVisible(md)
        self.mjurl.setVisible(html and self.cmbMath.currentText() == 'Mathjax')
        self.lblMJ.setVisible(html and self.cmbMath.currentText() == 'Mathjax')

    def get_setup(self):
        ''' Get dictionary of report format options '''
        fmt = ['html', 'md', 'tex', 'pdf', 'odt', 'docx'][self.cmbFormat.currentIndex()]
        math = ['latex', 'mpl'][self.cmbMath.currentIndex()]
        img = self.cmbImage.currentText().lower()
        if math == 'mpl':
            math = img
        return {'mathfmt': math, 'fmt': fmt, 'image': img, 'unicode': self.chkUnicode.isChecked()}


class ListSelectWidget(QtWidgets.QListWidget):
    ''' List Widget with multi-selection on click '''
    checkChange = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.itemChanged.connect(self.itemcheck)

    def addItems(self, itemlist):
        self.clear()
        for i in itemlist:
            item = QtWidgets.QListWidgetItem(i)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.addItem(item)

    def itemcheck(self, item):
        self.checkChange.emit(self.row(item))

    def getSelectedIndexes(self):
        sel = []
        for row in range(self.count()):
            if self.item(row).checkState() == QtCore.Qt.Checked:
                sel.append(row)
        return sel

    def selectAll(self):
        ''' Select all items '''
        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        for i in range(self.count()):
            self.item(i).setCheckState(QtCore.Qt.Checked)
        self.blockSignals(signalstate)

    def selectIndex(self, idxs):
        ''' Select items with index in idxs '''
        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        for i in range(self.count()):
            self.item(i).setCheckState(QtCore.Qt.Checked if i in idxs else QtCore.Qt.Unchecked)
        self.blockSignals(signalstate)

    def selectNone(self):
        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        for i in range(self.count()):
            self.item(i).setCheckState(QtCore.Qt.Unchecked)
        self.blockSignals(signalstate)


class SpinWidget(QtWidgets.QWidget):
    ''' Widget with label and spinbox '''
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, label=''):
        super().__init__()
        self.label = QtWidgets.QLabel(label)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(0, 1E8)
        self.spin.valueChanged.connect(self.valueChanged)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.spin)
        layout.addStretch()
        self.setLayout(layout)

    def value(self):
        return int(self.spin.text())

    def setValue(self, value):
        self.spin.setValue(int(value))


class CoverageButtons(QtWidgets.QWidget):
    ''' Widget for showing pushbuttons for selecting coverage interval percentages.

        Parameters
        ----------
        levels: list of strings
            Names for each button. Defaults to 99%, 95%, etc.
        dflt: list of int
            List of buttons to set checked by default
        multiselect: bool
            Allow selection of multiple coverage levels
    '''
    changed = QtCore.pyqtSignal()

    def __init__(self, levels=None, dflt=None, multiselect=False):
        super().__init__()
        self.btngroup = QtWidgets.QButtonGroup()
        self.btngroup.setExclusive(not multiselect)

        if levels is None:
            levels = ['99%', '95%', '90%', '85%', '68%']

        if dflt is None:
            dflt = [1]

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(0)
        layout.addStretch()
        for idx, p in enumerate(levels):
            b1 = QtWidgets.QToolButton()
            b1.setCheckable(True)
            b1.setText(p)
            b1.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            if idx in dflt:
                b1.setChecked(True)
            self.btngroup.addButton(b1)
            layout.addWidget(b1)
        layout.addStretch()
        self.setLayout(layout)
        self.btngroup.buttonClicked.connect(self.changed)

    def get_covlist(self):
        cov = []
        for b in self.btngroup.buttons():
            if b.isChecked():
                cov.append(b.text())
        return cov


class GUMExpandedWidget(QtWidgets.QWidget):
    ''' Widget with controls for changing coverage interval for GUM.

        Parameters
        ----------
        label: string
            Label for the widget
        multiselect: bool
            Allow selection of multiple coverage levels
        dflt: list of int
            Index(es) of buttons to check by default
    '''
    changed = QtCore.pyqtSignal()

    def __init__(self, label='GUM:', multiselect=True, dflt=None):
        super().__init__()
        self.usek = False
        self.GUMtype = QtWidgets.QComboBox()
        self.GUMtype.addItems(['Student-t', 'Normal/k'])
        self.covbuttons = CoverageButtons(dflt=dflt, multiselect=multiselect)
        self.kbuttons = CoverageButtons(['k = 1', 'k = 2', 'k = 3'], dflt=dflt, multiselect=multiselect)
        self.kbuttons.setVisible(False)
        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QFormLayout()
        hlayout.addRow(label, self.GUMtype)
        layout.addLayout(hlayout)
        layout.addWidget(self.covbuttons)
        layout.addWidget(self.kbuttons)
        self.setLayout(layout)
        self.GUMtype.currentIndexChanged.connect(self.typechange)
        self.kbuttons.changed.connect(self.changed)
        self.covbuttons.changed.connect(self.changed)

    def typechange(self):
        self.usek = (self.GUMtype.currentIndex() == 1)
        self.covbuttons.setVisible(not self.usek)
        self.kbuttons.setVisible(self.usek)
        self.changed.emit()

    def get_covlist(self):
        return self.kbuttons.get_covlist() if self.usek else self.covbuttons.get_covlist()

    def set_buttons(self, values):
        ''' Check buttons with label in values '''
        if values is None:
            values = []
        for b in self.covbuttons.btngroup.buttons():
            b.setChecked(b.text() in values)
        for b in self.kbuttons.btngroup.buttons():
            b.setChecked(b.text() in values)


class MCExpandedWidget(QtWidgets.QWidget):
    ''' Widget with controls for changing coverage interval for Monte-Carlo.

        Parameters
        ----------
        label: string
            Label for the widget
        multiselect: bool
            Allow selection of multiple coverage levels
        dflt: list of int
            Index(es) of buttons to check by default
    '''
    changed = QtCore.pyqtSignal()

    def __init__(self, label='Monte-Carlo:', multiselect=True, dflt=None):
        super().__init__()
        self.MCtype = QtWidgets.QComboBox()
        self.MCtype.addItems(['Symmetric', 'Shortest'])
        self.MCtype.currentIndexChanged.connect(self.changed)
        self.covbuttons = CoverageButtons(dflt=dflt, multiselect=multiselect)
        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QFormLayout()
        hlayout.addRow(label, self.MCtype)
        layout.addLayout(hlayout)
        layout.addWidget(self.covbuttons)
        layout.addStretch()
        self.setLayout(layout)
        self.covbuttons.changed.connect(self.changed)

    def get_covlist(self):
        return self.covbuttons.get_covlist()

    def set_buttons(self, values):
        ''' Check buttons with label in values '''
        if values is None:
            values = []
        for b in self.covbuttons.btngroup.buttons():
            b.setChecked(b.text() in values)


class DistributionEditTable(QtWidgets.QTableWidget):
    ''' Table for editing parameters of an uncertainty distribution

        Parameters
        ----------
        initargs: dict
            Initial arguments for distribution
        locslider: bool
            Show slider for median location of distribution
    '''
    changed = QtCore.pyqtSignal()

    def __init__(self, initargs=None, locslider=False):
        super().__init__()
        self.showlocslider = locslider
        self.range = (-2.0, 2.0)
        self.setMinimumWidth(200)
        self.setMaximumHeight(200)
        self.set_disttype(initargs)
        self.set_locrange(-4, 4)
        self.valuechanged()
        self.cellChanged.connect(self.valuechanged)

    def clear(self):
        ''' Clear and reset the table '''
        super().clear()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['Parameter', 'Value', ''])
        self.resizeColumnsToContents()
        self.setColumnWidth(0, 150)
        self.setColumnWidth(1, 100)

    def sliderchange(self):
        ''' Slider has changed, update loc in table '''
        rng = self.range[1] - self.range[0]
        val = self.locslide.value() / self.locslide.maximum() * rng + self.range[0]
        self.item(1, 1).setText('{}'.format(val))

    def set_locrange(self, low, high):
        ''' Set range for loc slider '''
        self.range = (low, high)

    def set_disttype(self, initargs=None):
        ''' Change distribution type, fill in required params '''
        if initargs is None:
            initargs = {}
        distname = initargs.pop('name', initargs.pop('dist', 'normal'))
        initargs.setdefault('median', initargs.get('mean', 0))
        bias = initargs['median'] - initargs.get('expected', initargs['median'])
        initargs.setdefault('bias', bias)

        # Want control to enter median value, not loc. Shift appropriately.
        stats_dist = distributions.get_distribution(distname, **initargs)
        argnames = stats_dist.argnames

        if 'loc' in argnames:
            argnames.remove('loc')
        if stats_dist.showshift:
            argnames = ['shift'] + argnames
            initargs.setdefault('shift', stats_dist.distargs.get('loc', 0))
        else:
            argnames = ['median'] + argnames

            try:
                median = stats_dist.median()
            except TypeError:
                median = 0
            else:
                median = median if np.isfinite(median) else 0
            initargs['median'] = median

        # Strip any units that may trickle in
        for k, v in initargs.items():
            if hasattr(v, 'magnitude'):
                initargs[k] = v.magnitude

        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        self.clear()
        dists = gui_common.settings.getDistributions()
        self.cmbdist = QtWidgets.QComboBox()
        self.cmbdist.addItems(dists)
        if distname not in dists:
            self.cmbdist.addItem(distname)
        self.cmbdist.setCurrentIndex(self.cmbdist.findText(distname))
        self.setRowCount(len(argnames) + 1 + self.showlocslider)  # header + argnames
        self.setItem(0, 0, ReadOnlyTableItem('Distribution'))
        self.setCellWidget(0, 1, self.cmbdist)

        if distname == 'histogram':
            self.setRowCount(1)
            if self.showlocslider:
                self.setRowCount(3)
                self.setItem(1, 0, ReadOnlyTableItem('measurement'))
                self.setItem(1, 1, QtWidgets.QTableWidgetItem(str(initargs.get('median', 0))))
                self.setItem(2, 0, ReadOnlyTableItem('bias'))
                self.setItem(2, 1, QtWidgets.QTableWidgetItem(str(initargs.get('bias', 0))))
            self.item(0, 0).setData(ROLE_HISTDATA, stats_dist.distargs)

        else:
            for row, arg in enumerate(argnames):
                self.setItem(row+1, 0, ReadOnlyTableItem(arg))
                self.setItem(row+1, 1, QtWidgets.QTableWidgetItem(str(initargs.get(arg, '1' if row > 0 else '0'))))
            if self.showlocslider:
                self.setItem(row+2, 0, ReadOnlyTableItem('bias'))
                self.setItem(row+2, 1, QtWidgets.QTableWidgetItem(str(initargs.get('bias', 0))))

        if self.showlocslider:
            self.locslidewidget = QtWidgets.QWidget()
            self.locslide = QtWidgets.QSlider(orientation=1)
            self.locslide.setRange(0, 200)  # Sliders always use ints.
            self.locslide.setValue(100)
            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(QtWidgets.QLabel('measurement'))
            layout.addWidget(self.locslide)
            self.locslidewidget.setLayout(layout)
            self.setCellWidget(1, 0, self.locslidewidget)
            self.locslide.valueChanged.connect(self.sliderchange)
            self.setItem(1, 0, ReadOnlyTableItem(''))

        for row in range(self.rowCount()):
            self.setRowHeight(row, 40)

        self.cmbdist.currentIndexChanged.connect(self.distchanged)
        self.blockSignals(signalstate)

    def distchanged(self):
        ''' Distribution combobox change '''
        self.set_disttype({'dist': self.cmbdist.currentText()})
        self.valuechanged()

    def valuechanged(self):
        ''' Table value has changed, update stats distribution. '''
        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        argvals = []
        distname = self.cmbdist.currentText()

        for r in range(self.rowCount()-1):
            try:
                argvals.append(float(self.item(r+1, 1).text()))
            except ValueError:
                argvals.append(None)
                self.item(r+1, 1).setBackground(gui_common.COLOR_INVALID)
                self.clearSelection()
            else:
                self.item(r+1, 1).setBackground(gui_common.COLOR_OK)

        argnames = [self.item(r+1, 0).text() for r in range(self.rowCount()-1)]
        args = dict(zip(argnames, argvals))
        if '' in args:
            args['median'] = args.pop('')  # 'median' label is hidden when slider is used

        changed = False
        if distname == 'histogram':
            distargs = self.item(0, 0).data(ROLE_HISTDATA)
            self.statsdist = distributions.get_distribution(distname, **distargs)
            self.distbias = args.pop('bias', 0)
            if 'median' in args or '' in args:
                median = args.pop('median', args.pop('', 0))
                self.statsdist.set_median(median - self.distbias)
                changed = True

        elif None not in argvals:
            self.distbias = args.pop('bias', 0)
            try:
                self.statsdist = distributions.get_distribution(distname, **args)
                if self.statsdist.showshift:
                    self.statsdist.set_shift(args.pop('shift', 0))
                else:
                    m = args.pop('median', args.pop('', 0))
                    self.statsdist.set_median(m - self.distbias)
            except ZeroDivisionError:
                self.statsdist = None
            else:
                changed = True

        self.blockSignals(signalstate)
        if changed:
            self.changed.emit()

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        actHelp = QtWidgets.QAction('Distribution Help...', self)
        menu.addAction(actHelp)
        actHelp.triggered.connect(self.helppopup)
        menu.popup(QtGui.QCursor.pos())
    
    def helppopup(self):
        dlg = PopupHelp(self.statsdist.helpstr())
        dlg.exec_()        


class PopupHelp(QtWidgets.QDialog):
    ''' Show a floating dialog window with a text message '''
    def __init__(self, text):
        super().__init__()
        centerWindow(self, 600, 400)
        self.setModal(False)
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        font = QtGui.QFont('Courier')
        font.setStyleHint(QtGui.QFont.TypeWriter)
        self.text.setCurrentFont(font)
        self.text.setText(text)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.text)
        self.setLayout(layout)


class ComboNoWheel(QtWidgets.QComboBox):
    ''' ComboBox with scroll wheel disabled '''
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, event):
        ''' Only pass on the event if we have focus '''
        if self.hasFocus():
            super().wheelEvent(event)


class QHLine(QtWidgets.QFrame):
    ''' Horizontal divider line '''
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class FloatLineEdit(QtWidgets.QLineEdit):
    ''' Line Edit with float validator '''
    def __init__(self, text='', low=None, high=None):
        super().__init__(text)
        self._validator = QtGui.QDoubleValidator()
        if low is not None:
            self._validator.setBottom(low)
        if high is not None:
            self._validator.setTop(high)
        self.setValidator(self._validator)

    def value(self):
        try:
            val = float(self.text())
        except ValueError:
            val = 0
        return val

    def setValue(self, value):
        self.setText(str(value))


class IntLineEdit(QtWidgets.QLineEdit):
    ''' Line Edit with integer validator '''
    def __init__(self, text='', low=None, high=None):
        super().__init__(text)
        self._validator = QtGui.QIntValidator()
        if low is not None:
            self._validator.setBottom(low)
        if high is not None:
            self._validator.setTop(high)
        self.setValidator(self._validator)

    def value(self):
        return int(self.text())

    def setValue(self, value):
        self.setText(str(int(value)))


class LineEditLabelWidget(QtWidgets.QWidget):
    ''' Class for a line edit and label '''
    def __init__(self, label='', text=''):
        super().__init__()
        self._label = QtWidgets.QLabel(label)
        self._text = QtWidgets.QLineEdit(text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._text)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def __getattr__(self, name):
        ''' Get all other attributes from the lineedit widget '''
        return getattr(self._text, name)


class SpinBoxLabelWidget(QtWidgets.QWidget):
    ''' Class for a DoubleSpinBox and label '''
    def __init__(self, label='', value=0, rng=None):
        super().__init__()
        self._label = QtWidgets.QLabel(label)
        self._spinbox = QtWidgets.QDoubleSpinBox()
        self._spinbox.setValue(value)
        if rng is not None:
            self._spinbox.setRange(*rng)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._spinbox)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def __getattr__(self, name):
        ''' Get all other attributes from the spinbox widget '''
        return getattr(self._spinbox, name)


class DateDialog(QtWidgets.QDialog):
    ''' Dialog for getting a date value '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dateedit = QtWidgets.QDateEdit()
        self.dateedit.setCalendarPopup(True)
        self.dateedit.setDate(QtCore.QDate.currentDate())
        btnbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btnbox.accepted.connect(self.accept)
        btnbox.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.dateedit)
        layout.addWidget(btnbox)
        self.setLayout(layout)

    def date(self):
        ''' Get the date (QDate) '''
        return self.dateedit.date()

    @staticmethod
    def getDate(parent=None):
        ''' Static method for getting a date value in one line. Returns (QDate date, bool valid) '''
        dialog = DateDialog(parent)
        result = dialog.exec_()
        return dialog.date(), result == QtWidgets.QDialog.Accepted
