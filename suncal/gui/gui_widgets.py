''' Common widgets for user interface '''

import numpy as np
from dateutil.parser import parse
import matplotlib.dates as mdates

from PyQt5 import QtWidgets, QtCore, QtGui

from . import gui_common
from .. import output
from .. import distributions


# Custom data roles for tree/table widgets
ROLE_ORIGDATA = QtCore.Qt.UserRole + 1    # Original, user-entered data
ROLE_HISTDATA = ROLE_ORIGDATA + 1         # Data for histogram distribution (tuple of (hist, edges))
ROLE_VALID = ROLE_HISTDATA + 1            # Histogram data


def centerWindow(window, w, h):
    ''' Set window geometry so it appears centered in the window
        If window size is too big, maximize it
    '''
    desktopsize = QtWidgets.QDesktopWidget().availableGeometry()
    window.setGeometry(desktopsize.width()//2 - w//2, desktopsize.height()//2 - h//2, w, h)  # Center window on screen
    if h >= desktopsize.height() or w >= desktopsize.width():
        window.showMaximized()


class ReadOnlyTableItem(QtWidgets.QTableWidgetItem):
    ''' Table Widget Item with read-only properties '''
    def __init__(self, *args, **kwargs):
        super(ReadOnlyTableItem, self).__init__(*args, **kwargs)
        self.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)


class EditableTableItem(QtWidgets.QTableWidgetItem):
    ''' Table Widget Item with editable flags '''
    def __init__(self, *args, **kwargs):
        super(EditableTableItem, self).__init__(*args, **kwargs)
        font = self.font()
        font.setBold(True)
        self.setFont(font)


# TableWidget and TreeWidget item.data() roles
class LatexDelegate(QtWidgets.QStyledItemDelegate):
    ''' Delegate class assigned to editable table/tree items. This overrides the editor so that the original
        user-entered (not calculated) expression is displayed for editing. Math expressions are rendered
        as graphics instead of text when not in edit mode.
    '''
    def __init__(self):
        super(LatexDelegate, self).__init__()

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
        model.setData(index, editor.text(), ROLE_ORIGDATA)    # Save for later
        px = QtGui.QPixmap()
        px.loadFromData(output.tex_to_buf(output.format_math(editor.text())).read())
        model.blockSignals(False)
        model.setData(index, px, QtCore.Qt.DecorationRole)


class EditableHeaderView(QtWidgets.QHeaderView):
    ''' Table Header that is user-editable by double-clicking.

        Credit: http://www.qtcentre.org/threads/12835-How-to-edit-Horizontal-Header-Item-in-QTableWidget
        Adapted for QT5.
    '''
    headeredited = QtCore.pyqtSignal()

    def __init__(self, orientation, floatonly=False, parent=None):
        super(EditableHeaderView, self).__init__(orientation, parent)
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
        xdates: bool
            Allow datetime values in first column. Will be converted to ordinal on get.
        xstrings: bool
            Allow string values in first column. Will be omitted from get_table.
        paste_multicol: bool
            Allow pasting multiple columns (and inserting columns as necessary)
    '''
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, movebyrows=False, headeredit=None, xdates=False, xstrings=False, paste_multicol=True, parent=None):
        super(FloatTableWidget, self).__init__(parent=parent)
        self.movebyrows = movebyrows
        self.paste_multicol = paste_multicol
        self.xdates = xdates
        self.xstrings = xstrings
        self.setRowCount(1)
        self.setColumnCount(0)
        if headeredit is not None:
            assert headeredit in ['str', 'float']
            self.setHorizontalHeader(EditableHeaderView(orientation=QtCore.Qt.Horizontal, floatonly=(headeredit == 'float')))
            self.horizontalHeader().headeredited.connect(self.valueChanged)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+v'), self).activated.connect(self._paste)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+c'), self).activated.connect(self._copy)
        self.cellChanged.connect(self._itemchanged)

    def _paste(self):
        ''' Handle pasting data into table '''
        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        clipboard = QtWidgets.QApplication.instance().clipboard().text()
        rowlist = clipboard.split('\n')
        startrow = self.currentRow()
        startcol = self.currentColumn()
        j = 0
        for i, row in enumerate(rowlist):
            collist = row.split()
            if self.paste_multicol:
                for j, st in enumerate(collist):
                    if j == 0 and self.xdates:
                        try:
                            parse(st)
                            val = st
                        except ValueError:
                            val = '-'
                    elif j == 0 and self.xstrings:
                        val = st
                    else:
                        try:
                            val = float(st)
                        except ValueError:
                            val = '-'
                    if self.rowCount() <= startrow+i:
                        self.setRowCount(startrow+i+1)
                    if self.columnCount() <= startcol+j:
                        self.setColumnCount(startcol+j+1)
                    self.setItem(startrow+i, startcol+j, QtWidgets.QTableWidgetItem(str(val)))
            else:
                if startcol == 0 and self.xdates:
                    try:
                        parse(st)
                        val = st
                    except ValueError:
                        val = '-'
                elif startcol == 0 and self.xstrings:
                    val = st
                else:
                    try:
                        val = float(collist[0])
                    except ValueError:
                        val = '-'
                j = 0
                self.setItem(startrow+i, startcol, QtWidgets.QTableWidgetItem(str(val)))
        self.clearSelection()
        self.setCurrentCell(startrow+i, startcol+j)
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
            super(FloatTableWidget, self).keyPressEvent(event)

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

    def set_xdates(self, xdates):
        ''' Change first column to for dates/floats. Converts any existing values. '''
        if self.xdates == xdates:
            return   # Nothing is changing

        signalstate = self.signalsBlocked()
        self.blockSignals(True)
        self.xdates = xdates
        if self.xdates:
            for row in range(self.rowCount()):
                if self.item(row, 0) is None:
                    continue
                try:
                    val = float(self.item(row, 0).text())
                    if val <= 1:
                        val = '-'
                    else:
                        val = mdates.num2date(val).strftime('%d-%b-%Y')
                except (AttributeError, ValueError):
                    val = '-'
                self.item(row, 0).setText(val)
        else:
            for row in range(self.rowCount()):
                if self.item(row, 0) is None:
                    continue
                try:
                    val = str(mdates.date2num(parse(self.item(row, 0).text())))
                except (AttributeError, ValueError):
                    val = '-'
                self.item(row, 0).setText(val)
        self.blockSignals(signalstate)

    def _itemchanged(self, row, col):
        ''' Item was changed. Add new row and move selected cell as appropriate. '''
        item = self.item(row, col)
        if item.text() != '':
            if col == 0 and self.xdates:
                try:
                    parse(item.text()).toordinal()
                except ValueError:
                    item.setText('-')
            elif col == 0 and self.xstrings:
                pass   # OK as string
            else:
                try:
                    float(item.text())
                except ValueError:
                    item.setText('-')

        if row == self.rowCount() - 1 and item is not None and item.text() != '':
            # Edited last row. Add a blank one
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

    def get_column(self, column):
        ''' Get array of values for one column '''
        vals = []
        for i in range(self.rowCount()):
            if column == 0 and self.xdates:
                try:
                    vals.append(mdates.date2num(parse(self.item(i, column).text())))
                except (AttributeError, ValueError):
                    vals.append(np.nan)
            elif column == 0 and self.xstrings:
                try:
                    vals.append(self.item(i, column).text())
                except AttributeError:
                    vals.append('')
            else:
                try:
                    vals.append(float(self.item(i, column).text()))
                except (AttributeError, ValueError):
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
        tbl = np.vstack(vals)
        while tbl.shape[1] > 0 and all(~np.isfinite(tbl[:, -1])):
            # Strip blank rows
            tbl = tbl[:, :-1]
        return tbl


class MarkdownTextEdit(QtWidgets.QTextEdit):
    ''' Text Edit widget with a Save option in context menu. '''
    def __init__(self):
        super(MarkdownTextEdit, self).__init__()
        self.setReadOnly(True)
        self.zoomIn(1)
        self.rawhtml = ''
        self.md = None
        self.showmd = False

    def contextMenuEvent(self, event):
        ''' Create custom context menu '''
        menu = self.createStandardContextMenu()
        actmd = menu.addAction('Show markdown')
        actmd.setCheckable(True)
        actmd.setChecked(self.showmd)
        actsave = menu.addAction('Save page...')
        actsave.triggered.connect(self.savepage)
        actmd.triggered.connect(self.toggledisplay)
        menu.exec(event.globalPos())

    def setMarkdown(self, md):
        ''' Set text to display in markdown format.

            Parameters
            ----------
            md: MDstring
                Markdown string to format and display as HTML.
        '''
        self.md = md
        html = self.md.get_html()

        self.setHtml(html)

    def setHtml(self, html):
        ''' Override setHtml to save off raw html as QTextEdit reformats it, strips css, etc. '''
        self.rawhtml = html
        super(MarkdownTextEdit, self).setHtml(html)
        self.repaint()

    def toggledisplay(self):
        self.showmd = not self.showmd
        if self.showmd:
            self.setPlainText(self.md.raw_md())
        else:
            self.setMarkdown(self.md)

    def savepage(self):
        ''' Save text edit contents to file '''
        savemarkdown(self.md)


def savemarkdown(md):
    ''' Save markdown object contents to file, prompting user for options and file name '''
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
                with output.report_format(math='mathjax', fig=setup.get('image')):
                    data = md.get_md(unicode=setup.get('unicode', True))
                with open(fname, 'w') as f:
                    f.write(data)
                err = None

            elif fmt == 'tex':
                err = md.save_tex(fname)

            elif fmt == 'html':
                with output.report_format(math=setup.get('math'), fig=setup.get('image')):
                    err = md.save_html(fname)

            elif fmt == 'docx':
                err = md.save_docx(fname)

            elif fmt == 'pdf':
                err = md.save_pdf(fname)

            elif fmt == 'odt':
                err = md.save_odt(fname)

            else:
                assert False

            if err:
                QtWidgets.QMessageBox.warning(None, 'Error saving report', 'Error saving report:\n\n{}'.format(err))


class SaveReportOptions(QtWidgets.QDialog):
    ''' Dialog for selecting save report options '''
    def __init__(self, parent=None):
        super(SaveReportOptions, self).__init__(parent)
        self.setWindowTitle('Save report options')
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
        self.mjurl = QtWidgets.QLineEdit(output._mathjaxurl)
        self.chkUnicode = QtWidgets.QCheckBox('Allow Unicode')

        if not output.pandoc_path:
            self.cmbFormat.setItemText(2, 'LaTeX (requires Pandoc)')
            self.cmbFormat.setItemText(3, 'PDF (requires Pandoc and LaTeX)')
            self.cmbFormat.setItemText(4, 'Open Office ODT (requires Pandoc)')
            self.cmbFormat.setItemText(5, 'Word DOCX (requires Pandoc)')
            self.cmbFormat.model().item(2).setEnabled(False)
            self.cmbFormat.model().item(3).setEnabled(False)
            self.cmbFormat.model().item(4).setEnabled(False)
            self.cmbFormat.model().item(5).setEnabled(False)

        if not output.latex_path:
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
        math = ['mathjax', 'mpl'][self.cmbMath.currentIndex()]
        img = self.cmbImage.currentText().lower()
        return {'math': math, 'fmt': fmt, 'image': img, 'unicode': self.chkUnicode.isChecked()}


class ListSelectWidget(QtWidgets.QListWidget):
    ''' List Widget with multi-selection on click '''
    checkChange = QtCore.pyqtSignal(int)

    def __init__(self):
        super(ListSelectWidget, self).__init__()
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
        super(SpinWidget, self).__init__()
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
        super(CoverageButtons, self).__init__()
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
        super(GUMExpandedWidget, self).__init__()
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
        super(MCExpandedWidget, self).__init__()
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
        super(DistributionEditTable, self).__init__()
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
        super(DistributionEditTable, self).clear()
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

        # Want control to enter median value, not loc. Shift appropriately.
        stats_dist = distributions.get_distribution(distname, **initargs)
        argnames = stats_dist.argnames

        if 'loc' in argnames:
            argnames.remove('loc')
        argnames = ['median'] + argnames

        try:
            median = stats_dist.median()
        except TypeError:
            median = 0
        else:
            median = median if np.isfinite(median) else 0
        initargs['median'] = median

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
            median = args.pop('median', args.pop('', 0))
            self.distbias = args.pop('bias', 0)
            try:
                self.statsdist = distributions.get_distribution(distname, **args)
                self.statsdist.set_median(median - self.distbias)
            except ZeroDivisionError:
                self.statsdist = None
            else:
                changed = True

        self.blockSignals(signalstate)
        if changed:
            self.changed.emit()


class ComboNoWheel(QtWidgets.QComboBox):
    ''' ComboBox with scroll wheel disabled '''
    def __init__(self):
        super(ComboNoWheel, self).__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, event):
        ''' Only pass on the event if we have focus '''
        if self.hasFocus():
            super(ComboNoWheel, self).wheelEvent(event)


class QHLine(QtWidgets.QFrame):
    ''' Horizontal divider line '''
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class LineEditLabelWidget(QtWidgets.QWidget):
    ''' Class for a line edit and label '''
    def __init__(self, label='', text=''):
        super(LineEditLabelWidget, self).__init__()
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
        super(SpinBoxLabelWidget, self).__init__()
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
        super(DateDialog, self).__init__(parent)
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
