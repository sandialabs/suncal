''' Markdown Text Viewer Widget and Save Report Dialog '''

from collections import ChainMap
from PyQt6 import QtCore, QtGui, QtWidgets
import markdown

from ...common.style import css
from ...common import report
from ..gui_settings import gui_settings
from .. import gui_styles
from .. import gui_math


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
        self.rpt = None
        self.sigfigs = gui_settings.sigfigs
        self.numformat = gui_settings.numformat
        gui_styles.darkmode_signal().connect(lambda x: self.setReport(self.rpt))

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

            Args:
                rpt: Report instance to format and display as HTML.
        '''
        # Don't just self.setHtml to self.rpt.get_html(), since that won't properly scale for
        # hi-dpi displays. Unfortunately, need to recreate report.get_md here but using
        # the TextCursor of TextEdit to add the images.

        self.rpt = rpt
        style = css.css + css.css_dark if gui_styles.isdark() else css.css
        try:
            document = self.document()
            document.clear()
            document.setDefaultStyleSheet(style)
        except RuntimeError:
            return  # Document was deleted from widget?
        if self.rpt is None or self.rpt._s is None:
            return

        cursor = self.textCursor()
        args = ChainMap({'n': self.sigfigs, 'fmt': self.numformat}, gui_settings.report_args)
        ratio = QtWidgets.QApplication.instance().primaryScreen().devicePixelRatio()

        # Convert markdown to HTML, but leave [[xxx]] image tags
        html = markdown.markdown(self.rpt._s, extensions=['markdown.extensions.tables'])
        self.insertHtml(html)

        regex = QtCore.QRegularExpression(r'(\[\[(?:EQN|VAL|PLT|UNT)[0-9].*?\]\])')
        cursor = document.find(regex)
        while cursor is not None and cursor.selectedText():
            tag = cursor.selectedText()
            tagindex = int(tag[5:-2])  # strip [[XXX and closing ]]
            if 'EQN' in tag:
                eqn = self.rpt._eqns[tagindex]
                px = gui_math.pixmap_from_reportmath(eqn)
                im = QtGui.QImage(px)
                cursor.removeSelectedText()
                cursor.insertImage(im)
            elif 'VAL' in tag:
                cursor.removeSelectedText()
                cursor.insertText(self.rpt._values[tagindex].string(**args))
            elif 'UNT' in tag:
                cursor.removeSelectedText()
                cursor.insertHtml(self.rpt._units[tagindex].html(**args))
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


class SaveReportOptions(QtWidgets.QDialog):
    ''' Dialog for selecting save report options '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Save report options')
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setMinimumWidth(500)
        self.btnbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                 QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.btnbox.rejected.connect(self.reject)
        self.btnbox.accepted.connect(self.accept)
        self.cmbFormat = QtWidgets.QComboBox()
        self.cmbFormat.addItems(['HTML', 'Markdown', 'LaTeX', 'PDF', 'Open Office ODT', 'Word DOCX'])

        self.cmbMath = QtWidgets.QComboBox()
        self.cmbMath.addItems(['Mathjax', 'Matplotlib'])
        self.cmbImage = QtWidgets.QComboBox()
        self.cmbImage.addItems(['SVG', 'PNG'])  # EPS?
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

        self.cmbFormat.setCurrentIndex(
            ['html', 'md', 'tex', 'pdf', 'odt', 'docx'].index(gui_settings.rptformat))
        self.cmbImage.setCurrentIndex(['svg', 'png'].index(gui_settings.rpt_imageformat))
        self.cmbMath.setCurrentIndex(['mathjax', 'mpl'].index(gui_settings.rpt_mathformat))
        self.chkUnicode.setChecked(gui_settings.rpt_unicode)

        self.lblMath = QtWidgets.QLabel('Math Renderer')
        self.lblImage = QtWidgets.QLabel('Image Format')

        glayout = QtWidgets.QGridLayout()
        glayout.addWidget(QtWidgets.QLabel('File Format'), 0, 0)
        glayout.addWidget(self.cmbFormat, 0, 1)
        glayout.addWidget(self.lblImage, 1, 0)
        glayout.addWidget(self.cmbImage, 1, 1)
        glayout.addWidget(self.lblMath, 2, 0)
        glayout.addWidget(self.cmbMath, 2, 1)
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

    def get_setup(self):
        ''' Get dictionary of report format options '''
        fmt = ['html', 'md', 'tex', 'pdf', 'odt', 'docx'][self.cmbFormat.currentIndex()]
        math = ['latex', 'mpl'][self.cmbMath.currentIndex()]
        img = self.cmbImage.currentText().lower()
        if math == 'mpl':
            math = img
        return {'mathfmt': math, 'fmt': fmt, 'image': img, 'unicode': self.chkUnicode.isChecked()}


def savereport(rpt, **kwargs):
    ''' Save Report object contents to file, prompting user for options and file name '''
    kargs = ChainMap(kwargs, gui_settings.report_args)
    dlg = SaveReportOptions()
    ok = dlg.exec()
    if ok:
        setup = dlg.get_setup()
        fmt = setup.get('fmt', 'html')
        filt = {'html': 'HTML (*.html)',
                'tex': 'LaTeX source (*.tex)',
                'md': 'Markdown (*.md *.txt)',
                'docx': 'Word DOCX (*.docx)',
                'pdf': 'PDF (*.pdf)',
                'odt': 'Open Document Text (*.odt)'}[fmt]

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='File to Save', filter=filt)
        if fname:
            if fmt == 'md':
                data = rpt.get_md(mathfmt='latex',
                                  figfmt=setup.get('image'),
                                  unicode=setup.get('unicode', True),
                                  **kargs)
                with open(fname, 'w', encoding='utf-8') as f:
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
                QtWidgets.QMessageBox.warning(None, 'Error saving report', f'Error saving report:\n\n{err}')
