''' Qt Widgets for selecting a matplotlib colormap. '''

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtCore, QtWidgets

# Sequential colormaps defined in matplotlib 2.0
DFLT_CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
              'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'gist_heat', 'copper',
              'twilight', 'hsv', 'ocean', 'gist_earth', 'terrain', 'gnuplot', 'cubehelix', 'brg', 'gist_rainbow', 'jet'
              ]


class ColorMapPickerWidget(QtWidgets.QWidget):
    ''' Widget for displaying and selecting a colormap '''
    cmappicked = QtCore.pyqtSignal(str)

    def __init__(self, selectedcmap=None, cmaps=None):
        super().__init__()

        self.cmaps = cmaps if cmaps is not None else DFLT_CMAPS
        if selectedcmap in self.cmaps:
            self.selectedidx = self.cmaps.index(selectedcmap)
        else:
            self.selectedidx = None
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.onclick)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.drawColormaps()

    def drawColormaps(self):
        ''' Use MPL to plot the available colormaps '''
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        self.axes = []
        self.labels = []
        for i, cm in enumerate(self.cmaps):
            ax = self.fig.add_subplot(len(self.cmaps), 1, (i+1))
            ax.imshow(gradient, aspect='auto', cmap=mpl.cm.get_cmap(cm))
            ax.set_axis_off()
            pos = list(ax.get_position().bounds)
            label = self.fig.text(pos[0]-.01, pos[1]+pos[3]/2, cm, va='center', ha='right', fontsize=10)
            if i == self.selectedidx:
                label.set_fontstyle('italic')
                label.set_color('red')
            self.labels.append(label)
            self.axes.append(ax)
        self.canvas.draw_idle()

    def onclick(self, event):
        ''' A colormap was clicked (single or double click) '''
        if event.inaxes:
            self.selectedidx = self.axes.index(event.inaxes)
            if event.dblclick:
                cmap = self.cmaps[self.selectedidx]
                self.cmappicked.emit(cmap)
            else:
                for i in range(len(self.axes)):
                    if i != self.selectedidx:
                        self.labels[i].set_fontstyle('normal')
                        self.labels[i].set_color('black')
                    else:
                        self.labels[i].set_fontstyle('italic')
                        self.labels[i].set_color('red')
                self.canvas.draw_idle()

    def get_selectedcmap(self):
        ''' Get the currently selected colormap '''
        if self.selectedidx is not None:
            return self.cmaps[self.selectedidx]
        return None


class ColorMapDialog(QtWidgets.QDialog):
    ''' Dialog for showing colormappicker and ok/cancel buttons '''
    def __init__(self, parent=None, selectedcmap=None):
        super().__init__(parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setGeometry(600, 200, 600, 800)
        self.setWindowTitle('Select Color Map')

        self.cpicker = ColorMapPickerWidget(selectedcmap)
        self.bbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        self.cpicker.cmappicked.connect(self.accept)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.cpicker)
        layout.addWidget(self.bbox)
        self.setLayout(layout)
        self.selectedcmap = selectedcmap

    def accept(self):
        ''' Accept the dialog with selected color '''
        self.selectedcmap = self.cpicker.get_selectedcmap()
        super().accept()
