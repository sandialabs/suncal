''' Pyscript interface to Suncal Distribution Explorer '''
from pyscript import Element, display
from js import alert

from scipy import stats
import matplotlib.pyplot as plt
import suncal
from suncal.common import uparser
from suncal.common.report import Report

Report.apply_css = False  # Use this page's CSS, not Suncal's built-in CSS


class DistExp:
    ''' Distribution Explorer State '''
    def __init__(self):
        self.distexplore = suncal.distexplore.DistExplore()
        self.fig = plt.figure()
        self.name = None

    def add_dists(self):
        ''' Add or remove distributions from the list '''
        number = int(Element('number_dists').value)
        for i in range(4):
            if i < number:
                Element(f'distribution{i+1}').remove_class('input-hidden')
            else:
                Element(f'distribution{i+1}').add_class('input-hidden')

    def set_samples(self):
        ''' Set the number of samples '''
        number = int(Element('samples').value)
        self.distexplore.set_numsamples(number)

    def set_expr(self, button='1'):
        ''' Set the expression for one distribution '''
        element = Element(f'expr{button}')
        name = element.value
        try:
            expr = uparser.parse_math(name)
        except ValueError as err:
            element.clear()
            Element(f'run{button}').element.disabled = True
            alert(str(err))
        else:
            Element(f'run{button}').element.disabled = False
            if hasattr(expr, 'is_symbol') and expr.is_symbol:
                # Single variable, can edit distribution
                Element(f'distvalues{button}').remove_class('input-hidden')
            else:
                # Monte Carlo, hide distribution controls
                Element(f'distvalues{button}').add_class('input-hidden')

    def set_dist(self, button):
        ''' Set the parameters for one distribution '''
        name = Element(f'expr{button}').value
        mean = float(Element(f'mean{button}').value)
        halfwidth = float(Element(f'a{button}').value)
        distname = Element(f'dist{button}').value

        if distname == 'uniform':
            dist = stats.uniform(loc=mean-halfwidth, scale=halfwidth*2)
        elif distname == 'triangular':
            dist = stats.triang(loc=mean-halfwidth, scale=halfwidth*2, c=0.5)
        else:
            dist = stats.norm(loc=mean, scale=halfwidth)

        self.distexplore.dists[name] = dist

    def update_plot(self):
        ''' Update the output plot and table '''
        if self.name:
            Element('stats').clear()   # append=False not working below?
            display(self.distexplore.report.single(
                self.name), target='stats', append=False)

            fitdist = Element('fitdist').value
            qq = Element('probplot').element.checked
            interval = Element('coverage').element.checked
            self.distexplore.report.plot.hist(
                self.name, fig=self.fig, fitdist=fitdist,
                qqplot=qq, interval=interval)
            self.fig.gca().set_title(self.name)
            display(self.fig, target='plot', append=False)

    def sample(self, button='1'):
        ''' Generate random samples and display results '''
        for i in range(1, 5):
            self.set_dist(i)

        try:
            self.distexplore.seed = int(Element('seed').value)
        except ValueError:
            self.distexplore.seed = None

        try:
            self.distexplore.calculate()
        except ValueError as err:
            alert(str(err))
            self.distexplore.dists = {}
        else:
            self.name = Element(f'expr{button}').value
            self.update_plot()
