''' Pyscript interface to Suncal Distribution Explorer '''
from pyscript import document, display
from js import alert

from scipy import stats
import matplotlib.pyplot as plt
from suncal import distexplore
from suncal.common import uparser
from suncal.common.report import Report

Report.apply_css = False  # Use this page's CSS, not Suncal's built-in CSS


class DistExp:
    ''' Distribution Explorer State '''
    def __init__(self):
        self.distexplore = distexplore.DistExplore()
        self.fig = plt.figure()
        self.name = None

    def add_dists(self, event):
        ''' Add or remove distributions from the list '''
        number = int(document.querySelector('#number_dists').value)
        for i in range(4):
            if i < number:
                # NOTE: document.querySelector returns something that won't allow changing style properties...
                document.querySelector(f'#distribution{i+1}').style.visibility = 'visible'
            else:
                document.querySelector(f'#distribution{i+1}').style.visibility = 'hidden'

    def set_samples(self, event):
        ''' Set the number of samples '''
        number = int(document.querySelector('#samples').value)
        self.distexplore.set_numsamples(number)

    def set_expr(self, event):
        ''' Set the expression for one distribution '''
        button = event.target.id[-1]  # id is "run1", "run2", etc.
        element = document.querySelector(f'#expr{button}')
        name = element.value
        try:
            expr = uparser.parse_math(name)
        except ValueError as err:
            display('', target=f'expr{button}', append=False)
            document.querySelector(f'#run{button}').disabled = True
            alert(str(err))
        else:
            document.querySelector(f'#run{button}').disabled = False
            if hasattr(expr, 'is_symbol') and expr.is_symbol:
                # Single variable, can edit distribution
                document.querySelector(f'#distvalues{button}').style.visibility = 'visible'
            else:
                # Monte Carlo, hide distribution controls
                document.querySelector(f'#distvalues{button}').style.visibility = 'hidden'

    def set_dist(self, button):
        ''' Set the parameters for one distribution '''
        name = document.querySelector(f'#expr{button}').value
        mean = float(document.querySelector(f'#mean{button}').value)
        halfwidth = float(document.querySelector(f'#a{button}').value)
        distname = document.querySelector(f'#dist{button}').value

        if distname == 'uniform':
            dist = stats.uniform(loc=mean-halfwidth, scale=halfwidth*2)
        elif distname == 'triangular':
            dist = stats.triang(loc=mean-halfwidth, scale=halfwidth*2, c=0.5)
        else:
            dist = stats.norm(loc=mean, scale=halfwidth)

        self.distexplore.dists[name] = dist

    def update_plot(self, event=None):
        ''' Update the output plot and table '''
        if self.name:
            display(self.distexplore.report.single(
                self.name), target='stats', append=False)

            fitdist = document.querySelector('#fitdist').value
            qq = document.querySelector('#probplot').checked
            interval = document.querySelector('#coverage').checked
            self.distexplore.report.plot.hist(
                self.name, fig=self.fig, fitdist=fitdist,
                qqplot=qq, interval=interval)
            self.fig.gca().set_title(self.name)
            display(self.fig, target='plot', append=False)

    def sample(self, event):
        ''' Generate random samples and display results '''
        button = event.target.id[-1]  # id is "run1", "run2", etc.

        for i in range(1, 5):
            self.set_dist(i)

        try:
            self.distexplore.seed = int(document.querySelector('#seed').value)
        except ValueError:
            self.distexplore.seed = None

        try:
            self.distexplore.calculate()
        except ValueError as err:
            alert(str(err))
            self.distexplore.dists = {}
        else:
            self.name = document.querySelector(f'#expr{button}').value
            self.update_plot()


distexp = DistExp()
