''' Inline help reports for Distribution Explorer tools '''
from ...common import report


class DistExploreHelp:
    @staticmethod
    def disthelp():
        rpt = report.Report()
        rpt.hdr('Distribution Explorer', level=2)
        rpt.txt('Mostly for training and educational purposes, the Distribution '
                'Explorer allows entry of probability distributions, '
                'generating random samples from distributions, and combining '
                'multiple distributions through the Monte Carlo method.\n\n')
        rpt.txt('Use the + and - buttons to add or remove new probability distributions. '
                'Give each distribution a name (but avoid Python keywords or an `ERROR` '
                'will be shown). The `normal...` button is used to change the '
                'distribution parameters. Press the `Sample` button to generate '
                'and display random samples from that distribution.\n\n'
                'To perform a Monte Carlo analysis, enter one or more distributions, '
                'then add a new distribution where the name is an expression, '
                'such as `a+b`. Common arithmetic functions, trigonomotric functions, '
                'exponential and log functions are recognized. When a Monte Carlo '
                'expression is entered, the `Sample` button changes to `Calculate`. '
                'Use this button to see the results of the combined distributions.')
        return rpt
