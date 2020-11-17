from . import report


class Output(object):
    ''' Generic output object. Each calculation type (uncertainty, reverse, curvefit, etc)
        subclasses this and must implement the report() method. Subclasses may also implement
        get_dists() and get_dataset() to provide raw output data.
    '''
    def __str__(self):
        ''' String representation of output '''
        return str(self.report())

    def __repr__(self):
        return self.__str__()

    def _repr_markdown_(self):
        ''' Markdown representation for display in Jupyter '''
        return self.report().get_md()

    def report(self, **kwargs):
        ''' Generate report.Report object for this calculation.

            Should return the essential calculation results in
            report.Report format.

            Keyword Arguments
            -----------------
            see report.Report

            Returns
            -------
            report.Report object
        '''
        return report.Report(**kwargs)

    def report_summary(self, **kwargs):
        ''' Generate a summary report. Includes the typical information and plots. '''
        return self.report(**kwargs)

    def report_all(self, **kwargs):
        ''' Generate a full report, including all results and plots. '''
        return self.report(**kwargs)

    def get_dists(self):
        ''' Return a distribution from the output '''
        return {}

    def get_dataset(self, name=None, **kwargs):
        ''' Return a DataSet from the output '''
        return None
