''' Decorator for adding Report object and Markdown representer to dataclass

    Usage:

        @reporter.reporter(ReportXYZ)
        @dataclass
        class ResultsXYZ:
            ...
'''


def reporter(reportclass):
    def decorator(resultclass):

        @property
        def report(self):
            return reportclass(self)

        def _repr_markdown_(self):
            return self.report.summary().get_md()

        setattr(resultclass, 'report', report)
        setattr(resultclass, '_repr_markdown_', _repr_markdown_)
        return resultclass
    return decorator
