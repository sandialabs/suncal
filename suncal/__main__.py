#!/usr/bin/env python
''' Sandia PSL Uncertainty Calculator - Sandia National Labs
    Command line interface to Uncertainty Calculator.

    Multiple commands are installed:
        suncal: Calculate normal uncertainty propagation problem
        suncalf: Calculate uncertainty problem from config (yaml) file
        suncalrev: Calculate reverse uncertainty problem
        suncalrisk: Calculate risk analysis
        suncalfit: Calculate curve fit uncertainty
'''
import os
import sys
import argparse
import numpy as np

from suncal.common import unitmgr, distributions
from suncal.project import Project, ProjectUncert, ProjectReverse, ProjectRisk, ProjectCurveFit
from suncal import Model
from suncal.reverse import ModelReverse
from suncal import curvefit


def main_setup(args=None):
    ''' Run calculations defined in YAML setup file '''
    parser = argparse.ArgumentParser(prog='suncalf', description='Run uncertainty calculation from setup file.')
    parser.add_argument('filename', help='Setup parameter file.', type=str)
    parser.add_argument('-o', help='Output filename. Extension determines file format.',
                        type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']",
                        type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('--verbose', '-v', help='Verbose mode. Include plots with one v, full report with two.',
                        default=0, action='count')
    args = parser.parse_args(args=args)

    u = Project.from_configfile(args.filename)
    u.calculate()

    fmt = args.f
    if args.o and hasattr(args, 'name') and args.o.name != '<stdout>':
        _, fmt = os.path.splitext(str(args.o.name))
        fmt = fmt[1:]  # remove '.'

    if args.verbose > 1:
        r = u.report_all()
    elif args.verbose > 0:
        r = u.report_summary()
    else:
        r = u.report_short()

    if fmt == 'docx':
        r.save_docx(args.o.name)
    elif fmt == 'odt':
        r.save_odt(args.o.name)
    elif fmt == 'pdf':
        r.save_pdf(args.o.name)
    elif fmt == 'html':
        strreport = r.get_html(mathfmt='latex', figfmt='svg')
        args.o.write(strreport)
    elif fmt == 'md':
        strreport = r.get_md(mathfmt='latex', figfmt='svg')
        args.o.write(strreport)
    else:
        strreport = r.get_md(mathfmt='text', figfmt='text')
        args.o.write(strreport)


def main_unc(args=None):
    ''' Run uncertainty propagation problem '''
    parser = argparse.ArgumentParser(prog='suncal', description='Compute combined uncertainty of a system.')
    parser.add_argument('funcs', nargs='+',
                        help='Measurement model functions (e.g. "f = x + y")', type=str)
    parser.add_argument('--units', nargs='+',
                        help='List of units for each function output', type=str, default=None)
    parser.add_argument('--variables', nargs='+',
                        help='List of variable measured values (e.g. "x=10")', type=str)
    parser.add_argument('--uncerts', nargs='+',
                        help='List of uncertainty components, parameters separated by semicolons. '
                             'First parameter must be variable name. (e.g. "x; unc=2; k=2")', type=str)
    parser.add_argument('--correlate', nargs='+', type=str,
                        help='List of correlation coefficients between pairs of variables. (e.g. "x; y; .8")')
    parser.add_argument('-o', help='Output filename. Extension determines file format.',
                        type=argparse.FileType('w', encoding='UTF-8'), default=sys.stdout)
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']",
                        type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('--samples', help='Number of Monte Carlo samples', type=int, default=1000000)
    parser.add_argument('--seed', help='Random Generator Seed', type=int, default=None)
    parser.add_argument('-s', help='Short output format, prints values only. Prints '
                                   '(GUM mean, GUM std. uncert, GUM expanded, GUM k, MC mean, MC std. uncert, '
                                   'MC expanded min, MC expanded max, MC k) for each function', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Verbose mode. Include plots with one v, full report with two.')
    args = parser.parse_args(args=args)

    u = ProjectUncert(Model(*args.funcs))
    u.outunits = args.units
    u.samples = args.samples
    u.seed = args.seed
    if args.variables is not None:
        for var in args.variables:
            name, val = var.split('=')
            units = None
            nom = unitmgr.parse_expression(val)
            if hasattr(nom, 'magnitude'):
                units = str(nom.units)
                nom = nom.magnitude
            u.model.var(name.strip()).measure(nom, units=units)

    if args.uncerts is not None:
        for uncert in args.uncerts:
            var, *uncargs = uncert.split(';')
            uncargs = dict(u.split('=') for u in uncargs)
            for key in list(uncargs.keys()):  # Use list since pop will modify keys
                uncargs[key.strip()] = uncargs.pop(key)
            u.model.var(var.strip()).typeb(**uncargs)

    if args.correlate is not None:
        for corr in args.correlate:
            x, y, c = corr.split(';')
            u.model.variables.correlate(x.strip(), y.strip(), float(c))
    result = u.calculate()

    if args.s:   # Print out short-format results
        expanded = result.gum.expanded(.95)
        mcexpanded = result.montecarlo.expanded(.95)
        for func in u.model.functionnames:
            gumexp, gumk, _ = expanded[func]
            mcmin, mcmax, mck, _ = mcexpanded[func]
            vals = [result.gum.expected[func], result.gum.uncertainty[func], gumexp, gumk,
                    result.montecarlo.expected[func], result.montecarlo.uncertainty[func],
                    mcmin, mcmax, mck]
            args.o.write(', '.join(f'{v:.9g}' if isinstance(v, float) else f'{v:.9g}' for v in vals))
            args.o.write('\n')

    else:    # Print a formatted report
        fmt = args.f
        if args.o and hasattr(args.o, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 1:
            r = result.report.all()
        elif args.verbose > 0:
            r = result.report.summary_withplots()
        else:
            r = result.report.summary()

        if fmt == 'docx':
            r.save_docx(args.o.name)
        elif fmt == 'odt':
            r.save_odt(args.o.name)
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        else:
            strreport = r.get_md(mathfmt='text', figfmt='text')
            args.o.write(strreport)


def main_reverse(args=None):
    ''' Run reverse uncertainty propagation problem '''
    parser = argparse.ArgumentParser(prog='suncalrev', description='Compute reverse uncertainty propagation.')
    parser.add_argument('funcs', nargs='+', help='Measurement model functions (e.g. "f = x + y")', type=str)
    parser.add_argument('--solvefor', help='Name of variable to solve for.', type=str, required=True)
    parser.add_argument('--target', help='Target nominal value for solve-for variable', type=float, required=True)
    parser.add_argument('--targetunc',
                        help='Target uncertainty value for solve-for variable', type=float, required=True)
    parser.add_argument('--fidx', help='Index of function (when more than one function is defined).',
                        type=int, default=-1)
    parser.add_argument('--variables', nargs='+', help='List of variable measured values (e.g. "x=10")', type=str)
    parser.add_argument('--uncerts', nargs='+',
                        help='List of uncertainty components, parameters separated by semicolons. '
                        'First parameter must be variable name. (e.g. "x; unc=2; k=2")', type=str)
    parser.add_argument('--correlate', nargs='+',
                        help='List of correlation coefficients between pairs of variables. (e.g. "x; y; .8")', type=str)
    parser.add_argument('-o', help='Output filename. Extension determines file format.',
                        type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']",
                        type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('--samples', help='Number of Monte Carlo samples', type=int, default=1000000)
    parser.add_argument('--seed', help='Random Number Seed', type=int, default=None)
    parser.add_argument('-s', help='Short output format, prints values only. '
                        'Prints (GUM mean, GUM std. uncert, MC mean, MC std. uncert) for solve-for variable.',
                        action='store_true')
    parser.add_argument('--verbose', '-v', help='Verbose mode (include plots)', action='count', default=0)
    args = parser.parse_args(args=args)

    model = ModelReverse(*args.funcs, targetnom=args.target, targetunc=args.targetunc, solvefor=args.solvefor)
    u = ProjectReverse(model)
    u.nsamples = args.samples
    u.seed = args.seed
    if args.variables is not None:
        for var in args.variables:
            name, val = var.split('=')
            u.model.var(name.strip()).measure(float(val))

    if args.uncerts is not None:
        for uncert in args.uncerts:
            var, *uncargs = uncert.split(';')
            uncargs = dict(u.split('=') for u in uncargs)
            for key in list(uncargs.keys()):
                uncargs[key.strip()] = uncargs.pop(key)
            u.model.var(var.strip()).typeb(**uncargs)

    if args.correlate is not None:
        for corr in args.correlate:
            x, y, c = corr.split(';')
            u.model.variables.correlate(x.strip(), y.strip(), float(c))
    result = u.calculate()

    if args.s:   # Print out short-format results
        vals = [result.gum.solvefor_value, result.gum.u_solvefor_value,
                result.montecarlosolvefor_value, result.montecarlo.u_solvefor_value]
        args.o.write(', '.join(f'{v:.9g}' for v in vals))
        args.o.write('\n')

    else:    # Print a formatted report
        fmt = args.f
        if args.o and hasattr(args, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 1:
            r = result.report.all()
        elif args.verbose > 0:
            r = result.report.summary_withplots()
        else:
            r = result.report.summary()

        if fmt == 'docx':
            r.save_docx(args.o.name)
        elif fmt == 'odt':
            r.save_odt(args.o.name)
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        else:
            strreport = r.get_md(mathfmt='text', figfmt='text')
            args.o.write(strreport)


def main_risk(args=None):
    ''' Calculate risk analysis '''
    parser = argparse.ArgumentParser(prog='suncalrisk', description='Risk Analysis Calculation')
    parser.add_argument('-LL', help='Lower specification limit', type=float, required=True)
    parser.add_argument('-UL', help='Upper specification limit', type=float, required=True)
    parser.add_argument('-GBL', help='Lower guardband as offset from lower limit', type=float, default=0)
    parser.add_argument('-GBU', help='Upper guardband as offset from upper limit', type=float, default=0)
    parser.add_argument('--procdist', type=str, required=True,
                        help='Process distribution parameters, semicolon-separated. '
                        '(e.g. "dist=uniform; median=10; a=3")')
    parser.add_argument('--testdist', type=str,
                        help='Test distribution parameters, semicolon-separated. (e.g. "median=10; std=.75")')
    parser.add_argument('-o', help='Output filename. Extension determines file format.',
                        type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']",
                        type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('-s', help='Short output format, prints values only (Process Risk, PFA, PFR)',
                        action='store_true')
    parser.add_argument('--verbose', '-v', help='Verbose mode (include plots)', action='count', default=0)
    args = parser.parse_args(args=args)

    dproc = {}
    for keyval in str(args.procdist).split(';'):
        key, val = keyval.split('=')
        if key.strip() != 'dist':
            dproc[key.strip()] = float(val)
        else:
            dproc[key.strip()] = val
    dproc = distributions.from_config(dproc)

    dtest = None
    if args.testdist is not None:
        dtest = {}
        for keyval in str(args.testdist).split(';'):
            key, val = keyval.split('=')
            if key.strip() != 'dist':
                dtest[key.strip()] = float(val)
            else:
                dtest[key.strip()] = val
        dtest = distributions.from_config(dtest)

    rsk = ProjectRisk()
    rsk.model.procdist = dproc
    rsk.model.testdist = dtest
    rsk.model.speclimits = (args.LL, args.UL)
    rsk.model.gbofsts = (args.GBL, args.GBU)

    if args.s:
        procrisk = rsk.model.process_risk()
        try:
            pfa = rsk.model.PFA()
            pfr = rsk.model.PFR()
        except ValueError:
            pfa = np.nan
            pfr = np.nan
        args.o.write(f'{procrisk:.5f}, {pfa:.5f}, {pfr:.5f}\n')

    else:
        result = rsk.calculate()

        fmt = args.f
        if args.o and hasattr(args, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 0:
            r = result.report.all()
        else:
            r = result.report.summary()

        fmt = str(args.f)
        if args.o and hasattr(args.o, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if fmt == 'docx':
            r.save_docx(args.o.name)
        elif fmt == 'odt':
            r.save_odt(args.o.name)
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        else:
            strreport = r.get_md(mathfmt='text', figfmt='text')
            args.o.write(strreport)


def main_curvefit(args=None):
    ''' Calculate curvefit uncertainty '''
    parser = argparse.ArgumentParser(prog='suncalfit', description='Curve fit uncertainty')
    parser.add_argument('--model', help='Curve model to fit.',
                        choices=['line', 'poly', 'exp'], default='line', type=str)
    parser.add_argument('-x', '--x', help='X-values', nargs='+', type=float)
    parser.add_argument('-y', '--y', help='Y-values', nargs='+', type=float)
    parser.add_argument('--uy', help='Y uncertainty value(s)', nargs='+', type=float)
    parser.add_argument('--ux', help='X uncertainty value(s)', nargs='+', type=float)
    parser.add_argument('--order', help='Order of polynomial fit', type=int, default=2)
    parser.add_argument('-o', help='Output filename. Extension determines file format.',
                        type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']",
                        type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('-s', help='Short output format, prints values only. First line is fit parameter values '
                        '(a, b, ...), second is parameter uncertainties. Lines repeat for each method.',
                        action='store_true')
    parser.add_argument('--verbose', '-v', help='Verbose mode. Includes plots.',
                        default=0, action='count')
    args = parser.parse_args(args=args)

    x = np.array(args.x)
    y = np.array(args.y)
    ux = args.ux if args.ux else 0
    uy = args.uy if args.uy else 0

    arr = curvefit.Array(x, y, ux, uy)
    fit = curvefit.CurveFit(arr, args.model, polyorder=args.order)
    proj = ProjectCurveFit(fit)
    result = proj.calculate()

    if args.s:
        args.o.write(', '.join(f'{m:.9g}' for m in result.lsq.coeffs) + '\n')
        args.o.write(', '.join(f'{m:.9g}' for m in result.lsq.uncerts) + '\n')

    else:
        fmt = args.f
        if args.o and hasattr(args, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 0:
            r = result.report.all()
        else:
            r = result.report.summary()

        if fmt == 'docx':
            r.save_docx(args.o.name)
        elif fmt == 'odt':
            r.save_odt(args.o.name)
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
            args.o.write(strreport)
        else:
            strreport = r.get_md(mathfmt='text', figfmt='text')
            args.o.write(strreport)


if __name__ == '__main__':
    main_setup()
