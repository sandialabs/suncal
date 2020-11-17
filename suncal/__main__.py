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

import suncal as uc
from suncal import project
from suncal import report
from suncal import reverse
from suncal import risk
from suncal import curvefit
from suncal import distributions
from suncal import unitmgr


def main_setup(args=None):
    ''' Run calculations defined in YAML setup file '''
    parser = argparse.ArgumentParser(prog='suncalf', description='Run uncertainty calculation from setup file.')
    parser.add_argument('filename', help='Setup parameter file.', type=str)
    parser.add_argument('-o', help='Output filename. Extension determines file format.', type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']", type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('--verbose', '-v', help='Verbose mode. Include plots with one v, full report with two.', default=0, action='count')
    args = parser.parse_args(args=args)

    u = project.Project.from_configfile(args.filename)
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
        return
    elif fmt == 'odt':
        r.save_odt(args.o.name)
        return
    elif fmt == 'pdf':
        r.save_pdf(args.o.name)
        return
    elif fmt == 'html':
        strreport = r.get_html(mathfmt='latex', figfmt='svg')
    elif fmt == 'md':
        strreport = r.get_md(mathfmt='latex', figfmt='svg')
    else:
        strreport = r.get_md(mathfmt='text', figfmt='text')

    args.o.write(strreport)


def main_unc(args=None):
    ''' Run uncertainty propagation problem '''
    parser = argparse.ArgumentParser(prog='suncal', description='Compute combined uncertainty of a system.')
    parser.add_argument('funcs', nargs='+', help='Measurement model functions (e.g. "f = x + y")', type=str)
    parser.add_argument('--units', nargs='+', help='List of units for each function output', type=str, default=None)
    parser.add_argument('--variables', nargs='+', help='List of variable measured values (e.g. "x=10")', type=str)
    parser.add_argument('--uncerts', nargs='+', help='List of uncertainty components, parameters separated by semicolons. First parameter must be variable name. (e.g. "x; unc=2; k=2")', type=str)
    parser.add_argument('--correlate', nargs='+', help='List of correlation coefficients between pairs of variables. (e.g. "x; y; .8")', type=str)
    parser.add_argument('-o', help='Output filename. Extension determines file format.', type=argparse.FileType('w', encoding='UTF-8'), default=sys.stdout)
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']", type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('--samples', help='Number of Monte Carlo samples', type=int, default=1000000)
    parser.add_argument('--seed', help='Random Generator Seed', type=int, default=None)
    parser.add_argument('-s', help='Short output format, prints values only. Prints (GUM mean, GUM std. uncert, GUM expanded, GUM k, MC mean, MC std. uncert, MC expanded min, MC expanded max, MC k) for each function', action='store_true')
    parser.add_argument('--verbose', '-v', help='Verbose mode. Include plots with one v, full report with two.', action='count', default=0)
    args = parser.parse_args(args=args)

    u = uc.UncertaintyCalc(args.funcs, units=args.units, samples=args.samples, seed=args.seed)
    if args.variables is not None:
        for var in args.variables:
            name, val = var.split('=')
            units = None
            nom = unitmgr.parse_expression(val)
            if hasattr(nom, 'magnitude'):
                units = str(nom.units)
                nom = nom.magnitude
            u.set_input(name.strip(), nom=nom, units=units)

    if args.uncerts is not None:
        for uncert in args.uncerts:
            var, *uncargs = uncert.split(';')
            uncargs = dict(u.split('=') for u in uncargs)
            for key in list(uncargs.keys()):  # Use list since pop will modify keys
                uncargs[key.strip()] = uncargs.pop(key)
            u.set_uncert(var.strip(), **uncargs)

    if args.correlate is not None:
        for corr in args.correlate:
            x, y, c = corr.split(';')
            u.correlate_vars(x.strip(), y.strip(), float(c))
    u.calculate()

    if args.s:   # Print out short-format results
        for func in u.functions:
            gumexp, gumk = func.out.gum.expanded(.95)
            mcmin, mcmax, mck = func.out.mc.expanded(.95)
            vals = [func.out.gum.mean.magnitude, func.out.gum.uncert.magnitude, gumexp.magnitude, gumk,
                    func.out.mc.mean.magnitude, func.out.mc.uncert.magnitude, mcmin.magnitude, mcmax.magnitude, mck]
            args.o.write(', '.join('{:.9g}'.format(v) if isinstance(v, float) else '{:.9g}'.format(v) for v in vals))  # expanded() returns length-1 arrays
            args.o.write('\n')

    else:    # Print a formatted report
        fmt = args.f
        if args.o and hasattr(args.o, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 1:
            r = u.out.report_all()
        elif args.verbose > 0:
            r = u.out.report_summary()
        else:
            r = u.out.report()

        if fmt == 'docx':
            r.save_docx(args.o.name)
            return
        elif fmt == 'odt':
            r.save_odt(args.o.name)
            return
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
            return
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
        else:
            strreport = r.get_md(mathfmt='text', figfmt='text')

        args.o.write(strreport)


def main_reverse(args=None):
    ''' Run reverse uncertainty propagation problem '''
    parser = argparse.ArgumentParser(prog='suncalrev', description='Compute reverse uncertainty propagation.')
    parser.add_argument('funcs', nargs='+', help='Measurement model functions (e.g. "f = x + y")', type=str)
    parser.add_argument('--solvefor', help='Name of variable to solve for.', type=str, required=True)
    parser.add_argument('--target', help='Target nominal value for solve-for variable', type=float, required=True)
    parser.add_argument('--targetunc', help='Target uncertainty value for solve-for variable', type=float, required=True)
    parser.add_argument('--fidx', help='Index of function (when more than one function is defined).', type=int, default=-1)
    parser.add_argument('--variables', nargs='+', help='List of variable measured values (e.g. "x=10")', type=str)
    parser.add_argument('--uncerts', nargs='+', help='List of uncertainty components, parameters separated by semicolons. First parameter must be variable name. (e.g. "x; unc=2; k=2")', type=str)
    parser.add_argument('--correlate', nargs='+', help='List of correlation coefficients between pairs of variables. (e.g. "x; y; .8")', type=str)
    parser.add_argument('-o', help='Output filename. Extension determines file format.', type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']", type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('--samples', help='Number of Monte Carlo samples', type=int, default=1000000)
    parser.add_argument('--seed', help='Random Number Seed', type=int, default=None)
    parser.add_argument('-s', help='Short output format, prints values only. Prints (GUM mean, GUM std. uncert, MC mean, MC std. uncert) for solve-for variable.', action='store_true')
    parser.add_argument('--verbose', '-v', help='Verbose mode (include plots)', action='count', default=0)
    args = parser.parse_args(args=args)

    u = reverse.UncertReverse(args.funcs, targetnom=args.target, targetunc=args.targetunc, solvefor=args.solvefor, samples=args.samples, seed=args.seed)
    if args.variables is not None:
        for var in args.variables:
            name, val = var.split('=')
            u.set_input(name.strip(), nom=float(val))

    if args.uncerts is not None:
        for uncert in args.uncerts:
            var, *uncargs = uncert.split(';')
            uncargs = dict(u.split('=') for u in uncargs)
            for key in list(uncargs.keys()):
                uncargs[key.strip()] = uncargs.pop(key)
            u.set_uncert(var.strip(), **uncargs)

    if args.correlate is not None:
        for corr in args.correlate:
            x, y, c = corr.split(';')
            u.correlate_vars(x.strip(), y.strip(), float(c))
    out = u.calculate()

    if args.s:   # Print out short-format results
        vals = [out.gumdata['i'], out.gumdata['u_i'],
                out.mcdata['i'], out.mcdata['u_i']]
        args.o.write(', '.join('{:.9g}'.format(v) for v in vals))
        args.o.write('\n')

    else:    # Print a formatted report
        fmt = args.f
        if args.o and hasattr(args, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 1:
            r = u.out.report_all()
        elif args.verbose > 0:
            r = u.out.report_summary()
        else:
            r = u.out.report()

        if fmt == 'docx':
            r.save_docx(args.o.name)
            return
        elif fmt == 'odt':
            r.save_odt(args.o.name)
            return
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
            return
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
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
    parser.add_argument('--procdist', help='Process distribution parameters, semicolon-separated. (e.g. "dist=uniform; median=10; a=3")', type=str, required=True)
    parser.add_argument('--testdist', help='Test distribution parameters, semicolon-separated. (e.g. "median=10; std=.75")', type=str)
    parser.add_argument('-o', help='Output filename. Extension determines file format.', type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']", type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('-s', help='Short output format, prints values only (Process Risk, PFA, PFR)', action='store_true')
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

    rsk = risk.Risk()
    rsk.set_procdist(dproc)
    rsk.set_testdist(dtest)
    rsk.set_speclimits(LL=args.LL, UL=args.UL)
    rsk.set_guardband(GBL=args.GBL, GBU=args.GBU)

    if args.s:
        procrisk = rsk.specific_risk()[1]
        try:
            pfa = rsk.PFA()
            pfr = rsk.PFR()
        except ValueError:
            pfa = np.nan
            pfr = np.nan
        args.o.write('{:.5f}, {:.5f}, {:.5f}\n'.format(procrisk, pfa, pfr))

    else:
        out = rsk.calculate()

        fmt = args.f
        if args.o and hasattr(args, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 0:
            r = out.report_all()
        else:
            r = out.report()

        fmt = str(args.f)
        if args.o and hasattr(args.o, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if fmt == 'docx':
            r.save_docx(args.o.name)
            return
        elif fmt == 'odt':
            r.save_odt(args.o.name)
            return
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
            return
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
        else:
            strreport = r.get_md(mathfmt='text', figfmt='text')

            args.o.write(strreport)


def main_curvefit(args=None):
    ''' Calculate curvefit uncertainty '''
    parser = argparse.ArgumentParser(prog='suncalfit', description='Curve fit uncertainty')
    parser.add_argument('--model', help='Curve model to fit.', choices=['line', 'poly', 'exp'], default='line', type=str)
    parser.add_argument('-x', '--x', help='X-values', nargs='+', type=float)
    parser.add_argument('-y', '--y', help='Y-values', nargs='+', type=float)
    parser.add_argument('--uy', help='Y uncertainty value(s)', nargs='+', type=float)
    parser.add_argument('--ux', help='X uncertainty value(s)', nargs='+', type=float)
    parser.add_argument('--order', help='Order of polynomial fit', type=int, default=2)
    parser.add_argument('--methods', help='Uncertainty calculation method', choices=['lsq', 'mc', 'mcmc', 'gum'], default='lsq', nargs='+', type=str)
    parser.add_argument('-o', help='Output filename. Extension determines file format.', type=argparse.FileType('w', encoding='UTF-8'), default='-')
    parser.add_argument('-f', help="Output format for when output filename not provided ['txt', 'html', 'md']", type=str, choices=['html', 'txt', 'md'])
    parser.add_argument('-s', help='Short output format, prints values only. First line is fit parameter values (a, b, ...), second is parameter uncertainties. Lines repeat for each method.', action='store_true')
    parser.add_argument('--verbose', '-v', help='Verbose mode. Include plots with one v, full report with two.', default=0, action='count')
    args = parser.parse_args(args=args)

    x = np.array(args.x)
    y = np.array(args.y)
    ux = args.ux if args.ux else 0
    uy = args.uy if args.uy else 0

    arr = curvefit.Array(x, y, ux, uy)
    fit = curvefit.CurveFit(arr, args.model, polyorder=args.order)
    methods = dict((m, True) for m in args.methods)
    fit.calculate(**methods)

    if args.s:
        for base in fit.out._baseoutputs:
            args.o.write(', '.join('{:.9g}'.format(m) for m in base.coeffs) + '\n')
            args.o.write(', '.join('{:.9g}'.format(m) for m in base.sigmas) + '\n')

    else:
        fmt = args.f
        if args.o and hasattr(args, 'name') and args.o.name != '<stdout>':
            _, fmt = os.path.splitext(str(args.o.name))
            fmt = fmt[1:]  # remove '.'

        if args.verbose > 1:
            r = fit.out.report_all()
        elif args.verbose > 0:
            r = fit.out.report_summary()
        else:
            r = fit.out.report()
        if fmt == 'docx':
            r.save_docx(args.o.name)
            return
        elif fmt == 'odt':
            r.save_odt(args.o.name)
            return
        elif fmt == 'pdf':
            r.save_pdf(args.o.name)
            return
        elif fmt == 'html':
            strreport = r.get_html(mathfmt='latex', figfmt='svg')
        elif fmt == 'md':
            strreport = r.get_md(mathfmt='latex', figfmt='svg')
        else:
            strreport = r.get_md(mathfmt='text', figfmt='text')

            args.o.write(strreport)


if __name__ == '__main__':
    main_setup()
