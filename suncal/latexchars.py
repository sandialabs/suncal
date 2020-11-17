r'''
Custom error handler for str.encode() that converts special unicode characters
into LaTeX codes that are pdflatex-compatible. Importing this module
installs the error handler named 'latex'.

Usage:

    '100 µΩ'.encode('ascii', 'latex')
    >>> '100 $\mu \Omega$'

'''

import codecs

# Could add others, but this gets all the greek letters, sub/superscripts,
# and the most common math symbols.
textable = {
    'Α': 'A',        # Greek capital alpha - replace with A
    'Β': 'B',        # Greek capital beta...
    'Γ': r'\Gamma',
    'Δ': r'\Delta',
    'Ε': 'E',
    'Ζ': 'Z',
    'Η': 'H',
    'Ι': 'I',
    'Κ': 'K',
    'Θ': r'\Theta',
    'Λ': r'\Lambda',
    'Μ': 'M',
    'Ν': 'N',
    'Ξ': r'\Xi',
    'Ο': 'O',
    'Π': r'\Pi',
    'Ρ': 'P',
    'Σ': r'\Sigma',
    'Τ': 'T',
    'Υ': r'\Upsilon',
    'Φ': r'\Phi',
    'Χ': 'X',
    'Ψ': r'\Psi',
    'Ω': r'\Omega',
    'α': r'\alpha',
    'β': r'\beta',
    'γ': r'\gamma',
    'δ': r'\delta',
    'ϵ': r'\epsilon',
    'ε': r'\varepsilon',
    'ζ': r'\zeta',
    'η': r'\eta',
    'θ': r'\theta',
    'ϑ': r'\vartheta',
    'ι': r'\iota',
    'κ': r'\kappa',
    'λ': r'\lambda',
    'μ': r'\mu',  # \u03BC - "Greek small letter mu"
    'µ': r'\mu',  # \uB5 - alt+m on mac "Micro Sign"
    'ν': r'\nu',
    'ξ': r'\xi',
    'ο': r'\omicron',
    'π': r'\pi',
    'ϖ': r'\varpi',
    'ρ': r'\rho',
    'σ': r'\sigma',
    'τ': r'\tau',
    'υ': r'\upsilon',
    'ϕ': r'\phi',
    'φ': r'\varphi',
    'χ': r'\chi',
    'ψ': r'\psi',
    'ω': r'\omega',

    '±': r'\pm',
    '∓': r'\mp',
    '×': r'\times',
    '÷': r'\div',
    '≠': r'\neq',
    '≤': r'\leq',
    '≥': r'\geq',
    '≪': r'\ll',
    '≫': r'\gg',
    '⊂': r'\subset',
    '⊆': r'\subseteq',
    '∑': r'\sum',
    '∏': r'\prod',
    '∐': r'\coprod',
    '∫': r'\int',
    '∬': r'\iint',
    '∭': r'\iiint',
    '∮': r'\oint',
    '∞': r'\infty',
    '∇': r'\nabla',
    'ℜ': r'\Re',
    'ℑ': r'\Im',
    '∠': r'\angle',
    '∡': r'\measuredangle',
    '℧': r'\mho',
    '∂': r'\partial',
    'Å': r'\mathrm{\mathring{A}}',
    'ℏ': r'\hbar',
    '°': r'^{\circ}',

    '⁰': r'^{0}',
    '¹': r'^{1}',
    '²': r'^{2}',
    '³': r'^{3}',
    '⁴': r'^{4}',
    '⁵': r'^{5}',
    '⁶': r'^{6}',
    '⁷': r'^{7}',
    '⁸': r'^{8}',
    '⁹': r'^{9}',
    '₀': r'_{0}',
    '₁': r'_{1}',
    '₂': r'_{2}',
    '₃': r'_{3}',
    '₄': r'_{4}',
    '₅': r'_{5}',
    '₆': r'_{6}',
    '₇': r'_{7}',
    '₈': r'_{8}',
    '₉': r'_{9}',
    }


def texhandler(err):
    ''' Codec handler for replacing special characters with LaTeX
        math codes, wrapped in $..$ if necessary.
    '''
    tex = [textable.get(err.object[p], '?') for p in range(err.start, err.end)]
    hasdollar = err.object[:err.start].count('$') % 2  # odd number of $ signs, already in math mode
    if not hasdollar:
        return (r'${}$'.format(' '.join(tex)), err.end)
    else:
        return (r'{} '.format(' '.join(tex)), err.end)


codecs.register_error('latex', texhandler)
