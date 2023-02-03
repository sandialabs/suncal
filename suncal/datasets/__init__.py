''' DataSets are 1D or 2D sets of measured data, for computing repeatability and reproducibility information '''

from .dataset import autocorrelation, uncert_autocorrelated, anova, group_stats, pooled_stats, standarderror

__all__ = ['autocorrelation', 'uncert_autocorrelated', 'anova', 'group_stats', 'pooled_stats', 'standarderror']
