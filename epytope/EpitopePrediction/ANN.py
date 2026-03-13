# coding=utf-8
# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: EpitopePrediction.ANN
   :synopsis: This module contains all classes for ANN-based epitope prediction.
.. moduleauthor:: heumos, krakau

"""
import itertools
import logging
import subprocess
import math

from abc import abstractmethod

import pandas
from collections import defaultdict

from epytope.Core import EpitopePredictionResult
from epytope.Core.Base import AEpitopePrediction
from epytope.Core.Allele import Allele, MouseAllele


class AANNEpitopePrediction(AEpitopePrediction):
    """
        Abstract base class for ANN predictions.
        Implements predict functionality
    """

    @abstractmethod
    def predict(self, peptides, alleles=None, binary=False, **kwargs):
        """
        All ANN based predictors have to implement their custom predict method.

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a nested dictionary {allele1: {scoreType1: {pep1: score1, pep2:..}, scoreType2: {..}, ..}, allele2:..}
        :rtype: :class:`pandas.DataFrame`
        """


class MHCFlurryPredictor_1_2_2(AANNEpitopePrediction):
    """
    Implements MHCFlurry

    .. note::
        T. J. O'Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher,
         "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," Cell Systems, 2018.
          Available at: https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.
    """
    __name = "mhcflurry"
    __version = "1.2.2"
    __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

    @property
    def name(self):
        return self.__name

    @property
    def supportedAlleles(self):
        return self.__alleles

    @property
    def supportedLength(self):
        return self.__supported_length

    @property
    def version(self):
        return self.__version

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)
        else:
            return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

    # Converts epytopes internal allele representation into the format required by MHCFlurry
    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
        of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return [self._represent(a) for a in alleles]

    # Converts the internal MHCFlurry representation back into a epytope representation
    def revert_allele_repr(self, name):
        if name.startswith("H-2-"):
            return MouseAllele(name)
        else:
            return Allele(name[:5] + '*' + name[5:7] + ':' + name[7:])

    # predicts the binding affinity for a set of peptides and alleles
    def predict(self, peptides, alleles=None, binary=False, **kwargs):

        try:
            from mhcflurry import Class1AffinityPredictor
        except ImportError:
            raise ImportError("mhcflurry is required for MHCFlurry predictions. "
                              "Install with: pip install epytope[mhcflurry]")

        # test whether one peptide or a list
        if not isinstance(peptides, list):
            peptides = [peptides]

        # if no alleles are specified do predictions for all supported alleles
        if alleles is None:
            alleles = self.supportedAlleles
        else:
            # filter for supported alleles
            alleles = [a for a in alleles if a in self.supportedAlleles]

        # Create a dictionary with Allele Obj as key and the respective allele predictor representation as value
        alleles_repr = {allele: self._represent(allele) for allele in alleles}

        # test mhcflurry models are available => download if not
        p = subprocess.call(['mhcflurry-downloads', 'path', 'models_class1'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p != 0:
            logging.warning("mhcflurry models must be downloaded, as they were not found locally.")
            cp = subprocess.run(['mhcflurry-downloads', 'fetch', 'models_class1'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if cp.returncode != 0:
                for line in cp.stdout.decode().splitlines():
                    logging.error(line)
                raise RuntimeError("mhcflurry failed to download model file")

        # load model
        predictor = Class1AffinityPredictor.load()

        # prepare results dictionary
        scores = defaultdict(defaultdict)

        # keep input peptide objects for later use
        peptide_objects = {}
        for peptide in peptides:
            peptide_objects[str(peptide)] = peptide

        # group peptides by length
        pep_groups = list(peptide_objects.keys())
        pep_groups.sort(key=len)

        for length, peps in itertools.groupby(peptides, key=len):
            if length not in self.supportedLength:
                logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                    self.name, length))
                continue
            peps = list(peps)

            # predict and assign binding affinities
            for a in alleles:
                for p in peps:
                    binding_affinity = predictor.predict(allele=alleles_repr[a], peptides=[str(p)])[0]
                    if binary:
                        if binding_affinity <= 500:
                            scores[a][p] = 1.0
                        else:
                            scores[a][p] = 0.0
                    else:
                        # convert ic50 to raw prediction score
                        scores[a][p] = 1- math.log(binding_affinity, 50000)

        if not scores:
            raise ValueError("No predictions could be made with " + self.name +
                            " for given input. Check your epitope length and HLA allele combination.")

        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': AffScore1, 'Pep2': AffScore2,..}, 'Allele2':...}
        result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

        # create EpitopePredictionResult object. This is a multi-indexed DataFrame
        # with Allele, Method and Score type as multi-columns and peptides as rows
        df_result = EpitopePredictionResult.from_dict(result, peptide_objects.values(), self.name)
        return df_result


class MHCFlurryPredictor_1_4_3(MHCFlurryPredictor_1_2_2):
    """
    Implements MHCFlurry

    .. note::
        T. J. O'Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher,
         "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," Cell Systems, 2018.
          Available at: https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.
    """
    __name = "mhcflurry"
    __version = "1.4.3"
    __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

    @property
    def name(self):
        return self.__name

    @property
    def supportedAlleles(self):
        return self.__alleles

    @property
    def supportedLength(self):
        return self.__supported_length

    @property
    def version(self):
        return self.__version

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)
        else:
            return "%s-%s*%s:%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal :class:`~epytope.Core.Allele.Allele` representation
        of the predictor and returns a string representation

        :param alleles: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return [self._represent(a) for a in alleles]

    # Converts the internal MHCFlurry representation back into a epytope representation
    def revert_allele_repr(self, name):
        if name.startswith("H-2-"):
            return MouseAllele(name)
        else:
            return Allele(name)


class MHCFlurryPredictor_2_0(MHCFlurryPredictor_1_4_3):
    """
    Implements MHCFlurry 2.0

    Uses Class1PresentationPredictor which combines binding affinity with
    antigen processing predictions and provides percentile rank scores.

    .. note::
        T. J. O'Donnell, A. Rubinsteyn, and U. Laserson,
         "MHCflurry 2.0: Improved Pan-Allele Prediction of MHC Class I-Presented Peptides by
          Incorporating Antigen Processing," Cell Systems, 2020.
    """
    __name = "mhcflurry"
    __version = "2.0"
    __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
    __allele_import_name = f"{__name}_{__version}".replace('.', '_')
    __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                   fromlist=[__allele_import_name])
                        , __allele_import_name)

    @property
    def name(self):
        return self.__name

    @property
    def supportedAlleles(self):
        return self.__alleles

    @property
    def supportedLength(self):
        return self.__supported_length

    @property
    def version(self):
        return self.__version

    def predict(self, peptides, alleles=None, binary=False, **kwargs):

        try:
            from mhcflurry import Class1PresentationPredictor
        except ImportError:
            raise ImportError("mhcflurry >= 2.0 is required for MHCFlurry 2.0 predictions. "
                              "Install with: pip install epytope[mhcflurry]")

        if not isinstance(peptides, list):
            peptides = [peptides]

        if alleles is None:
            alleles = self.supportedAlleles
        else:
            alleles = [a for a in alleles if a in self.supportedAlleles]

        alleles_repr = {allele: self._represent(allele) for allele in alleles}

        # test mhcflurry models are available => download if not
        p = subprocess.call(['mhcflurry-downloads', 'path', 'models_class1_presentation'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p != 0:
            logging.warning("mhcflurry presentation models must be downloaded.")
            cp = subprocess.run(['mhcflurry-downloads', 'fetch', 'models_class1_presentation'],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if cp.returncode != 0:
                for line in cp.stdout.decode().splitlines():
                    logging.error(line)
                raise RuntimeError("mhcflurry failed to download presentation model files")

        predictor = Class1PresentationPredictor.load()

        peptide_objects = {}
        for peptide in peptides:
            peptide_objects[str(peptide)] = peptide

        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)

        for length, peps in itertools.groupby(peptides, key=len):
            if length not in self.supportedLength:
                logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (
                    min(self.supportedLength), max(self.supportedLength), self.name, length))
                continue
            peps = list(peps)

            for a in alleles:
                pep_strs = [str(p) for p in peps]
                df = predictor.predict(
                    peptides=pep_strs,
                    alleles=[alleles_repr[a]] * len(pep_strs),
                    verbose=0
                )
                for p, row in zip(peps, df.itertuples()):
                    affinity = row.affinity
                    percentile = row.affinity_percentile
                    if binary:
                        scores[a][p] = 1.0 if affinity <= 500 else 0.0
                    else:
                        scores[a][p] = 1 - math.log(max(affinity, 1e-10), 50000)
                    ranks[a][p] = percentile

        if not scores:
            raise ValueError("No predictions could be made with " + self.name +
                            " for given input. Check your epitope length and HLA allele combination.")

        result = {allele: {
            "Score": list(scores.values())[j],
            "Rank": list(ranks.values())[j]
        } for j, allele in enumerate(alleles)}

        df_result = EpitopePredictionResult.from_dict(result, peptide_objects.values(), self.name)
        return df_result
