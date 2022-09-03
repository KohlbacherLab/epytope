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
import csv
import sys
import tempfile
import io
import math

from abc import abstractmethod
from enum import IntEnum

import pandas
from collections import defaultdict

from mhcflurry import Class1AffinityPredictor
from mhcnuggets.src.predict import predict as mhcnuggets_predict
from epytope.Core import EpitopePredictionResult
from epytope.Core.Base import AEpitopePrediction
from epytope.Core.Allele import Allele, CombinedAllele, MouseAllele
from epytope.IO.Utils import capture_stdout

import inspect


class BadSignatureException(Exception):
    pass


class SignatureCheckerMeta(type):
    def __new__(cls, name, baseClasses, d):
        # For each method in d, check to see if any base class already
        # defined a method with that name. If so, make sure the
        # signatures are the same.
        for methodName in d:
            f = d[methodName]
            for baseClass in baseClasses:
                try:
                    fBase = getattr(baseClass, methodName).__func__
                    if not inspect.getfullargspec(f) == inspect.getfullargspec(fBase):
                        raise BadSignatureException(str(methodName))
                except AttributeError:
                    # This method was not defined in this base class,
                    # So just go to the next base class.
                    continue

        return type(name, baseClasses, d)


class AANNEpitopePrediction(AEpitopePrediction):
    """
        Abstract base class for ANN predictions.
        Implements predict functionality
    """

    @abstractmethod
    def predict(self, peptides, alleles=None, binary=False, **kwargs):
        """
        All ANN based predictors have to implement their custom predict method.
        Furthermore, all of them have to use the metaclass SignatureCheckerMeta to check for any contract violations.
        They have to adhere to the following contract

        :param peptides: A single :class:`~epytope.Core.Peptide.Peptide` or a list of :class:`~epytope.Core.Peptide.Peptide`
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param kwargs: optional parameter (not used yet)
        :return: Returns a nested dictionary {allele1: {scoreType1: {pep1: score1, pep2:..}, scoreType2: {..}, ..}, allele2:..}
        :rtype: :class:`pandas.DataFrame`
        """

try:
    class MHCNuggetsPredictor_class1_2_0(AANNEpitopePrediction):
        """
        Implements MHCNuggets Class I

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
        __name = "mhcnuggets-class-1"
        __version = "2.0"
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = __name.replace('-', '_') + '_' + __version.replace('.', '_')
        __alleles = getattr(__import__("epytope.Data.supportedAlleles.ann." + __allele_import_name,
                                       fromlist=[__allele_import_name])
                            , __allele_import_name)

        @property
        def name(self):
            # returns the name of the predictor
            return self.__name

        @property
        def supportedAlleles(self):
            # returns the supported alleles as strings (without the HLA prefix)
            return self.__alleles

        @property
        def supportedLength(self):
            # returns the supported epitope lengths as iterable
            return self.__supported_length

        @property
        def version(self):
            # returns the version of the predictor
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
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype.upper(), allele.subtype)
            else:
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

        # Converts epytopes internal allele representation into the format required by mhcnuggets
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

        # Converts the internal mhcnuggets-class-1 HLA representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                name = name[:5] + '*' + name[5:7] + ':' + name[7:]
                return Allele(name)

        # predicts the binding affinity for a set of peptides and alleles
        def predict(self, peptides, alleles=None, binary=False, **kwargs):

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

            # prepare results dictionary
            scores = defaultdict(defaultdict)

            # keep input peptide objects for later use
            peptide_objects = {}
            for peptide in peptides:
                peptide_objects[str(peptide)] = peptide

            # group peptides by length
            pep_groups = list(peptide_objects.keys())
            pep_groups.sort(key=len)
            for length, peps in itertools.groupby(pep_groups, key=len):
                if length not in self.supportedLength:
                    logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                        self.name, length))
                    continue
                peps = list(peps)

                # write peptides temporarily, new line separated
                tmp_input_file = tempfile.NamedTemporaryFile().name
                with open(tmp_input_file, 'w') as file:
                    for peptide in peps:
                        file.write(peptide + "\n")

                # predict binding affinities
                for a in alleles:
                    # workaround for mhcnuggets file i/o buffer bug
                    mhcnuggets_output = io.StringIO()
                    with capture_stdout(mhcnuggets_output):
                        mhcnuggets_predict(class_='I',
                                            peptides_path=tmp_input_file,
                                            mhc=alleles_repr[a])

                    # read predicted binding affinities back
                    mhcnuggets_output.seek(0)
                    reader = csv.reader(mhcnuggets_output, delimiter=' ', quotechar='|')
                    # skip log statements from mhcnuggets and header
                    for row in reader:
                        if row[0] == 'peptide,ic50':
                            break
                        logging.info(' '.join(row))

                    # assign binding affinities
                    for row in reader:
                        content = row[0].split(',')
                        # get original peptide object
                        peptide = content[0]
                        binding_affinity = float(content[ScoreIndex.MHCNUGGETS_CLASS1_2_0])
                        if binary:
                            if binding_affinity <= 500:
                                scores[a][peptide] = 1.0
                            else:
                                scores[a][peptide] = 0.0
                        else:
                            # convert ic50 to raw prediction score
                            scores[a][peptide] = 1- math.log(binding_affinity, 50000)

            if not scores:
                raise ValueError("No predictions could be made with " + self.name +
                                " for given input. Check your epitope length and HLA allele combination.")
            
            # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': AffScore1, 'Pep2': AffScore2,..}, 'Allele2':...}
            result = {alleles: {"Score":(list(scores.values())[j])} for j, alleles in enumerate(alleles)}

            # create EpitopePredictionResult object. This is a multi-indexed DataFrame
            # with Allele, Method and Score type as multi-columns and peptides as rows
            df_result = EpitopePredictionResult.from_dict(result, peptide_objects.values(), self.name)

            return df_result

except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class1_2_0 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCNuggetsPredictor_class1_2_3_2(MHCNuggetsPredictor_class1_2_0):
        """
        Implements MHCNuggets Class I

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
        __name = "mhcnuggets-class-1"
        __version = "2.3.2"
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = __name.replace('-', '_') + '_' + __version.replace('.', '_')
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
                return "%s-%s%s:%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

        # Converts the internal mhcnuggets-class-1 HLA representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                name = name[:5] + '*' + name[5:]
                return Allele(name)

        # predict(): same workaround for mhcnuggets file i/o buffer bug needed here
except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class1_2_3_2 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCNuggetsPredictor_class2_2_0(AANNEpitopePrediction):
        """
        Implements MHCNuggets Class II

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
        __name = "mhcnuggets-class-2"
        __version = "2.0"
        __supported_length = frozenset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        __allele_import_name = __name.replace('-', '_') + '_' + __version.replace('.', '_')
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
                # expects H-2-XXx
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype.upper(), allele.subtype)
            elif isinstance(allele, CombinedAllele):
                return "%s-%s%s%s-%s%s%s" % (allele.organism, allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                        allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
            else:
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

        # Converts epytopes internal allele representation into the format required by mhcnuggets
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

        # Converts the internal mhcnuggets-class-2 representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                # since we need to support single and double mhc2 alleles
                name_split = name.split('-')
                if len(name_split) > 2:
                    return CombinedAllele(name_split[0] + '-' +
                                          name_split[1][:4] + '*' + name_split[1][4:6] + ':' + name_split[1][6:] + '-' +
                                          name_split[2][:4] + '*' + name_split[2][4:6] + ':' + name_split[2][6:])
                else:
                    return Allele(name_split[0] + '-' +
                                  name_split[1][:4] + '*' + name_split[1][4:6] + ':' + name_split[1][6:])

        # predicts the binding affinity for a set of peptides and alleles
        def predict(self, peptides, alleles=None, binary=False, **kwargs):

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

            # prepare results dictionary
            scores = defaultdict(defaultdict)

            # keep input peptide objects for later use
            peptide_objects = {}
            for peptide in peptides:
                peptide_objects[str(peptide)] = peptide

            # group peptides by length
            pep_groups = list(peptide_objects.keys())
            pep_groups.sort(key=len)
            for length, peps in itertools.groupby(pep_groups, key=len):
                if length not in self.supportedLength:
                    logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                        self.name, length))
                    continue
                peps = list(peps)

                # write peptides temporarily, new line separated
                tmp_input_file = tempfile.NamedTemporaryFile().name
                with open(tmp_input_file, 'w') as file:
                    for peptide in peps:
                        file.write(peptide + "\n")

                # predict bindings
                for a in alleles:
                    # workaround for mhcnuggets file i/o buffer bug
                    mhcnuggets_output = io.StringIO()
                    with capture_stdout(mhcnuggets_output):
                        mhcnuggets_predict(class_='II',
                                        peptides_path=tmp_input_file,
                                        mhc=alleles_repr[a])

                    # read predicted binding affinities back
                    mhcnuggets_output.seek(0)
                    reader = csv.reader(mhcnuggets_output, delimiter=' ', quotechar='|')
                    # skip log statements from mhcnuggets and header
                    for row in reader:
                        if row[0] == 'peptide,ic50':
                            break
                        logging.warning(' '.join(row))

                    for row in reader:
                        content = row[0].split(',')
                        # get original peptide object
                        peptide = peptide_objects[content[0]]
                        binding_affinity = float(content[ScoreIndex.MHCNUGGETS_CLASS2_2_0])
                        if binary:
                            if binding_affinity <= 500:
                                scores[a][peptide] = 1.0
                            else:
                                scores[a][peptide] = 0.0
                        else:
                            # convert ic50 to raw prediction score
                            scores[a][peptide] = 1- math.log(binding_affinity, 50000)

            if not scores:
                raise ValueError("No predictions could be made with " + self.name +
                                " for given input. Check your epitope length and HLA allele combination.")

            # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': AffScore1, 'Pep2': AffScore2,..}, 'Allele2':...}
            result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

            # create EpitopePredictionResult object. This is a multi-indexed DataFrame
            # with Allele, Method and Score type as multi-columns and peptides as rows
            df_result = EpitopePredictionResult.from_dict(result, peptide_objects.values(), self.name)
            return df_result

except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class2_2_0 cannot be constructed, because of a bad method signature (predict)")

try:
    class MHCNuggetsPredictor_class2_2_3_2(MHCNuggetsPredictor_class2_2_0):
        """
        Implements MHCNuggets Class II

        .. note::
            Evaluation of machine learning methods to predict peptide binding to MHC Class I proteins
            Rohit Bhattacharya, Ashok Sivakumar, Collin Tokheim, Violeta Beleva Guthrie, Valsamo Anagnostou,
            Victor E. Velculescu, Rachel Karchin (2017) bioRxiv
        """
        __name = "mhcnuggets-class-2"
        __version = "2.3.2"
        __supported_length = frozenset(
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        __allele_import_name = __name.replace('-', '_') + '_' + __version.replace('.', '_')
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
                # expects H-2-XXx
                return "%s-%s%s%s" % (allele.organism, allele.locus, allele.supertype.upper(), allele.subtype)
            elif isinstance(allele, CombinedAllele):
                return "%s-%s%s:%s-%s%s:%s" % (allele.organism, allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                        allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
            else:
                return "%s-%s%s:%s" % (allele.organism, allele.locus, allele.supertype, allele.subtype)

        # Converts the internal mhcnuggets-class-2 representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                # since we need to support single and double mhc2 alleles
                name_split = name.split('-')
                if len(name_split) > 2:
                    return CombinedAllele(name_split[0] + '-' +
                            name_split[1][:4] + '*' + name_split[1][4:]  + '-' +
                            name_split[2][:4] + '*' + name_split[2][4:])
                else:
                    return Allele(name_split[0] + '-' + name_split[1][:4] + '*' + name_split[1][4:])

        # predict(): same workaround for mhcnuggets file i/o buffer bug needed here
except BadSignatureException:
    logging.warning("Class MHCNuggetsPredictor_class2_2_3_2 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCFlurryPredictor_1_2_2(AANNEpitopePrediction):
        """
        Implements MHCFlurry

        .. note::
            T. J. O’Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher,
             "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," Cell Systems, 2018.
              Available at: https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.
        """
        __name = "mhcflurry"
        __version = "1.2.2"
        __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = __name + '_' + __version.replace('.', '_')
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
                logging.warn("mhcflurry models must be downloaded, as they were not found locally.")
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
                        binding_affinity = predictor.predict(allele=alleles_repr[a], peptides=[str(p)])[ScoreIndex.MHCFLURRY]
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

except BadSignatureException:
    logging.warning("Class MHCFlurryPredictor_1_2_2 cannot be constructed, because of a bad method signature (predict)")


try:
    class MHCFlurryPredictor_1_4_3(MHCFlurryPredictor_1_2_2):
        """
        Implements MHCFlurry

        .. note::
            T. J. O’Donnell, A. Rubinsteyn, M. Bonsack, A. B. Riemer, U. Laserson, and J. Hammerbacher,
             "MHCflurry: Open-Source Class I MHC Binding Affinity Prediction," Cell Systems, 2018.
              Available at: https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30232-1.
        """
        __name = "mhcflurry"
        __version = "1.4.3"
        __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14, 15])
        __allele_import_name = __name + '_' + __version.replace('.', '_')
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

        # Converts the internal MHCFlurry representation back into a epytope representation
        def revert_allele_repr(self, name):
            if name.startswith("H-2-"):
                return MouseAllele(name)
            else:
                return Allele(name)
except BadSignatureException:
    logging.warning("Class MHCFlurryPredictor_1_4_3 cannot be constructed, because of a bad method signature (predict)")



class ScoreIndex(IntEnum):
    """
    Specifies the score column index in the respective output format
    """
    MHCNUGGETS_CLASS1_2_0 = 1
    MHCNUGGETS_CLASS2_2_0 = 1
    MHCFLURRY = 0
