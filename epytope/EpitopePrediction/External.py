# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: EpitopePrediction.ANN
   :synopsis: This module contains all classes for ANN-based epitope prediction methods.
.. moduleauthor:: schubert, walzer

"""
import abc

import itertools
import warnings
import logging
import pandas
import subprocess
import csv
import os
import math
import re

from collections import defaultdict
from enum import IntEnum

from epytope.Core.Allele import Allele, CombinedAllele, MouseAllele
from epytope.Core.Peptide import Peptide
from epytope.Core.Result import EpitopePredictionResult
from epytope.Core.Base import AEpitopePrediction, AExternal
from tempfile import NamedTemporaryFile, mkstemp

class AExternalEpitopePrediction(AEpitopePrediction, AExternal):
    """
        Abstract class representing an external prediction function. Implementations shall wrap external binaries by
        following the given abstraction.
    """

    @abc.abstractmethod
    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to _file in the specific format

        NO return value!

        :param: list(str) _input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        return NotImplementedError

    def predict(self, peptides, alleles=None, command=None, options=None, **kwargs):
        """
        Overwrites AEpitopePrediction.predict

        :param peptides: A list of or a single :class:`~epytope.Core.Peptide.Peptide` object
        :type peptides: list(:class:`~epytope.Core.Peptide.Peptide`) or :class:`~epytope.Core.Peptide.Peptide`
        :param alleles: A list of or a single :class:`~epytope.Core.Allele.Allele` object. If no
                        :class:`~epytope.Core.Allele.Allele` are provided, predictions are made for all
                        :class:`~epytope.Core.Allele.Allele` supported by the prediction method
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)/:class:`~epytope.Core.Allele.Allele`
        :param str command: The path to a alternative binary (can be used if binary is not globally executable)
        :param str options: A string of additional options directly past to the external tool.
        :keyword chunksize: denotes the chunksize in which the number of peptides are bulk processed
        :return: A :class:`~epytope.Core.Result.EpitopePredictionResult` object
        :rtype: :class:`~epytope.Core.Result.EpitopePredictionResult`
        """
        if not self.is_in_path() and command is None:
            raise RuntimeError("{name} {version} could not be found in PATH".format(name=self.name,
                                                                                    version=self.version))
        external_version = self.get_external_version(path=command)
        if self.version != external_version and external_version is not None:
            raise RuntimeError("Internal version {internal_version} does "
                               "not match external version {external_version}".format(internal_version=self.version,
                                                                                      external_version=external_version))

        if isinstance(peptides, Peptide):
            pep_seqs = {str(peptides): peptides}
        else:
            pep_seqs = {}
            for p in peptides:
                if not isinstance(p, Peptide):
                    raise ValueError("Input is not of type Protein or Peptide")
                pep_seqs[str(p)] = p

        chunksize = len(pep_seqs)
        if 'chunks' in kwargs:
            chunksize = kwargs['chunks']

        if alleles is None:
            alleles = [Allele(a) for a in self.supportedAlleles]
        else:
            if isinstance(alleles, Allele):
                alleles = [alleles]
            if any(not isinstance(p, Allele) for p in alleles):
                raise ValueError("Input is not of type Allele")

        # Create dictionary containing the predictors string representation and the Allele Obj representation of the allele
        alleles_string = {conv_a: a for conv_a, a in zip(self.convert_alleles(alleles), alleles)}
        
        # Create empty result dictionary to fill downstream
        result = {}

        # group alleles in blocks of 80 alleles (NetMHC can't deal with more)
        _MAX_ALLELES = 50

        # allow custom executable specification
        if command is not None:
            exe = self.command.split()[0]
            _command = self.command.replace(exe, command)
        else:
            _command = self.command

        allele_groups = []
        c_a = 0
        allele_group = []
        for a in alleles_string.keys():
            if c_a >= _MAX_ALLELES:
                c_a = 0
                allele_groups.append(allele_group)
                if str(alleles_string[a]) not in self.supportedAlleles:
                    logging.warning("Allele %s is not supported by %s" % (str(alleles_string[a]), self.name))
                    allele_group = []
                    continue
                allele_group = [a]
            else:
                if str(alleles_string[a]) not in self.supportedAlleles:
                    logging.warning("Allele %s is not supported by %s" % (str(alleles_string[a]), self.name))
                    continue
                allele_group.append(a)
                c_a += 1

        if len(allele_group) > 0:
            allele_groups.append(allele_group)
        # export peptides to peptide list

        pep_groups = list(pep_seqs.keys())
        pep_groups.sort(key=len)
        for length, peps in itertools.groupby(pep_groups, key=len):
            if length not in self.supportedLength:
                logging.warning("Peptide length must be at least %i or at most %i for %s but is %i" % (min(self.supportedLength), max(self.supportedLength),
                                                                                       self.name, length))
                continue
            peps = list(peps)
            
            for i in range(0, len(peps), chunksize):
                # Create a temporary file for subprocess to write to. The
                # handle is not needed on the python end, as only the path will
                # be passed to the subprocess.
                _, tmp_out_path = mkstemp()
                # Create a temporary file to be used for the peptide input
                tmp_file = NamedTemporaryFile(mode="r+", delete=False)
                self.prepare_input(peps[i:i+chunksize], tmp_file)
                tmp_file.close()

                # generate cmd command
                for allele_group in allele_groups:
                    try:
                        stdo = None
                        stde = None
                        cmd = _command.format(peptides=tmp_file.name, alleles=",".join(allele_group),
                                              options="" if options is None else options, out=tmp_out_path, length=str(length))
                        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT)
                        stdo, stde = p.communicate()
                        stdr = p.returncode
                        if stdr > 0:
                            raise RuntimeError("Unsuccessful execution of " + cmd + " (EXIT!=0) with output:\n" + stdo.decode())
                        if os.path.getsize(tmp_out_path) == 0:
                            raise RuntimeError("Unsuccessful execution of " + cmd + " (empty output file) with output:\n" + stdo.decode())
                    except Exception as e:
                        raise RuntimeError(e)

                    # Obtain parsed output dataframe containing the peptide scores and/or ranks
                    res_tmp = self.parse_external_result(tmp_out_path)
                    for allele_string, scores in res_tmp.items():
                        allele = alleles_string[allele_string]
                        if allele not in result.keys():
                            result[allele] = {}
                        for scoretype, pep_scores in scores.items():
                            if scoretype not in result[allele].keys():
                                result[allele][scoretype] = {}
                            for pep, score in pep_scores.items():
                                result[allele][scoretype][pep_seqs[pep]] = score
                                
                os.remove(tmp_file.name)
                os.remove(tmp_out_path)
        
        if not result:
            raise ValueError("No predictions could be made with " + self.name +
                             " for given input. Check your epitope length and HLA allele combination.")
        
        df_result = EpitopePredictionResult.from_dict(result, list(pep_seqs.values()), self.name)

        return df_result


class NetMHC_3_4(AExternalEpitopePrediction):
    """
    Implements the NetMHC binding (in current form for netMHC3.4).

    .. note::

        NetMHC-3.0: accurate web accessible predictions of human, mouse and monkey MHC class I affinities for peptides
        of length 8-11. Lundegaard C, Lamberth K, Harndahl M, Buus S, Lund O, Nielsen M.
        Nucleic Acids Res. 1;36(Web Server issue):W509-12. 2008

        Accurate approximation method for prediction of class I MHC affinities for peptides of length 8, 10 and 11 using
        prediction tools trained on 9mers. Lundegaard C, Lund O, Nielsen M. Bioinformatics, 24(11):1397-98, 2008.

    """

    __alleles = frozenset(['HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03', 'HLA-A*02:06', 'HLA-A*02:11', 'HLA-A*02:12', 'HLA-A*02:16',
                           'HLA-A*02:17', 'HLA-A*02:19', 'HLA-A*02:50', 'HLA-A*03:01', 'HLA-A*11:01', 'HLA-A*23:01', 'HLA-A*24:02', 'HLA-A*24:03',
                           'HLA-A*25:01', 'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*29:02', 'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*31:01',
                           'HLA-A*32:01', 'HLA-A*32:07', 'HLA-A*32:15', 'HLA-A*33:01', 'HLA-A*66:01', 'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*68:23',
                           'HLA-A*69:01', 'HLA-A*80:01', 'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*14:02', 'HLA-B*15:01',
                           'HLA-B*15:02', 'HLA-B*15:03', 'HLA-B*15:09', 'HLA-B*15:17', 'HLA-B*18:01', 'HLA-B*27:05', 'HLA-B*27:20', 'HLA-B*35:01',
                           'HLA-B*35:03', 'HLA-B*38:01', 'HLA-B*39:01', 'HLA-B*40:01', 'HLA-B*40:02', 'HLA-B*40:13', 'HLA-B*42:01', 'HLA-B*44:02',
                           'HLA-B*44:03', 'HLA-B*45:01', 'HLA-B*46:01', 'HLA-B*48:01', 'HLA-B*51:01', 'HLA-B*53:01', 'HLA-B*54:01', 'HLA-B*57:01',
                           'HLA-B*58:01', 'HLA-B*73:01', 'HLA-B*83:01', 'HLA-C*03:03', 'HLA-C*04:01', 'HLA-C*05:01', 'HLA-C*06:02', 'HLA-C*07:01',
                           'HLA-C*07:02', 'HLA-C*08:02', 'HLA-C*12:03', 'HLA-C*14:02', 'HLA-C*15:02', 'HLA-E*01:01',
                           'H-2-Db', 'H-2-Dd', 'H-2-Kb', 'H-2-Kd', 'H-2-Kk', 'H-2-Ld'])
    __supported_length = frozenset([8, 9, 10, 11])
    __name = "netmhc"
    __command = "netMHC -p {peptides} -a {alleles} -x {out} {options}"
    __version = "3.4"

    @property
    def version(self):
        """The version of the predictor"""
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
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    @property
    def supportedAlleles(self):
        """
        A list of valid allele models
        """
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results 
        :rtype: dict
        """
        result = defaultdict(defaultdict)
        f = csv.reader(open(file, "r"), delimiter='\t')
        next(f)
        next(f)
        alleles = [x.split()[0] for x in f.next()[3:]]
        for l in f:
            if not l:
                continue
            pep_seq = l[PeptideIndex.NETMHC_3_4]
            for ic_50, a in zip(l[ScoreIndex.NETMHC_3_0:], alleles):
                sc = 1.0 - math.log(float(ic_50), 50000)
                result[a][pep_seq] = sc if sc > 0.0 else 0.0
        return dict(result)

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: dict
        """
        return super(NetMHC_3_4, self).get_external_version()

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to file in the specific format

        NO return value!

        :param: list(str) input: The : sequences to write into _file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetMHC_3_0(NetMHC_3_4):
    """
    Implements the NetMHC binding (for netMHC3.0)::


    .. note::

        NetMHC-3.0: accurate web accessible predictions of human, mouse and monkey MHC class I affinities for peptides
        of length 8-11. Lundegaard C, Lamberth K, Harndahl M, Buus S, Lund O, Nielsen M.
        Nucleic Acids Res. 1;36(Web Server issue):W509-12. 2008

        Accurate approximation method for prediction of class I MHC affinities for peptides of length 8, 10 and 11
        using prediction tools trained on 9mers. Lundegaard C, Lund O, Nielsen M. Bioinformatics, 24(11):1397-98, 2008.
    """

    __alleles = frozenset(['HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03', 'HLA-A*02:04', 'HLA-A*02:06', 'HLA-A*02:11', 'HLA-A*02:12',
                           'HLA-A*02:16', 'HLA-A*02:19', 'HLA-A*03:01', 'HLA-A*11:01', 'HLA-A*23:01', 'HLA-A*24:02', 'HLA-A*24:03', 'HLA-A*26:01',
                           'HLA-A*26:02', 'HLA-A*29:02', 'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*31:01', 'HLA-A*33:01', 'HLA-A*68:01', 'HLA-A*68:02',
                           'HLA-A*69:01', 'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*15:01', 'HLA-B*18:01', 'HLA-B*27:05', 'HLA-B*35:01',
                           'HLA-B*39:01', 'HLA-B*40:01', 'HLA-B*40:02', 'HLA-B*44:02', 'HLA-B*44:03', 'HLA-B*45:01', 'HLA-B*51:01', 'HLA-B*53:01',
                           'HLA-B*54:01', 'HLA-B*57:01', 'HLA-B*58:01',
                           'H-2-Db', 'H-2-Dd', 'H-2-Kb', 'H-2-Kd', 'H-2-Kk', 'H-2-Ld'])  # no PSSM predictors

    __supported_length = frozenset([8, 9, 10, 11])
    __name = "netmhc"
    __version = "3.0a"
    __command = "netMHC-3.0 -p {peptides} -a {alleles} -x {out} -l {length} {options}"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedAlleles(self):
        """
        A list of valid :class:`~epytope.Core.Allele.Allele` models
        """
        return self.__alleles

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        result = defaultdict(dict)
        with open(file, 'r') as f:
            next(f, None)  # skip first line with logging stuff
            next(f, None)  # skip first line with nothing
            csvr = csv.reader(f, delimiter='\t')
            alleles = [x.split()[0] for x in csvr.next()[3:]]
            for l in csvr:
                if not l:
                    continue
                pep_seq = l[PeptideIndex.NETMHC_3_0]
                for ic_50, a in zip(l[ScoreIndex.NETMHC_3_0:], alleles):
                    sc = 1.0 - math.log(float(ic_50), 50000)
                    result[a][pep_seq] = sc if sc > 0.0 else 0.0
        if 'Average' in result:
            result.pop('Average')
        return dict(result)


class NetMHC_4_0(NetMHC_3_4):
    """
    Implements the NetMHC 4.0 binding

    .. note::
        Andreatta M, Nielsen M. Gapped sequence alignment using artificial neural networks:
        application to the MHC class I system. Bioinformatics (2016) Feb 15;32(4):511-7
    """
    __command = "netMHC -p {peptides} -a {alleles} -xls -xlsfile {out} {options}"
    __version = "4.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        f = csv.reader(open(file, "r"), delimiter='\t')
        alleles = [x.split()[0] for x in [x for x in next(f) if x.strip() != ""]]
        next(f)
        for l in f:
            if not l:
                continue
            pep_seq = l[PeptideIndex.NETMHC_4_0]
            for i, a in enumerate(alleles):
                ic_50 = l[(i+1) * Offset.NETMHC_4_0]
                sc = 1.0 - math.log(float(ic_50), 50000)
                rank = l[(i+1)* Offset.NETMHC_4_0 + 1]
                scores[a][pep_seq] = sc if sc > 0.0 else 0.0
                ranks[a][pep_seq] = float(rank)
        
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        # can not be determined netmhcpan does not support --version or similar
        return None


class NetMHCpan_2_4(AExternalEpitopePrediction):
    """
    Implements the NetMHC binding (in current form for netMHCpan 2.4).
    Supported  MHC alleles currently only restricted to HLA alleles.

    .. note::

        Nielsen, Morten, et al. "NetMHCpan, a method for quantitative predictions of peptide binding to any HLA-A and-B
        locus protein of known sequence." PloS one 2.8 (2007): e796.
    """
    __supported_length = frozenset([8, 9, 10, 11])
    __name = "netmhcpan"
    __command = "netMHCpan-2.4 -p {peptides} -a {alleles} {options} -ic50 -xls -xlsfile {out}"
    __alleles = frozenset(
        ['HLA-A*01:01', 'HLA-A*01:02', 'HLA-A*01:03', 'HLA-A*01:06', 'HLA-A*01:07', 'HLA-A*01:08', 'HLA-A*01:09', 'HLA-A*01:10', 'HLA-A*01:12',
         'HLA-A*01:13', 'HLA-A*01:14', 'HLA-A*01:17', 'HLA-A*01:19', 'HLA-A*01:20', 'HLA-A*01:21', 'HLA-A*01:23', 'HLA-A*01:24', 'HLA-A*01:25',
         'HLA-A*01:26', 'HLA-A*01:28', 'HLA-A*01:29', 'HLA-A*01:30', 'HLA-A*01:32', 'HLA-A*01:33', 'HLA-A*01:35', 'HLA-A*01:36', 'HLA-A*01:37',
         'HLA-A*01:38', 'HLA-A*01:39', 'HLA-A*01:40', 'HLA-A*01:41', 'HLA-A*01:42', 'HLA-A*01:43', 'HLA-A*01:44', 'HLA-A*01:45', 'HLA-A*01:46',
         'HLA-A*01:47', 'HLA-A*01:48', 'HLA-A*01:49', 'HLA-A*01:50', 'HLA-A*01:51', 'HLA-A*01:54', 'HLA-A*01:55', 'HLA-A*01:58', 'HLA-A*01:59',
         'HLA-A*01:60', 'HLA-A*01:61', 'HLA-A*01:62', 'HLA-A*01:63', 'HLA-A*01:64', 'HLA-A*01:65', 'HLA-A*01:66', 'HLA-A*02:01', 'HLA-A*02:02',
         'HLA-A*02:03', 'HLA-A*02:04', 'HLA-A*02:05', 'HLA-A*02:06', 'HLA-A*02:07', 'HLA-A*02:08', 'HLA-A*02:09', 'HLA-A*02:10', 'HLA-A*02:101',
         'HLA-A*02:102', 'HLA-A*02:103', 'HLA-A*02:104', 'HLA-A*02:105', 'HLA-A*02:106', 'HLA-A*02:107', 'HLA-A*02:108', 'HLA-A*02:109',
         'HLA-A*02:11', 'HLA-A*02:110', 'HLA-A*02:111', 'HLA-A*02:112', 'HLA-A*02:114', 'HLA-A*02:115', 'HLA-A*02:116', 'HLA-A*02:117',
         'HLA-A*02:118', 'HLA-A*02:119', 'HLA-A*02:12', 'HLA-A*02:120', 'HLA-A*02:121', 'HLA-A*02:122', 'HLA-A*02:123', 'HLA-A*02:124',
         'HLA-A*02:126', 'HLA-A*02:127', 'HLA-A*02:128', 'HLA-A*02:129', 'HLA-A*02:13', 'HLA-A*02:130', 'HLA-A*02:131', 'HLA-A*02:132',
         'HLA-A*02:133', 'HLA-A*02:134', 'HLA-A*02:135', 'HLA-A*02:136', 'HLA-A*02:137', 'HLA-A*02:138', 'HLA-A*02:139', 'HLA-A*02:14',
         'HLA-A*02:140', 'HLA-A*02:141', 'HLA-A*02:142', 'HLA-A*02:143', 'HLA-A*02:144', 'HLA-A*02:145', 'HLA-A*02:146', 'HLA-A*02:147',
         'HLA-A*02:148', 'HLA-A*02:149', 'HLA-A*02:150', 'HLA-A*02:151', 'HLA-A*02:152', 'HLA-A*02:153', 'HLA-A*02:154', 'HLA-A*02:155',
         'HLA-A*02:156', 'HLA-A*02:157', 'HLA-A*02:158', 'HLA-A*02:159', 'HLA-A*02:16', 'HLA-A*02:160', 'HLA-A*02:161', 'HLA-A*02:162',
         'HLA-A*02:163', 'HLA-A*02:164', 'HLA-A*02:165', 'HLA-A*02:166', 'HLA-A*02:167', 'HLA-A*02:168', 'HLA-A*02:169', 'HLA-A*02:17',
         'HLA-A*02:170', 'HLA-A*02:171', 'HLA-A*02:172', 'HLA-A*02:173', 'HLA-A*02:174', 'HLA-A*02:175', 'HLA-A*02:176', 'HLA-A*02:177',
         'HLA-A*02:178', 'HLA-A*02:179', 'HLA-A*02:18', 'HLA-A*02:180', 'HLA-A*02:181', 'HLA-A*02:182', 'HLA-A*02:183', 'HLA-A*02:184',
         'HLA-A*02:185', 'HLA-A*02:186', 'HLA-A*02:187', 'HLA-A*02:188', 'HLA-A*02:189', 'HLA-A*02:19', 'HLA-A*02:190', 'HLA-A*02:191',
         'HLA-A*02:192', 'HLA-A*02:193', 'HLA-A*02:194', 'HLA-A*02:195', 'HLA-A*02:196', 'HLA-A*02:197', 'HLA-A*02:198', 'HLA-A*02:199',
         'HLA-A*02:20', 'HLA-A*02:200', 'HLA-A*02:201', 'HLA-A*02:202', 'HLA-A*02:203', 'HLA-A*02:204', 'HLA-A*02:205', 'HLA-A*02:206',
         'HLA-A*02:207', 'HLA-A*02:208', 'HLA-A*02:209', 'HLA-A*02:21', 'HLA-A*02:210', 'HLA-A*02:211', 'HLA-A*02:212', 'HLA-A*02:213',
         'HLA-A*02:214', 'HLA-A*02:215', 'HLA-A*02:216', 'HLA-A*02:217', 'HLA-A*02:218', 'HLA-A*02:219', 'HLA-A*02:22', 'HLA-A*02:220',
         'HLA-A*02:221', 'HLA-A*02:224', 'HLA-A*02:228', 'HLA-A*02:229', 'HLA-A*02:230', 'HLA-A*02:231', 'HLA-A*02:232', 'HLA-A*02:233',
         'HLA-A*02:234', 'HLA-A*02:235', 'HLA-A*02:236', 'HLA-A*02:237', 'HLA-A*02:238', 'HLA-A*02:239', 'HLA-A*02:24', 'HLA-A*02:240',
         'HLA-A*02:241', 'HLA-A*02:242', 'HLA-A*02:243', 'HLA-A*02:244', 'HLA-A*02:245', 'HLA-A*02:246', 'HLA-A*02:247', 'HLA-A*02:248',
         'HLA-A*02:249', 'HLA-A*02:25', 'HLA-A*02:251', 'HLA-A*02:252', 'HLA-A*02:253', 'HLA-A*02:254', 'HLA-A*02:255', 'HLA-A*02:256',
         'HLA-A*02:257', 'HLA-A*02:258', 'HLA-A*02:259', 'HLA-A*02:26', 'HLA-A*02:260', 'HLA-A*02:261', 'HLA-A*02:262', 'HLA-A*02:263',
         'HLA-A*02:264', 'HLA-A*02:265', 'HLA-A*02:266', 'HLA-A*02:27', 'HLA-A*02:28', 'HLA-A*02:29', 'HLA-A*02:30', 'HLA-A*02:31', 'HLA-A*02:33',
         'HLA-A*02:34', 'HLA-A*02:35', 'HLA-A*02:36', 'HLA-A*02:37', 'HLA-A*02:38', 'HLA-A*02:39', 'HLA-A*02:40', 'HLA-A*02:41', 'HLA-A*02:42',
         'HLA-A*02:44', 'HLA-A*02:45', 'HLA-A*02:46', 'HLA-A*02:47', 'HLA-A*02:48', 'HLA-A*02:49', 'HLA-A*02:50', 'HLA-A*02:51', 'HLA-A*02:52',
         'HLA-A*02:54', 'HLA-A*02:55', 'HLA-A*02:56', 'HLA-A*02:57', 'HLA-A*02:58', 'HLA-A*02:59', 'HLA-A*02:60', 'HLA-A*02:61', 'HLA-A*02:62',
         'HLA-A*02:63', 'HLA-A*02:64', 'HLA-A*02:65', 'HLA-A*02:66', 'HLA-A*02:67', 'HLA-A*02:68', 'HLA-A*02:69', 'HLA-A*02:70', 'HLA-A*02:71',
         'HLA-A*02:72', 'HLA-A*02:73', 'HLA-A*02:74', 'HLA-A*02:75', 'HLA-A*02:76', 'HLA-A*02:77', 'HLA-A*02:78', 'HLA-A*02:79', 'HLA-A*02:80',
         'HLA-A*02:81', 'HLA-A*02:84', 'HLA-A*02:85', 'HLA-A*02:86', 'HLA-A*02:87', 'HLA-A*02:89', 'HLA-A*02:90', 'HLA-A*02:91', 'HLA-A*02:92',
         'HLA-A*02:93', 'HLA-A*02:95', 'HLA-A*02:96', 'HLA-A*02:97', 'HLA-A*02:99', 'HLA-A*03:01', 'HLA-A*03:02', 'HLA-A*03:04', 'HLA-A*03:05',
         'HLA-A*03:06', 'HLA-A*03:07', 'HLA-A*03:08', 'HLA-A*03:09', 'HLA-A*03:10', 'HLA-A*03:12', 'HLA-A*03:13', 'HLA-A*03:14', 'HLA-A*03:15',
         'HLA-A*03:16', 'HLA-A*03:17', 'HLA-A*03:18', 'HLA-A*03:19', 'HLA-A*03:20', 'HLA-A*03:22', 'HLA-A*03:23', 'HLA-A*03:24', 'HLA-A*03:25',
         'HLA-A*03:26', 'HLA-A*03:27', 'HLA-A*03:28', 'HLA-A*03:29', 'HLA-A*03:30', 'HLA-A*03:31', 'HLA-A*03:32', 'HLA-A*03:33', 'HLA-A*03:34',
         'HLA-A*03:35', 'HLA-A*03:37', 'HLA-A*03:38', 'HLA-A*03:39', 'HLA-A*03:40', 'HLA-A*03:41', 'HLA-A*03:42', 'HLA-A*03:43', 'HLA-A*03:44',
         'HLA-A*03:45', 'HLA-A*03:46', 'HLA-A*03:47', 'HLA-A*03:48', 'HLA-A*03:49', 'HLA-A*03:50', 'HLA-A*03:51', 'HLA-A*03:52', 'HLA-A*03:53',
         'HLA-A*03:54', 'HLA-A*03:55', 'HLA-A*03:56', 'HLA-A*03:57', 'HLA-A*03:58', 'HLA-A*03:59', 'HLA-A*03:60', 'HLA-A*03:61', 'HLA-A*03:62',
         'HLA-A*03:63', 'HLA-A*03:64', 'HLA-A*03:65', 'HLA-A*03:66', 'HLA-A*03:67', 'HLA-A*03:70', 'HLA-A*03:71', 'HLA-A*03:72', 'HLA-A*03:73',
         'HLA-A*03:74', 'HLA-A*03:75', 'HLA-A*03:76', 'HLA-A*03:77', 'HLA-A*03:78', 'HLA-A*03:79', 'HLA-A*03:80', 'HLA-A*03:81', 'HLA-A*03:82',
         'HLA-A*11:01', 'HLA-A*11:02', 'HLA-A*11:03', 'HLA-A*11:04', 'HLA-A*11:05', 'HLA-A*11:06', 'HLA-A*11:07', 'HLA-A*11:08', 'HLA-A*11:09',
         'HLA-A*11:10', 'HLA-A*11:11', 'HLA-A*11:12', 'HLA-A*11:13', 'HLA-A*11:14', 'HLA-A*11:15', 'HLA-A*11:16', 'HLA-A*11:17', 'HLA-A*11:18',
         'HLA-A*11:19', 'HLA-A*11:20', 'HLA-A*11:22', 'HLA-A*11:23', 'HLA-A*11:24', 'HLA-A*11:25', 'HLA-A*11:26', 'HLA-A*11:27', 'HLA-A*11:29',
         'HLA-A*11:30', 'HLA-A*11:31', 'HLA-A*11:32', 'HLA-A*11:33', 'HLA-A*11:34', 'HLA-A*11:35', 'HLA-A*11:36', 'HLA-A*11:37', 'HLA-A*11:38',
         'HLA-A*11:39', 'HLA-A*11:40', 'HLA-A*11:41', 'HLA-A*11:42', 'HLA-A*11:43', 'HLA-A*11:44', 'HLA-A*11:45', 'HLA-A*11:46', 'HLA-A*11:47',
         'HLA-A*11:48', 'HLA-A*11:49', 'HLA-A*11:51', 'HLA-A*11:53', 'HLA-A*11:54', 'HLA-A*11:55', 'HLA-A*11:56', 'HLA-A*11:57', 'HLA-A*11:58',
         'HLA-A*11:59', 'HLA-A*11:60', 'HLA-A*11:61', 'HLA-A*11:62', 'HLA-A*11:63', 'HLA-A*11:64', 'HLA-A*23:01', 'HLA-A*23:02', 'HLA-A*23:03',
         'HLA-A*23:04', 'HLA-A*23:05', 'HLA-A*23:06', 'HLA-A*23:09', 'HLA-A*23:10', 'HLA-A*23:12', 'HLA-A*23:13', 'HLA-A*23:14', 'HLA-A*23:15',
         'HLA-A*23:16', 'HLA-A*23:17', 'HLA-A*23:18', 'HLA-A*23:20', 'HLA-A*23:21', 'HLA-A*23:22', 'HLA-A*23:23', 'HLA-A*23:24', 'HLA-A*23:25',
         'HLA-A*23:26', 'HLA-A*24:02', 'HLA-A*24:03', 'HLA-A*24:04', 'HLA-A*24:05', 'HLA-A*24:06', 'HLA-A*24:07', 'HLA-A*24:08', 'HLA-A*24:10',
         'HLA-A*24:100', 'HLA-A*24:101', 'HLA-A*24:102', 'HLA-A*24:103', 'HLA-A*24:104', 'HLA-A*24:105', 'HLA-A*24:106', 'HLA-A*24:107',
         'HLA-A*24:108', 'HLA-A*24:109', 'HLA-A*24:110', 'HLA-A*24:111', 'HLA-A*24:112', 'HLA-A*24:113', 'HLA-A*24:114', 'HLA-A*24:115',
         'HLA-A*24:116', 'HLA-A*24:117', 'HLA-A*24:118', 'HLA-A*24:119', 'HLA-A*24:120', 'HLA-A*24:121', 'HLA-A*24:122', 'HLA-A*24:123',
         'HLA-A*24:124', 'HLA-A*24:125', 'HLA-A*24:126', 'HLA-A*24:127', 'HLA-A*24:128', 'HLA-A*24:129', 'HLA-A*24:13', 'HLA-A*24:130',
         'HLA-A*24:131', 'HLA-A*24:133', 'HLA-A*24:134', 'HLA-A*24:135', 'HLA-A*24:136', 'HLA-A*24:137', 'HLA-A*24:138', 'HLA-A*24:139',
         'HLA-A*24:14', 'HLA-A*24:140', 'HLA-A*24:141', 'HLA-A*24:142', 'HLA-A*24:143', 'HLA-A*24:144', 'HLA-A*24:15', 'HLA-A*24:17', 'HLA-A*24:18',
         'HLA-A*24:19', 'HLA-A*24:20', 'HLA-A*24:21', 'HLA-A*24:22', 'HLA-A*24:23', 'HLA-A*24:24', 'HLA-A*24:25', 'HLA-A*24:26', 'HLA-A*24:27',
         'HLA-A*24:28', 'HLA-A*24:29', 'HLA-A*24:30', 'HLA-A*24:31', 'HLA-A*24:32', 'HLA-A*24:33', 'HLA-A*24:34', 'HLA-A*24:35', 'HLA-A*24:37',
         'HLA-A*24:38', 'HLA-A*24:39', 'HLA-A*24:41', 'HLA-A*24:42', 'HLA-A*24:43', 'HLA-A*24:44', 'HLA-A*24:46', 'HLA-A*24:47', 'HLA-A*24:49',
         'HLA-A*24:50', 'HLA-A*24:51', 'HLA-A*24:52', 'HLA-A*24:53', 'HLA-A*24:54', 'HLA-A*24:55', 'HLA-A*24:56', 'HLA-A*24:57', 'HLA-A*24:58',
         'HLA-A*24:59', 'HLA-A*24:61', 'HLA-A*24:62', 'HLA-A*24:63', 'HLA-A*24:64', 'HLA-A*24:66', 'HLA-A*24:67', 'HLA-A*24:68', 'HLA-A*24:69',
         'HLA-A*24:70', 'HLA-A*24:71', 'HLA-A*24:72', 'HLA-A*24:73', 'HLA-A*24:74', 'HLA-A*24:75', 'HLA-A*24:76', 'HLA-A*24:77', 'HLA-A*24:78',
         'HLA-A*24:79', 'HLA-A*24:80', 'HLA-A*24:81', 'HLA-A*24:82', 'HLA-A*24:85', 'HLA-A*24:87', 'HLA-A*24:88', 'HLA-A*24:89', 'HLA-A*24:91',
         'HLA-A*24:92', 'HLA-A*24:93', 'HLA-A*24:94', 'HLA-A*24:95', 'HLA-A*24:96', 'HLA-A*24:97', 'HLA-A*24:98', 'HLA-A*24:99', 'HLA-A*25:01',
         'HLA-A*25:02', 'HLA-A*25:03', 'HLA-A*25:04', 'HLA-A*25:05', 'HLA-A*25:06', 'HLA-A*25:07', 'HLA-A*25:08', 'HLA-A*25:09', 'HLA-A*25:10',
         'HLA-A*25:11', 'HLA-A*25:13', 'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*26:04', 'HLA-A*26:05', 'HLA-A*26:06', 'HLA-A*26:07',
         'HLA-A*26:08', 'HLA-A*26:09', 'HLA-A*26:10', 'HLA-A*26:12', 'HLA-A*26:13', 'HLA-A*26:14', 'HLA-A*26:15', 'HLA-A*26:16', 'HLA-A*26:17',
         'HLA-A*26:18', 'HLA-A*26:19', 'HLA-A*26:20', 'HLA-A*26:21', 'HLA-A*26:22', 'HLA-A*26:23', 'HLA-A*26:24', 'HLA-A*26:26', 'HLA-A*26:27',
         'HLA-A*26:28', 'HLA-A*26:29', 'HLA-A*26:30', 'HLA-A*26:31', 'HLA-A*26:32', 'HLA-A*26:33', 'HLA-A*26:34', 'HLA-A*26:35', 'HLA-A*26:36',
         'HLA-A*26:37', 'HLA-A*26:38', 'HLA-A*26:39', 'HLA-A*26:40', 'HLA-A*26:41', 'HLA-A*26:42', 'HLA-A*26:43', 'HLA-A*26:45', 'HLA-A*26:46',
         'HLA-A*26:47', 'HLA-A*26:48', 'HLA-A*26:49', 'HLA-A*26:50', 'HLA-A*29:01', 'HLA-A*29:02', 'HLA-A*29:03', 'HLA-A*29:04', 'HLA-A*29:05',
         'HLA-A*29:06', 'HLA-A*29:07', 'HLA-A*29:09', 'HLA-A*29:10', 'HLA-A*29:11', 'HLA-A*29:12', 'HLA-A*29:13', 'HLA-A*29:14', 'HLA-A*29:15',
         'HLA-A*29:16', 'HLA-A*29:17', 'HLA-A*29:18', 'HLA-A*29:19', 'HLA-A*29:20', 'HLA-A*29:21', 'HLA-A*29:22', 'HLA-A*30:01', 'HLA-A*30:02',
         'HLA-A*30:03', 'HLA-A*30:04', 'HLA-A*30:06', 'HLA-A*30:07', 'HLA-A*30:08', 'HLA-A*30:09', 'HLA-A*30:10', 'HLA-A*30:11', 'HLA-A*30:12',
         'HLA-A*30:13', 'HLA-A*30:15', 'HLA-A*30:16', 'HLA-A*30:17', 'HLA-A*30:18', 'HLA-A*30:19', 'HLA-A*30:20', 'HLA-A*30:22', 'HLA-A*30:23',
         'HLA-A*30:24', 'HLA-A*30:25', 'HLA-A*30:26', 'HLA-A*30:28', 'HLA-A*30:29', 'HLA-A*30:30', 'HLA-A*30:31', 'HLA-A*30:32', 'HLA-A*30:33',
         'HLA-A*30:34', 'HLA-A*30:35', 'HLA-A*30:36', 'HLA-A*30:37', 'HLA-A*30:38', 'HLA-A*30:39', 'HLA-A*30:40', 'HLA-A*30:41', 'HLA-A*31:01',
         'HLA-A*31:02', 'HLA-A*31:03', 'HLA-A*31:04', 'HLA-A*31:05', 'HLA-A*31:06', 'HLA-A*31:07', 'HLA-A*31:08', 'HLA-A*31:09', 'HLA-A*31:10',
         'HLA-A*31:11', 'HLA-A*31:12', 'HLA-A*31:13', 'HLA-A*31:15', 'HLA-A*31:16', 'HLA-A*31:17', 'HLA-A*31:18', 'HLA-A*31:19', 'HLA-A*31:20',
         'HLA-A*31:21', 'HLA-A*31:22', 'HLA-A*31:23', 'HLA-A*31:24', 'HLA-A*31:25', 'HLA-A*31:26', 'HLA-A*31:27', 'HLA-A*31:28', 'HLA-A*31:29',
         'HLA-A*31:30', 'HLA-A*31:31', 'HLA-A*31:32', 'HLA-A*31:33', 'HLA-A*31:34', 'HLA-A*31:35', 'HLA-A*31:36', 'HLA-A*31:37', 'HLA-A*32:01',
         'HLA-A*32:02', 'HLA-A*32:03', 'HLA-A*32:04', 'HLA-A*32:05', 'HLA-A*32:06', 'HLA-A*32:07', 'HLA-A*32:08', 'HLA-A*32:09', 'HLA-A*32:10',
         'HLA-A*32:12', 'HLA-A*32:13', 'HLA-A*32:14', 'HLA-A*32:15', 'HLA-A*32:16', 'HLA-A*32:17', 'HLA-A*32:18', 'HLA-A*32:20', 'HLA-A*32:21',
         'HLA-A*32:22', 'HLA-A*32:23', 'HLA-A*32:24', 'HLA-A*32:25', 'HLA-A*33:01', 'HLA-A*33:03', 'HLA-A*33:04', 'HLA-A*33:05', 'HLA-A*33:06',
         'HLA-A*33:07', 'HLA-A*33:08', 'HLA-A*33:09', 'HLA-A*33:10', 'HLA-A*33:11', 'HLA-A*33:12', 'HLA-A*33:13', 'HLA-A*33:14', 'HLA-A*33:15',
         'HLA-A*33:16', 'HLA-A*33:17', 'HLA-A*33:18', 'HLA-A*33:19', 'HLA-A*33:20', 'HLA-A*33:21', 'HLA-A*33:22', 'HLA-A*33:23', 'HLA-A*33:24',
         'HLA-A*33:25', 'HLA-A*33:26', 'HLA-A*33:27', 'HLA-A*33:28', 'HLA-A*33:29', 'HLA-A*33:30', 'HLA-A*33:31', 'HLA-A*34:01', 'HLA-A*34:02',
         'HLA-A*34:03', 'HLA-A*34:04', 'HLA-A*34:05', 'HLA-A*34:06', 'HLA-A*34:07', 'HLA-A*34:08', 'HLA-A*36:01', 'HLA-A*36:02', 'HLA-A*36:03',
         'HLA-A*36:04', 'HLA-A*36:05', 'HLA-A*43:01', 'HLA-A*66:01', 'HLA-A*66:02', 'HLA-A*66:03', 'HLA-A*66:04', 'HLA-A*66:05', 'HLA-A*66:06',
         'HLA-A*66:07', 'HLA-A*66:08', 'HLA-A*66:09', 'HLA-A*66:10', 'HLA-A*66:11', 'HLA-A*66:12', 'HLA-A*66:13', 'HLA-A*66:14', 'HLA-A*66:15',
         'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*68:03', 'HLA-A*68:04', 'HLA-A*68:05', 'HLA-A*68:06', 'HLA-A*68:07', 'HLA-A*68:08', 'HLA-A*68:09',
         'HLA-A*68:10', 'HLA-A*68:12', 'HLA-A*68:13', 'HLA-A*68:14', 'HLA-A*68:15', 'HLA-A*68:16', 'HLA-A*68:17', 'HLA-A*68:19', 'HLA-A*68:20',
         'HLA-A*68:21', 'HLA-A*68:22', 'HLA-A*68:23', 'HLA-A*68:24', 'HLA-A*68:25', 'HLA-A*68:26', 'HLA-A*68:27', 'HLA-A*68:28', 'HLA-A*68:29',
         'HLA-A*68:30', 'HLA-A*68:31', 'HLA-A*68:32', 'HLA-A*68:33', 'HLA-A*68:34', 'HLA-A*68:35', 'HLA-A*68:36', 'HLA-A*68:37', 'HLA-A*68:38',
         'HLA-A*68:39', 'HLA-A*68:40', 'HLA-A*68:41', 'HLA-A*68:42', 'HLA-A*68:43', 'HLA-A*68:44', 'HLA-A*68:45', 'HLA-A*68:46', 'HLA-A*68:47',
         'HLA-A*68:48', 'HLA-A*68:50', 'HLA-A*68:51', 'HLA-A*68:52', 'HLA-A*68:53', 'HLA-A*68:54', 'HLA-A*69:01', 'HLA-A*74:01', 'HLA-A*74:02',
         'HLA-A*74:03', 'HLA-A*74:04', 'HLA-A*74:05', 'HLA-A*74:06', 'HLA-A*74:07', 'HLA-A*74:08', 'HLA-A*74:09', 'HLA-A*74:10', 'HLA-A*74:11',
         'HLA-A*74:13', 'HLA-A*80:01', 'HLA-A*80:02', 'HLA-B*07:02', 'HLA-B*07:03', 'HLA-B*07:04', 'HLA-B*07:05', 'HLA-B*07:06', 'HLA-B*07:07',
         'HLA-B*07:08', 'HLA-B*07:09', 'HLA-B*07:10', 'HLA-B*07:100', 'HLA-B*07:101', 'HLA-B*07:102', 'HLA-B*07:103', 'HLA-B*07:104',
         'HLA-B*07:105', 'HLA-B*07:106', 'HLA-B*07:107', 'HLA-B*07:108', 'HLA-B*07:109', 'HLA-B*07:11', 'HLA-B*07:110', 'HLA-B*07:112',
         'HLA-B*07:113', 'HLA-B*07:114', 'HLA-B*07:115', 'HLA-B*07:12', 'HLA-B*07:13', 'HLA-B*07:14', 'HLA-B*07:15', 'HLA-B*07:16', 'HLA-B*07:17',
         'HLA-B*07:18', 'HLA-B*07:19', 'HLA-B*07:20', 'HLA-B*07:21', 'HLA-B*07:22', 'HLA-B*07:23', 'HLA-B*07:24', 'HLA-B*07:25', 'HLA-B*07:26',
         'HLA-B*07:27', 'HLA-B*07:28', 'HLA-B*07:29', 'HLA-B*07:30', 'HLA-B*07:31', 'HLA-B*07:32', 'HLA-B*07:33', 'HLA-B*07:34', 'HLA-B*07:35',
         'HLA-B*07:36', 'HLA-B*07:37', 'HLA-B*07:38', 'HLA-B*07:39', 'HLA-B*07:40', 'HLA-B*07:41', 'HLA-B*07:42', 'HLA-B*07:43', 'HLA-B*07:44',
         'HLA-B*07:45', 'HLA-B*07:46', 'HLA-B*07:47', 'HLA-B*07:48', 'HLA-B*07:50', 'HLA-B*07:51', 'HLA-B*07:52', 'HLA-B*07:53', 'HLA-B*07:54',
         'HLA-B*07:55', 'HLA-B*07:56', 'HLA-B*07:57', 'HLA-B*07:58', 'HLA-B*07:59', 'HLA-B*07:60', 'HLA-B*07:61', 'HLA-B*07:62', 'HLA-B*07:63',
         'HLA-B*07:64', 'HLA-B*07:65', 'HLA-B*07:66', 'HLA-B*07:68', 'HLA-B*07:69', 'HLA-B*07:70', 'HLA-B*07:71', 'HLA-B*07:72', 'HLA-B*07:73',
         'HLA-B*07:74', 'HLA-B*07:75', 'HLA-B*07:76', 'HLA-B*07:77', 'HLA-B*07:78', 'HLA-B*07:79', 'HLA-B*07:80', 'HLA-B*07:81', 'HLA-B*07:82',
         'HLA-B*07:83', 'HLA-B*07:84', 'HLA-B*07:85', 'HLA-B*07:86', 'HLA-B*07:87', 'HLA-B*07:88', 'HLA-B*07:89', 'HLA-B*07:90', 'HLA-B*07:91',
         'HLA-B*07:92', 'HLA-B*07:93', 'HLA-B*07:94', 'HLA-B*07:95', 'HLA-B*07:96', 'HLA-B*07:97', 'HLA-B*07:98', 'HLA-B*07:99', 'HLA-B*08:01',
         'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*08:04', 'HLA-B*08:05', 'HLA-B*08:07', 'HLA-B*08:09', 'HLA-B*08:10', 'HLA-B*08:11', 'HLA-B*08:12',
         'HLA-B*08:13', 'HLA-B*08:14', 'HLA-B*08:15', 'HLA-B*08:16', 'HLA-B*08:17', 'HLA-B*08:18', 'HLA-B*08:20', 'HLA-B*08:21', 'HLA-B*08:22',
         'HLA-B*08:23', 'HLA-B*08:24', 'HLA-B*08:25', 'HLA-B*08:26', 'HLA-B*08:27', 'HLA-B*08:28', 'HLA-B*08:29', 'HLA-B*08:31', 'HLA-B*08:32',
         'HLA-B*08:33', 'HLA-B*08:34', 'HLA-B*08:35', 'HLA-B*08:36', 'HLA-B*08:37', 'HLA-B*08:38', 'HLA-B*08:39', 'HLA-B*08:40', 'HLA-B*08:41',
         'HLA-B*08:42', 'HLA-B*08:43', 'HLA-B*08:44', 'HLA-B*08:45', 'HLA-B*08:46', 'HLA-B*08:47', 'HLA-B*08:48', 'HLA-B*08:49', 'HLA-B*08:50',
         'HLA-B*08:51', 'HLA-B*08:52', 'HLA-B*08:53', 'HLA-B*08:54', 'HLA-B*08:55', 'HLA-B*08:56', 'HLA-B*08:57', 'HLA-B*08:58', 'HLA-B*08:59',
         'HLA-B*08:60', 'HLA-B*08:61', 'HLA-B*08:62', 'HLA-B*13:01', 'HLA-B*13:02', 'HLA-B*13:03', 'HLA-B*13:04', 'HLA-B*13:06', 'HLA-B*13:09',
         'HLA-B*13:10', 'HLA-B*13:11', 'HLA-B*13:12', 'HLA-B*13:13', 'HLA-B*13:14', 'HLA-B*13:15', 'HLA-B*13:16', 'HLA-B*13:17', 'HLA-B*13:18',
         'HLA-B*13:19', 'HLA-B*13:20', 'HLA-B*13:21', 'HLA-B*13:22', 'HLA-B*13:23', 'HLA-B*13:25', 'HLA-B*13:26', 'HLA-B*13:27', 'HLA-B*13:28',
         'HLA-B*13:29', 'HLA-B*13:30', 'HLA-B*13:31', 'HLA-B*13:32', 'HLA-B*13:33', 'HLA-B*13:34', 'HLA-B*13:35', 'HLA-B*13:36', 'HLA-B*13:37',
         'HLA-B*13:38', 'HLA-B*13:39', 'HLA-B*14:01', 'HLA-B*14:02', 'HLA-B*14:03', 'HLA-B*14:04', 'HLA-B*14:05', 'HLA-B*14:06', 'HLA-B*14:08',
         'HLA-B*14:09', 'HLA-B*14:10', 'HLA-B*14:11', 'HLA-B*14:12', 'HLA-B*14:13', 'HLA-B*14:14', 'HLA-B*14:15', 'HLA-B*14:16', 'HLA-B*14:17',
         'HLA-B*14:18', 'HLA-B*15:01', 'HLA-B*15:02', 'HLA-B*15:03', 'HLA-B*15:04', 'HLA-B*15:05', 'HLA-B*15:06', 'HLA-B*15:07', 'HLA-B*15:08',
         'HLA-B*15:09', 'HLA-B*15:10', 'HLA-B*15:101', 'HLA-B*15:102', 'HLA-B*15:103', 'HLA-B*15:104', 'HLA-B*15:105', 'HLA-B*15:106',
         'HLA-B*15:107', 'HLA-B*15:108', 'HLA-B*15:109', 'HLA-B*15:11', 'HLA-B*15:110', 'HLA-B*15:112', 'HLA-B*15:113', 'HLA-B*15:114',
         'HLA-B*15:115', 'HLA-B*15:116', 'HLA-B*15:117', 'HLA-B*15:118', 'HLA-B*15:119', 'HLA-B*15:12', 'HLA-B*15:120', 'HLA-B*15:121',
         'HLA-B*15:122', 'HLA-B*15:123', 'HLA-B*15:124', 'HLA-B*15:125', 'HLA-B*15:126', 'HLA-B*15:127', 'HLA-B*15:128', 'HLA-B*15:129',
         'HLA-B*15:13', 'HLA-B*15:131', 'HLA-B*15:132', 'HLA-B*15:133', 'HLA-B*15:134', 'HLA-B*15:135', 'HLA-B*15:136', 'HLA-B*15:137',
         'HLA-B*15:138', 'HLA-B*15:139', 'HLA-B*15:14', 'HLA-B*15:140', 'HLA-B*15:141', 'HLA-B*15:142', 'HLA-B*15:143', 'HLA-B*15:144',
         'HLA-B*15:145', 'HLA-B*15:146', 'HLA-B*15:147', 'HLA-B*15:148', 'HLA-B*15:15', 'HLA-B*15:150', 'HLA-B*15:151', 'HLA-B*15:152',
         'HLA-B*15:153', 'HLA-B*15:154', 'HLA-B*15:155', 'HLA-B*15:156', 'HLA-B*15:157', 'HLA-B*15:158', 'HLA-B*15:159', 'HLA-B*15:16',
         'HLA-B*15:160', 'HLA-B*15:161', 'HLA-B*15:162', 'HLA-B*15:163', 'HLA-B*15:164', 'HLA-B*15:165', 'HLA-B*15:166', 'HLA-B*15:167',
         'HLA-B*15:168', 'HLA-B*15:169', 'HLA-B*15:17', 'HLA-B*15:170', 'HLA-B*15:171', 'HLA-B*15:172', 'HLA-B*15:173', 'HLA-B*15:174',
         'HLA-B*15:175', 'HLA-B*15:176', 'HLA-B*15:177', 'HLA-B*15:178', 'HLA-B*15:179', 'HLA-B*15:18', 'HLA-B*15:180', 'HLA-B*15:183',
         'HLA-B*15:184', 'HLA-B*15:185', 'HLA-B*15:186', 'HLA-B*15:187', 'HLA-B*15:188', 'HLA-B*15:189', 'HLA-B*15:19', 'HLA-B*15:191',
         'HLA-B*15:192', 'HLA-B*15:193', 'HLA-B*15:194', 'HLA-B*15:195', 'HLA-B*15:196', 'HLA-B*15:197', 'HLA-B*15:198', 'HLA-B*15:199',
         'HLA-B*15:20', 'HLA-B*15:200', 'HLA-B*15:201', 'HLA-B*15:202', 'HLA-B*15:21', 'HLA-B*15:23', 'HLA-B*15:24', 'HLA-B*15:25', 'HLA-B*15:27',
         'HLA-B*15:28', 'HLA-B*15:29', 'HLA-B*15:30', 'HLA-B*15:31', 'HLA-B*15:32', 'HLA-B*15:33', 'HLA-B*15:34', 'HLA-B*15:35', 'HLA-B*15:36',
         'HLA-B*15:37', 'HLA-B*15:38', 'HLA-B*15:39', 'HLA-B*15:40', 'HLA-B*15:42', 'HLA-B*15:43', 'HLA-B*15:44', 'HLA-B*15:45', 'HLA-B*15:46',
         'HLA-B*15:47', 'HLA-B*15:48', 'HLA-B*15:49', 'HLA-B*15:50', 'HLA-B*15:51', 'HLA-B*15:52', 'HLA-B*15:53', 'HLA-B*15:54', 'HLA-B*15:55',
         'HLA-B*15:56', 'HLA-B*15:57', 'HLA-B*15:58', 'HLA-B*15:60', 'HLA-B*15:61', 'HLA-B*15:62', 'HLA-B*15:63', 'HLA-B*15:64', 'HLA-B*15:65',
         'HLA-B*15:66', 'HLA-B*15:67', 'HLA-B*15:68', 'HLA-B*15:69', 'HLA-B*15:70', 'HLA-B*15:71', 'HLA-B*15:72', 'HLA-B*15:73', 'HLA-B*15:74',
         'HLA-B*15:75', 'HLA-B*15:76', 'HLA-B*15:77', 'HLA-B*15:78', 'HLA-B*15:80', 'HLA-B*15:81', 'HLA-B*15:82', 'HLA-B*15:83', 'HLA-B*15:84',
         'HLA-B*15:85', 'HLA-B*15:86', 'HLA-B*15:87', 'HLA-B*15:88', 'HLA-B*15:89', 'HLA-B*15:90', 'HLA-B*15:91', 'HLA-B*15:92', 'HLA-B*15:93',
         'HLA-B*15:95', 'HLA-B*15:96', 'HLA-B*15:97', 'HLA-B*15:98', 'HLA-B*15:99', 'HLA-B*18:01', 'HLA-B*18:02', 'HLA-B*18:03', 'HLA-B*18:04',
         'HLA-B*18:05', 'HLA-B*18:06', 'HLA-B*18:07', 'HLA-B*18:08', 'HLA-B*18:09', 'HLA-B*18:10', 'HLA-B*18:11', 'HLA-B*18:12', 'HLA-B*18:13',
         'HLA-B*18:14', 'HLA-B*18:15', 'HLA-B*18:18', 'HLA-B*18:19', 'HLA-B*18:20', 'HLA-B*18:21', 'HLA-B*18:22', 'HLA-B*18:24', 'HLA-B*18:25',
         'HLA-B*18:26', 'HLA-B*18:27', 'HLA-B*18:28', 'HLA-B*18:29', 'HLA-B*18:30', 'HLA-B*18:31', 'HLA-B*18:32', 'HLA-B*18:33', 'HLA-B*18:34',
         'HLA-B*18:35', 'HLA-B*18:36', 'HLA-B*18:37', 'HLA-B*18:38', 'HLA-B*18:39', 'HLA-B*18:40', 'HLA-B*18:41', 'HLA-B*18:42', 'HLA-B*18:43',
         'HLA-B*18:44', 'HLA-B*18:45', 'HLA-B*18:46', 'HLA-B*18:47', 'HLA-B*18:48', 'HLA-B*18:49', 'HLA-B*18:50', 'HLA-B*27:01', 'HLA-B*27:02',
         'HLA-B*27:03', 'HLA-B*27:04', 'HLA-B*27:05', 'HLA-B*27:06', 'HLA-B*27:07', 'HLA-B*27:08', 'HLA-B*27:09', 'HLA-B*27:10', 'HLA-B*27:11',
         'HLA-B*27:12', 'HLA-B*27:13', 'HLA-B*27:14', 'HLA-B*27:15', 'HLA-B*27:16', 'HLA-B*27:17', 'HLA-B*27:18', 'HLA-B*27:19', 'HLA-B*27:20',
         'HLA-B*27:21', 'HLA-B*27:23', 'HLA-B*27:24', 'HLA-B*27:25', 'HLA-B*27:26', 'HLA-B*27:27', 'HLA-B*27:28', 'HLA-B*27:29', 'HLA-B*27:30',
         'HLA-B*27:31', 'HLA-B*27:32', 'HLA-B*27:33', 'HLA-B*27:34', 'HLA-B*27:35', 'HLA-B*27:36', 'HLA-B*27:37', 'HLA-B*27:38', 'HLA-B*27:39',
         'HLA-B*27:40', 'HLA-B*27:41', 'HLA-B*27:42', 'HLA-B*27:43', 'HLA-B*27:44', 'HLA-B*27:45', 'HLA-B*27:46', 'HLA-B*27:47', 'HLA-B*27:48',
         'HLA-B*27:49', 'HLA-B*27:50', 'HLA-B*27:51', 'HLA-B*27:52', 'HLA-B*27:53', 'HLA-B*27:54', 'HLA-B*27:55', 'HLA-B*27:56', 'HLA-B*27:57',
         'HLA-B*27:58', 'HLA-B*27:60', 'HLA-B*27:61', 'HLA-B*27:62', 'HLA-B*27:63', 'HLA-B*27:67', 'HLA-B*27:68', 'HLA-B*27:69', 'HLA-B*35:01',
         'HLA-B*35:02', 'HLA-B*35:03', 'HLA-B*35:04', 'HLA-B*35:05', 'HLA-B*35:06', 'HLA-B*35:07', 'HLA-B*35:08', 'HLA-B*35:09', 'HLA-B*35:10',
         'HLA-B*35:100', 'HLA-B*35:101', 'HLA-B*35:102', 'HLA-B*35:103', 'HLA-B*35:104', 'HLA-B*35:105', 'HLA-B*35:106', 'HLA-B*35:107',
         'HLA-B*35:108', 'HLA-B*35:109', 'HLA-B*35:11', 'HLA-B*35:110', 'HLA-B*35:111', 'HLA-B*35:112', 'HLA-B*35:113', 'HLA-B*35:114',
         'HLA-B*35:115', 'HLA-B*35:116', 'HLA-B*35:117', 'HLA-B*35:118', 'HLA-B*35:119', 'HLA-B*35:12', 'HLA-B*35:120', 'HLA-B*35:121',
         'HLA-B*35:122', 'HLA-B*35:123', 'HLA-B*35:124', 'HLA-B*35:125', 'HLA-B*35:126', 'HLA-B*35:127', 'HLA-B*35:128', 'HLA-B*35:13',
         'HLA-B*35:131', 'HLA-B*35:132', 'HLA-B*35:133', 'HLA-B*35:135', 'HLA-B*35:136', 'HLA-B*35:137', 'HLA-B*35:138', 'HLA-B*35:139',
         'HLA-B*35:14', 'HLA-B*35:140', 'HLA-B*35:141', 'HLA-B*35:142', 'HLA-B*35:143', 'HLA-B*35:144', 'HLA-B*35:15', 'HLA-B*35:16', 'HLA-B*35:17',
         'HLA-B*35:18', 'HLA-B*35:19', 'HLA-B*35:20', 'HLA-B*35:21', 'HLA-B*35:22', 'HLA-B*35:23', 'HLA-B*35:24', 'HLA-B*35:25', 'HLA-B*35:26',
         'HLA-B*35:27', 'HLA-B*35:28', 'HLA-B*35:29', 'HLA-B*35:30', 'HLA-B*35:31', 'HLA-B*35:32', 'HLA-B*35:33', 'HLA-B*35:34', 'HLA-B*35:35',
         'HLA-B*35:36', 'HLA-B*35:37', 'HLA-B*35:38', 'HLA-B*35:39', 'HLA-B*35:41', 'HLA-B*35:42', 'HLA-B*35:43', 'HLA-B*35:44', 'HLA-B*35:45',
         'HLA-B*35:46', 'HLA-B*35:47', 'HLA-B*35:48', 'HLA-B*35:49', 'HLA-B*35:50', 'HLA-B*35:51', 'HLA-B*35:52', 'HLA-B*35:54', 'HLA-B*35:55',
         'HLA-B*35:56', 'HLA-B*35:57', 'HLA-B*35:58', 'HLA-B*35:59', 'HLA-B*35:60', 'HLA-B*35:61', 'HLA-B*35:62', 'HLA-B*35:63', 'HLA-B*35:64',
         'HLA-B*35:66', 'HLA-B*35:67', 'HLA-B*35:68', 'HLA-B*35:69', 'HLA-B*35:70', 'HLA-B*35:71', 'HLA-B*35:72', 'HLA-B*35:74', 'HLA-B*35:75',
         'HLA-B*35:76', 'HLA-B*35:77', 'HLA-B*35:78', 'HLA-B*35:79', 'HLA-B*35:80', 'HLA-B*35:81', 'HLA-B*35:82', 'HLA-B*35:83', 'HLA-B*35:84',
         'HLA-B*35:85', 'HLA-B*35:86', 'HLA-B*35:87', 'HLA-B*35:88', 'HLA-B*35:89', 'HLA-B*35:90', 'HLA-B*35:91', 'HLA-B*35:92', 'HLA-B*35:93',
         'HLA-B*35:94', 'HLA-B*35:95', 'HLA-B*35:96', 'HLA-B*35:97', 'HLA-B*35:98', 'HLA-B*35:99', 'HLA-B*37:01', 'HLA-B*37:02', 'HLA-B*37:04',
         'HLA-B*37:05', 'HLA-B*37:06', 'HLA-B*37:07', 'HLA-B*37:08', 'HLA-B*37:09', 'HLA-B*37:10', 'HLA-B*37:11', 'HLA-B*37:12', 'HLA-B*37:13',
         'HLA-B*37:14', 'HLA-B*37:15', 'HLA-B*37:17', 'HLA-B*37:18', 'HLA-B*37:19', 'HLA-B*37:20', 'HLA-B*37:21', 'HLA-B*37:22', 'HLA-B*37:23',
         'HLA-B*38:01', 'HLA-B*38:02', 'HLA-B*38:03', 'HLA-B*38:04', 'HLA-B*38:05', 'HLA-B*38:06', 'HLA-B*38:07', 'HLA-B*38:08', 'HLA-B*38:09',
         'HLA-B*38:10', 'HLA-B*38:11', 'HLA-B*38:12', 'HLA-B*38:13', 'HLA-B*38:14', 'HLA-B*38:15', 'HLA-B*38:16', 'HLA-B*38:17', 'HLA-B*38:18',
         'HLA-B*38:19', 'HLA-B*38:20', 'HLA-B*38:21', 'HLA-B*38:22', 'HLA-B*38:23', 'HLA-B*39:01', 'HLA-B*39:02', 'HLA-B*39:03', 'HLA-B*39:04',
         'HLA-B*39:05', 'HLA-B*39:06', 'HLA-B*39:07', 'HLA-B*39:08', 'HLA-B*39:09', 'HLA-B*39:10', 'HLA-B*39:11', 'HLA-B*39:12', 'HLA-B*39:13',
         'HLA-B*39:14', 'HLA-B*39:15', 'HLA-B*39:16', 'HLA-B*39:17', 'HLA-B*39:18', 'HLA-B*39:19', 'HLA-B*39:20', 'HLA-B*39:22', 'HLA-B*39:23',
         'HLA-B*39:24', 'HLA-B*39:26', 'HLA-B*39:27', 'HLA-B*39:28', 'HLA-B*39:29', 'HLA-B*39:30', 'HLA-B*39:31', 'HLA-B*39:32', 'HLA-B*39:33',
         'HLA-B*39:34', 'HLA-B*39:35', 'HLA-B*39:36', 'HLA-B*39:37', 'HLA-B*39:39', 'HLA-B*39:41', 'HLA-B*39:42', 'HLA-B*39:43', 'HLA-B*39:44',
         'HLA-B*39:45', 'HLA-B*39:46', 'HLA-B*39:47', 'HLA-B*39:48', 'HLA-B*39:49', 'HLA-B*39:50', 'HLA-B*39:51', 'HLA-B*39:52', 'HLA-B*39:53',
         'HLA-B*39:54', 'HLA-B*39:55', 'HLA-B*39:56', 'HLA-B*39:57', 'HLA-B*39:58', 'HLA-B*39:59', 'HLA-B*39:60', 'HLA-B*40:01', 'HLA-B*40:02',
         'HLA-B*40:03', 'HLA-B*40:04', 'HLA-B*40:05', 'HLA-B*40:06', 'HLA-B*40:07', 'HLA-B*40:08', 'HLA-B*40:09', 'HLA-B*40:10', 'HLA-B*40:100',
         'HLA-B*40:101', 'HLA-B*40:102', 'HLA-B*40:103', 'HLA-B*40:104', 'HLA-B*40:105', 'HLA-B*40:106', 'HLA-B*40:107', 'HLA-B*40:108',
         'HLA-B*40:109', 'HLA-B*40:11', 'HLA-B*40:110', 'HLA-B*40:111', 'HLA-B*40:112', 'HLA-B*40:113', 'HLA-B*40:114', 'HLA-B*40:115',
         'HLA-B*40:116', 'HLA-B*40:117', 'HLA-B*40:119', 'HLA-B*40:12', 'HLA-B*40:120', 'HLA-B*40:121', 'HLA-B*40:122', 'HLA-B*40:123',
         'HLA-B*40:124', 'HLA-B*40:125', 'HLA-B*40:126', 'HLA-B*40:127', 'HLA-B*40:128', 'HLA-B*40:129', 'HLA-B*40:13', 'HLA-B*40:130',
         'HLA-B*40:131', 'HLA-B*40:132', 'HLA-B*40:134', 'HLA-B*40:135', 'HLA-B*40:136', 'HLA-B*40:137', 'HLA-B*40:138', 'HLA-B*40:139',
         'HLA-B*40:14', 'HLA-B*40:140', 'HLA-B*40:141', 'HLA-B*40:143', 'HLA-B*40:145', 'HLA-B*40:146', 'HLA-B*40:147', 'HLA-B*40:15',
         'HLA-B*40:16', 'HLA-B*40:18', 'HLA-B*40:19', 'HLA-B*40:20', 'HLA-B*40:21', 'HLA-B*40:23', 'HLA-B*40:24', 'HLA-B*40:25', 'HLA-B*40:26',
         'HLA-B*40:27', 'HLA-B*40:28', 'HLA-B*40:29', 'HLA-B*40:30', 'HLA-B*40:31', 'HLA-B*40:32', 'HLA-B*40:33', 'HLA-B*40:34', 'HLA-B*40:35',
         'HLA-B*40:36', 'HLA-B*40:37', 'HLA-B*40:38', 'HLA-B*40:39', 'HLA-B*40:40', 'HLA-B*40:42', 'HLA-B*40:43', 'HLA-B*40:44', 'HLA-B*40:45',
         'HLA-B*40:46', 'HLA-B*40:47', 'HLA-B*40:48', 'HLA-B*40:49', 'HLA-B*40:50', 'HLA-B*40:51', 'HLA-B*40:52', 'HLA-B*40:53', 'HLA-B*40:54',
         'HLA-B*40:55', 'HLA-B*40:56', 'HLA-B*40:57', 'HLA-B*40:58', 'HLA-B*40:59', 'HLA-B*40:60', 'HLA-B*40:61', 'HLA-B*40:62', 'HLA-B*40:63',
         'HLA-B*40:64', 'HLA-B*40:65', 'HLA-B*40:66', 'HLA-B*40:67', 'HLA-B*40:68', 'HLA-B*40:69', 'HLA-B*40:70', 'HLA-B*40:71', 'HLA-B*40:72',
         'HLA-B*40:73', 'HLA-B*40:74', 'HLA-B*40:75', 'HLA-B*40:76', 'HLA-B*40:77', 'HLA-B*40:78', 'HLA-B*40:79', 'HLA-B*40:80', 'HLA-B*40:81',
         'HLA-B*40:82', 'HLA-B*40:83', 'HLA-B*40:84', 'HLA-B*40:85', 'HLA-B*40:86', 'HLA-B*40:87', 'HLA-B*40:88', 'HLA-B*40:89', 'HLA-B*40:90',
         'HLA-B*40:91', 'HLA-B*40:92', 'HLA-B*40:93', 'HLA-B*40:94', 'HLA-B*40:95', 'HLA-B*40:96', 'HLA-B*40:97', 'HLA-B*40:98', 'HLA-B*40:99',
         'HLA-B*41:01', 'HLA-B*41:02', 'HLA-B*41:03', 'HLA-B*41:04', 'HLA-B*41:05', 'HLA-B*41:06', 'HLA-B*41:07', 'HLA-B*41:08', 'HLA-B*41:09',
         'HLA-B*41:10', 'HLA-B*41:11', 'HLA-B*41:12', 'HLA-B*42:01', 'HLA-B*42:02', 'HLA-B*42:04', 'HLA-B*42:05', 'HLA-B*42:06', 'HLA-B*42:07',
         'HLA-B*42:08', 'HLA-B*42:09', 'HLA-B*42:10', 'HLA-B*42:11', 'HLA-B*42:12', 'HLA-B*42:13', 'HLA-B*42:14', 'HLA-B*44:02', 'HLA-B*44:03',
         'HLA-B*44:04', 'HLA-B*44:05', 'HLA-B*44:06', 'HLA-B*44:07', 'HLA-B*44:08', 'HLA-B*44:09', 'HLA-B*44:10', 'HLA-B*44:100', 'HLA-B*44:101',
         'HLA-B*44:102', 'HLA-B*44:103', 'HLA-B*44:104', 'HLA-B*44:105', 'HLA-B*44:106', 'HLA-B*44:107', 'HLA-B*44:109', 'HLA-B*44:11',
         'HLA-B*44:110', 'HLA-B*44:12', 'HLA-B*44:13', 'HLA-B*44:14', 'HLA-B*44:15', 'HLA-B*44:16', 'HLA-B*44:17', 'HLA-B*44:18', 'HLA-B*44:20',
         'HLA-B*44:21', 'HLA-B*44:22', 'HLA-B*44:24', 'HLA-B*44:25', 'HLA-B*44:26', 'HLA-B*44:27', 'HLA-B*44:28', 'HLA-B*44:29', 'HLA-B*44:30',
         'HLA-B*44:31', 'HLA-B*44:32', 'HLA-B*44:33', 'HLA-B*44:34', 'HLA-B*44:35', 'HLA-B*44:36', 'HLA-B*44:37', 'HLA-B*44:38', 'HLA-B*44:39',
         'HLA-B*44:40', 'HLA-B*44:41', 'HLA-B*44:42', 'HLA-B*44:43', 'HLA-B*44:44', 'HLA-B*44:45', 'HLA-B*44:46', 'HLA-B*44:47', 'HLA-B*44:48',
         'HLA-B*44:49', 'HLA-B*44:50', 'HLA-B*44:51', 'HLA-B*44:53', 'HLA-B*44:54', 'HLA-B*44:55', 'HLA-B*44:57', 'HLA-B*44:59', 'HLA-B*44:60',
         'HLA-B*44:62', 'HLA-B*44:63', 'HLA-B*44:64', 'HLA-B*44:65', 'HLA-B*44:66', 'HLA-B*44:67', 'HLA-B*44:68', 'HLA-B*44:69', 'HLA-B*44:70',
         'HLA-B*44:71', 'HLA-B*44:72', 'HLA-B*44:73', 'HLA-B*44:74', 'HLA-B*44:75', 'HLA-B*44:76', 'HLA-B*44:77', 'HLA-B*44:78', 'HLA-B*44:79',
         'HLA-B*44:80', 'HLA-B*44:81', 'HLA-B*44:82', 'HLA-B*44:83', 'HLA-B*44:84', 'HLA-B*44:85', 'HLA-B*44:86', 'HLA-B*44:87', 'HLA-B*44:88',
         'HLA-B*44:89', 'HLA-B*44:90', 'HLA-B*44:91', 'HLA-B*44:92', 'HLA-B*44:93', 'HLA-B*44:94', 'HLA-B*44:95', 'HLA-B*44:96', 'HLA-B*44:97',
         'HLA-B*44:98', 'HLA-B*44:99', 'HLA-B*45:01', 'HLA-B*45:02', 'HLA-B*45:03', 'HLA-B*45:04', 'HLA-B*45:05', 'HLA-B*45:06', 'HLA-B*45:07',
         'HLA-B*45:08', 'HLA-B*45:09', 'HLA-B*45:10', 'HLA-B*45:11', 'HLA-B*45:12', 'HLA-B*46:01', 'HLA-B*46:02', 'HLA-B*46:03', 'HLA-B*46:04',
         'HLA-B*46:05', 'HLA-B*46:06', 'HLA-B*46:08', 'HLA-B*46:09', 'HLA-B*46:10', 'HLA-B*46:11', 'HLA-B*46:12', 'HLA-B*46:13', 'HLA-B*46:14',
         'HLA-B*46:16', 'HLA-B*46:17', 'HLA-B*46:18', 'HLA-B*46:19', 'HLA-B*46:20', 'HLA-B*46:21', 'HLA-B*46:22', 'HLA-B*46:23', 'HLA-B*46:24',
         'HLA-B*47:01', 'HLA-B*47:02', 'HLA-B*47:03', 'HLA-B*47:04', 'HLA-B*47:05', 'HLA-B*47:06', 'HLA-B*47:07', 'HLA-B*48:01', 'HLA-B*48:02',
         'HLA-B*48:03', 'HLA-B*48:04', 'HLA-B*48:05', 'HLA-B*48:06', 'HLA-B*48:07', 'HLA-B*48:08', 'HLA-B*48:09', 'HLA-B*48:10', 'HLA-B*48:11',
         'HLA-B*48:12', 'HLA-B*48:13', 'HLA-B*48:14', 'HLA-B*48:15', 'HLA-B*48:16', 'HLA-B*48:17', 'HLA-B*48:18', 'HLA-B*48:19', 'HLA-B*48:20',
         'HLA-B*48:21', 'HLA-B*48:22', 'HLA-B*48:23', 'HLA-B*49:01', 'HLA-B*49:02', 'HLA-B*49:03', 'HLA-B*49:04', 'HLA-B*49:05', 'HLA-B*49:06',
         'HLA-B*49:07', 'HLA-B*49:08', 'HLA-B*49:09', 'HLA-B*49:10', 'HLA-B*50:01', 'HLA-B*50:02', 'HLA-B*50:04', 'HLA-B*50:05', 'HLA-B*50:06',
         'HLA-B*50:07', 'HLA-B*50:08', 'HLA-B*50:09', 'HLA-B*51:01', 'HLA-B*51:02', 'HLA-B*51:03', 'HLA-B*51:04', 'HLA-B*51:05', 'HLA-B*51:06',
         'HLA-B*51:07', 'HLA-B*51:08', 'HLA-B*51:09', 'HLA-B*51:12', 'HLA-B*51:13', 'HLA-B*51:14', 'HLA-B*51:15', 'HLA-B*51:16', 'HLA-B*51:17',
         'HLA-B*51:18', 'HLA-B*51:19', 'HLA-B*51:20', 'HLA-B*51:21', 'HLA-B*51:22', 'HLA-B*51:23', 'HLA-B*51:24', 'HLA-B*51:26', 'HLA-B*51:28',
         'HLA-B*51:29', 'HLA-B*51:30', 'HLA-B*51:31', 'HLA-B*51:32', 'HLA-B*51:33', 'HLA-B*51:34', 'HLA-B*51:35', 'HLA-B*51:36', 'HLA-B*51:37',
         'HLA-B*51:38', 'HLA-B*51:39', 'HLA-B*51:40', 'HLA-B*51:42', 'HLA-B*51:43', 'HLA-B*51:45', 'HLA-B*51:46', 'HLA-B*51:48', 'HLA-B*51:49',
         'HLA-B*51:50', 'HLA-B*51:51', 'HLA-B*51:52', 'HLA-B*51:53', 'HLA-B*51:54', 'HLA-B*51:55', 'HLA-B*51:56', 'HLA-B*51:57', 'HLA-B*51:58',
         'HLA-B*51:59', 'HLA-B*51:60', 'HLA-B*51:61', 'HLA-B*51:62', 'HLA-B*51:63', 'HLA-B*51:64', 'HLA-B*51:65', 'HLA-B*51:66', 'HLA-B*51:67',
         'HLA-B*51:68', 'HLA-B*51:69', 'HLA-B*51:70', 'HLA-B*51:71', 'HLA-B*51:72', 'HLA-B*51:73', 'HLA-B*51:74', 'HLA-B*51:75', 'HLA-B*51:76',
         'HLA-B*51:77', 'HLA-B*51:78', 'HLA-B*51:79', 'HLA-B*51:80', 'HLA-B*51:81', 'HLA-B*51:82', 'HLA-B*51:83', 'HLA-B*51:84', 'HLA-B*51:85',
         'HLA-B*51:86', 'HLA-B*51:87', 'HLA-B*51:88', 'HLA-B*51:89', 'HLA-B*51:90', 'HLA-B*51:91', 'HLA-B*51:92', 'HLA-B*51:93', 'HLA-B*51:94',
         'HLA-B*51:95', 'HLA-B*51:96', 'HLA-B*52:01', 'HLA-B*52:02', 'HLA-B*52:03', 'HLA-B*52:04', 'HLA-B*52:05', 'HLA-B*52:06', 'HLA-B*52:07',
         'HLA-B*52:08', 'HLA-B*52:09', 'HLA-B*52:10', 'HLA-B*52:11', 'HLA-B*52:12', 'HLA-B*52:13', 'HLA-B*52:14', 'HLA-B*52:15', 'HLA-B*52:16',
         'HLA-B*52:17', 'HLA-B*52:18', 'HLA-B*52:19', 'HLA-B*52:20', 'HLA-B*52:21', 'HLA-B*53:01', 'HLA-B*53:02', 'HLA-B*53:03', 'HLA-B*53:04',
         'HLA-B*53:05', 'HLA-B*53:06', 'HLA-B*53:07', 'HLA-B*53:08', 'HLA-B*53:09', 'HLA-B*53:10', 'HLA-B*53:11', 'HLA-B*53:12', 'HLA-B*53:13',
         'HLA-B*53:14', 'HLA-B*53:15', 'HLA-B*53:16', 'HLA-B*53:17', 'HLA-B*53:18', 'HLA-B*53:19', 'HLA-B*53:20', 'HLA-B*53:21', 'HLA-B*53:22',
         'HLA-B*53:23', 'HLA-B*54:01', 'HLA-B*54:02', 'HLA-B*54:03', 'HLA-B*54:04', 'HLA-B*54:06', 'HLA-B*54:07', 'HLA-B*54:09', 'HLA-B*54:10',
         'HLA-B*54:11', 'HLA-B*54:12', 'HLA-B*54:13', 'HLA-B*54:14', 'HLA-B*54:15', 'HLA-B*54:16', 'HLA-B*54:17', 'HLA-B*54:18', 'HLA-B*54:19',
         'HLA-B*54:20', 'HLA-B*54:21', 'HLA-B*54:22', 'HLA-B*54:23', 'HLA-B*55:01', 'HLA-B*55:02', 'HLA-B*55:03', 'HLA-B*55:04', 'HLA-B*55:05',
         'HLA-B*55:07', 'HLA-B*55:08', 'HLA-B*55:09', 'HLA-B*55:10', 'HLA-B*55:11', 'HLA-B*55:12', 'HLA-B*55:13', 'HLA-B*55:14', 'HLA-B*55:15',
         'HLA-B*55:16', 'HLA-B*55:17', 'HLA-B*55:18', 'HLA-B*55:19', 'HLA-B*55:20', 'HLA-B*55:21', 'HLA-B*55:22', 'HLA-B*55:23', 'HLA-B*55:24',
         'HLA-B*55:25', 'HLA-B*55:26', 'HLA-B*55:27', 'HLA-B*55:28', 'HLA-B*55:29', 'HLA-B*55:30', 'HLA-B*55:31', 'HLA-B*55:32', 'HLA-B*55:33',
         'HLA-B*55:34', 'HLA-B*55:35', 'HLA-B*55:36', 'HLA-B*55:37', 'HLA-B*55:38', 'HLA-B*55:39', 'HLA-B*55:40', 'HLA-B*55:41', 'HLA-B*55:42',
         'HLA-B*55:43', 'HLA-B*56:01', 'HLA-B*56:02', 'HLA-B*56:03', 'HLA-B*56:04', 'HLA-B*56:05', 'HLA-B*56:06', 'HLA-B*56:07', 'HLA-B*56:08',
         'HLA-B*56:09', 'HLA-B*56:10', 'HLA-B*56:11', 'HLA-B*56:12', 'HLA-B*56:13', 'HLA-B*56:14', 'HLA-B*56:15', 'HLA-B*56:16', 'HLA-B*56:17',
         'HLA-B*56:18', 'HLA-B*56:20', 'HLA-B*56:21', 'HLA-B*56:22', 'HLA-B*56:23', 'HLA-B*56:24', 'HLA-B*56:25', 'HLA-B*56:26', 'HLA-B*56:27',
         'HLA-B*56:29', 'HLA-B*57:01', 'HLA-B*57:02', 'HLA-B*57:03', 'HLA-B*57:04', 'HLA-B*57:05', 'HLA-B*57:06', 'HLA-B*57:07', 'HLA-B*57:08',
         'HLA-B*57:09', 'HLA-B*57:10', 'HLA-B*57:11', 'HLA-B*57:12', 'HLA-B*57:13', 'HLA-B*57:14', 'HLA-B*57:15', 'HLA-B*57:16', 'HLA-B*57:17',
         'HLA-B*57:18', 'HLA-B*57:19', 'HLA-B*57:20', 'HLA-B*57:21', 'HLA-B*57:22', 'HLA-B*57:23', 'HLA-B*57:24', 'HLA-B*57:25', 'HLA-B*57:26',
         'HLA-B*57:27', 'HLA-B*57:29', 'HLA-B*57:30', 'HLA-B*57:31', 'HLA-B*57:32', 'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*58:04', 'HLA-B*58:05',
         'HLA-B*58:06', 'HLA-B*58:07', 'HLA-B*58:08', 'HLA-B*58:09', 'HLA-B*58:11', 'HLA-B*58:12', 'HLA-B*58:13', 'HLA-B*58:14', 'HLA-B*58:15',
         'HLA-B*58:16', 'HLA-B*58:18', 'HLA-B*58:19', 'HLA-B*58:20', 'HLA-B*58:21', 'HLA-B*58:22', 'HLA-B*58:23', 'HLA-B*58:24', 'HLA-B*58:25',
         'HLA-B*58:26', 'HLA-B*58:27', 'HLA-B*58:28', 'HLA-B*58:29', 'HLA-B*58:30', 'HLA-B*59:01', 'HLA-B*59:02', 'HLA-B*59:03', 'HLA-B*59:04',
         'HLA-B*59:05', 'HLA-B*67:01', 'HLA-B*67:02', 'HLA-B*73:01', 'HLA-B*73:02', 'HLA-B*78:01', 'HLA-B*78:02', 'HLA-B*78:03', 'HLA-B*78:04',
         'HLA-B*78:05', 'HLA-B*78:06', 'HLA-B*78:07', 'HLA-B*81:01', 'HLA-B*81:02', 'HLA-B*81:03', 'HLA-B*81:05', 'HLA-B*82:01', 'HLA-B*82:02',
         'HLA-B*82:03', 'HLA-B*83:01', 'HLA-C*01:02', 'HLA-C*01:03', 'HLA-C*01:04', 'HLA-C*01:05', 'HLA-C*01:06', 'HLA-C*01:07', 'HLA-C*01:08',
         'HLA-C*01:09', 'HLA-C*01:10', 'HLA-C*01:11', 'HLA-C*01:12', 'HLA-C*01:13', 'HLA-C*01:14', 'HLA-C*01:15', 'HLA-C*01:16', 'HLA-C*01:17',
         'HLA-C*01:18', 'HLA-C*01:19', 'HLA-C*01:20', 'HLA-C*01:21', 'HLA-C*01:22', 'HLA-C*01:23', 'HLA-C*01:24', 'HLA-C*01:25', 'HLA-C*01:26',
         'HLA-C*01:27', 'HLA-C*01:28', 'HLA-C*01:29', 'HLA-C*01:30', 'HLA-C*01:31', 'HLA-C*01:32', 'HLA-C*01:33', 'HLA-C*01:34', 'HLA-C*01:35',
         'HLA-C*01:36', 'HLA-C*01:38', 'HLA-C*01:39', 'HLA-C*01:40', 'HLA-C*02:02', 'HLA-C*02:03', 'HLA-C*02:04', 'HLA-C*02:05', 'HLA-C*02:06',
         'HLA-C*02:07', 'HLA-C*02:08', 'HLA-C*02:09', 'HLA-C*02:10', 'HLA-C*02:11', 'HLA-C*02:12', 'HLA-C*02:13', 'HLA-C*02:14', 'HLA-C*02:15',
         'HLA-C*02:16', 'HLA-C*02:17', 'HLA-C*02:18', 'HLA-C*02:19', 'HLA-C*02:20', 'HLA-C*02:21', 'HLA-C*02:22', 'HLA-C*02:23', 'HLA-C*02:24',
         'HLA-C*02:26', 'HLA-C*02:27', 'HLA-C*02:28', 'HLA-C*02:29', 'HLA-C*02:30', 'HLA-C*02:31', 'HLA-C*02:32', 'HLA-C*02:33', 'HLA-C*02:34',
         'HLA-C*02:35', 'HLA-C*02:36', 'HLA-C*02:37', 'HLA-C*02:39', 'HLA-C*02:40', 'HLA-C*03:01', 'HLA-C*03:02', 'HLA-C*03:03', 'HLA-C*03:04',
         'HLA-C*03:05', 'HLA-C*03:06', 'HLA-C*03:07', 'HLA-C*03:08', 'HLA-C*03:09', 'HLA-C*03:10', 'HLA-C*03:11', 'HLA-C*03:12', 'HLA-C*03:13',
         'HLA-C*03:14', 'HLA-C*03:15', 'HLA-C*03:16', 'HLA-C*03:17', 'HLA-C*03:18', 'HLA-C*03:19', 'HLA-C*03:21', 'HLA-C*03:23', 'HLA-C*03:24',
         'HLA-C*03:25', 'HLA-C*03:26', 'HLA-C*03:27', 'HLA-C*03:28', 'HLA-C*03:29', 'HLA-C*03:30', 'HLA-C*03:31', 'HLA-C*03:32', 'HLA-C*03:33',
         'HLA-C*03:34', 'HLA-C*03:35', 'HLA-C*03:36', 'HLA-C*03:37', 'HLA-C*03:38', 'HLA-C*03:39', 'HLA-C*03:40', 'HLA-C*03:41', 'HLA-C*03:42',
         'HLA-C*03:43', 'HLA-C*03:44', 'HLA-C*03:45', 'HLA-C*03:46', 'HLA-C*03:47', 'HLA-C*03:48', 'HLA-C*03:49', 'HLA-C*03:50', 'HLA-C*03:51',
         'HLA-C*03:52', 'HLA-C*03:53', 'HLA-C*03:54', 'HLA-C*03:55', 'HLA-C*03:56', 'HLA-C*03:57', 'HLA-C*03:58', 'HLA-C*03:59', 'HLA-C*03:60',
         'HLA-C*03:61', 'HLA-C*03:62', 'HLA-C*03:63', 'HLA-C*03:64', 'HLA-C*03:65', 'HLA-C*03:66', 'HLA-C*03:67', 'HLA-C*03:68', 'HLA-C*03:69',
         'HLA-C*03:70', 'HLA-C*03:71', 'HLA-C*03:72', 'HLA-C*03:73', 'HLA-C*03:74', 'HLA-C*03:75', 'HLA-C*03:76', 'HLA-C*03:77', 'HLA-C*03:78',
         'HLA-C*03:79', 'HLA-C*03:80', 'HLA-C*03:81', 'HLA-C*03:82', 'HLA-C*03:83', 'HLA-C*03:84', 'HLA-C*03:85', 'HLA-C*03:86', 'HLA-C*03:87',
         'HLA-C*03:88', 'HLA-C*03:89', 'HLA-C*03:90', 'HLA-C*03:91', 'HLA-C*03:92', 'HLA-C*03:93', 'HLA-C*03:94', 'HLA-C*04:01', 'HLA-C*04:03',
         'HLA-C*04:04', 'HLA-C*04:05', 'HLA-C*04:06', 'HLA-C*04:07', 'HLA-C*04:08', 'HLA-C*04:10', 'HLA-C*04:11', 'HLA-C*04:12', 'HLA-C*04:13',
         'HLA-C*04:14', 'HLA-C*04:15', 'HLA-C*04:16', 'HLA-C*04:17', 'HLA-C*04:18', 'HLA-C*04:19', 'HLA-C*04:20', 'HLA-C*04:23', 'HLA-C*04:24',
         'HLA-C*04:25', 'HLA-C*04:26', 'HLA-C*04:27', 'HLA-C*04:28', 'HLA-C*04:29', 'HLA-C*04:30', 'HLA-C*04:31', 'HLA-C*04:32', 'HLA-C*04:33',
         'HLA-C*04:34', 'HLA-C*04:35', 'HLA-C*04:36', 'HLA-C*04:37', 'HLA-C*04:38', 'HLA-C*04:39', 'HLA-C*04:40', 'HLA-C*04:41', 'HLA-C*04:42',
         'HLA-C*04:43', 'HLA-C*04:44', 'HLA-C*04:45', 'HLA-C*04:46', 'HLA-C*04:47', 'HLA-C*04:48', 'HLA-C*04:49', 'HLA-C*04:50', 'HLA-C*04:51',
         'HLA-C*04:52', 'HLA-C*04:53', 'HLA-C*04:54', 'HLA-C*04:55', 'HLA-C*04:56', 'HLA-C*04:57', 'HLA-C*04:58', 'HLA-C*04:60', 'HLA-C*04:61',
         'HLA-C*04:62', 'HLA-C*04:63', 'HLA-C*04:64', 'HLA-C*04:65', 'HLA-C*04:66', 'HLA-C*04:67', 'HLA-C*04:68', 'HLA-C*04:69', 'HLA-C*04:70',
         'HLA-C*05:01', 'HLA-C*05:03', 'HLA-C*05:04', 'HLA-C*05:05', 'HLA-C*05:06', 'HLA-C*05:08', 'HLA-C*05:09', 'HLA-C*05:10', 'HLA-C*05:11',
         'HLA-C*05:12', 'HLA-C*05:13', 'HLA-C*05:14', 'HLA-C*05:15', 'HLA-C*05:16', 'HLA-C*05:17', 'HLA-C*05:18', 'HLA-C*05:19', 'HLA-C*05:20',
         'HLA-C*05:21', 'HLA-C*05:22', 'HLA-C*05:23', 'HLA-C*05:24', 'HLA-C*05:25', 'HLA-C*05:26', 'HLA-C*05:27', 'HLA-C*05:28', 'HLA-C*05:29',
         'HLA-C*05:30', 'HLA-C*05:31', 'HLA-C*05:32', 'HLA-C*05:33', 'HLA-C*05:34', 'HLA-C*05:35', 'HLA-C*05:36', 'HLA-C*05:37', 'HLA-C*05:38',
         'HLA-C*05:39', 'HLA-C*05:40', 'HLA-C*05:41', 'HLA-C*05:42', 'HLA-C*05:43', 'HLA-C*05:44', 'HLA-C*05:45', 'HLA-C*06:02', 'HLA-C*06:03',
         'HLA-C*06:04', 'HLA-C*06:05', 'HLA-C*06:06', 'HLA-C*06:07', 'HLA-C*06:08', 'HLA-C*06:09', 'HLA-C*06:10', 'HLA-C*06:11', 'HLA-C*06:12',
         'HLA-C*06:13', 'HLA-C*06:14', 'HLA-C*06:15', 'HLA-C*06:17', 'HLA-C*06:18', 'HLA-C*06:19', 'HLA-C*06:20', 'HLA-C*06:21', 'HLA-C*06:22',
         'HLA-C*06:23', 'HLA-C*06:24', 'HLA-C*06:25', 'HLA-C*06:26', 'HLA-C*06:27', 'HLA-C*06:28', 'HLA-C*06:29', 'HLA-C*06:30', 'HLA-C*06:31',
         'HLA-C*06:32', 'HLA-C*06:33', 'HLA-C*06:34', 'HLA-C*06:35', 'HLA-C*06:36', 'HLA-C*06:37', 'HLA-C*06:38', 'HLA-C*06:39', 'HLA-C*06:40',
         'HLA-C*06:41', 'HLA-C*06:42', 'HLA-C*06:43', 'HLA-C*06:44', 'HLA-C*06:45', 'HLA-C*07:01', 'HLA-C*07:02', 'HLA-C*07:03', 'HLA-C*07:04',
         'HLA-C*07:05', 'HLA-C*07:06', 'HLA-C*07:07', 'HLA-C*07:08', 'HLA-C*07:09', 'HLA-C*07:10', 'HLA-C*07:100', 'HLA-C*07:101', 'HLA-C*07:102',
         'HLA-C*07:103', 'HLA-C*07:105', 'HLA-C*07:106', 'HLA-C*07:107', 'HLA-C*07:108', 'HLA-C*07:109', 'HLA-C*07:11', 'HLA-C*07:110',
         'HLA-C*07:111', 'HLA-C*07:112', 'HLA-C*07:113', 'HLA-C*07:114', 'HLA-C*07:115', 'HLA-C*07:116', 'HLA-C*07:117', 'HLA-C*07:118',
         'HLA-C*07:119', 'HLA-C*07:12', 'HLA-C*07:120', 'HLA-C*07:122', 'HLA-C*07:123', 'HLA-C*07:124', 'HLA-C*07:125', 'HLA-C*07:126',
         'HLA-C*07:127', 'HLA-C*07:128', 'HLA-C*07:129', 'HLA-C*07:13', 'HLA-C*07:130', 'HLA-C*07:131', 'HLA-C*07:132', 'HLA-C*07:133',
         'HLA-C*07:134', 'HLA-C*07:135', 'HLA-C*07:136', 'HLA-C*07:137', 'HLA-C*07:138', 'HLA-C*07:139', 'HLA-C*07:14', 'HLA-C*07:140',
         'HLA-C*07:141', 'HLA-C*07:142', 'HLA-C*07:143', 'HLA-C*07:144', 'HLA-C*07:145', 'HLA-C*07:146', 'HLA-C*07:147', 'HLA-C*07:148',
         'HLA-C*07:149', 'HLA-C*07:15', 'HLA-C*07:16', 'HLA-C*07:17', 'HLA-C*07:18', 'HLA-C*07:19', 'HLA-C*07:20', 'HLA-C*07:21', 'HLA-C*07:22',
         'HLA-C*07:23', 'HLA-C*07:24', 'HLA-C*07:25', 'HLA-C*07:26', 'HLA-C*07:27', 'HLA-C*07:28', 'HLA-C*07:29', 'HLA-C*07:30', 'HLA-C*07:31',
         'HLA-C*07:35', 'HLA-C*07:36', 'HLA-C*07:37', 'HLA-C*07:38', 'HLA-C*07:39', 'HLA-C*07:40', 'HLA-C*07:41', 'HLA-C*07:42', 'HLA-C*07:43',
         'HLA-C*07:44', 'HLA-C*07:45', 'HLA-C*07:46', 'HLA-C*07:47', 'HLA-C*07:48', 'HLA-C*07:49', 'HLA-C*07:50', 'HLA-C*07:51', 'HLA-C*07:52',
         'HLA-C*07:53', 'HLA-C*07:54', 'HLA-C*07:56', 'HLA-C*07:57', 'HLA-C*07:58', 'HLA-C*07:59', 'HLA-C*07:60', 'HLA-C*07:62', 'HLA-C*07:63',
         'HLA-C*07:64', 'HLA-C*07:65', 'HLA-C*07:66', 'HLA-C*07:67', 'HLA-C*07:68', 'HLA-C*07:69', 'HLA-C*07:70', 'HLA-C*07:71', 'HLA-C*07:72',
         'HLA-C*07:73', 'HLA-C*07:74', 'HLA-C*07:75', 'HLA-C*07:76', 'HLA-C*07:77', 'HLA-C*07:78', 'HLA-C*07:79', 'HLA-C*07:80', 'HLA-C*07:81',
         'HLA-C*07:82', 'HLA-C*07:83', 'HLA-C*07:84', 'HLA-C*07:85', 'HLA-C*07:86', 'HLA-C*07:87', 'HLA-C*07:88', 'HLA-C*07:89', 'HLA-C*07:90',
         'HLA-C*07:91', 'HLA-C*07:92', 'HLA-C*07:93', 'HLA-C*07:94', 'HLA-C*07:95', 'HLA-C*07:96', 'HLA-C*07:97', 'HLA-C*07:99', 'HLA-C*08:01',
         'HLA-C*08:02', 'HLA-C*08:03', 'HLA-C*08:04', 'HLA-C*08:05', 'HLA-C*08:06', 'HLA-C*08:07', 'HLA-C*08:08', 'HLA-C*08:09', 'HLA-C*08:10',
         'HLA-C*08:11', 'HLA-C*08:12', 'HLA-C*08:13', 'HLA-C*08:14', 'HLA-C*08:15', 'HLA-C*08:16', 'HLA-C*08:17', 'HLA-C*08:18', 'HLA-C*08:19',
         'HLA-C*08:20', 'HLA-C*08:21', 'HLA-C*08:22', 'HLA-C*08:23', 'HLA-C*08:24', 'HLA-C*08:25', 'HLA-C*08:27', 'HLA-C*08:28', 'HLA-C*08:29',
         'HLA-C*08:30', 'HLA-C*08:31', 'HLA-C*08:32', 'HLA-C*08:33', 'HLA-C*08:34', 'HLA-C*08:35', 'HLA-C*12:02', 'HLA-C*12:03', 'HLA-C*12:04',
         'HLA-C*12:05', 'HLA-C*12:06', 'HLA-C*12:07', 'HLA-C*12:08', 'HLA-C*12:09', 'HLA-C*12:10', 'HLA-C*12:11', 'HLA-C*12:12', 'HLA-C*12:13',
         'HLA-C*12:14', 'HLA-C*12:15', 'HLA-C*12:16', 'HLA-C*12:17', 'HLA-C*12:18', 'HLA-C*12:19', 'HLA-C*12:20', 'HLA-C*12:21', 'HLA-C*12:22',
         'HLA-C*12:23', 'HLA-C*12:24', 'HLA-C*12:25', 'HLA-C*12:26', 'HLA-C*12:27', 'HLA-C*12:28', 'HLA-C*12:29', 'HLA-C*12:30', 'HLA-C*12:31',
         'HLA-C*12:32', 'HLA-C*12:33', 'HLA-C*12:34', 'HLA-C*12:35', 'HLA-C*12:36', 'HLA-C*12:37', 'HLA-C*12:38', 'HLA-C*12:40', 'HLA-C*12:41',
         'HLA-C*12:43', 'HLA-C*12:44', 'HLA-C*14:02', 'HLA-C*14:03', 'HLA-C*14:04', 'HLA-C*14:05', 'HLA-C*14:06', 'HLA-C*14:08', 'HLA-C*14:09',
         'HLA-C*14:10', 'HLA-C*14:11', 'HLA-C*14:12', 'HLA-C*14:13', 'HLA-C*14:14', 'HLA-C*14:15', 'HLA-C*14:16', 'HLA-C*14:17', 'HLA-C*14:18',
         'HLA-C*14:19', 'HLA-C*14:20', 'HLA-C*15:02', 'HLA-C*15:03', 'HLA-C*15:04', 'HLA-C*15:05', 'HLA-C*15:06', 'HLA-C*15:07', 'HLA-C*15:08',
         'HLA-C*15:09', 'HLA-C*15:10', 'HLA-C*15:11', 'HLA-C*15:12', 'HLA-C*15:13', 'HLA-C*15:15', 'HLA-C*15:16', 'HLA-C*15:17', 'HLA-C*15:18',
         'HLA-C*15:19', 'HLA-C*15:20', 'HLA-C*15:21', 'HLA-C*15:22', 'HLA-C*15:23', 'HLA-C*15:24', 'HLA-C*15:25', 'HLA-C*15:26', 'HLA-C*15:27',
         'HLA-C*15:28', 'HLA-C*15:29', 'HLA-C*15:30', 'HLA-C*15:31', 'HLA-C*15:33', 'HLA-C*15:34', 'HLA-C*15:35', 'HLA-C*16:01', 'HLA-C*16:02',
         'HLA-C*16:04', 'HLA-C*16:06', 'HLA-C*16:07', 'HLA-C*16:08', 'HLA-C*16:09', 'HLA-C*16:10', 'HLA-C*16:11', 'HLA-C*16:12', 'HLA-C*16:13',
         'HLA-C*16:14', 'HLA-C*16:15', 'HLA-C*16:17', 'HLA-C*16:18', 'HLA-C*16:19', 'HLA-C*16:20', 'HLA-C*16:21', 'HLA-C*16:22', 'HLA-C*16:23',
         'HLA-C*16:24', 'HLA-C*16:25', 'HLA-C*16:26', 'HLA-C*17:01', 'HLA-C*17:02', 'HLA-C*17:03', 'HLA-C*17:04', 'HLA-C*17:05', 'HLA-C*17:06',
         'HLA-C*17:07', 'HLA-C*18:01', 'HLA-C*18:02', 'HLA-C*18:03', 'HLA-E*01:01', 'HLA-G*01:01', 'HLA-G*01:02', 'HLA-G*01:03', 'HLA-G*01:04',
         'HLA-G*01:06', 'HLA-G*01:07', 'HLA-G*01:08', 'HLA-G*01:09',
         'H-2-Db', 'H-2-Dd', 'H-2-Kb', 'H-2-Kd', 'H-2-Kk', 'H-2-Ld'])
    __version = "2.4"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def supportedAlleles(self):
        """
        A list of valid :class:`~epytope.Core.Allele.Allele` models
        """
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        alleles = [x for x in next(f) if "HLA" in x]
        # Rank is not supported in command line tool of NetMHCpan 2.4
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_2_4]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_2_4 + i])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Allele2':...}
        result = {allele: {"Score":(list(scores.values())[j])} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        # can not be determined netmhcpan does not support --version or similar
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        NO return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetMHCpan_2_8(AExternalEpitopePrediction):
    """
    Implements the NetMHC binding (in current form for netMHCpan 2.8).
    Supported  MHC alleles currently only restricted to HLA alleles.

    .. note::

        Nielsen, Morten, et al. "NetMHCpan, a method for quantitative predictions of peptide binding to any HLA-A and-B
        locus protein of known sequence." PloS one 2.8 (2007): e796.
    """
    __version = "2.8"
    __supported_length = frozenset([8, 9, 10, 11, 12, 13, 14])
    __name = "netmhcpan"
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -ic50 -xls -xlsfile {out}"
    __alleles = frozenset(
        ['HLA-A*01:01', 'HLA-A*01:02', 'HLA-A*01:03', 'HLA-A*01:06', 'HLA-A*01:07', 'HLA-A*01:08', 'HLA-A*01:09', 'HLA-A*01:10', 'HLA-A*01:12',
         'HLA-A*01:13', 'HLA-A*01:14', 'HLA-A*01:17', 'HLA-A*01:19', 'HLA-A*01:20', 'HLA-A*01:21', 'HLA-A*01:23', 'HLA-A*01:24', 'HLA-A*01:25',
         'HLA-A*01:26', 'HLA-A*01:28', 'HLA-A*01:29', 'HLA-A*01:30', 'HLA-A*01:32', 'HLA-A*01:33', 'HLA-A*01:35', 'HLA-A*01:36', 'HLA-A*01:37',
         'HLA-A*01:38', 'HLA-A*01:39', 'HLA-A*01:40', 'HLA-A*01:41', 'HLA-A*01:42', 'HLA-A*01:43', 'HLA-A*01:44', 'HLA-A*01:45', 'HLA-A*01:46',
         'HLA-A*01:47', 'HLA-A*01:48', 'HLA-A*01:49', 'HLA-A*01:50', 'HLA-A*01:51', 'HLA-A*01:54', 'HLA-A*01:55', 'HLA-A*01:58', 'HLA-A*01:59',
         'HLA-A*01:60', 'HLA-A*01:61', 'HLA-A*01:62', 'HLA-A*01:63', 'HLA-A*01:64', 'HLA-A*01:65', 'HLA-A*01:66', 'HLA-A*02:01', 'HLA-A*02:02',
         'HLA-A*02:03', 'HLA-A*02:04', 'HLA-A*02:05', 'HLA-A*02:06', 'HLA-A*02:07', 'HLA-A*02:08', 'HLA-A*02:09', 'HLA-A*02:10', 'HLA-A*02:101',
         'HLA-A*02:102', 'HLA-A*02:103', 'HLA-A*02:104', 'HLA-A*02:105', 'HLA-A*02:106', 'HLA-A*02:107', 'HLA-A*02:108', 'HLA-A*02:109',
         'HLA-A*02:11', 'HLA-A*02:110', 'HLA-A*02:111', 'HLA-A*02:112', 'HLA-A*02:114', 'HLA-A*02:115', 'HLA-A*02:116', 'HLA-A*02:117',
         'HLA-A*02:118', 'HLA-A*02:119', 'HLA-A*02:12', 'HLA-A*02:120', 'HLA-A*02:121', 'HLA-A*02:122', 'HLA-A*02:123', 'HLA-A*02:124',
         'HLA-A*02:126', 'HLA-A*02:127', 'HLA-A*02:128', 'HLA-A*02:129', 'HLA-A*02:13', 'HLA-A*02:130', 'HLA-A*02:131', 'HLA-A*02:132',
         'HLA-A*02:133', 'HLA-A*02:134', 'HLA-A*02:135', 'HLA-A*02:136', 'HLA-A*02:137', 'HLA-A*02:138', 'HLA-A*02:139', 'HLA-A*02:14',
         'HLA-A*02:140', 'HLA-A*02:141', 'HLA-A*02:142', 'HLA-A*02:143', 'HLA-A*02:144', 'HLA-A*02:145', 'HLA-A*02:146', 'HLA-A*02:147',
         'HLA-A*02:148', 'HLA-A*02:149', 'HLA-A*02:150', 'HLA-A*02:151', 'HLA-A*02:152', 'HLA-A*02:153', 'HLA-A*02:154', 'HLA-A*02:155',
         'HLA-A*02:156', 'HLA-A*02:157', 'HLA-A*02:158', 'HLA-A*02:159', 'HLA-A*02:16', 'HLA-A*02:160', 'HLA-A*02:161', 'HLA-A*02:162',
         'HLA-A*02:163', 'HLA-A*02:164', 'HLA-A*02:165', 'HLA-A*02:166', 'HLA-A*02:167', 'HLA-A*02:168', 'HLA-A*02:169', 'HLA-A*02:17',
         'HLA-A*02:170', 'HLA-A*02:171', 'HLA-A*02:172', 'HLA-A*02:173', 'HLA-A*02:174', 'HLA-A*02:175', 'HLA-A*02:176', 'HLA-A*02:177',
         'HLA-A*02:178', 'HLA-A*02:179', 'HLA-A*02:18', 'HLA-A*02:180', 'HLA-A*02:181', 'HLA-A*02:182', 'HLA-A*02:183', 'HLA-A*02:184',
         'HLA-A*02:185', 'HLA-A*02:186', 'HLA-A*02:187', 'HLA-A*02:188', 'HLA-A*02:189', 'HLA-A*02:19', 'HLA-A*02:190', 'HLA-A*02:191',
         'HLA-A*02:192', 'HLA-A*02:193', 'HLA-A*02:194', 'HLA-A*02:195', 'HLA-A*02:196', 'HLA-A*02:197', 'HLA-A*02:198', 'HLA-A*02:199',
         'HLA-A*02:20', 'HLA-A*02:200', 'HLA-A*02:201', 'HLA-A*02:202', 'HLA-A*02:203', 'HLA-A*02:204', 'HLA-A*02:205', 'HLA-A*02:206',
         'HLA-A*02:207', 'HLA-A*02:208', 'HLA-A*02:209', 'HLA-A*02:21', 'HLA-A*02:210', 'HLA-A*02:211', 'HLA-A*02:212', 'HLA-A*02:213',
         'HLA-A*02:214', 'HLA-A*02:215', 'HLA-A*02:216', 'HLA-A*02:217', 'HLA-A*02:218', 'HLA-A*02:219', 'HLA-A*02:22', 'HLA-A*02:220',
         'HLA-A*02:221', 'HLA-A*02:224', 'HLA-A*02:228', 'HLA-A*02:229', 'HLA-A*02:230', 'HLA-A*02:231', 'HLA-A*02:232', 'HLA-A*02:233',
         'HLA-A*02:234', 'HLA-A*02:235', 'HLA-A*02:236', 'HLA-A*02:237', 'HLA-A*02:238', 'HLA-A*02:239', 'HLA-A*02:24', 'HLA-A*02:240',
         'HLA-A*02:241', 'HLA-A*02:242', 'HLA-A*02:243', 'HLA-A*02:244', 'HLA-A*02:245', 'HLA-A*02:246', 'HLA-A*02:247', 'HLA-A*02:248',
         'HLA-A*02:249', 'HLA-A*02:25', 'HLA-A*02:251', 'HLA-A*02:252', 'HLA-A*02:253', 'HLA-A*02:254', 'HLA-A*02:255', 'HLA-A*02:256',
         'HLA-A*02:257', 'HLA-A*02:258', 'HLA-A*02:259', 'HLA-A*02:26', 'HLA-A*02:260', 'HLA-A*02:261', 'HLA-A*02:262', 'HLA-A*02:263',
         'HLA-A*02:264', 'HLA-A*02:265', 'HLA-A*02:266', 'HLA-A*02:27', 'HLA-A*02:28', 'HLA-A*02:29', 'HLA-A*02:30', 'HLA-A*02:31', 'HLA-A*02:33',
         'HLA-A*02:34', 'HLA-A*02:35', 'HLA-A*02:36', 'HLA-A*02:37', 'HLA-A*02:38', 'HLA-A*02:39', 'HLA-A*02:40', 'HLA-A*02:41', 'HLA-A*02:42',
         'HLA-A*02:44', 'HLA-A*02:45', 'HLA-A*02:46', 'HLA-A*02:47', 'HLA-A*02:48', 'HLA-A*02:49', 'HLA-A*02:50', 'HLA-A*02:51', 'HLA-A*02:52',
         'HLA-A*02:54', 'HLA-A*02:55', 'HLA-A*02:56', 'HLA-A*02:57', 'HLA-A*02:58', 'HLA-A*02:59', 'HLA-A*02:60', 'HLA-A*02:61', 'HLA-A*02:62',
         'HLA-A*02:63', 'HLA-A*02:64', 'HLA-A*02:65', 'HLA-A*02:66', 'HLA-A*02:67', 'HLA-A*02:68', 'HLA-A*02:69', 'HLA-A*02:70', 'HLA-A*02:71',
         'HLA-A*02:72', 'HLA-A*02:73', 'HLA-A*02:74', 'HLA-A*02:75', 'HLA-A*02:76', 'HLA-A*02:77', 'HLA-A*02:78', 'HLA-A*02:79', 'HLA-A*02:80',
         'HLA-A*02:81', 'HLA-A*02:84', 'HLA-A*02:85', 'HLA-A*02:86', 'HLA-A*02:87', 'HLA-A*02:89', 'HLA-A*02:90', 'HLA-A*02:91', 'HLA-A*02:92',
         'HLA-A*02:93', 'HLA-A*02:95', 'HLA-A*02:96', 'HLA-A*02:97', 'HLA-A*02:99', 'HLA-A*03:01', 'HLA-A*03:02', 'HLA-A*03:04', 'HLA-A*03:05',
         'HLA-A*03:06', 'HLA-A*03:07', 'HLA-A*03:08', 'HLA-A*03:09', 'HLA-A*03:10', 'HLA-A*03:12', 'HLA-A*03:13', 'HLA-A*03:14', 'HLA-A*03:15',
         'HLA-A*03:16', 'HLA-A*03:17', 'HLA-A*03:18', 'HLA-A*03:19', 'HLA-A*03:20', 'HLA-A*03:22', 'HLA-A*03:23', 'HLA-A*03:24', 'HLA-A*03:25',
         'HLA-A*03:26', 'HLA-A*03:27', 'HLA-A*03:28', 'HLA-A*03:29', 'HLA-A*03:30', 'HLA-A*03:31', 'HLA-A*03:32', 'HLA-A*03:33', 'HLA-A*03:34',
         'HLA-A*03:35', 'HLA-A*03:37', 'HLA-A*03:38', 'HLA-A*03:39', 'HLA-A*03:40', 'HLA-A*03:41', 'HLA-A*03:42', 'HLA-A*03:43', 'HLA-A*03:44',
         'HLA-A*03:45', 'HLA-A*03:46', 'HLA-A*03:47', 'HLA-A*03:48', 'HLA-A*03:49', 'HLA-A*03:50', 'HLA-A*03:51', 'HLA-A*03:52', 'HLA-A*03:53',
         'HLA-A*03:54', 'HLA-A*03:55', 'HLA-A*03:56', 'HLA-A*03:57', 'HLA-A*03:58', 'HLA-A*03:59', 'HLA-A*03:60', 'HLA-A*03:61', 'HLA-A*03:62',
         'HLA-A*03:63', 'HLA-A*03:64', 'HLA-A*03:65', 'HLA-A*03:66', 'HLA-A*03:67', 'HLA-A*03:70', 'HLA-A*03:71', 'HLA-A*03:72', 'HLA-A*03:73',
         'HLA-A*03:74', 'HLA-A*03:75', 'HLA-A*03:76', 'HLA-A*03:77', 'HLA-A*03:78', 'HLA-A*03:79', 'HLA-A*03:80', 'HLA-A*03:81', 'HLA-A*03:82',
         'HLA-A*11:01', 'HLA-A*11:02', 'HLA-A*11:03', 'HLA-A*11:04', 'HLA-A*11:05', 'HLA-A*11:06', 'HLA-A*11:07', 'HLA-A*11:08', 'HLA-A*11:09',
         'HLA-A*11:10', 'HLA-A*11:11', 'HLA-A*11:12', 'HLA-A*11:13', 'HLA-A*11:14', 'HLA-A*11:15', 'HLA-A*11:16', 'HLA-A*11:17', 'HLA-A*11:18',
         'HLA-A*11:19', 'HLA-A*11:20', 'HLA-A*11:22', 'HLA-A*11:23', 'HLA-A*11:24', 'HLA-A*11:25', 'HLA-A*11:26', 'HLA-A*11:27', 'HLA-A*11:29',
         'HLA-A*11:30', 'HLA-A*11:31', 'HLA-A*11:32', 'HLA-A*11:33', 'HLA-A*11:34', 'HLA-A*11:35', 'HLA-A*11:36', 'HLA-A*11:37', 'HLA-A*11:38',
         'HLA-A*11:39', 'HLA-A*11:40', 'HLA-A*11:41', 'HLA-A*11:42', 'HLA-A*11:43', 'HLA-A*11:44', 'HLA-A*11:45', 'HLA-A*11:46', 'HLA-A*11:47',
         'HLA-A*11:48', 'HLA-A*11:49', 'HLA-A*11:51', 'HLA-A*11:53', 'HLA-A*11:54', 'HLA-A*11:55', 'HLA-A*11:56', 'HLA-A*11:57', 'HLA-A*11:58',
         'HLA-A*11:59', 'HLA-A*11:60', 'HLA-A*11:61', 'HLA-A*11:62', 'HLA-A*11:63', 'HLA-A*11:64', 'HLA-A*23:01', 'HLA-A*23:02', 'HLA-A*23:03',
         'HLA-A*23:04', 'HLA-A*23:05', 'HLA-A*23:06', 'HLA-A*23:09', 'HLA-A*23:10', 'HLA-A*23:12', 'HLA-A*23:13', 'HLA-A*23:14', 'HLA-A*23:15',
         'HLA-A*23:16', 'HLA-A*23:17', 'HLA-A*23:18', 'HLA-A*23:20', 'HLA-A*23:21', 'HLA-A*23:22', 'HLA-A*23:23', 'HLA-A*23:24', 'HLA-A*23:25',
         'HLA-A*23:26', 'HLA-A*24:02', 'HLA-A*24:03', 'HLA-A*24:04', 'HLA-A*24:05', 'HLA-A*24:06', 'HLA-A*24:07', 'HLA-A*24:08', 'HLA-A*24:10',
         'HLA-A*24:100', 'HLA-A*24:101', 'HLA-A*24:102', 'HLA-A*24:103', 'HLA-A*24:104', 'HLA-A*24:105', 'HLA-A*24:106', 'HLA-A*24:107',
         'HLA-A*24:108', 'HLA-A*24:109', 'HLA-A*24:110', 'HLA-A*24:111', 'HLA-A*24:112', 'HLA-A*24:113', 'HLA-A*24:114', 'HLA-A*24:115',
         'HLA-A*24:116', 'HLA-A*24:117', 'HLA-A*24:118', 'HLA-A*24:119', 'HLA-A*24:120', 'HLA-A*24:121', 'HLA-A*24:122', 'HLA-A*24:123',
         'HLA-A*24:124', 'HLA-A*24:125', 'HLA-A*24:126', 'HLA-A*24:127', 'HLA-A*24:128', 'HLA-A*24:129', 'HLA-A*24:13', 'HLA-A*24:130',
         'HLA-A*24:131', 'HLA-A*24:133', 'HLA-A*24:134', 'HLA-A*24:135', 'HLA-A*24:136', 'HLA-A*24:137', 'HLA-A*24:138', 'HLA-A*24:139',
         'HLA-A*24:14', 'HLA-A*24:140', 'HLA-A*24:141', 'HLA-A*24:142', 'HLA-A*24:143', 'HLA-A*24:144', 'HLA-A*24:15', 'HLA-A*24:17', 'HLA-A*24:18',
         'HLA-A*24:19', 'HLA-A*24:20', 'HLA-A*24:21', 'HLA-A*24:22', 'HLA-A*24:23', 'HLA-A*24:24', 'HLA-A*24:25', 'HLA-A*24:26', 'HLA-A*24:27',
         'HLA-A*24:28', 'HLA-A*24:29', 'HLA-A*24:30', 'HLA-A*24:31', 'HLA-A*24:32', 'HLA-A*24:33', 'HLA-A*24:34', 'HLA-A*24:35', 'HLA-A*24:37',
         'HLA-A*24:38', 'HLA-A*24:39', 'HLA-A*24:41', 'HLA-A*24:42', 'HLA-A*24:43', 'HLA-A*24:44', 'HLA-A*24:46', 'HLA-A*24:47', 'HLA-A*24:49',
         'HLA-A*24:50', 'HLA-A*24:51', 'HLA-A*24:52', 'HLA-A*24:53', 'HLA-A*24:54', 'HLA-A*24:55', 'HLA-A*24:56', 'HLA-A*24:57', 'HLA-A*24:58',
         'HLA-A*24:59', 'HLA-A*24:61', 'HLA-A*24:62', 'HLA-A*24:63', 'HLA-A*24:64', 'HLA-A*24:66', 'HLA-A*24:67', 'HLA-A*24:68', 'HLA-A*24:69',
         'HLA-A*24:70', 'HLA-A*24:71', 'HLA-A*24:72', 'HLA-A*24:73', 'HLA-A*24:74', 'HLA-A*24:75', 'HLA-A*24:76', 'HLA-A*24:77', 'HLA-A*24:78',
         'HLA-A*24:79', 'HLA-A*24:80', 'HLA-A*24:81', 'HLA-A*24:82', 'HLA-A*24:85', 'HLA-A*24:87', 'HLA-A*24:88', 'HLA-A*24:89', 'HLA-A*24:91',
         'HLA-A*24:92', 'HLA-A*24:93', 'HLA-A*24:94', 'HLA-A*24:95', 'HLA-A*24:96', 'HLA-A*24:97', 'HLA-A*24:98', 'HLA-A*24:99', 'HLA-A*25:01',
         'HLA-A*25:02', 'HLA-A*25:03', 'HLA-A*25:04', 'HLA-A*25:05', 'HLA-A*25:06', 'HLA-A*25:07', 'HLA-A*25:08', 'HLA-A*25:09', 'HLA-A*25:10',
         'HLA-A*25:11', 'HLA-A*25:13', 'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*26:04', 'HLA-A*26:05', 'HLA-A*26:06', 'HLA-A*26:07',
         'HLA-A*26:08', 'HLA-A*26:09', 'HLA-A*26:10', 'HLA-A*26:12', 'HLA-A*26:13', 'HLA-A*26:14', 'HLA-A*26:15', 'HLA-A*26:16', 'HLA-A*26:17',
         'HLA-A*26:18', 'HLA-A*26:19', 'HLA-A*26:20', 'HLA-A*26:21', 'HLA-A*26:22', 'HLA-A*26:23', 'HLA-A*26:24', 'HLA-A*26:26', 'HLA-A*26:27',
         'HLA-A*26:28', 'HLA-A*26:29', 'HLA-A*26:30', 'HLA-A*26:31', 'HLA-A*26:32', 'HLA-A*26:33', 'HLA-A*26:34', 'HLA-A*26:35', 'HLA-A*26:36',
         'HLA-A*26:37', 'HLA-A*26:38', 'HLA-A*26:39', 'HLA-A*26:40', 'HLA-A*26:41', 'HLA-A*26:42', 'HLA-A*26:43', 'HLA-A*26:45', 'HLA-A*26:46',
         'HLA-A*26:47', 'HLA-A*26:48', 'HLA-A*26:49', 'HLA-A*26:50', 'HLA-A*29:01', 'HLA-A*29:02', 'HLA-A*29:03', 'HLA-A*29:04', 'HLA-A*29:05',
         'HLA-A*29:06', 'HLA-A*29:07', 'HLA-A*29:09', 'HLA-A*29:10', 'HLA-A*29:11', 'HLA-A*29:12', 'HLA-A*29:13', 'HLA-A*29:14', 'HLA-A*29:15',
         'HLA-A*29:16', 'HLA-A*29:17', 'HLA-A*29:18', 'HLA-A*29:19', 'HLA-A*29:20', 'HLA-A*29:21', 'HLA-A*29:22', 'HLA-A*30:01', 'HLA-A*30:02',
         'HLA-A*30:03', 'HLA-A*30:04', 'HLA-A*30:06', 'HLA-A*30:07', 'HLA-A*30:08', 'HLA-A*30:09', 'HLA-A*30:10', 'HLA-A*30:11', 'HLA-A*30:12',
         'HLA-A*30:13', 'HLA-A*30:15', 'HLA-A*30:16', 'HLA-A*30:17', 'HLA-A*30:18', 'HLA-A*30:19', 'HLA-A*30:20', 'HLA-A*30:22', 'HLA-A*30:23',
         'HLA-A*30:24', 'HLA-A*30:25', 'HLA-A*30:26', 'HLA-A*30:28', 'HLA-A*30:29', 'HLA-A*30:30', 'HLA-A*30:31', 'HLA-A*30:32', 'HLA-A*30:33',
         'HLA-A*30:34', 'HLA-A*30:35', 'HLA-A*30:36', 'HLA-A*30:37', 'HLA-A*30:38', 'HLA-A*30:39', 'HLA-A*30:40', 'HLA-A*30:41', 'HLA-A*31:01',
         'HLA-A*31:02', 'HLA-A*31:03', 'HLA-A*31:04', 'HLA-A*31:05', 'HLA-A*31:06', 'HLA-A*31:07', 'HLA-A*31:08', 'HLA-A*31:09', 'HLA-A*31:10',
         'HLA-A*31:11', 'HLA-A*31:12', 'HLA-A*31:13', 'HLA-A*31:15', 'HLA-A*31:16', 'HLA-A*31:17', 'HLA-A*31:18', 'HLA-A*31:19', 'HLA-A*31:20',
         'HLA-A*31:21', 'HLA-A*31:22', 'HLA-A*31:23', 'HLA-A*31:24', 'HLA-A*31:25', 'HLA-A*31:26', 'HLA-A*31:27', 'HLA-A*31:28', 'HLA-A*31:29',
         'HLA-A*31:30', 'HLA-A*31:31', 'HLA-A*31:32', 'HLA-A*31:33', 'HLA-A*31:34', 'HLA-A*31:35', 'HLA-A*31:36', 'HLA-A*31:37', 'HLA-A*32:01',
         'HLA-A*32:02', 'HLA-A*32:03', 'HLA-A*32:04', 'HLA-A*32:05', 'HLA-A*32:06', 'HLA-A*32:07', 'HLA-A*32:08', 'HLA-A*32:09', 'HLA-A*32:10',
         'HLA-A*32:12', 'HLA-A*32:13', 'HLA-A*32:14', 'HLA-A*32:15', 'HLA-A*32:16', 'HLA-A*32:17', 'HLA-A*32:18', 'HLA-A*32:20', 'HLA-A*32:21',
         'HLA-A*32:22', 'HLA-A*32:23', 'HLA-A*32:24', 'HLA-A*32:25', 'HLA-A*33:01', 'HLA-A*33:03', 'HLA-A*33:04', 'HLA-A*33:05', 'HLA-A*33:06',
         'HLA-A*33:07', 'HLA-A*33:08', 'HLA-A*33:09', 'HLA-A*33:10', 'HLA-A*33:11', 'HLA-A*33:12', 'HLA-A*33:13', 'HLA-A*33:14', 'HLA-A*33:15',
         'HLA-A*33:16', 'HLA-A*33:17', 'HLA-A*33:18', 'HLA-A*33:19', 'HLA-A*33:20', 'HLA-A*33:21', 'HLA-A*33:22', 'HLA-A*33:23', 'HLA-A*33:24',
         'HLA-A*33:25', 'HLA-A*33:26', 'HLA-A*33:27', 'HLA-A*33:28', 'HLA-A*33:29', 'HLA-A*33:30', 'HLA-A*33:31', 'HLA-A*34:01', 'HLA-A*34:02',
         'HLA-A*34:03', 'HLA-A*34:04', 'HLA-A*34:05', 'HLA-A*34:06', 'HLA-A*34:07', 'HLA-A*34:08', 'HLA-A*36:01', 'HLA-A*36:02', 'HLA-A*36:03',
         'HLA-A*36:04', 'HLA-A*36:05', 'HLA-A*43:01', 'HLA-A*66:01', 'HLA-A*66:02', 'HLA-A*66:03', 'HLA-A*66:04', 'HLA-A*66:05', 'HLA-A*66:06',
         'HLA-A*66:07', 'HLA-A*66:08', 'HLA-A*66:09', 'HLA-A*66:10', 'HLA-A*66:11', 'HLA-A*66:12', 'HLA-A*66:13', 'HLA-A*66:14', 'HLA-A*66:15',
         'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*68:03', 'HLA-A*68:04', 'HLA-A*68:05', 'HLA-A*68:06', 'HLA-A*68:07', 'HLA-A*68:08', 'HLA-A*68:09',
         'HLA-A*68:10', 'HLA-A*68:12', 'HLA-A*68:13', 'HLA-A*68:14', 'HLA-A*68:15', 'HLA-A*68:16', 'HLA-A*68:17', 'HLA-A*68:19', 'HLA-A*68:20',
         'HLA-A*68:21', 'HLA-A*68:22', 'HLA-A*68:23', 'HLA-A*68:24', 'HLA-A*68:25', 'HLA-A*68:26', 'HLA-A*68:27', 'HLA-A*68:28', 'HLA-A*68:29',
         'HLA-A*68:30', 'HLA-A*68:31', 'HLA-A*68:32', 'HLA-A*68:33', 'HLA-A*68:34', 'HLA-A*68:35', 'HLA-A*68:36', 'HLA-A*68:37', 'HLA-A*68:38',
         'HLA-A*68:39', 'HLA-A*68:40', 'HLA-A*68:41', 'HLA-A*68:42', 'HLA-A*68:43', 'HLA-A*68:44', 'HLA-A*68:45', 'HLA-A*68:46', 'HLA-A*68:47',
         'HLA-A*68:48', 'HLA-A*68:50', 'HLA-A*68:51', 'HLA-A*68:52', 'HLA-A*68:53', 'HLA-A*68:54', 'HLA-A*69:01', 'HLA-A*74:01', 'HLA-A*74:02',
         'HLA-A*74:03', 'HLA-A*74:04', 'HLA-A*74:05', 'HLA-A*74:06', 'HLA-A*74:07', 'HLA-A*74:08', 'HLA-A*74:09', 'HLA-A*74:10', 'HLA-A*74:11',
         'HLA-A*74:13', 'HLA-A*80:01', 'HLA-A*80:02', 'HLA-B*07:02', 'HLA-B*07:03', 'HLA-B*07:04', 'HLA-B*07:05', 'HLA-B*07:06', 'HLA-B*07:07',
         'HLA-B*07:08', 'HLA-B*07:09', 'HLA-B*07:10', 'HLA-B*07:100', 'HLA-B*07:101', 'HLA-B*07:102', 'HLA-B*07:103', 'HLA-B*07:104',
         'HLA-B*07:105', 'HLA-B*07:106', 'HLA-B*07:107', 'HLA-B*07:108', 'HLA-B*07:109', 'HLA-B*07:11', 'HLA-B*07:110', 'HLA-B*07:112',
         'HLA-B*07:113', 'HLA-B*07:114', 'HLA-B*07:115', 'HLA-B*07:12', 'HLA-B*07:13', 'HLA-B*07:14', 'HLA-B*07:15', 'HLA-B*07:16', 'HLA-B*07:17',
         'HLA-B*07:18', 'HLA-B*07:19', 'HLA-B*07:20', 'HLA-B*07:21', 'HLA-B*07:22', 'HLA-B*07:23', 'HLA-B*07:24', 'HLA-B*07:25', 'HLA-B*07:26',
         'HLA-B*07:27', 'HLA-B*07:28', 'HLA-B*07:29', 'HLA-B*07:30', 'HLA-B*07:31', 'HLA-B*07:32', 'HLA-B*07:33', 'HLA-B*07:34', 'HLA-B*07:35',
         'HLA-B*07:36', 'HLA-B*07:37', 'HLA-B*07:38', 'HLA-B*07:39', 'HLA-B*07:40', 'HLA-B*07:41', 'HLA-B*07:42', 'HLA-B*07:43', 'HLA-B*07:44',
         'HLA-B*07:45', 'HLA-B*07:46', 'HLA-B*07:47', 'HLA-B*07:48', 'HLA-B*07:50', 'HLA-B*07:51', 'HLA-B*07:52', 'HLA-B*07:53', 'HLA-B*07:54',
         'HLA-B*07:55', 'HLA-B*07:56', 'HLA-B*07:57', 'HLA-B*07:58', 'HLA-B*07:59', 'HLA-B*07:60', 'HLA-B*07:61', 'HLA-B*07:62', 'HLA-B*07:63',
         'HLA-B*07:64', 'HLA-B*07:65', 'HLA-B*07:66', 'HLA-B*07:68', 'HLA-B*07:69', 'HLA-B*07:70', 'HLA-B*07:71', 'HLA-B*07:72', 'HLA-B*07:73',
         'HLA-B*07:74', 'HLA-B*07:75', 'HLA-B*07:76', 'HLA-B*07:77', 'HLA-B*07:78', 'HLA-B*07:79', 'HLA-B*07:80', 'HLA-B*07:81', 'HLA-B*07:82',
         'HLA-B*07:83', 'HLA-B*07:84', 'HLA-B*07:85', 'HLA-B*07:86', 'HLA-B*07:87', 'HLA-B*07:88', 'HLA-B*07:89', 'HLA-B*07:90', 'HLA-B*07:91',
         'HLA-B*07:92', 'HLA-B*07:93', 'HLA-B*07:94', 'HLA-B*07:95', 'HLA-B*07:96', 'HLA-B*07:97', 'HLA-B*07:98', 'HLA-B*07:99', 'HLA-B*08:01',
         'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*08:04', 'HLA-B*08:05', 'HLA-B*08:07', 'HLA-B*08:09', 'HLA-B*08:10', 'HLA-B*08:11', 'HLA-B*08:12',
         'HLA-B*08:13', 'HLA-B*08:14', 'HLA-B*08:15', 'HLA-B*08:16', 'HLA-B*08:17', 'HLA-B*08:18', 'HLA-B*08:20', 'HLA-B*08:21', 'HLA-B*08:22',
         'HLA-B*08:23', 'HLA-B*08:24', 'HLA-B*08:25', 'HLA-B*08:26', 'HLA-B*08:27', 'HLA-B*08:28', 'HLA-B*08:29', 'HLA-B*08:31', 'HLA-B*08:32',
         'HLA-B*08:33', 'HLA-B*08:34', 'HLA-B*08:35', 'HLA-B*08:36', 'HLA-B*08:37', 'HLA-B*08:38', 'HLA-B*08:39', 'HLA-B*08:40', 'HLA-B*08:41',
         'HLA-B*08:42', 'HLA-B*08:43', 'HLA-B*08:44', 'HLA-B*08:45', 'HLA-B*08:46', 'HLA-B*08:47', 'HLA-B*08:48', 'HLA-B*08:49', 'HLA-B*08:50',
         'HLA-B*08:51', 'HLA-B*08:52', 'HLA-B*08:53', 'HLA-B*08:54', 'HLA-B*08:55', 'HLA-B*08:56', 'HLA-B*08:57', 'HLA-B*08:58', 'HLA-B*08:59',
         'HLA-B*08:60', 'HLA-B*08:61', 'HLA-B*08:62', 'HLA-B*13:01', 'HLA-B*13:02', 'HLA-B*13:03', 'HLA-B*13:04', 'HLA-B*13:06', 'HLA-B*13:09',
         'HLA-B*13:10', 'HLA-B*13:11', 'HLA-B*13:12', 'HLA-B*13:13', 'HLA-B*13:14', 'HLA-B*13:15', 'HLA-B*13:16', 'HLA-B*13:17', 'HLA-B*13:18',
         'HLA-B*13:19', 'HLA-B*13:20', 'HLA-B*13:21', 'HLA-B*13:22', 'HLA-B*13:23', 'HLA-B*13:25', 'HLA-B*13:26', 'HLA-B*13:27', 'HLA-B*13:28',
         'HLA-B*13:29', 'HLA-B*13:30', 'HLA-B*13:31', 'HLA-B*13:32', 'HLA-B*13:33', 'HLA-B*13:34', 'HLA-B*13:35', 'HLA-B*13:36', 'HLA-B*13:37',
         'HLA-B*13:38', 'HLA-B*13:39', 'HLA-B*14:01', 'HLA-B*14:02', 'HLA-B*14:03', 'HLA-B*14:04', 'HLA-B*14:05', 'HLA-B*14:06', 'HLA-B*14:08',
         'HLA-B*14:09', 'HLA-B*14:10', 'HLA-B*14:11', 'HLA-B*14:12', 'HLA-B*14:13', 'HLA-B*14:14', 'HLA-B*14:15', 'HLA-B*14:16', 'HLA-B*14:17',
         'HLA-B*14:18', 'HLA-B*15:01', 'HLA-B*15:02', 'HLA-B*15:03', 'HLA-B*15:04', 'HLA-B*15:05', 'HLA-B*15:06', 'HLA-B*15:07', 'HLA-B*15:08',
         'HLA-B*15:09', 'HLA-B*15:10', 'HLA-B*15:101', 'HLA-B*15:102', 'HLA-B*15:103', 'HLA-B*15:104', 'HLA-B*15:105', 'HLA-B*15:106',
         'HLA-B*15:107', 'HLA-B*15:108', 'HLA-B*15:109', 'HLA-B*15:11', 'HLA-B*15:110', 'HLA-B*15:112', 'HLA-B*15:113', 'HLA-B*15:114',
         'HLA-B*15:115', 'HLA-B*15:116', 'HLA-B*15:117', 'HLA-B*15:118', 'HLA-B*15:119', 'HLA-B*15:12', 'HLA-B*15:120', 'HLA-B*15:121',
         'HLA-B*15:122', 'HLA-B*15:123', 'HLA-B*15:124', 'HLA-B*15:125', 'HLA-B*15:126', 'HLA-B*15:127', 'HLA-B*15:128', 'HLA-B*15:129',
         'HLA-B*15:13', 'HLA-B*15:131', 'HLA-B*15:132', 'HLA-B*15:133', 'HLA-B*15:134', 'HLA-B*15:135', 'HLA-B*15:136', 'HLA-B*15:137',
         'HLA-B*15:138', 'HLA-B*15:139', 'HLA-B*15:14', 'HLA-B*15:140', 'HLA-B*15:141', 'HLA-B*15:142', 'HLA-B*15:143', 'HLA-B*15:144',
         'HLA-B*15:145', 'HLA-B*15:146', 'HLA-B*15:147', 'HLA-B*15:148', 'HLA-B*15:15', 'HLA-B*15:150', 'HLA-B*15:151', 'HLA-B*15:152',
         'HLA-B*15:153', 'HLA-B*15:154', 'HLA-B*15:155', 'HLA-B*15:156', 'HLA-B*15:157', 'HLA-B*15:158', 'HLA-B*15:159', 'HLA-B*15:16',
         'HLA-B*15:160', 'HLA-B*15:161', 'HLA-B*15:162', 'HLA-B*15:163', 'HLA-B*15:164', 'HLA-B*15:165', 'HLA-B*15:166', 'HLA-B*15:167',
         'HLA-B*15:168', 'HLA-B*15:169', 'HLA-B*15:17', 'HLA-B*15:170', 'HLA-B*15:171', 'HLA-B*15:172', 'HLA-B*15:173', 'HLA-B*15:174',
         'HLA-B*15:175', 'HLA-B*15:176', 'HLA-B*15:177', 'HLA-B*15:178', 'HLA-B*15:179', 'HLA-B*15:18', 'HLA-B*15:180', 'HLA-B*15:183',
         'HLA-B*15:184', 'HLA-B*15:185', 'HLA-B*15:186', 'HLA-B*15:187', 'HLA-B*15:188', 'HLA-B*15:189', 'HLA-B*15:19', 'HLA-B*15:191',
         'HLA-B*15:192', 'HLA-B*15:193', 'HLA-B*15:194', 'HLA-B*15:195', 'HLA-B*15:196', 'HLA-B*15:197', 'HLA-B*15:198', 'HLA-B*15:199',
         'HLA-B*15:20', 'HLA-B*15:200', 'HLA-B*15:201', 'HLA-B*15:202', 'HLA-B*15:21', 'HLA-B*15:23', 'HLA-B*15:24', 'HLA-B*15:25', 'HLA-B*15:27',
         'HLA-B*15:28', 'HLA-B*15:29', 'HLA-B*15:30', 'HLA-B*15:31', 'HLA-B*15:32', 'HLA-B*15:33', 'HLA-B*15:34', 'HLA-B*15:35', 'HLA-B*15:36',
         'HLA-B*15:37', 'HLA-B*15:38', 'HLA-B*15:39', 'HLA-B*15:40', 'HLA-B*15:42', 'HLA-B*15:43', 'HLA-B*15:44', 'HLA-B*15:45', 'HLA-B*15:46',
         'HLA-B*15:47', 'HLA-B*15:48', 'HLA-B*15:49', 'HLA-B*15:50', 'HLA-B*15:51', 'HLA-B*15:52', 'HLA-B*15:53', 'HLA-B*15:54', 'HLA-B*15:55',
         'HLA-B*15:56', 'HLA-B*15:57', 'HLA-B*15:58', 'HLA-B*15:60', 'HLA-B*15:61', 'HLA-B*15:62', 'HLA-B*15:63', 'HLA-B*15:64', 'HLA-B*15:65',
         'HLA-B*15:66', 'HLA-B*15:67', 'HLA-B*15:68', 'HLA-B*15:69', 'HLA-B*15:70', 'HLA-B*15:71', 'HLA-B*15:72', 'HLA-B*15:73', 'HLA-B*15:74',
         'HLA-B*15:75', 'HLA-B*15:76', 'HLA-B*15:77', 'HLA-B*15:78', 'HLA-B*15:80', 'HLA-B*15:81', 'HLA-B*15:82', 'HLA-B*15:83', 'HLA-B*15:84',
         'HLA-B*15:85', 'HLA-B*15:86', 'HLA-B*15:87', 'HLA-B*15:88', 'HLA-B*15:89', 'HLA-B*15:90', 'HLA-B*15:91', 'HLA-B*15:92', 'HLA-B*15:93',
         'HLA-B*15:95', 'HLA-B*15:96', 'HLA-B*15:97', 'HLA-B*15:98', 'HLA-B*15:99', 'HLA-B*18:01', 'HLA-B*18:02', 'HLA-B*18:03', 'HLA-B*18:04',
         'HLA-B*18:05', 'HLA-B*18:06', 'HLA-B*18:07', 'HLA-B*18:08', 'HLA-B*18:09', 'HLA-B*18:10', 'HLA-B*18:11', 'HLA-B*18:12', 'HLA-B*18:13',
         'HLA-B*18:14', 'HLA-B*18:15', 'HLA-B*18:18', 'HLA-B*18:19', 'HLA-B*18:20', 'HLA-B*18:21', 'HLA-B*18:22', 'HLA-B*18:24', 'HLA-B*18:25',
         'HLA-B*18:26', 'HLA-B*18:27', 'HLA-B*18:28', 'HLA-B*18:29', 'HLA-B*18:30', 'HLA-B*18:31', 'HLA-B*18:32', 'HLA-B*18:33', 'HLA-B*18:34',
         'HLA-B*18:35', 'HLA-B*18:36', 'HLA-B*18:37', 'HLA-B*18:38', 'HLA-B*18:39', 'HLA-B*18:40', 'HLA-B*18:41', 'HLA-B*18:42', 'HLA-B*18:43',
         'HLA-B*18:44', 'HLA-B*18:45', 'HLA-B*18:46', 'HLA-B*18:47', 'HLA-B*18:48', 'HLA-B*18:49', 'HLA-B*18:50', 'HLA-B*27:01', 'HLA-B*27:02',
         'HLA-B*27:03', 'HLA-B*27:04', 'HLA-B*27:05', 'HLA-B*27:06', 'HLA-B*27:07', 'HLA-B*27:08', 'HLA-B*27:09', 'HLA-B*27:10', 'HLA-B*27:11',
         'HLA-B*27:12', 'HLA-B*27:13', 'HLA-B*27:14', 'HLA-B*27:15', 'HLA-B*27:16', 'HLA-B*27:17', 'HLA-B*27:18', 'HLA-B*27:19', 'HLA-B*27:20',
         'HLA-B*27:21', 'HLA-B*27:23', 'HLA-B*27:24', 'HLA-B*27:25', 'HLA-B*27:26', 'HLA-B*27:27', 'HLA-B*27:28', 'HLA-B*27:29', 'HLA-B*27:30',
         'HLA-B*27:31', 'HLA-B*27:32', 'HLA-B*27:33', 'HLA-B*27:34', 'HLA-B*27:35', 'HLA-B*27:36', 'HLA-B*27:37', 'HLA-B*27:38', 'HLA-B*27:39',
         'HLA-B*27:40', 'HLA-B*27:41', 'HLA-B*27:42', 'HLA-B*27:43', 'HLA-B*27:44', 'HLA-B*27:45', 'HLA-B*27:46', 'HLA-B*27:47', 'HLA-B*27:48',
         'HLA-B*27:49', 'HLA-B*27:50', 'HLA-B*27:51', 'HLA-B*27:52', 'HLA-B*27:53', 'HLA-B*27:54', 'HLA-B*27:55', 'HLA-B*27:56', 'HLA-B*27:57',
         'HLA-B*27:58', 'HLA-B*27:60', 'HLA-B*27:61', 'HLA-B*27:62', 'HLA-B*27:63', 'HLA-B*27:67', 'HLA-B*27:68', 'HLA-B*27:69', 'HLA-B*35:01',
         'HLA-B*35:02', 'HLA-B*35:03', 'HLA-B*35:04', 'HLA-B*35:05', 'HLA-B*35:06', 'HLA-B*35:07', 'HLA-B*35:08', 'HLA-B*35:09', 'HLA-B*35:10',
         'HLA-B*35:100', 'HLA-B*35:101', 'HLA-B*35:102', 'HLA-B*35:103', 'HLA-B*35:104', 'HLA-B*35:105', 'HLA-B*35:106', 'HLA-B*35:107',
         'HLA-B*35:108', 'HLA-B*35:109', 'HLA-B*35:11', 'HLA-B*35:110', 'HLA-B*35:111', 'HLA-B*35:112', 'HLA-B*35:113', 'HLA-B*35:114',
         'HLA-B*35:115', 'HLA-B*35:116', 'HLA-B*35:117', 'HLA-B*35:118', 'HLA-B*35:119', 'HLA-B*35:12', 'HLA-B*35:120', 'HLA-B*35:121',
         'HLA-B*35:122', 'HLA-B*35:123', 'HLA-B*35:124', 'HLA-B*35:125', 'HLA-B*35:126', 'HLA-B*35:127', 'HLA-B*35:128', 'HLA-B*35:13',
         'HLA-B*35:131', 'HLA-B*35:132', 'HLA-B*35:133', 'HLA-B*35:135', 'HLA-B*35:136', 'HLA-B*35:137', 'HLA-B*35:138', 'HLA-B*35:139',
         'HLA-B*35:14', 'HLA-B*35:140', 'HLA-B*35:141', 'HLA-B*35:142', 'HLA-B*35:143', 'HLA-B*35:144', 'HLA-B*35:15', 'HLA-B*35:16', 'HLA-B*35:17',
         'HLA-B*35:18', 'HLA-B*35:19', 'HLA-B*35:20', 'HLA-B*35:21', 'HLA-B*35:22', 'HLA-B*35:23', 'HLA-B*35:24', 'HLA-B*35:25', 'HLA-B*35:26',
         'HLA-B*35:27', 'HLA-B*35:28', 'HLA-B*35:29', 'HLA-B*35:30', 'HLA-B*35:31', 'HLA-B*35:32', 'HLA-B*35:33', 'HLA-B*35:34', 'HLA-B*35:35',
         'HLA-B*35:36', 'HLA-B*35:37', 'HLA-B*35:38', 'HLA-B*35:39', 'HLA-B*35:41', 'HLA-B*35:42', 'HLA-B*35:43', 'HLA-B*35:44', 'HLA-B*35:45',
         'HLA-B*35:46', 'HLA-B*35:47', 'HLA-B*35:48', 'HLA-B*35:49', 'HLA-B*35:50', 'HLA-B*35:51', 'HLA-B*35:52', 'HLA-B*35:54', 'HLA-B*35:55',
         'HLA-B*35:56', 'HLA-B*35:57', 'HLA-B*35:58', 'HLA-B*35:59', 'HLA-B*35:60', 'HLA-B*35:61', 'HLA-B*35:62', 'HLA-B*35:63', 'HLA-B*35:64',
         'HLA-B*35:66', 'HLA-B*35:67', 'HLA-B*35:68', 'HLA-B*35:69', 'HLA-B*35:70', 'HLA-B*35:71', 'HLA-B*35:72', 'HLA-B*35:74', 'HLA-B*35:75',
         'HLA-B*35:76', 'HLA-B*35:77', 'HLA-B*35:78', 'HLA-B*35:79', 'HLA-B*35:80', 'HLA-B*35:81', 'HLA-B*35:82', 'HLA-B*35:83', 'HLA-B*35:84',
         'HLA-B*35:85', 'HLA-B*35:86', 'HLA-B*35:87', 'HLA-B*35:88', 'HLA-B*35:89', 'HLA-B*35:90', 'HLA-B*35:91', 'HLA-B*35:92', 'HLA-B*35:93',
         'HLA-B*35:94', 'HLA-B*35:95', 'HLA-B*35:96', 'HLA-B*35:97', 'HLA-B*35:98', 'HLA-B*35:99', 'HLA-B*37:01', 'HLA-B*37:02', 'HLA-B*37:04',
         'HLA-B*37:05', 'HLA-B*37:06', 'HLA-B*37:07', 'HLA-B*37:08', 'HLA-B*37:09', 'HLA-B*37:10', 'HLA-B*37:11', 'HLA-B*37:12', 'HLA-B*37:13',
         'HLA-B*37:14', 'HLA-B*37:15', 'HLA-B*37:17', 'HLA-B*37:18', 'HLA-B*37:19', 'HLA-B*37:20', 'HLA-B*37:21', 'HLA-B*37:22', 'HLA-B*37:23',
         'HLA-B*38:01', 'HLA-B*38:02', 'HLA-B*38:03', 'HLA-B*38:04', 'HLA-B*38:05', 'HLA-B*38:06', 'HLA-B*38:07', 'HLA-B*38:08', 'HLA-B*38:09',
         'HLA-B*38:10', 'HLA-B*38:11', 'HLA-B*38:12', 'HLA-B*38:13', 'HLA-B*38:14', 'HLA-B*38:15', 'HLA-B*38:16', 'HLA-B*38:17', 'HLA-B*38:18',
         'HLA-B*38:19', 'HLA-B*38:20', 'HLA-B*38:21', 'HLA-B*38:22', 'HLA-B*38:23', 'HLA-B*39:01', 'HLA-B*39:02', 'HLA-B*39:03', 'HLA-B*39:04',
         'HLA-B*39:05', 'HLA-B*39:06', 'HLA-B*39:07', 'HLA-B*39:08', 'HLA-B*39:09', 'HLA-B*39:10', 'HLA-B*39:11', 'HLA-B*39:12', 'HLA-B*39:13',
         'HLA-B*39:14', 'HLA-B*39:15', 'HLA-B*39:16', 'HLA-B*39:17', 'HLA-B*39:18', 'HLA-B*39:19', 'HLA-B*39:20', 'HLA-B*39:22', 'HLA-B*39:23',
         'HLA-B*39:24', 'HLA-B*39:26', 'HLA-B*39:27', 'HLA-B*39:28', 'HLA-B*39:29', 'HLA-B*39:30', 'HLA-B*39:31', 'HLA-B*39:32', 'HLA-B*39:33',
         'HLA-B*39:34', 'HLA-B*39:35', 'HLA-B*39:36', 'HLA-B*39:37', 'HLA-B*39:39', 'HLA-B*39:41', 'HLA-B*39:42', 'HLA-B*39:43', 'HLA-B*39:44',
         'HLA-B*39:45', 'HLA-B*39:46', 'HLA-B*39:47', 'HLA-B*39:48', 'HLA-B*39:49', 'HLA-B*39:50', 'HLA-B*39:51', 'HLA-B*39:52', 'HLA-B*39:53',
         'HLA-B*39:54', 'HLA-B*39:55', 'HLA-B*39:56', 'HLA-B*39:57', 'HLA-B*39:58', 'HLA-B*39:59', 'HLA-B*39:60', 'HLA-B*40:01', 'HLA-B*40:02',
         'HLA-B*40:03', 'HLA-B*40:04', 'HLA-B*40:05', 'HLA-B*40:06', 'HLA-B*40:07', 'HLA-B*40:08', 'HLA-B*40:09', 'HLA-B*40:10', 'HLA-B*40:100',
         'HLA-B*40:101', 'HLA-B*40:102', 'HLA-B*40:103', 'HLA-B*40:104', 'HLA-B*40:105', 'HLA-B*40:106', 'HLA-B*40:107', 'HLA-B*40:108',
         'HLA-B*40:109', 'HLA-B*40:11', 'HLA-B*40:110', 'HLA-B*40:111', 'HLA-B*40:112', 'HLA-B*40:113', 'HLA-B*40:114', 'HLA-B*40:115',
         'HLA-B*40:116', 'HLA-B*40:117', 'HLA-B*40:119', 'HLA-B*40:12', 'HLA-B*40:120', 'HLA-B*40:121', 'HLA-B*40:122', 'HLA-B*40:123',
         'HLA-B*40:124', 'HLA-B*40:125', 'HLA-B*40:126', 'HLA-B*40:127', 'HLA-B*40:128', 'HLA-B*40:129', 'HLA-B*40:13', 'HLA-B*40:130',
         'HLA-B*40:131', 'HLA-B*40:132', 'HLA-B*40:134', 'HLA-B*40:135', 'HLA-B*40:136', 'HLA-B*40:137', 'HLA-B*40:138', 'HLA-B*40:139',
         'HLA-B*40:14', 'HLA-B*40:140', 'HLA-B*40:141', 'HLA-B*40:143', 'HLA-B*40:145', 'HLA-B*40:146', 'HLA-B*40:147', 'HLA-B*40:15',
         'HLA-B*40:16', 'HLA-B*40:18', 'HLA-B*40:19', 'HLA-B*40:20', 'HLA-B*40:21', 'HLA-B*40:23', 'HLA-B*40:24', 'HLA-B*40:25', 'HLA-B*40:26',
         'HLA-B*40:27', 'HLA-B*40:28', 'HLA-B*40:29', 'HLA-B*40:30', 'HLA-B*40:31', 'HLA-B*40:32', 'HLA-B*40:33', 'HLA-B*40:34', 'HLA-B*40:35',
         'HLA-B*40:36', 'HLA-B*40:37', 'HLA-B*40:38', 'HLA-B*40:39', 'HLA-B*40:40', 'HLA-B*40:42', 'HLA-B*40:43', 'HLA-B*40:44', 'HLA-B*40:45',
         'HLA-B*40:46', 'HLA-B*40:47', 'HLA-B*40:48', 'HLA-B*40:49', 'HLA-B*40:50', 'HLA-B*40:51', 'HLA-B*40:52', 'HLA-B*40:53', 'HLA-B*40:54',
         'HLA-B*40:55', 'HLA-B*40:56', 'HLA-B*40:57', 'HLA-B*40:58', 'HLA-B*40:59', 'HLA-B*40:60', 'HLA-B*40:61', 'HLA-B*40:62', 'HLA-B*40:63',
         'HLA-B*40:64', 'HLA-B*40:65', 'HLA-B*40:66', 'HLA-B*40:67', 'HLA-B*40:68', 'HLA-B*40:69', 'HLA-B*40:70', 'HLA-B*40:71', 'HLA-B*40:72',
         'HLA-B*40:73', 'HLA-B*40:74', 'HLA-B*40:75', 'HLA-B*40:76', 'HLA-B*40:77', 'HLA-B*40:78', 'HLA-B*40:79', 'HLA-B*40:80', 'HLA-B*40:81',
         'HLA-B*40:82', 'HLA-B*40:83', 'HLA-B*40:84', 'HLA-B*40:85', 'HLA-B*40:86', 'HLA-B*40:87', 'HLA-B*40:88', 'HLA-B*40:89', 'HLA-B*40:90',
         'HLA-B*40:91', 'HLA-B*40:92', 'HLA-B*40:93', 'HLA-B*40:94', 'HLA-B*40:95', 'HLA-B*40:96', 'HLA-B*40:97', 'HLA-B*40:98', 'HLA-B*40:99',
         'HLA-B*41:01', 'HLA-B*41:02', 'HLA-B*41:03', 'HLA-B*41:04', 'HLA-B*41:05', 'HLA-B*41:06', 'HLA-B*41:07', 'HLA-B*41:08', 'HLA-B*41:09',
         'HLA-B*41:10', 'HLA-B*41:11', 'HLA-B*41:12', 'HLA-B*42:01', 'HLA-B*42:02', 'HLA-B*42:04', 'HLA-B*42:05', 'HLA-B*42:06', 'HLA-B*42:07',
         'HLA-B*42:08', 'HLA-B*42:09', 'HLA-B*42:10', 'HLA-B*42:11', 'HLA-B*42:12', 'HLA-B*42:13', 'HLA-B*42:14', 'HLA-B*44:02', 'HLA-B*44:03',
         'HLA-B*44:04', 'HLA-B*44:05', 'HLA-B*44:06', 'HLA-B*44:07', 'HLA-B*44:08', 'HLA-B*44:09', 'HLA-B*44:10', 'HLA-B*44:100', 'HLA-B*44:101',
         'HLA-B*44:102', 'HLA-B*44:103', 'HLA-B*44:104', 'HLA-B*44:105', 'HLA-B*44:106', 'HLA-B*44:107', 'HLA-B*44:109', 'HLA-B*44:11',
         'HLA-B*44:110', 'HLA-B*44:12', 'HLA-B*44:13', 'HLA-B*44:14', 'HLA-B*44:15', 'HLA-B*44:16', 'HLA-B*44:17', 'HLA-B*44:18', 'HLA-B*44:20',
         'HLA-B*44:21', 'HLA-B*44:22', 'HLA-B*44:24', 'HLA-B*44:25', 'HLA-B*44:26', 'HLA-B*44:27', 'HLA-B*44:28', 'HLA-B*44:29', 'HLA-B*44:30',
         'HLA-B*44:31', 'HLA-B*44:32', 'HLA-B*44:33', 'HLA-B*44:34', 'HLA-B*44:35', 'HLA-B*44:36', 'HLA-B*44:37', 'HLA-B*44:38', 'HLA-B*44:39',
         'HLA-B*44:40', 'HLA-B*44:41', 'HLA-B*44:42', 'HLA-B*44:43', 'HLA-B*44:44', 'HLA-B*44:45', 'HLA-B*44:46', 'HLA-B*44:47', 'HLA-B*44:48',
         'HLA-B*44:49', 'HLA-B*44:50', 'HLA-B*44:51', 'HLA-B*44:53', 'HLA-B*44:54', 'HLA-B*44:55', 'HLA-B*44:57', 'HLA-B*44:59', 'HLA-B*44:60',
         'HLA-B*44:62', 'HLA-B*44:63', 'HLA-B*44:64', 'HLA-B*44:65', 'HLA-B*44:66', 'HLA-B*44:67', 'HLA-B*44:68', 'HLA-B*44:69', 'HLA-B*44:70',
         'HLA-B*44:71', 'HLA-B*44:72', 'HLA-B*44:73', 'HLA-B*44:74', 'HLA-B*44:75', 'HLA-B*44:76', 'HLA-B*44:77', 'HLA-B*44:78', 'HLA-B*44:79',
         'HLA-B*44:80', 'HLA-B*44:81', 'HLA-B*44:82', 'HLA-B*44:83', 'HLA-B*44:84', 'HLA-B*44:85', 'HLA-B*44:86', 'HLA-B*44:87', 'HLA-B*44:88',
         'HLA-B*44:89', 'HLA-B*44:90', 'HLA-B*44:91', 'HLA-B*44:92', 'HLA-B*44:93', 'HLA-B*44:94', 'HLA-B*44:95', 'HLA-B*44:96', 'HLA-B*44:97',
         'HLA-B*44:98', 'HLA-B*44:99', 'HLA-B*45:01', 'HLA-B*45:02', 'HLA-B*45:03', 'HLA-B*45:04', 'HLA-B*45:05', 'HLA-B*45:06', 'HLA-B*45:07',
         'HLA-B*45:08', 'HLA-B*45:09', 'HLA-B*45:10', 'HLA-B*45:11', 'HLA-B*45:12', 'HLA-B*46:01', 'HLA-B*46:02', 'HLA-B*46:03', 'HLA-B*46:04',
         'HLA-B*46:05', 'HLA-B*46:06', 'HLA-B*46:08', 'HLA-B*46:09', 'HLA-B*46:10', 'HLA-B*46:11', 'HLA-B*46:12', 'HLA-B*46:13', 'HLA-B*46:14',
         'HLA-B*46:16', 'HLA-B*46:17', 'HLA-B*46:18', 'HLA-B*46:19', 'HLA-B*46:20', 'HLA-B*46:21', 'HLA-B*46:22', 'HLA-B*46:23', 'HLA-B*46:24',
         'HLA-B*47:01', 'HLA-B*47:02', 'HLA-B*47:03', 'HLA-B*47:04', 'HLA-B*47:05', 'HLA-B*47:06', 'HLA-B*47:07', 'HLA-B*48:01', 'HLA-B*48:02',
         'HLA-B*48:03', 'HLA-B*48:04', 'HLA-B*48:05', 'HLA-B*48:06', 'HLA-B*48:07', 'HLA-B*48:08', 'HLA-B*48:09', 'HLA-B*48:10', 'HLA-B*48:11',
         'HLA-B*48:12', 'HLA-B*48:13', 'HLA-B*48:14', 'HLA-B*48:15', 'HLA-B*48:16', 'HLA-B*48:17', 'HLA-B*48:18', 'HLA-B*48:19', 'HLA-B*48:20',
         'HLA-B*48:21', 'HLA-B*48:22', 'HLA-B*48:23', 'HLA-B*49:01', 'HLA-B*49:02', 'HLA-B*49:03', 'HLA-B*49:04', 'HLA-B*49:05', 'HLA-B*49:06',
         'HLA-B*49:07', 'HLA-B*49:08', 'HLA-B*49:09', 'HLA-B*49:10', 'HLA-B*50:01', 'HLA-B*50:02', 'HLA-B*50:04', 'HLA-B*50:05', 'HLA-B*50:06',
         'HLA-B*50:07', 'HLA-B*50:08', 'HLA-B*50:09', 'HLA-B*51:01', 'HLA-B*51:02', 'HLA-B*51:03', 'HLA-B*51:04', 'HLA-B*51:05', 'HLA-B*51:06',
         'HLA-B*51:07', 'HLA-B*51:08', 'HLA-B*51:09', 'HLA-B*51:12', 'HLA-B*51:13', 'HLA-B*51:14', 'HLA-B*51:15', 'HLA-B*51:16', 'HLA-B*51:17',
         'HLA-B*51:18', 'HLA-B*51:19', 'HLA-B*51:20', 'HLA-B*51:21', 'HLA-B*51:22', 'HLA-B*51:23', 'HLA-B*51:24', 'HLA-B*51:26', 'HLA-B*51:28',
         'HLA-B*51:29', 'HLA-B*51:30', 'HLA-B*51:31', 'HLA-B*51:32', 'HLA-B*51:33', 'HLA-B*51:34', 'HLA-B*51:35', 'HLA-B*51:36', 'HLA-B*51:37',
         'HLA-B*51:38', 'HLA-B*51:39', 'HLA-B*51:40', 'HLA-B*51:42', 'HLA-B*51:43', 'HLA-B*51:45', 'HLA-B*51:46', 'HLA-B*51:48', 'HLA-B*51:49',
         'HLA-B*51:50', 'HLA-B*51:51', 'HLA-B*51:52', 'HLA-B*51:53', 'HLA-B*51:54', 'HLA-B*51:55', 'HLA-B*51:56', 'HLA-B*51:57', 'HLA-B*51:58',
         'HLA-B*51:59', 'HLA-B*51:60', 'HLA-B*51:61', 'HLA-B*51:62', 'HLA-B*51:63', 'HLA-B*51:64', 'HLA-B*51:65', 'HLA-B*51:66', 'HLA-B*51:67',
         'HLA-B*51:68', 'HLA-B*51:69', 'HLA-B*51:70', 'HLA-B*51:71', 'HLA-B*51:72', 'HLA-B*51:73', 'HLA-B*51:74', 'HLA-B*51:75', 'HLA-B*51:76',
         'HLA-B*51:77', 'HLA-B*51:78', 'HLA-B*51:79', 'HLA-B*51:80', 'HLA-B*51:81', 'HLA-B*51:82', 'HLA-B*51:83', 'HLA-B*51:84', 'HLA-B*51:85',
         'HLA-B*51:86', 'HLA-B*51:87', 'HLA-B*51:88', 'HLA-B*51:89', 'HLA-B*51:90', 'HLA-B*51:91', 'HLA-B*51:92', 'HLA-B*51:93', 'HLA-B*51:94',
         'HLA-B*51:95', 'HLA-B*51:96', 'HLA-B*52:01', 'HLA-B*52:02', 'HLA-B*52:03', 'HLA-B*52:04', 'HLA-B*52:05', 'HLA-B*52:06', 'HLA-B*52:07',
         'HLA-B*52:08', 'HLA-B*52:09', 'HLA-B*52:10', 'HLA-B*52:11', 'HLA-B*52:12', 'HLA-B*52:13', 'HLA-B*52:14', 'HLA-B*52:15', 'HLA-B*52:16',
         'HLA-B*52:17', 'HLA-B*52:18', 'HLA-B*52:19', 'HLA-B*52:20', 'HLA-B*52:21', 'HLA-B*53:01', 'HLA-B*53:02', 'HLA-B*53:03', 'HLA-B*53:04',
         'HLA-B*53:05', 'HLA-B*53:06', 'HLA-B*53:07', 'HLA-B*53:08', 'HLA-B*53:09', 'HLA-B*53:10', 'HLA-B*53:11', 'HLA-B*53:12', 'HLA-B*53:13',
         'HLA-B*53:14', 'HLA-B*53:15', 'HLA-B*53:16', 'HLA-B*53:17', 'HLA-B*53:18', 'HLA-B*53:19', 'HLA-B*53:20', 'HLA-B*53:21', 'HLA-B*53:22',
         'HLA-B*53:23', 'HLA-B*54:01', 'HLA-B*54:02', 'HLA-B*54:03', 'HLA-B*54:04', 'HLA-B*54:06', 'HLA-B*54:07', 'HLA-B*54:09', 'HLA-B*54:10',
         'HLA-B*54:11', 'HLA-B*54:12', 'HLA-B*54:13', 'HLA-B*54:14', 'HLA-B*54:15', 'HLA-B*54:16', 'HLA-B*54:17', 'HLA-B*54:18', 'HLA-B*54:19',
         'HLA-B*54:20', 'HLA-B*54:21', 'HLA-B*54:22', 'HLA-B*54:23', 'HLA-B*55:01', 'HLA-B*55:02', 'HLA-B*55:03', 'HLA-B*55:04', 'HLA-B*55:05',
         'HLA-B*55:07', 'HLA-B*55:08', 'HLA-B*55:09', 'HLA-B*55:10', 'HLA-B*55:11', 'HLA-B*55:12', 'HLA-B*55:13', 'HLA-B*55:14', 'HLA-B*55:15',
         'HLA-B*55:16', 'HLA-B*55:17', 'HLA-B*55:18', 'HLA-B*55:19', 'HLA-B*55:20', 'HLA-B*55:21', 'HLA-B*55:22', 'HLA-B*55:23', 'HLA-B*55:24',
         'HLA-B*55:25', 'HLA-B*55:26', 'HLA-B*55:27', 'HLA-B*55:28', 'HLA-B*55:29', 'HLA-B*55:30', 'HLA-B*55:31', 'HLA-B*55:32', 'HLA-B*55:33',
         'HLA-B*55:34', 'HLA-B*55:35', 'HLA-B*55:36', 'HLA-B*55:37', 'HLA-B*55:38', 'HLA-B*55:39', 'HLA-B*55:40', 'HLA-B*55:41', 'HLA-B*55:42',
         'HLA-B*55:43', 'HLA-B*56:01', 'HLA-B*56:02', 'HLA-B*56:03', 'HLA-B*56:04', 'HLA-B*56:05', 'HLA-B*56:06', 'HLA-B*56:07', 'HLA-B*56:08',
         'HLA-B*56:09', 'HLA-B*56:10', 'HLA-B*56:11', 'HLA-B*56:12', 'HLA-B*56:13', 'HLA-B*56:14', 'HLA-B*56:15', 'HLA-B*56:16', 'HLA-B*56:17',
         'HLA-B*56:18', 'HLA-B*56:20', 'HLA-B*56:21', 'HLA-B*56:22', 'HLA-B*56:23', 'HLA-B*56:24', 'HLA-B*56:25', 'HLA-B*56:26', 'HLA-B*56:27',
         'HLA-B*56:29', 'HLA-B*57:01', 'HLA-B*57:02', 'HLA-B*57:03', 'HLA-B*57:04', 'HLA-B*57:05', 'HLA-B*57:06', 'HLA-B*57:07', 'HLA-B*57:08',
         'HLA-B*57:09', 'HLA-B*57:10', 'HLA-B*57:11', 'HLA-B*57:12', 'HLA-B*57:13', 'HLA-B*57:14', 'HLA-B*57:15', 'HLA-B*57:16', 'HLA-B*57:17',
         'HLA-B*57:18', 'HLA-B*57:19', 'HLA-B*57:20', 'HLA-B*57:21', 'HLA-B*57:22', 'HLA-B*57:23', 'HLA-B*57:24', 'HLA-B*57:25', 'HLA-B*57:26',
         'HLA-B*57:27', 'HLA-B*57:29', 'HLA-B*57:30', 'HLA-B*57:31', 'HLA-B*57:32', 'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*58:04', 'HLA-B*58:05',
         'HLA-B*58:06', 'HLA-B*58:07', 'HLA-B*58:08', 'HLA-B*58:09', 'HLA-B*58:11', 'HLA-B*58:12', 'HLA-B*58:13', 'HLA-B*58:14', 'HLA-B*58:15',
         'HLA-B*58:16', 'HLA-B*58:18', 'HLA-B*58:19', 'HLA-B*58:20', 'HLA-B*58:21', 'HLA-B*58:22', 'HLA-B*58:23', 'HLA-B*58:24', 'HLA-B*58:25',
         'HLA-B*58:26', 'HLA-B*58:27', 'HLA-B*58:28', 'HLA-B*58:29', 'HLA-B*58:30', 'HLA-B*59:01', 'HLA-B*59:02', 'HLA-B*59:03', 'HLA-B*59:04',
         'HLA-B*59:05', 'HLA-B*67:01', 'HLA-B*67:02', 'HLA-B*73:01', 'HLA-B*73:02', 'HLA-B*78:01', 'HLA-B*78:02', 'HLA-B*78:03', 'HLA-B*78:04',
         'HLA-B*78:05', 'HLA-B*78:06', 'HLA-B*78:07', 'HLA-B*81:01', 'HLA-B*81:02', 'HLA-B*81:03', 'HLA-B*81:05', 'HLA-B*82:01', 'HLA-B*82:02',
         'HLA-B*82:03', 'HLA-B*83:01', 'HLA-C*01:02', 'HLA-C*01:03', 'HLA-C*01:04', 'HLA-C*01:05', 'HLA-C*01:06', 'HLA-C*01:07', 'HLA-C*01:08',
         'HLA-C*01:09', 'HLA-C*01:10', 'HLA-C*01:11', 'HLA-C*01:12', 'HLA-C*01:13', 'HLA-C*01:14', 'HLA-C*01:15', 'HLA-C*01:16', 'HLA-C*01:17',
         'HLA-C*01:18', 'HLA-C*01:19', 'HLA-C*01:20', 'HLA-C*01:21', 'HLA-C*01:22', 'HLA-C*01:23', 'HLA-C*01:24', 'HLA-C*01:25', 'HLA-C*01:26',
         'HLA-C*01:27', 'HLA-C*01:28', 'HLA-C*01:29', 'HLA-C*01:30', 'HLA-C*01:31', 'HLA-C*01:32', 'HLA-C*01:33', 'HLA-C*01:34', 'HLA-C*01:35',
         'HLA-C*01:36', 'HLA-C*01:38', 'HLA-C*01:39', 'HLA-C*01:40', 'HLA-C*02:02', 'HLA-C*02:03', 'HLA-C*02:04', 'HLA-C*02:05', 'HLA-C*02:06',
         'HLA-C*02:07', 'HLA-C*02:08', 'HLA-C*02:09', 'HLA-C*02:10', 'HLA-C*02:11', 'HLA-C*02:12', 'HLA-C*02:13', 'HLA-C*02:14', 'HLA-C*02:15',
         'HLA-C*02:16', 'HLA-C*02:17', 'HLA-C*02:18', 'HLA-C*02:19', 'HLA-C*02:20', 'HLA-C*02:21', 'HLA-C*02:22', 'HLA-C*02:23', 'HLA-C*02:24',
         'HLA-C*02:26', 'HLA-C*02:27', 'HLA-C*02:28', 'HLA-C*02:29', 'HLA-C*02:30', 'HLA-C*02:31', 'HLA-C*02:32', 'HLA-C*02:33', 'HLA-C*02:34',
         'HLA-C*02:35', 'HLA-C*02:36', 'HLA-C*02:37', 'HLA-C*02:39', 'HLA-C*02:40', 'HLA-C*03:01', 'HLA-C*03:02', 'HLA-C*03:03', 'HLA-C*03:04',
         'HLA-C*03:05', 'HLA-C*03:06', 'HLA-C*03:07', 'HLA-C*03:08', 'HLA-C*03:09', 'HLA-C*03:10', 'HLA-C*03:11', 'HLA-C*03:12', 'HLA-C*03:13',
         'HLA-C*03:14', 'HLA-C*03:15', 'HLA-C*03:16', 'HLA-C*03:17', 'HLA-C*03:18', 'HLA-C*03:19', 'HLA-C*03:21', 'HLA-C*03:23', 'HLA-C*03:24',
         'HLA-C*03:25', 'HLA-C*03:26', 'HLA-C*03:27', 'HLA-C*03:28', 'HLA-C*03:29', 'HLA-C*03:30', 'HLA-C*03:31', 'HLA-C*03:32', 'HLA-C*03:33',
         'HLA-C*03:34', 'HLA-C*03:35', 'HLA-C*03:36', 'HLA-C*03:37', 'HLA-C*03:38', 'HLA-C*03:39', 'HLA-C*03:40', 'HLA-C*03:41', 'HLA-C*03:42',
         'HLA-C*03:43', 'HLA-C*03:44', 'HLA-C*03:45', 'HLA-C*03:46', 'HLA-C*03:47', 'HLA-C*03:48', 'HLA-C*03:49', 'HLA-C*03:50', 'HLA-C*03:51',
         'HLA-C*03:52', 'HLA-C*03:53', 'HLA-C*03:54', 'HLA-C*03:55', 'HLA-C*03:56', 'HLA-C*03:57', 'HLA-C*03:58', 'HLA-C*03:59', 'HLA-C*03:60',
         'HLA-C*03:61', 'HLA-C*03:62', 'HLA-C*03:63', 'HLA-C*03:64', 'HLA-C*03:65', 'HLA-C*03:66', 'HLA-C*03:67', 'HLA-C*03:68', 'HLA-C*03:69',
         'HLA-C*03:70', 'HLA-C*03:71', 'HLA-C*03:72', 'HLA-C*03:73', 'HLA-C*03:74', 'HLA-C*03:75', 'HLA-C*03:76', 'HLA-C*03:77', 'HLA-C*03:78',
         'HLA-C*03:79', 'HLA-C*03:80', 'HLA-C*03:81', 'HLA-C*03:82', 'HLA-C*03:83', 'HLA-C*03:84', 'HLA-C*03:85', 'HLA-C*03:86', 'HLA-C*03:87',
         'HLA-C*03:88', 'HLA-C*03:89', 'HLA-C*03:90', 'HLA-C*03:91', 'HLA-C*03:92', 'HLA-C*03:93', 'HLA-C*03:94', 'HLA-C*04:01', 'HLA-C*04:03',
         'HLA-C*04:04', 'HLA-C*04:05', 'HLA-C*04:06', 'HLA-C*04:07', 'HLA-C*04:08', 'HLA-C*04:10', 'HLA-C*04:11', 'HLA-C*04:12', 'HLA-C*04:13',
         'HLA-C*04:14', 'HLA-C*04:15', 'HLA-C*04:16', 'HLA-C*04:17', 'HLA-C*04:18', 'HLA-C*04:19', 'HLA-C*04:20', 'HLA-C*04:23', 'HLA-C*04:24',
         'HLA-C*04:25', 'HLA-C*04:26', 'HLA-C*04:27', 'HLA-C*04:28', 'HLA-C*04:29', 'HLA-C*04:30', 'HLA-C*04:31', 'HLA-C*04:32', 'HLA-C*04:33',
         'HLA-C*04:34', 'HLA-C*04:35', 'HLA-C*04:36', 'HLA-C*04:37', 'HLA-C*04:38', 'HLA-C*04:39', 'HLA-C*04:40', 'HLA-C*04:41', 'HLA-C*04:42',
         'HLA-C*04:43', 'HLA-C*04:44', 'HLA-C*04:45', 'HLA-C*04:46', 'HLA-C*04:47', 'HLA-C*04:48', 'HLA-C*04:49', 'HLA-C*04:50', 'HLA-C*04:51',
         'HLA-C*04:52', 'HLA-C*04:53', 'HLA-C*04:54', 'HLA-C*04:55', 'HLA-C*04:56', 'HLA-C*04:57', 'HLA-C*04:58', 'HLA-C*04:60', 'HLA-C*04:61',
         'HLA-C*04:62', 'HLA-C*04:63', 'HLA-C*04:64', 'HLA-C*04:65', 'HLA-C*04:66', 'HLA-C*04:67', 'HLA-C*04:68', 'HLA-C*04:69', 'HLA-C*04:70',
         'HLA-C*05:01', 'HLA-C*05:03', 'HLA-C*05:04', 'HLA-C*05:05', 'HLA-C*05:06', 'HLA-C*05:08', 'HLA-C*05:09', 'HLA-C*05:10', 'HLA-C*05:11',
         'HLA-C*05:12', 'HLA-C*05:13', 'HLA-C*05:14', 'HLA-C*05:15', 'HLA-C*05:16', 'HLA-C*05:17', 'HLA-C*05:18', 'HLA-C*05:19', 'HLA-C*05:20',
         'HLA-C*05:21', 'HLA-C*05:22', 'HLA-C*05:23', 'HLA-C*05:24', 'HLA-C*05:25', 'HLA-C*05:26', 'HLA-C*05:27', 'HLA-C*05:28', 'HLA-C*05:29',
         'HLA-C*05:30', 'HLA-C*05:31', 'HLA-C*05:32', 'HLA-C*05:33', 'HLA-C*05:34', 'HLA-C*05:35', 'HLA-C*05:36', 'HLA-C*05:37', 'HLA-C*05:38',
         'HLA-C*05:39', 'HLA-C*05:40', 'HLA-C*05:41', 'HLA-C*05:42', 'HLA-C*05:43', 'HLA-C*05:44', 'HLA-C*05:45', 'HLA-C*06:02', 'HLA-C*06:03',
         'HLA-C*06:04', 'HLA-C*06:05', 'HLA-C*06:06', 'HLA-C*06:07', 'HLA-C*06:08', 'HLA-C*06:09', 'HLA-C*06:10', 'HLA-C*06:11', 'HLA-C*06:12',
         'HLA-C*06:13', 'HLA-C*06:14', 'HLA-C*06:15', 'HLA-C*06:17', 'HLA-C*06:18', 'HLA-C*06:19', 'HLA-C*06:20', 'HLA-C*06:21', 'HLA-C*06:22',
         'HLA-C*06:23', 'HLA-C*06:24', 'HLA-C*06:25', 'HLA-C*06:26', 'HLA-C*06:27', 'HLA-C*06:28', 'HLA-C*06:29', 'HLA-C*06:30', 'HLA-C*06:31',
         'HLA-C*06:32', 'HLA-C*06:33', 'HLA-C*06:34', 'HLA-C*06:35', 'HLA-C*06:36', 'HLA-C*06:37', 'HLA-C*06:38', 'HLA-C*06:39', 'HLA-C*06:40',
         'HLA-C*06:41', 'HLA-C*06:42', 'HLA-C*06:43', 'HLA-C*06:44', 'HLA-C*06:45', 'HLA-C*07:01', 'HLA-C*07:02', 'HLA-C*07:03', 'HLA-C*07:04',
         'HLA-C*07:05', 'HLA-C*07:06', 'HLA-C*07:07', 'HLA-C*07:08', 'HLA-C*07:09', 'HLA-C*07:10', 'HLA-C*07:100', 'HLA-C*07:101', 'HLA-C*07:102',
         'HLA-C*07:103', 'HLA-C*07:105', 'HLA-C*07:106', 'HLA-C*07:107', 'HLA-C*07:108', 'HLA-C*07:109', 'HLA-C*07:11', 'HLA-C*07:110',
         'HLA-C*07:111', 'HLA-C*07:112', 'HLA-C*07:113', 'HLA-C*07:114', 'HLA-C*07:115', 'HLA-C*07:116', 'HLA-C*07:117', 'HLA-C*07:118',
         'HLA-C*07:119', 'HLA-C*07:12', 'HLA-C*07:120', 'HLA-C*07:122', 'HLA-C*07:123', 'HLA-C*07:124', 'HLA-C*07:125', 'HLA-C*07:126',
         'HLA-C*07:127', 'HLA-C*07:128', 'HLA-C*07:129', 'HLA-C*07:13', 'HLA-C*07:130', 'HLA-C*07:131', 'HLA-C*07:132', 'HLA-C*07:133',
         'HLA-C*07:134', 'HLA-C*07:135', 'HLA-C*07:136', 'HLA-C*07:137', 'HLA-C*07:138', 'HLA-C*07:139', 'HLA-C*07:14', 'HLA-C*07:140',
         'HLA-C*07:141', 'HLA-C*07:142', 'HLA-C*07:143', 'HLA-C*07:144', 'HLA-C*07:145', 'HLA-C*07:146', 'HLA-C*07:147', 'HLA-C*07:148',
         'HLA-C*07:149', 'HLA-C*07:15', 'HLA-C*07:16', 'HLA-C*07:17', 'HLA-C*07:18', 'HLA-C*07:19', 'HLA-C*07:20', 'HLA-C*07:21', 'HLA-C*07:22',
         'HLA-C*07:23', 'HLA-C*07:24', 'HLA-C*07:25', 'HLA-C*07:26', 'HLA-C*07:27', 'HLA-C*07:28', 'HLA-C*07:29', 'HLA-C*07:30', 'HLA-C*07:31',
         'HLA-C*07:35', 'HLA-C*07:36', 'HLA-C*07:37', 'HLA-C*07:38', 'HLA-C*07:39', 'HLA-C*07:40', 'HLA-C*07:41', 'HLA-C*07:42', 'HLA-C*07:43',
         'HLA-C*07:44', 'HLA-C*07:45', 'HLA-C*07:46', 'HLA-C*07:47', 'HLA-C*07:48', 'HLA-C*07:49', 'HLA-C*07:50', 'HLA-C*07:51', 'HLA-C*07:52',
         'HLA-C*07:53', 'HLA-C*07:54', 'HLA-C*07:56', 'HLA-C*07:57', 'HLA-C*07:58', 'HLA-C*07:59', 'HLA-C*07:60', 'HLA-C*07:62', 'HLA-C*07:63',
         'HLA-C*07:64', 'HLA-C*07:65', 'HLA-C*07:66', 'HLA-C*07:67', 'HLA-C*07:68', 'HLA-C*07:69', 'HLA-C*07:70', 'HLA-C*07:71', 'HLA-C*07:72',
         'HLA-C*07:73', 'HLA-C*07:74', 'HLA-C*07:75', 'HLA-C*07:76', 'HLA-C*07:77', 'HLA-C*07:78', 'HLA-C*07:79', 'HLA-C*07:80', 'HLA-C*07:81',
         'HLA-C*07:82', 'HLA-C*07:83', 'HLA-C*07:84', 'HLA-C*07:85', 'HLA-C*07:86', 'HLA-C*07:87', 'HLA-C*07:88', 'HLA-C*07:89', 'HLA-C*07:90',
         'HLA-C*07:91', 'HLA-C*07:92', 'HLA-C*07:93', 'HLA-C*07:94', 'HLA-C*07:95', 'HLA-C*07:96', 'HLA-C*07:97', 'HLA-C*07:99', 'HLA-C*08:01',
         'HLA-C*08:02', 'HLA-C*08:03', 'HLA-C*08:04', 'HLA-C*08:05', 'HLA-C*08:06', 'HLA-C*08:07', 'HLA-C*08:08', 'HLA-C*08:09', 'HLA-C*08:10',
         'HLA-C*08:11', 'HLA-C*08:12', 'HLA-C*08:13', 'HLA-C*08:14', 'HLA-C*08:15', 'HLA-C*08:16', 'HLA-C*08:17', 'HLA-C*08:18', 'HLA-C*08:19',
         'HLA-C*08:20', 'HLA-C*08:21', 'HLA-C*08:22', 'HLA-C*08:23', 'HLA-C*08:24', 'HLA-C*08:25', 'HLA-C*08:27', 'HLA-C*08:28', 'HLA-C*08:29',
         'HLA-C*08:30', 'HLA-C*08:31', 'HLA-C*08:32', 'HLA-C*08:33', 'HLA-C*08:34', 'HLA-C*08:35', 'HLA-C*12:02', 'HLA-C*12:03', 'HLA-C*12:04',
         'HLA-C*12:05', 'HLA-C*12:06', 'HLA-C*12:07', 'HLA-C*12:08', 'HLA-C*12:09', 'HLA-C*12:10', 'HLA-C*12:11', 'HLA-C*12:12', 'HLA-C*12:13',
         'HLA-C*12:14', 'HLA-C*12:15', 'HLA-C*12:16', 'HLA-C*12:17', 'HLA-C*12:18', 'HLA-C*12:19', 'HLA-C*12:20', 'HLA-C*12:21', 'HLA-C*12:22',
         'HLA-C*12:23', 'HLA-C*12:24', 'HLA-C*12:25', 'HLA-C*12:26', 'HLA-C*12:27', 'HLA-C*12:28', 'HLA-C*12:29', 'HLA-C*12:30', 'HLA-C*12:31',
         'HLA-C*12:32', 'HLA-C*12:33', 'HLA-C*12:34', 'HLA-C*12:35', 'HLA-C*12:36', 'HLA-C*12:37', 'HLA-C*12:38', 'HLA-C*12:40', 'HLA-C*12:41',
         'HLA-C*12:43', 'HLA-C*12:44', 'HLA-C*14:02', 'HLA-C*14:03', 'HLA-C*14:04', 'HLA-C*14:05', 'HLA-C*14:06', 'HLA-C*14:08', 'HLA-C*14:09',
         'HLA-C*14:10', 'HLA-C*14:11', 'HLA-C*14:12', 'HLA-C*14:13', 'HLA-C*14:14', 'HLA-C*14:15', 'HLA-C*14:16', 'HLA-C*14:17', 'HLA-C*14:18',
         'HLA-C*14:19', 'HLA-C*14:20', 'HLA-C*15:02', 'HLA-C*15:03', 'HLA-C*15:04', 'HLA-C*15:05', 'HLA-C*15:06', 'HLA-C*15:07', 'HLA-C*15:08',
         'HLA-C*15:09', 'HLA-C*15:10', 'HLA-C*15:11', 'HLA-C*15:12', 'HLA-C*15:13', 'HLA-C*15:15', 'HLA-C*15:16', 'HLA-C*15:17', 'HLA-C*15:18',
         'HLA-C*15:19', 'HLA-C*15:20', 'HLA-C*15:21', 'HLA-C*15:22', 'HLA-C*15:23', 'HLA-C*15:24', 'HLA-C*15:25', 'HLA-C*15:26', 'HLA-C*15:27',
         'HLA-C*15:28', 'HLA-C*15:29', 'HLA-C*15:30', 'HLA-C*15:31', 'HLA-C*15:33', 'HLA-C*15:34', 'HLA-C*15:35', 'HLA-C*16:01', 'HLA-C*16:02',
         'HLA-C*16:04', 'HLA-C*16:06', 'HLA-C*16:07', 'HLA-C*16:08', 'HLA-C*16:09', 'HLA-C*16:10', 'HLA-C*16:11', 'HLA-C*16:12', 'HLA-C*16:13',
         'HLA-C*16:14', 'HLA-C*16:15', 'HLA-C*16:17', 'HLA-C*16:18', 'HLA-C*16:19', 'HLA-C*16:20', 'HLA-C*16:21', 'HLA-C*16:22', 'HLA-C*16:23',
         'HLA-C*16:24', 'HLA-C*16:25', 'HLA-C*16:26', 'HLA-C*17:01', 'HLA-C*17:02', 'HLA-C*17:03', 'HLA-C*17:04', 'HLA-C*17:05', 'HLA-C*17:06',
         'HLA-C*17:07', 'HLA-C*18:01', 'HLA-C*18:02', 'HLA-C*18:03', 'HLA-E*01:01', 'HLA-G*01:01', 'HLA-G*01:02', 'HLA-G*01:03', 'HLA-G*01:04',
         'HLA-G*01:06', 'HLA-G*01:07', 'HLA-G*01:08', 'HLA-G*01:09',
         'H-2-Db', 'H-2-Dd', 'H-2-Kb', 'H-2-Kd', 'H-2-Kk', 'H-2-Ld', "H-2-Qa1", "H-2-Qa2"])

    @property
    def version(self):
        """The version of the predictor"""
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
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_2_8]
            for i, a in enumerate(alleles):
                if row[ScoreIndex.NETMHCPAN_2_8 + i * Offset.NETMHCPAN_2_8] != "1-log50k":     # Avoid header column, only access raw and rank scores
                    scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_2_8 + i * Offset.NETMHCPAN_2_8])
                    ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_2_8 + i * Offset.NETMHCPAN_2_8])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result


class NetMHCpan_3_0(NetMHCpan_2_8):
    """
        Implements the NetMHC binding version 3.0
        Supported  MHC alleles currently only restricted to HLA alleles.

    .. note::

        Nielsen, M., & Andreatta, M. (2016).
        NetMHCpan-3.0; improved prediction of binding to MHC class I molecules integrating information from multiple
        receptor and peptide length datasets. Genome Medicine, 8(1), 1.
    """

    __version = "3.0"
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"


    @property
    def version(self):
        return self.__version

    @property
    def command(self):
        return self.__command

    def parse_external_result(self, file):
        """
        Parses external results and returns the result

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_3_0]
            for i, a in enumerate(alleles):
                if row[ScoreIndex.NETMHCPAN_3_0 + i * Offset.NETMHCPAN_3_0] != "1-log50k":     # Avoid header column, only access raw and rank scores
                    scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_3_0 + i * Offset.NETMHCPAN_3_0])
                    ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_3_0 + i * Offset.NETMHCPAN_3_0])
                    
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result


class NetMHCpan_4_0(NetMHCpan_3_0):
    """
        Implements the NetMHC binding version 4.0
        Supported  MHC alleles currently only restricted to HLA alleles.
    """
    __version = "4.0"
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"
    @property
    def version(self):
        return self.__version

    @property
    def command(self):
        return self.__command

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_4_0]
            for i, a in enumerate(alleles):
                if row[ScoreIndex.NETMHCPAN_4_0 + i * Offset.NETMHCPAN_4_0] != "1-log50k":     # Avoid header column, only access raw and rank scores
                    scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_4_0 + i * Offset.NETMHCPAN_4_0])
                    ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_4_0 + i * Offset.NETMHCPAN_4_0])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result



class NetMHCpan_4_1(NetMHCpan_4_0):
    """
        Implements the NetMHC binding version 4.1
        Supported  MHC alleles currently only restricted to HLA alleles.
    """
    __version = "4.1"
    __command = "netMHCpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"
    @property
    def version(self):
        return self.__version

    @property
    def command(self):
        return self.__command

    __alleles = frozenset(['H-2-Db','H-2-Dd','H-2-Kb','H-2-Kd','H-2-Kk','H-2-Ld','H-2-Lq','H-2-Qa1','H-2-Qa2','H2-Db','H2-Dd','H2-Kb','H2-Kd','H2-Kk','H2-Ld','H2-Qa1','H2-Qa2','HLA-A*0101','HLA-A*0102','HLA-A*0103','HLA-A*0104','HLA-A*0106','HLA-A*0107','HLA-A*0108','HLA-A*0109','HLA-A*0110','HLA-A*0111','HLA-A*0112','HLA-A*0113','HLA-A*0114','HLA-A*0115','HLA-A*0117','HLA-A*0118','HLA-A*0119','HLA-A*0120','HLA-A*0121','HLA-A*0122','HLA-A*0123','HLA-A*0124','HLA-A*0125','HLA-A*0126','HLA-A*01:01','HLA-A*01:02','HLA-A*01:03','HLA-A*01:06','HLA-A*01:07','HLA-A*01:08','HLA-A*01:09','HLA-A*01:10','HLA-A*01:12','HLA-A*01:13','HLA-A*01:14','HLA-A*01:17','HLA-A*01:19','HLA-A*01:20','HLA-A*01:21','HLA-A*01:23','HLA-A*01:24','HLA-A*01:25','HLA-A*01:26','HLA-A*01:28','HLA-A*01:29','HLA-A*01:30','HLA-A*01:32','HLA-A*01:33','HLA-A*01:35','HLA-A*01:36','HLA-A*01:37','HLA-A*01:38','HLA-A*01:39','HLA-A*01:40','HLA-A*01:41','HLA-A*01:42','HLA-A*01:43','HLA-A*01:44','HLA-A*01:45','HLA-A*01:46','HLA-A*01:47','HLA-A*01:48','HLA-A*01:49','HLA-A*01:50','HLA-A*01:51','HLA-A*01:54','HLA-A*01:55','HLA-A*01:58','HLA-A*01:59','HLA-A*01:60','HLA-A*01:61','HLA-A*01:62','HLA-A*01:63','HLA-A*01:64','HLA-A*01:65','HLA-A*01:66','HLA-A*0201','HLA-A*0202','HLA-A*0203','HLA-A*0204','HLA-A*0205','HLA-A*0206','HLA-A*0207','HLA-A*0208','HLA-A*0209','HLA-A*0210','HLA-A*0211','HLA-A*0212','HLA-A*0213','HLA-A*0214','HLA-A*0215','HLA-A*0216','HLA-A*0217','HLA-A*0218','HLA-A*0219','HLA-A*0220','HLA-A*0221','HLA-A*0222','HLA-A*0224','HLA-A*0225','HLA-A*0226','HLA-A*0227','HLA-A*0228','HLA-A*0229','HLA-A*0230','HLA-A*0231','HLA-A*0233','HLA-A*0234','HLA-A*0235','HLA-A*0236','HLA-A*0237','HLA-A*0238','HLA-A*0239','HLA-A*0240','HLA-A*0241','HLA-A*0242','HLA-A*0243','HLA-A*0244','HLA-A*0245','HLA-A*0246','HLA-A*0247','HLA-A*0248','HLA-A*0249','HLA-A*0250','HLA-A*0251','HLA-A*0252','HLA-A*0254','HLA-A*0255','HLA-A*0256','HLA-A*0257','HLA-A*0258','HLA-A*0259','HLA-A*0260','HLA-A*0261','HLA-A*0262','HLA-A*0263','HLA-A*0264','HLA-A*0265','HLA-A*0266','HLA-A*0267','HLA-A*0268','HLA-A*0269','HLA-A*0270','HLA-A*0271','HLA-A*0272','HLA-A*0273','HLA-A*0274','HLA-A*0275','HLA-A*0276','HLA-A*0277','HLA-A*0278','HLA-A*0279','HLA-A*0280','HLA-A*0281','HLA-A*0283','HLA-A*0284','HLA-A*0285','HLA-A*0286','HLA-A*0287','HLA-A*0289','HLA-A*0290','HLA-A*0291','HLA-A*0292','HLA-A*0293','HLA-A*0295','HLA-A*0296','HLA-A*0297','HLA-A*0299','HLA-A*02:01','HLA-A*02:02','HLA-A*02:03','HLA-A*02:04','HLA-A*02:05','HLA-A*02:06','HLA-A*02:07','HLA-A*02:08','HLA-A*02:09','HLA-A*02:10','HLA-A*02:101','HLA-A*02:102','HLA-A*02:103','HLA-A*02:104','HLA-A*02:105','HLA-A*02:106','HLA-A*02:107','HLA-A*02:108','HLA-A*02:109','HLA-A*02:11','HLA-A*02:110','HLA-A*02:111','HLA-A*02:112','HLA-A*02:114','HLA-A*02:115','HLA-A*02:116','HLA-A*02:117','HLA-A*02:118','HLA-A*02:119','HLA-A*02:12','HLA-A*02:120','HLA-A*02:121','HLA-A*02:122','HLA-A*02:123','HLA-A*02:124','HLA-A*02:126','HLA-A*02:127','HLA-A*02:128','HLA-A*02:129','HLA-A*02:13','HLA-A*02:130','HLA-A*02:131','HLA-A*02:132','HLA-A*02:133','HLA-A*02:134','HLA-A*02:135','HLA-A*02:136','HLA-A*02:137','HLA-A*02:138','HLA-A*02:139','HLA-A*02:14','HLA-A*02:140','HLA-A*02:141','HLA-A*02:142','HLA-A*02:143','HLA-A*02:144','HLA-A*02:145','HLA-A*02:146','HLA-A*02:147','HLA-A*02:148','HLA-A*02:149','HLA-A*02:150','HLA-A*02:151','HLA-A*02:152','HLA-A*02:153','HLA-A*02:154','HLA-A*02:155','HLA-A*02:156','HLA-A*02:157','HLA-A*02:158','HLA-A*02:159','HLA-A*02:16','HLA-A*02:160','HLA-A*02:161','HLA-A*02:162','HLA-A*02:163','HLA-A*02:164','HLA-A*02:165','HLA-A*02:166','HLA-A*02:167','HLA-A*02:168','HLA-A*02:169','HLA-A*02:17','HLA-A*02:170','HLA-A*02:171','HLA-A*02:172','HLA-A*02:173','HLA-A*02:174','HLA-A*02:175','HLA-A*02:176','HLA-A*02:177','HLA-A*02:178','HLA-A*02:179','HLA-A*02:18','HLA-A*02:180','HLA-A*02:181','HLA-A*02:182','HLA-A*02:183','HLA-A*02:184','HLA-A*02:185','HLA-A*02:186','HLA-A*02:187','HLA-A*02:188','HLA-A*02:189','HLA-A*02:19','HLA-A*02:190','HLA-A*02:191','HLA-A*02:192','HLA-A*02:193','HLA-A*02:194','HLA-A*02:195','HLA-A*02:196','HLA-A*02:197','HLA-A*02:198','HLA-A*02:199','HLA-A*02:20','HLA-A*02:200','HLA-A*02:201','HLA-A*02:202','HLA-A*02:203','HLA-A*02:204','HLA-A*02:205','HLA-A*02:206','HLA-A*02:207','HLA-A*02:208','HLA-A*02:209','HLA-A*02:21','HLA-A*02:210','HLA-A*02:211','HLA-A*02:212','HLA-A*02:213','HLA-A*02:214','HLA-A*02:215','HLA-A*02:216','HLA-A*02:217','HLA-A*02:218','HLA-A*02:219','HLA-A*02:22','HLA-A*02:220','HLA-A*02:221','HLA-A*02:224','HLA-A*02:228','HLA-A*02:229','HLA-A*02:230','HLA-A*02:231','HLA-A*02:232','HLA-A*02:233','HLA-A*02:234','HLA-A*02:235','HLA-A*02:236','HLA-A*02:237','HLA-A*02:238','HLA-A*02:239','HLA-A*02:24','HLA-A*02:240','HLA-A*02:241','HLA-A*02:242','HLA-A*02:243','HLA-A*02:244','HLA-A*02:245','HLA-A*02:246','HLA-A*02:247','HLA-A*02:248','HLA-A*02:249','HLA-A*02:25','HLA-A*02:251','HLA-A*02:252','HLA-A*02:253','HLA-A*02:254','HLA-A*02:255','HLA-A*02:256','HLA-A*02:257','HLA-A*02:258','HLA-A*02:259','HLA-A*02:26','HLA-A*02:260','HLA-A*02:261','HLA-A*02:262','HLA-A*02:263','HLA-A*02:264','HLA-A*02:265','HLA-A*02:266','HLA-A*02:27','HLA-A*02:28','HLA-A*02:29','HLA-A*02:30','HLA-A*02:31','HLA-A*02:33','HLA-A*02:34','HLA-A*02:35','HLA-A*02:36','HLA-A*02:37','HLA-A*02:38','HLA-A*02:39','HLA-A*02:40','HLA-A*02:41','HLA-A*02:42','HLA-A*02:44','HLA-A*02:45','HLA-A*02:46','HLA-A*02:47','HLA-A*02:48','HLA-A*02:49','HLA-A*02:50','HLA-A*02:51','HLA-A*02:52','HLA-A*02:54','HLA-A*02:55','HLA-A*02:56','HLA-A*02:57','HLA-A*02:58','HLA-A*02:59','HLA-A*02:60','HLA-A*02:61','HLA-A*02:62','HLA-A*02:63','HLA-A*02:64','HLA-A*02:65','HLA-A*02:66','HLA-A*02:67','HLA-A*02:68','HLA-A*02:69','HLA-A*02:70','HLA-A*02:71','HLA-A*02:72','HLA-A*02:73','HLA-A*02:74','HLA-A*02:75','HLA-A*02:76','HLA-A*02:77','HLA-A*02:78','HLA-A*02:79','HLA-A*02:80','HLA-A*02:81','HLA-A*02:84','HLA-A*02:85','HLA-A*02:86','HLA-A*02:87','HLA-A*02:89','HLA-A*02:90','HLA-A*02:91','HLA-A*02:92','HLA-A*02:93','HLA-A*02:95','HLA-A*02:96','HLA-A*02:97','HLA-A*02:99','HLA-A*0301','HLA-A*0302','HLA-A*0303','HLA-A*0304','HLA-A*0305','HLA-A*0306','HLA-A*0307','HLA-A*0308','HLA-A*0309','HLA-A*0310','HLA-A*0312','HLA-A*0313','HLA-A*0314','HLA-A*0315','HLA-A*0316','HLA-A*0317','HLA-A*0318','HLA-A*0319','HLA-A*0320','HLA-A*0321','HLA-A*0322','HLA-A*0323','HLA-A*0324','HLA-A*0325','HLA-A*0326','HLA-A*0327','HLA-A*0328','HLA-A*0329','HLA-A*0330','HLA-A*03:01','HLA-A*03:02','HLA-A*03:04','HLA-A*03:05','HLA-A*03:06','HLA-A*03:07','HLA-A*03:08','HLA-A*03:09','HLA-A*03:10','HLA-A*03:12','HLA-A*03:13','HLA-A*03:14','HLA-A*03:15','HLA-A*03:16','HLA-A*03:17','HLA-A*03:18','HLA-A*03:19','HLA-A*03:20','HLA-A*03:22','HLA-A*03:23','HLA-A*03:24','HLA-A*03:25','HLA-A*03:26','HLA-A*03:27','HLA-A*03:28','HLA-A*03:29','HLA-A*03:30','HLA-A*03:31','HLA-A*03:32','HLA-A*03:33','HLA-A*03:34','HLA-A*03:35','HLA-A*03:37','HLA-A*03:38','HLA-A*03:39','HLA-A*03:40','HLA-A*03:41','HLA-A*03:42','HLA-A*03:43','HLA-A*03:44','HLA-A*03:45','HLA-A*03:46','HLA-A*03:47','HLA-A*03:48','HLA-A*03:49','HLA-A*03:50','HLA-A*03:51','HLA-A*03:52','HLA-A*03:53','HLA-A*03:54','HLA-A*03:55','HLA-A*03:56','HLA-A*03:57','HLA-A*03:58','HLA-A*03:59','HLA-A*03:60','HLA-A*03:61','HLA-A*03:62','HLA-A*03:63','HLA-A*03:64','HLA-A*03:65','HLA-A*03:66','HLA-A*03:67','HLA-A*03:70','HLA-A*03:71','HLA-A*03:72','HLA-A*03:73','HLA-A*03:74','HLA-A*03:75','HLA-A*03:76','HLA-A*03:77','HLA-A*03:78','HLA-A*03:79','HLA-A*03:80','HLA-A*03:81','HLA-A*03:82','HLA-A*1101','HLA-A*1102','HLA-A*1103','HLA-A*1104','HLA-A*1105','HLA-A*1106','HLA-A*1107','HLA-A*1108','HLA-A*1109','HLA-A*1110','HLA-A*1111','HLA-A*1112','HLA-A*1113','HLA-A*1114','HLA-A*1115','HLA-A*1116','HLA-A*1117','HLA-A*1118','HLA-A*1119','HLA-A*1120','HLA-A*1121','HLA-A*1122','HLA-A*1123','HLA-A*1124','HLA-A*1125','HLA-A*1126','HLA-A*1127','HLA-A*1128','HLA-A*1129','HLA-A*1130','HLA-A*1131','HLA-A*1132','HLA-A*11:01','HLA-A*11:02','HLA-A*11:03','HLA-A*11:04','HLA-A*11:05','HLA-A*11:06','HLA-A*11:07','HLA-A*11:08','HLA-A*11:09','HLA-A*11:10','HLA-A*11:11','HLA-A*11:12','HLA-A*11:13','HLA-A*11:14','HLA-A*11:15','HLA-A*11:16','HLA-A*11:17','HLA-A*11:18','HLA-A*11:19','HLA-A*11:20','HLA-A*11:22','HLA-A*11:23','HLA-A*11:24','HLA-A*11:25','HLA-A*11:26','HLA-A*11:27','HLA-A*11:29','HLA-A*11:30','HLA-A*11:31','HLA-A*11:32','HLA-A*11:33','HLA-A*11:34','HLA-A*11:35','HLA-A*11:36','HLA-A*11:37','HLA-A*11:38','HLA-A*11:39','HLA-A*11:40','HLA-A*11:41','HLA-A*11:42','HLA-A*11:43','HLA-A*11:44','HLA-A*11:45','HLA-A*11:46','HLA-A*11:47','HLA-A*11:48','HLA-A*11:49','HLA-A*11:51','HLA-A*11:53','HLA-A*11:54','HLA-A*11:55','HLA-A*11:56','HLA-A*11:57','HLA-A*11:58','HLA-A*11:59','HLA-A*11:60','HLA-A*11:61','HLA-A*11:62','HLA-A*11:63','HLA-A*11:64','HLA-A*2301','HLA-A*2302','HLA-A*2303','HLA-A*2304','HLA-A*2305','HLA-A*2306','HLA-A*2307','HLA-A*2309','HLA-A*2310','HLA-A*2312','HLA-A*2313','HLA-A*2314','HLA-A*2315','HLA-A*2316','HLA-A*23:01','HLA-A*23:02','HLA-A*23:03','HLA-A*23:04','HLA-A*23:05','HLA-A*23:06','HLA-A*23:09','HLA-A*23:10','HLA-A*23:12','HLA-A*23:13','HLA-A*23:14','HLA-A*23:15','HLA-A*23:16','HLA-A*23:17','HLA-A*23:18','HLA-A*23:20','HLA-A*23:21','HLA-A*23:22','HLA-A*23:23','HLA-A*23:24','HLA-A*23:25','HLA-A*23:26','HLA-A*2402','HLA-A*2403','HLA-A*2404','HLA-A*2405','HLA-A*2406','HLA-A*2407','HLA-A*2408','HLA-A*2409','HLA-A*2410','HLA-A*2411','HLA-A*2413','HLA-A*2414','HLA-A*2415','HLA-A*2417','HLA-A*2418','HLA-A*2419','HLA-A*2420','HLA-A*2421','HLA-A*2422','HLA-A*2423','HLA-A*2424','HLA-A*2425','HLA-A*2426','HLA-A*2427','HLA-A*2428','HLA-A*2429','HLA-A*2430','HLA-A*2431','HLA-A*2432','HLA-A*2433','HLA-A*2434','HLA-A*2435','HLA-A*2437','HLA-A*2438','HLA-A*2439','HLA-A*2440','HLA-A*2441','HLA-A*2442','HLA-A*2443','HLA-A*2444','HLA-A*2446','HLA-A*2447','HLA-A*2449','HLA-A*2450','HLA-A*2451','HLA-A*2452','HLA-A*2453','HLA-A*2454','HLA-A*2455','HLA-A*2456','HLA-A*2457','HLA-A*2458','HLA-A*2459','HLA-A*2461','HLA-A*2462','HLA-A*2463','HLA-A*2464','HLA-A*2465','HLA-A*2466','HLA-A*2467','HLA-A*2468','HLA-A*2469','HLA-A*2470','HLA-A*2471','HLA-A*2472','HLA-A*2473','HLA-A*2474','HLA-A*2475','HLA-A*2476','HLA-A*2477','HLA-A*2478','HLA-A*2479','HLA-A*24:02','HLA-A*24:03','HLA-A*24:04','HLA-A*24:05','HLA-A*24:06','HLA-A*24:07','HLA-A*24:08','HLA-A*24:10','HLA-A*24:100','HLA-A*24:101','HLA-A*24:102','HLA-A*24:103','HLA-A*24:104','HLA-A*24:105','HLA-A*24:106','HLA-A*24:107','HLA-A*24:108','HLA-A*24:109','HLA-A*24:110','HLA-A*24:111','HLA-A*24:112','HLA-A*24:113','HLA-A*24:114','HLA-A*24:115','HLA-A*24:116','HLA-A*24:117','HLA-A*24:118','HLA-A*24:119','HLA-A*24:120','HLA-A*24:121','HLA-A*24:122','HLA-A*24:123','HLA-A*24:124','HLA-A*24:125','HLA-A*24:126','HLA-A*24:127','HLA-A*24:128','HLA-A*24:129','HLA-A*24:13','HLA-A*24:130','HLA-A*24:131','HLA-A*24:133','HLA-A*24:134','HLA-A*24:135','HLA-A*24:136','HLA-A*24:137','HLA-A*24:138','HLA-A*24:139','HLA-A*24:14','HLA-A*24:140','HLA-A*24:141','HLA-A*24:142','HLA-A*24:143','HLA-A*24:144','HLA-A*24:15','HLA-A*24:17','HLA-A*24:18','HLA-A*24:19','HLA-A*24:20','HLA-A*24:21','HLA-A*24:22','HLA-A*24:23','HLA-A*24:24','HLA-A*24:25','HLA-A*24:26','HLA-A*24:27','HLA-A*24:28','HLA-A*24:29','HLA-A*24:30','HLA-A*24:31','HLA-A*24:32','HLA-A*24:33','HLA-A*24:34','HLA-A*24:35','HLA-A*24:37','HLA-A*24:38','HLA-A*24:39','HLA-A*24:41','HLA-A*24:42','HLA-A*24:43','HLA-A*24:44','HLA-A*24:46','HLA-A*24:47','HLA-A*24:49','HLA-A*24:50','HLA-A*24:51','HLA-A*24:52','HLA-A*24:53','HLA-A*24:54','HLA-A*24:55','HLA-A*24:56','HLA-A*24:57','HLA-A*24:58','HLA-A*24:59','HLA-A*24:61','HLA-A*24:62','HLA-A*24:63','HLA-A*24:64','HLA-A*24:66','HLA-A*24:67','HLA-A*24:68','HLA-A*24:69','HLA-A*24:70','HLA-A*24:71','HLA-A*24:72','HLA-A*24:73','HLA-A*24:74','HLA-A*24:75','HLA-A*24:76','HLA-A*24:77','HLA-A*24:78','HLA-A*24:79','HLA-A*24:80','HLA-A*24:81','HLA-A*24:82','HLA-A*24:85','HLA-A*24:87','HLA-A*24:88','HLA-A*24:89','HLA-A*24:91','HLA-A*24:92','HLA-A*24:93','HLA-A*24:94','HLA-A*24:95','HLA-A*24:96','HLA-A*24:97','HLA-A*24:98','HLA-A*24:99','HLA-A*2501','HLA-A*2502','HLA-A*2503','HLA-A*2504','HLA-A*2505','HLA-A*2506','HLA-A*25:01','HLA-A*25:02','HLA-A*25:03','HLA-A*25:04','HLA-A*25:05','HLA-A*25:06','HLA-A*25:07','HLA-A*25:08','HLA-A*25:09','HLA-A*25:10','HLA-A*25:11','HLA-A*25:13','HLA-A*2601','HLA-A*2602','HLA-A*2603','HLA-A*2604','HLA-A*2605','HLA-A*2606','HLA-A*2607','HLA-A*2608','HLA-A*2609','HLA-A*2610','HLA-A*2611','HLA-A*2612','HLA-A*2613','HLA-A*2614','HLA-A*2615','HLA-A*2616','HLA-A*2617','HLA-A*2618','HLA-A*2619','HLA-A*2620','HLA-A*2621','HLA-A*2622','HLA-A*2623','HLA-A*2624','HLA-A*2626','HLA-A*2627','HLA-A*2628','HLA-A*2629','HLA-A*2630','HLA-A*2631','HLA-A*2632','HLA-A*2633','HLA-A*2634','HLA-A*2635','HLA-A*26:01','HLA-A*26:02','HLA-A*26:03','HLA-A*26:04','HLA-A*26:05','HLA-A*26:06','HLA-A*26:07','HLA-A*26:08','HLA-A*26:09','HLA-A*26:10','HLA-A*26:12','HLA-A*26:13','HLA-A*26:14','HLA-A*26:15','HLA-A*26:16','HLA-A*26:17','HLA-A*26:18','HLA-A*26:19','HLA-A*26:20','HLA-A*26:21','HLA-A*26:22','HLA-A*26:23','HLA-A*26:24','HLA-A*26:26','HLA-A*26:27','HLA-A*26:28','HLA-A*26:29','HLA-A*26:30','HLA-A*26:31','HLA-A*26:32','HLA-A*26:33','HLA-A*26:34','HLA-A*26:35','HLA-A*26:36','HLA-A*26:37','HLA-A*26:38','HLA-A*26:39','HLA-A*26:40','HLA-A*26:41','HLA-A*26:42','HLA-A*26:43','HLA-A*26:45','HLA-A*26:46','HLA-A*26:47','HLA-A*26:48','HLA-A*26:49','HLA-A*26:50','HLA-A*2901','HLA-A*2902','HLA-A*2903','HLA-A*2904','HLA-A*2905','HLA-A*2906','HLA-A*2907','HLA-A*2909','HLA-A*2910','HLA-A*2911','HLA-A*2912','HLA-A*2913','HLA-A*2914','HLA-A*2915','HLA-A*2916','HLA-A*29:01','HLA-A*29:02','HLA-A*29:03','HLA-A*29:04','HLA-A*29:05','HLA-A*29:06','HLA-A*29:07','HLA-A*29:09','HLA-A*29:10','HLA-A*29:11','HLA-A*29:12','HLA-A*29:13','HLA-A*29:14','HLA-A*29:15','HLA-A*29:16','HLA-A*29:17','HLA-A*29:18','HLA-A*29:19','HLA-A*29:20','HLA-A*29:21','HLA-A*29:22','HLA-A*3001','HLA-A*3002','HLA-A*3003','HLA-A*3004','HLA-A*3006','HLA-A*3007','HLA-A*3008','HLA-A*3009','HLA-A*3010','HLA-A*3011','HLA-A*3012','HLA-A*3013','HLA-A*3014','HLA-A*3015','HLA-A*3016','HLA-A*3017','HLA-A*3018','HLA-A*3019','HLA-A*3020','HLA-A*3021','HLA-A*3022','HLA-A*30:01','HLA-A*30:02','HLA-A*30:03','HLA-A*30:04','HLA-A*30:06','HLA-A*30:07','HLA-A*30:08','HLA-A*30:09','HLA-A*30:10','HLA-A*30:11','HLA-A*30:12','HLA-A*30:13','HLA-A*30:15','HLA-A*30:16','HLA-A*30:17','HLA-A*30:18','HLA-A*30:19','HLA-A*30:20','HLA-A*30:22','HLA-A*30:23','HLA-A*30:24','HLA-A*30:25','HLA-A*30:26','HLA-A*30:28','HLA-A*30:29','HLA-A*30:30','HLA-A*30:31','HLA-A*30:32','HLA-A*30:33','HLA-A*30:34','HLA-A*30:35','HLA-A*30:36','HLA-A*30:37','HLA-A*30:38','HLA-A*30:39','HLA-A*30:40','HLA-A*30:41','HLA-A*3101','HLA-A*3102','HLA-A*3103','HLA-A*3104','HLA-A*3105','HLA-A*3106','HLA-A*3107','HLA-A*3108','HLA-A*3109','HLA-A*3110','HLA-A*3111','HLA-A*3112','HLA-A*3113','HLA-A*3114','HLA-A*3115','HLA-A*3116','HLA-A*3117','HLA-A*3118','HLA-A*31:01','HLA-A*31:02','HLA-A*31:03','HLA-A*31:04','HLA-A*31:05','HLA-A*31:06','HLA-A*31:07','HLA-A*31:08','HLA-A*31:09','HLA-A*31:10','HLA-A*31:11','HLA-A*31:12','HLA-A*31:13','HLA-A*31:15','HLA-A*31:16','HLA-A*31:17','HLA-A*31:18','HLA-A*31:19','HLA-A*31:20','HLA-A*31:21','HLA-A*31:22','HLA-A*31:23','HLA-A*31:24','HLA-A*31:25','HLA-A*31:26','HLA-A*31:27','HLA-A*31:28','HLA-A*31:29','HLA-A*31:30','HLA-A*31:31','HLA-A*31:32','HLA-A*31:33','HLA-A*31:34','HLA-A*31:35','HLA-A*31:36','HLA-A*31:37','HLA-A*3201','HLA-A*3202','HLA-A*3203','HLA-A*3204','HLA-A*3205','HLA-A*3206','HLA-A*3207','HLA-A*3208','HLA-A*3209','HLA-A*3210','HLA-A*3211','HLA-A*3212','HLA-A*3213','HLA-A*3214','HLA-A*3215','HLA-A*32:01','HLA-A*32:02','HLA-A*32:03','HLA-A*32:04','HLA-A*32:05','HLA-A*32:06','HLA-A*32:07','HLA-A*32:08','HLA-A*32:09','HLA-A*32:10','HLA-A*32:12','HLA-A*32:13','HLA-A*32:14','HLA-A*32:15','HLA-A*32:16','HLA-A*32:17','HLA-A*32:18','HLA-A*32:20','HLA-A*32:21','HLA-A*32:22','HLA-A*32:23','HLA-A*32:24','HLA-A*32:25','HLA-A*3301','HLA-A*3303','HLA-A*3304','HLA-A*3305','HLA-A*3306','HLA-A*3307','HLA-A*3308','HLA-A*3309','HLA-A*3310','HLA-A*3311','HLA-A*3312','HLA-A*3313','HLA-A*33:01','HLA-A*33:03','HLA-A*33:04','HLA-A*33:05','HLA-A*33:06','HLA-A*33:07','HLA-A*33:08','HLA-A*33:09','HLA-A*33:10','HLA-A*33:11','HLA-A*33:12','HLA-A*33:13','HLA-A*33:14','HLA-A*33:15','HLA-A*33:16','HLA-A*33:17','HLA-A*33:18','HLA-A*33:19','HLA-A*33:20','HLA-A*33:21','HLA-A*33:22','HLA-A*33:23','HLA-A*33:24','HLA-A*33:25','HLA-A*33:26','HLA-A*33:27','HLA-A*33:28','HLA-A*33:29','HLA-A*33:30','HLA-A*33:31','HLA-A*3401','HLA-A*3402','HLA-A*3403','HLA-A*3404','HLA-A*3405','HLA-A*3406','HLA-A*3407','HLA-A*3408','HLA-A*34:01','HLA-A*34:02','HLA-A*34:03','HLA-A*34:04','HLA-A*34:05','HLA-A*34:06','HLA-A*34:07','HLA-A*34:08','HLA-A*3601','HLA-A*3602','HLA-A*3603','HLA-A*3604','HLA-A*36:01','HLA-A*36:02','HLA-A*36:03','HLA-A*36:04','HLA-A*36:05','HLA-A*4301','HLA-A*43:01','HLA-A*6601','HLA-A*6602','HLA-A*6603','HLA-A*6604','HLA-A*6605','HLA-A*6606','HLA-A*66:01','HLA-A*66:02','HLA-A*66:03','HLA-A*66:04','HLA-A*66:05','HLA-A*66:06','HLA-A*66:07','HLA-A*66:08','HLA-A*66:09','HLA-A*66:10','HLA-A*66:11','HLA-A*66:12','HLA-A*66:13','HLA-A*66:14','HLA-A*66:15','HLA-A*6801','HLA-A*6802','HLA-A*6803','HLA-A*6804','HLA-A*6805','HLA-A*6806','HLA-A*6807','HLA-A*6808','HLA-A*6809','HLA-A*6810','HLA-A*6812','HLA-A*6813','HLA-A*6814','HLA-A*6815','HLA-A*6816','HLA-A*6817','HLA-A*6819','HLA-A*6820','HLA-A*6821','HLA-A*6822','HLA-A*6823','HLA-A*6824','HLA-A*6825','HLA-A*6826','HLA-A*6827','HLA-A*6828','HLA-A*6829','HLA-A*6830','HLA-A*6831','HLA-A*6832','HLA-A*6833','HLA-A*6834','HLA-A*6835','HLA-A*6836','HLA-A*6837','HLA-A*6838','HLA-A*6839','HLA-A*6840','HLA-A*68:01','HLA-A*68:02','HLA-A*68:03','HLA-A*68:04','HLA-A*68:05','HLA-A*68:06','HLA-A*68:07','HLA-A*68:08','HLA-A*68:09','HLA-A*68:10','HLA-A*68:12','HLA-A*68:13','HLA-A*68:14','HLA-A*68:15','HLA-A*68:16','HLA-A*68:17','HLA-A*68:19','HLA-A*68:20','HLA-A*68:21','HLA-A*68:22','HLA-A*68:23','HLA-A*68:24','HLA-A*68:25','HLA-A*68:26','HLA-A*68:27','HLA-A*68:28','HLA-A*68:29','HLA-A*68:30','HLA-A*68:31','HLA-A*68:32','HLA-A*68:33','HLA-A*68:34','HLA-A*68:35','HLA-A*68:36','HLA-A*68:37','HLA-A*68:38','HLA-A*68:39','HLA-A*68:40','HLA-A*68:41','HLA-A*68:42','HLA-A*68:43','HLA-A*68:44','HLA-A*68:45','HLA-A*68:46','HLA-A*68:47','HLA-A*68:48','HLA-A*68:50','HLA-A*68:51','HLA-A*68:52','HLA-A*68:53','HLA-A*68:54','HLA-A*6901','HLA-A*69:01','HLA-A*7401','HLA-A*7402','HLA-A*7403','HLA-A*7404','HLA-A*7405','HLA-A*7406','HLA-A*7407','HLA-A*7408','HLA-A*7409','HLA-A*7410','HLA-A*7411','HLA-A*74:01','HLA-A*74:02','HLA-A*74:03','HLA-A*74:04','HLA-A*74:05','HLA-A*74:06','HLA-A*74:07','HLA-A*74:08','HLA-A*74:09','HLA-A*74:10','HLA-A*74:11','HLA-A*74:13','HLA-A*8001','HLA-A*80:01','HLA-A*80:02','HLA-A*9201','HLA-A*9202','HLA-A*9203','HLA-A*9204','HLA-A*9205','HLA-A*9206','HLA-A*9207','HLA-A*9208','HLA-A*9209','HLA-A*9210','HLA-A*9211','HLA-A*9212','HLA-A*9214','HLA-A*9215','HLA-A*9216','HLA-A*9217','HLA-A*9218','HLA-A*9219','HLA-A*9220','HLA-A*9221','HLA-A*9222','HLA-A*9223','HLA-A*9224','HLA-A*9226','HLA-B*0702','HLA-B*0703','HLA-B*0704','HLA-B*0705','HLA-B*0706','HLA-B*0707','HLA-B*0708','HLA-B*0709','HLA-B*0710','HLA-B*0711','HLA-B*0712','HLA-B*0713','HLA-B*0714','HLA-B*0715','HLA-B*0716','HLA-B*0717','HLA-B*0718','HLA-B*0719','HLA-B*0720','HLA-B*0721','HLA-B*0722','HLA-B*0723','HLA-B*0724','HLA-B*0725','HLA-B*0726','HLA-B*0727','HLA-B*0728','HLA-B*0729','HLA-B*0730','HLA-B*0731','HLA-B*0732','HLA-B*0733','HLA-B*0734','HLA-B*0735','HLA-B*0736','HLA-B*0737','HLA-B*0738','HLA-B*0739','HLA-B*0740','HLA-B*0741','HLA-B*0742','HLA-B*0743','HLA-B*0744','HLA-B*0745','HLA-B*0746','HLA-B*0747','HLA-B*0748','HLA-B*0749','HLA-B*0750','HLA-B*0751','HLA-B*0752','HLA-B*0753','HLA-B*0754','HLA-B*0755','HLA-B*0756','HLA-B*0757','HLA-B*0758','HLA-B*07:02','HLA-B*07:03','HLA-B*07:04','HLA-B*07:05','HLA-B*07:06','HLA-B*07:07','HLA-B*07:08','HLA-B*07:09','HLA-B*07:10','HLA-B*07:100','HLA-B*07:101','HLA-B*07:102','HLA-B*07:103','HLA-B*07:104','HLA-B*07:105','HLA-B*07:106','HLA-B*07:107','HLA-B*07:108','HLA-B*07:109','HLA-B*07:11','HLA-B*07:110','HLA-B*07:112','HLA-B*07:113','HLA-B*07:114','HLA-B*07:115','HLA-B*07:12','HLA-B*07:13','HLA-B*07:14','HLA-B*07:15','HLA-B*07:16','HLA-B*07:17','HLA-B*07:18','HLA-B*07:19','HLA-B*07:20','HLA-B*07:21','HLA-B*07:22','HLA-B*07:23','HLA-B*07:24','HLA-B*07:25','HLA-B*07:26','HLA-B*07:27','HLA-B*07:28','HLA-B*07:29','HLA-B*07:30','HLA-B*07:31','HLA-B*07:32','HLA-B*07:33','HLA-B*07:34','HLA-B*07:35','HLA-B*07:36','HLA-B*07:37','HLA-B*07:38','HLA-B*07:39','HLA-B*07:40','HLA-B*07:41','HLA-B*07:42','HLA-B*07:43','HLA-B*07:44','HLA-B*07:45','HLA-B*07:46','HLA-B*07:47','HLA-B*07:48','HLA-B*07:50','HLA-B*07:51','HLA-B*07:52','HLA-B*07:53','HLA-B*07:54','HLA-B*07:55','HLA-B*07:56','HLA-B*07:57','HLA-B*07:58','HLA-B*07:59','HLA-B*07:60','HLA-B*07:61','HLA-B*07:62','HLA-B*07:63','HLA-B*07:64','HLA-B*07:65','HLA-B*07:66','HLA-B*07:68','HLA-B*07:69','HLA-B*07:70','HLA-B*07:71','HLA-B*07:72','HLA-B*07:73','HLA-B*07:74','HLA-B*07:75','HLA-B*07:76','HLA-B*07:77','HLA-B*07:78','HLA-B*07:79','HLA-B*07:80','HLA-B*07:81','HLA-B*07:82','HLA-B*07:83','HLA-B*07:84','HLA-B*07:85','HLA-B*07:86','HLA-B*07:87','HLA-B*07:88','HLA-B*07:89','HLA-B*07:90','HLA-B*07:91','HLA-B*07:92','HLA-B*07:93','HLA-B*07:94','HLA-B*07:95','HLA-B*07:96','HLA-B*07:97','HLA-B*07:98','HLA-B*07:99','HLA-B*0801','HLA-B*0802','HLA-B*0803','HLA-B*0804','HLA-B*0805','HLA-B*0806','HLA-B*0807','HLA-B*0808','HLA-B*0809','HLA-B*0810','HLA-B*0811','HLA-B*0812','HLA-B*0813','HLA-B*0814','HLA-B*0815','HLA-B*0816','HLA-B*0817','HLA-B*0818','HLA-B*0819','HLA-B*0820','HLA-B*0821','HLA-B*0822','HLA-B*0823','HLA-B*0824','HLA-B*0825','HLA-B*0826','HLA-B*0827','HLA-B*0828','HLA-B*0829','HLA-B*0831','HLA-B*0832','HLA-B*0833','HLA-B*08:01','HLA-B*08:02','HLA-B*08:03','HLA-B*08:04','HLA-B*08:05','HLA-B*08:07','HLA-B*08:09','HLA-B*08:10','HLA-B*08:11','HLA-B*08:12','HLA-B*08:13','HLA-B*08:14','HLA-B*08:15','HLA-B*08:16','HLA-B*08:17','HLA-B*08:18','HLA-B*08:20','HLA-B*08:21','HLA-B*08:22','HLA-B*08:23','HLA-B*08:24','HLA-B*08:25','HLA-B*08:26','HLA-B*08:27','HLA-B*08:28','HLA-B*08:29','HLA-B*08:31','HLA-B*08:32','HLA-B*08:33','HLA-B*08:34','HLA-B*08:35','HLA-B*08:36','HLA-B*08:37','HLA-B*08:38','HLA-B*08:39','HLA-B*08:40','HLA-B*08:41','HLA-B*08:42','HLA-B*08:43','HLA-B*08:44','HLA-B*08:45','HLA-B*08:46','HLA-B*08:47','HLA-B*08:48','HLA-B*08:49','HLA-B*08:50','HLA-B*08:51','HLA-B*08:52','HLA-B*08:53','HLA-B*08:54','HLA-B*08:55','HLA-B*08:56','HLA-B*08:57','HLA-B*08:58','HLA-B*08:59','HLA-B*08:60','HLA-B*08:61','HLA-B*08:62','HLA-B*1301','HLA-B*1302','HLA-B*1303','HLA-B*1304','HLA-B*1306','HLA-B*1308','HLA-B*1309','HLA-B*1310','HLA-B*1311','HLA-B*1312','HLA-B*1313','HLA-B*1314','HLA-B*1315','HLA-B*1316','HLA-B*1317','HLA-B*1318','HLA-B*1319','HLA-B*1320','HLA-B*13:01','HLA-B*13:02','HLA-B*13:03','HLA-B*13:04','HLA-B*13:06','HLA-B*13:09','HLA-B*13:10','HLA-B*13:11','HLA-B*13:12','HLA-B*13:13','HLA-B*13:14','HLA-B*13:15','HLA-B*13:16','HLA-B*13:17','HLA-B*13:18','HLA-B*13:19','HLA-B*13:20','HLA-B*13:21','HLA-B*13:22','HLA-B*13:23','HLA-B*13:25','HLA-B*13:26','HLA-B*13:27','HLA-B*13:28','HLA-B*13:29','HLA-B*13:30','HLA-B*13:31','HLA-B*13:32','HLA-B*13:33','HLA-B*13:34','HLA-B*13:35','HLA-B*13:36','HLA-B*13:37','HLA-B*13:38','HLA-B*13:39','HLA-B*1401','HLA-B*1402','HLA-B*1403','HLA-B*1404','HLA-B*1405','HLA-B*1406','HLA-B*14:01','HLA-B*14:02','HLA-B*14:03','HLA-B*14:04','HLA-B*14:05','HLA-B*14:06','HLA-B*14:08','HLA-B*14:09','HLA-B*14:10','HLA-B*14:11','HLA-B*14:12','HLA-B*14:13','HLA-B*14:14','HLA-B*14:15','HLA-B*14:16','HLA-B*14:17','HLA-B*14:18','HLA-B*1501','HLA-B*1502','HLA-B*1503','HLA-B*1504','HLA-B*1505','HLA-B*1506','HLA-B*1507','HLA-B*1508','HLA-B*1509','HLA-B*1510','HLA-B*1511','HLA-B*1512','HLA-B*1513','HLA-B*1514','HLA-B*1515','HLA-B*1516','HLA-B*1517','HLA-B*1518','HLA-B*1519','HLA-B*1520','HLA-B*1521','HLA-B*1523','HLA-B*1524','HLA-B*1525','HLA-B*1527','HLA-B*1528','HLA-B*1529','HLA-B*1530','HLA-B*1531','HLA-B*1532','HLA-B*1533','HLA-B*1534','HLA-B*1535','HLA-B*1536','HLA-B*1537','HLA-B*1538','HLA-B*1539','HLA-B*1540','HLA-B*1542','HLA-B*1543','HLA-B*1544','HLA-B*1545','HLA-B*1546','HLA-B*1547','HLA-B*1548','HLA-B*1549','HLA-B*1550','HLA-B*1551','HLA-B*1552','HLA-B*1553','HLA-B*1554','HLA-B*1555','HLA-B*1556','HLA-B*1557','HLA-B*1558','HLA-B*1560','HLA-B*1561','HLA-B*1562','HLA-B*1563','HLA-B*1564','HLA-B*1565','HLA-B*1566','HLA-B*1567','HLA-B*1568','HLA-B*1569','HLA-B*1570','HLA-B*1571','HLA-B*1572','HLA-B*1573','HLA-B*1574','HLA-B*1575','HLA-B*1576','HLA-B*1577','HLA-B*1578','HLA-B*1580','HLA-B*1581','HLA-B*1582','HLA-B*1583','HLA-B*1584','HLA-B*1585','HLA-B*1586','HLA-B*1587','HLA-B*1588','HLA-B*1589','HLA-B*1590','HLA-B*1591','HLA-B*1592','HLA-B*1593','HLA-B*1595','HLA-B*1596','HLA-B*1597','HLA-B*1598','HLA-B*1599','HLA-B*15:01','HLA-B*15:02','HLA-B*15:03','HLA-B*15:04','HLA-B*15:05','HLA-B*15:06','HLA-B*15:07','HLA-B*15:08','HLA-B*15:09','HLA-B*15:10','HLA-B*15:101','HLA-B*15:102','HLA-B*15:103','HLA-B*15:104','HLA-B*15:105','HLA-B*15:106','HLA-B*15:107','HLA-B*15:108','HLA-B*15:109','HLA-B*15:11','HLA-B*15:110','HLA-B*15:112','HLA-B*15:113','HLA-B*15:114','HLA-B*15:115','HLA-B*15:116','HLA-B*15:117','HLA-B*15:118','HLA-B*15:119','HLA-B*15:12','HLA-B*15:120','HLA-B*15:121','HLA-B*15:122','HLA-B*15:123','HLA-B*15:124','HLA-B*15:125','HLA-B*15:126','HLA-B*15:127','HLA-B*15:128','HLA-B*15:129','HLA-B*15:13','HLA-B*15:131','HLA-B*15:132','HLA-B*15:133','HLA-B*15:134','HLA-B*15:135','HLA-B*15:136','HLA-B*15:137','HLA-B*15:138','HLA-B*15:139','HLA-B*15:14','HLA-B*15:140','HLA-B*15:141','HLA-B*15:142','HLA-B*15:143','HLA-B*15:144','HLA-B*15:145','HLA-B*15:146','HLA-B*15:147','HLA-B*15:148','HLA-B*15:15','HLA-B*15:150','HLA-B*15:151','HLA-B*15:152','HLA-B*15:153','HLA-B*15:154','HLA-B*15:155','HLA-B*15:156','HLA-B*15:157','HLA-B*15:158','HLA-B*15:159','HLA-B*15:16','HLA-B*15:160','HLA-B*15:161','HLA-B*15:162','HLA-B*15:163','HLA-B*15:164','HLA-B*15:165','HLA-B*15:166','HLA-B*15:167','HLA-B*15:168','HLA-B*15:169','HLA-B*15:17','HLA-B*15:170','HLA-B*15:171','HLA-B*15:172','HLA-B*15:173','HLA-B*15:174','HLA-B*15:175','HLA-B*15:176','HLA-B*15:177','HLA-B*15:178','HLA-B*15:179','HLA-B*15:18','HLA-B*15:180','HLA-B*15:183','HLA-B*15:184','HLA-B*15:185','HLA-B*15:186','HLA-B*15:187','HLA-B*15:188','HLA-B*15:189','HLA-B*15:19','HLA-B*15:191','HLA-B*15:192','HLA-B*15:193','HLA-B*15:194','HLA-B*15:195','HLA-B*15:196','HLA-B*15:197','HLA-B*15:198','HLA-B*15:199','HLA-B*15:20','HLA-B*15:200','HLA-B*15:201','HLA-B*15:202','HLA-B*15:21','HLA-B*15:23','HLA-B*15:24','HLA-B*15:25','HLA-B*15:27','HLA-B*15:28','HLA-B*15:29','HLA-B*15:30','HLA-B*15:31','HLA-B*15:32','HLA-B*15:33','HLA-B*15:34','HLA-B*15:35','HLA-B*15:36','HLA-B*15:37','HLA-B*15:38','HLA-B*15:39','HLA-B*15:40','HLA-B*15:42','HLA-B*15:43','HLA-B*15:44','HLA-B*15:45','HLA-B*15:46','HLA-B*15:47','HLA-B*15:48','HLA-B*15:49','HLA-B*15:50','HLA-B*15:51','HLA-B*15:52','HLA-B*15:53','HLA-B*15:54','HLA-B*15:55','HLA-B*15:56','HLA-B*15:57','HLA-B*15:58','HLA-B*15:60','HLA-B*15:61','HLA-B*15:62','HLA-B*15:63','HLA-B*15:64','HLA-B*15:65','HLA-B*15:66','HLA-B*15:67','HLA-B*15:68','HLA-B*15:69','HLA-B*15:70','HLA-B*15:71','HLA-B*15:72','HLA-B*15:73','HLA-B*15:74','HLA-B*15:75','HLA-B*15:76','HLA-B*15:77','HLA-B*15:78','HLA-B*15:80','HLA-B*15:81','HLA-B*15:82','HLA-B*15:83','HLA-B*15:84','HLA-B*15:85','HLA-B*15:86','HLA-B*15:87','HLA-B*15:88','HLA-B*15:89','HLA-B*15:90','HLA-B*15:91','HLA-B*15:92','HLA-B*15:93','HLA-B*15:95','HLA-B*15:96','HLA-B*15:97','HLA-B*15:98','HLA-B*15:99','HLA-B*1801','HLA-B*1802','HLA-B*1803','HLA-B*1804','HLA-B*1805','HLA-B*1806','HLA-B*1807','HLA-B*1808','HLA-B*1809','HLA-B*1810','HLA-B*1811','HLA-B*1812','HLA-B*1813','HLA-B*1814','HLA-B*1815','HLA-B*1818','HLA-B*1819','HLA-B*1820','HLA-B*1821','HLA-B*1822','HLA-B*1823','HLA-B*1824','HLA-B*1825','HLA-B*1826','HLA-B*18:01','HLA-B*18:02','HLA-B*18:03','HLA-B*18:04','HLA-B*18:05','HLA-B*18:06','HLA-B*18:07','HLA-B*18:08','HLA-B*18:09','HLA-B*18:10','HLA-B*18:11','HLA-B*18:12','HLA-B*18:13','HLA-B*18:14','HLA-B*18:15','HLA-B*18:18','HLA-B*18:19','HLA-B*18:20','HLA-B*18:21','HLA-B*18:22','HLA-B*18:24','HLA-B*18:25','HLA-B*18:26','HLA-B*18:27','HLA-B*18:28','HLA-B*18:29','HLA-B*18:30','HLA-B*18:31','HLA-B*18:32','HLA-B*18:33','HLA-B*18:34','HLA-B*18:35','HLA-B*18:36','HLA-B*18:37','HLA-B*18:38','HLA-B*18:39','HLA-B*18:40','HLA-B*18:41','HLA-B*18:42','HLA-B*18:43','HLA-B*18:44','HLA-B*18:45','HLA-B*18:46','HLA-B*18:47','HLA-B*18:48','HLA-B*18:49','HLA-B*18:50','HLA-B*2701','HLA-B*2702','HLA-B*2703','HLA-B*2704','HLA-B*2705','HLA-B*2706','HLA-B*2707','HLA-B*2708','HLA-B*2709','HLA-B*2710','HLA-B*2711','HLA-B*2712','HLA-B*2713','HLA-B*2714','HLA-B*2715','HLA-B*2716','HLA-B*2717','HLA-B*2718','HLA-B*2719','HLA-B*2720','HLA-B*2721','HLA-B*2723','HLA-B*2724','HLA-B*2725','HLA-B*2726','HLA-B*2727','HLA-B*2728','HLA-B*2729','HLA-B*2730','HLA-B*2731','HLA-B*2732','HLA-B*2733','HLA-B*2734','HLA-B*2735','HLA-B*2736','HLA-B*2737','HLA-B*2738','HLA-B*27:01','HLA-B*27:02','HLA-B*27:03','HLA-B*27:04','HLA-B*27:05','HLA-B*27:06','HLA-B*27:07','HLA-B*27:08','HLA-B*27:09','HLA-B*27:10','HLA-B*27:11','HLA-B*27:12','HLA-B*27:13','HLA-B*27:14','HLA-B*27:15','HLA-B*27:16','HLA-B*27:17','HLA-B*27:18','HLA-B*27:19','HLA-B*27:20','HLA-B*27:21','HLA-B*27:23','HLA-B*27:24','HLA-B*27:25','HLA-B*27:26','HLA-B*27:27','HLA-B*27:28','HLA-B*27:29','HLA-B*27:30','HLA-B*27:31','HLA-B*27:32','HLA-B*27:33','HLA-B*27:34','HLA-B*27:35','HLA-B*27:36','HLA-B*27:37','HLA-B*27:38','HLA-B*27:39','HLA-B*27:40','HLA-B*27:41','HLA-B*27:42','HLA-B*27:43','HLA-B*27:44','HLA-B*27:45','HLA-B*27:46','HLA-B*27:47','HLA-B*27:48','HLA-B*27:49','HLA-B*27:50','HLA-B*27:51','HLA-B*27:52','HLA-B*27:53','HLA-B*27:54','HLA-B*27:55','HLA-B*27:56','HLA-B*27:57','HLA-B*27:58','HLA-B*27:60','HLA-B*27:61','HLA-B*27:62','HLA-B*27:63','HLA-B*27:67','HLA-B*27:68','HLA-B*27:69','HLA-B*3501','HLA-B*3502','HLA-B*3503','HLA-B*3504','HLA-B*3505','HLA-B*3506','HLA-B*3507','HLA-B*3508','HLA-B*3509','HLA-B*3510','HLA-B*3511','HLA-B*3512','HLA-B*3513','HLA-B*3514','HLA-B*3515','HLA-B*3516','HLA-B*3517','HLA-B*3518','HLA-B*3519','HLA-B*3520','HLA-B*3521','HLA-B*3522','HLA-B*3523','HLA-B*3524','HLA-B*3525','HLA-B*3526','HLA-B*3527','HLA-B*3528','HLA-B*3529','HLA-B*3530','HLA-B*3531','HLA-B*3532','HLA-B*3533','HLA-B*3534','HLA-B*3535','HLA-B*3536','HLA-B*3537','HLA-B*3538','HLA-B*3539','HLA-B*3540','HLA-B*3541','HLA-B*3542','HLA-B*3543','HLA-B*3544','HLA-B*3545','HLA-B*3546','HLA-B*3547','HLA-B*3548','HLA-B*3549','HLA-B*3550','HLA-B*3551','HLA-B*3552','HLA-B*3554','HLA-B*3555','HLA-B*3556','HLA-B*3557','HLA-B*3558','HLA-B*3559','HLA-B*3560','HLA-B*3561','HLA-B*3562','HLA-B*3563','HLA-B*3564','HLA-B*3565','HLA-B*3566','HLA-B*3567','HLA-B*3568','HLA-B*3569','HLA-B*3570','HLA-B*3571','HLA-B*3572','HLA-B*3573','HLA-B*3574','HLA-B*3575','HLA-B*3576','HLA-B*3577','HLA-B*35:01','HLA-B*35:02','HLA-B*35:03','HLA-B*35:04','HLA-B*35:05','HLA-B*35:06','HLA-B*35:07','HLA-B*35:08','HLA-B*35:09','HLA-B*35:10','HLA-B*35:100','HLA-B*35:101','HLA-B*35:102','HLA-B*35:103','HLA-B*35:104','HLA-B*35:105','HLA-B*35:106','HLA-B*35:107','HLA-B*35:108','HLA-B*35:109','HLA-B*35:11','HLA-B*35:110','HLA-B*35:111','HLA-B*35:112','HLA-B*35:113','HLA-B*35:114','HLA-B*35:115','HLA-B*35:116','HLA-B*35:117','HLA-B*35:118','HLA-B*35:119','HLA-B*35:12','HLA-B*35:120','HLA-B*35:121','HLA-B*35:122','HLA-B*35:123','HLA-B*35:124','HLA-B*35:125','HLA-B*35:126','HLA-B*35:127','HLA-B*35:128','HLA-B*35:13','HLA-B*35:131','HLA-B*35:132','HLA-B*35:133','HLA-B*35:135','HLA-B*35:136','HLA-B*35:137','HLA-B*35:138','HLA-B*35:139','HLA-B*35:14','HLA-B*35:140','HLA-B*35:141','HLA-B*35:142','HLA-B*35:143','HLA-B*35:144','HLA-B*35:15','HLA-B*35:16','HLA-B*35:17','HLA-B*35:18','HLA-B*35:19','HLA-B*35:20','HLA-B*35:21','HLA-B*35:22','HLA-B*35:23','HLA-B*35:24','HLA-B*35:25','HLA-B*35:26','HLA-B*35:27','HLA-B*35:28','HLA-B*35:29','HLA-B*35:30','HLA-B*35:31','HLA-B*35:32','HLA-B*35:33','HLA-B*35:34','HLA-B*35:35','HLA-B*35:36','HLA-B*35:37','HLA-B*35:38','HLA-B*35:39','HLA-B*35:41','HLA-B*35:42','HLA-B*35:43','HLA-B*35:44','HLA-B*35:45','HLA-B*35:46','HLA-B*35:47','HLA-B*35:48','HLA-B*35:49','HLA-B*35:50','HLA-B*35:51','HLA-B*35:52','HLA-B*35:54','HLA-B*35:55','HLA-B*35:56','HLA-B*35:57','HLA-B*35:58','HLA-B*35:59','HLA-B*35:60','HLA-B*35:61','HLA-B*35:62','HLA-B*35:63','HLA-B*35:64','HLA-B*35:66','HLA-B*35:67','HLA-B*35:68','HLA-B*35:69','HLA-B*35:70','HLA-B*35:71','HLA-B*35:72','HLA-B*35:74','HLA-B*35:75','HLA-B*35:76','HLA-B*35:77','HLA-B*35:78','HLA-B*35:79','HLA-B*35:80','HLA-B*35:81','HLA-B*35:82','HLA-B*35:83','HLA-B*35:84','HLA-B*35:85','HLA-B*35:86','HLA-B*35:87','HLA-B*35:88','HLA-B*35:89','HLA-B*35:90','HLA-B*35:91','HLA-B*35:92','HLA-B*35:93','HLA-B*35:94','HLA-B*35:95','HLA-B*35:96','HLA-B*35:97','HLA-B*35:98','HLA-B*35:99','HLA-B*3701','HLA-B*3702','HLA-B*3704','HLA-B*3705','HLA-B*3706','HLA-B*3707','HLA-B*3708','HLA-B*3709','HLA-B*3710','HLA-B*3711','HLA-B*3712','HLA-B*3713','HLA-B*37:01','HLA-B*37:02','HLA-B*37:04','HLA-B*37:05','HLA-B*37:06','HLA-B*37:07','HLA-B*37:08','HLA-B*37:09','HLA-B*37:10','HLA-B*37:11','HLA-B*37:12','HLA-B*37:13','HLA-B*37:14','HLA-B*37:15','HLA-B*37:17','HLA-B*37:18','HLA-B*37:19','HLA-B*37:20','HLA-B*37:21','HLA-B*37:22','HLA-B*37:23','HLA-B*3801','HLA-B*3802','HLA-B*3803','HLA-B*3804','HLA-B*3805','HLA-B*3806','HLA-B*3807','HLA-B*3808','HLA-B*3809','HLA-B*3810','HLA-B*3811','HLA-B*3812','HLA-B*3813','HLA-B*3814','HLA-B*3815','HLA-B*3816','HLA-B*38:01','HLA-B*38:02','HLA-B*38:03','HLA-B*38:04','HLA-B*38:05','HLA-B*38:06','HLA-B*38:07','HLA-B*38:08','HLA-B*38:09','HLA-B*38:10','HLA-B*38:11','HLA-B*38:12','HLA-B*38:13','HLA-B*38:14','HLA-B*38:15','HLA-B*38:16','HLA-B*38:17','HLA-B*38:18','HLA-B*38:19','HLA-B*38:20','HLA-B*38:21','HLA-B*38:22','HLA-B*38:23','HLA-B*3901','HLA-B*3902','HLA-B*3903','HLA-B*3904','HLA-B*3905','HLA-B*3906','HLA-B*3908','HLA-B*3909','HLA-B*3910','HLA-B*3912','HLA-B*3913','HLA-B*3914','HLA-B*3915','HLA-B*3916','HLA-B*3917','HLA-B*3918','HLA-B*3919','HLA-B*3920','HLA-B*3922','HLA-B*3923','HLA-B*3924','HLA-B*3926','HLA-B*3927','HLA-B*3928','HLA-B*3929','HLA-B*3930','HLA-B*3931','HLA-B*3932','HLA-B*3933','HLA-B*3934','HLA-B*3935','HLA-B*3936','HLA-B*3937','HLA-B*3938','HLA-B*3939','HLA-B*3941','HLA-B*3942','HLA-B*39:01','HLA-B*39:02','HLA-B*39:03','HLA-B*39:04','HLA-B*39:05','HLA-B*39:06','HLA-B*39:07','HLA-B*39:08','HLA-B*39:09','HLA-B*39:10','HLA-B*39:11','HLA-B*39:12','HLA-B*39:13','HLA-B*39:14','HLA-B*39:15','HLA-B*39:16','HLA-B*39:17','HLA-B*39:18','HLA-B*39:19','HLA-B*39:20','HLA-B*39:22','HLA-B*39:23','HLA-B*39:24','HLA-B*39:26','HLA-B*39:27','HLA-B*39:28','HLA-B*39:29','HLA-B*39:30','HLA-B*39:31','HLA-B*39:32','HLA-B*39:33','HLA-B*39:34','HLA-B*39:35','HLA-B*39:36','HLA-B*39:37','HLA-B*39:39','HLA-B*39:41','HLA-B*39:42','HLA-B*39:43','HLA-B*39:44','HLA-B*39:45','HLA-B*39:46','HLA-B*39:47','HLA-B*39:48','HLA-B*39:49','HLA-B*39:50','HLA-B*39:51','HLA-B*39:52','HLA-B*39:53','HLA-B*39:54','HLA-B*39:55','HLA-B*39:56','HLA-B*39:57','HLA-B*39:58','HLA-B*39:59','HLA-B*39:60','HLA-B*4001','HLA-B*4002','HLA-B*4003','HLA-B*4004','HLA-B*4005','HLA-B*4006','HLA-B*4007','HLA-B*4008','HLA-B*4009','HLA-B*4010','HLA-B*4011','HLA-B*4012','HLA-B*4013','HLA-B*4014','HLA-B*4015','HLA-B*4016','HLA-B*4018','HLA-B*4019','HLA-B*4020','HLA-B*4021','HLA-B*4023','HLA-B*4024','HLA-B*4025','HLA-B*4026','HLA-B*4027','HLA-B*4028','HLA-B*4029','HLA-B*4030','HLA-B*4031','HLA-B*4032','HLA-B*4033','HLA-B*4034','HLA-B*4035','HLA-B*4036','HLA-B*4037','HLA-B*4038','HLA-B*4039','HLA-B*4040','HLA-B*4042','HLA-B*4043','HLA-B*4044','HLA-B*4045','HLA-B*4046','HLA-B*4047','HLA-B*4048','HLA-B*4049','HLA-B*4050','HLA-B*4051','HLA-B*4052','HLA-B*4053','HLA-B*4054','HLA-B*4055','HLA-B*4056','HLA-B*4057','HLA-B*4058','HLA-B*4059','HLA-B*4060','HLA-B*4061','HLA-B*4062','HLA-B*4063','HLA-B*4064','HLA-B*4065','HLA-B*4066','HLA-B*4067','HLA-B*4068','HLA-B*4069','HLA-B*4070','HLA-B*4071','HLA-B*4072','HLA-B*4073','HLA-B*4074','HLA-B*4075','HLA-B*4076','HLA-B*4077','HLA-B*40:01','HLA-B*40:02','HLA-B*40:03','HLA-B*40:04','HLA-B*40:05','HLA-B*40:06','HLA-B*40:07','HLA-B*40:08','HLA-B*40:09','HLA-B*40:10','HLA-B*40:100','HLA-B*40:101','HLA-B*40:102','HLA-B*40:103','HLA-B*40:104','HLA-B*40:105','HLA-B*40:106','HLA-B*40:107','HLA-B*40:108','HLA-B*40:109','HLA-B*40:11','HLA-B*40:110','HLA-B*40:111','HLA-B*40:112','HLA-B*40:113','HLA-B*40:114','HLA-B*40:115','HLA-B*40:116','HLA-B*40:117','HLA-B*40:119','HLA-B*40:12','HLA-B*40:120','HLA-B*40:121','HLA-B*40:122','HLA-B*40:123','HLA-B*40:124','HLA-B*40:125','HLA-B*40:126','HLA-B*40:127','HLA-B*40:128','HLA-B*40:129','HLA-B*40:13','HLA-B*40:130','HLA-B*40:131','HLA-B*40:132','HLA-B*40:134','HLA-B*40:135','HLA-B*40:136','HLA-B*40:137','HLA-B*40:138','HLA-B*40:139','HLA-B*40:14','HLA-B*40:140','HLA-B*40:141','HLA-B*40:143','HLA-B*40:145','HLA-B*40:146','HLA-B*40:147','HLA-B*40:15','HLA-B*40:16','HLA-B*40:18','HLA-B*40:19','HLA-B*40:20','HLA-B*40:21','HLA-B*40:23','HLA-B*40:24','HLA-B*40:25','HLA-B*40:26','HLA-B*40:27','HLA-B*40:28','HLA-B*40:29','HLA-B*40:30','HLA-B*40:31','HLA-B*40:32','HLA-B*40:33','HLA-B*40:34','HLA-B*40:35','HLA-B*40:36','HLA-B*40:37','HLA-B*40:38','HLA-B*40:39','HLA-B*40:40','HLA-B*40:42','HLA-B*40:43','HLA-B*40:44','HLA-B*40:45','HLA-B*40:46','HLA-B*40:47','HLA-B*40:48','HLA-B*40:49','HLA-B*40:50','HLA-B*40:51','HLA-B*40:52','HLA-B*40:53','HLA-B*40:54','HLA-B*40:55','HLA-B*40:56','HLA-B*40:57','HLA-B*40:58','HLA-B*40:59','HLA-B*40:60','HLA-B*40:61','HLA-B*40:62','HLA-B*40:63','HLA-B*40:64','HLA-B*40:65','HLA-B*40:66','HLA-B*40:67','HLA-B*40:68','HLA-B*40:69','HLA-B*40:70','HLA-B*40:71','HLA-B*40:72','HLA-B*40:73','HLA-B*40:74','HLA-B*40:75','HLA-B*40:76','HLA-B*40:77','HLA-B*40:78','HLA-B*40:79','HLA-B*40:80','HLA-B*40:81','HLA-B*40:82','HLA-B*40:83','HLA-B*40:84','HLA-B*40:85','HLA-B*40:86','HLA-B*40:87','HLA-B*40:88','HLA-B*40:89','HLA-B*40:90','HLA-B*40:91','HLA-B*40:92','HLA-B*40:93','HLA-B*40:94','HLA-B*40:95','HLA-B*40:96','HLA-B*40:97','HLA-B*40:98','HLA-B*40:99','HLA-B*4101','HLA-B*4102','HLA-B*4103','HLA-B*4104','HLA-B*4105','HLA-B*4106','HLA-B*4107','HLA-B*4108','HLA-B*41:01','HLA-B*41:02','HLA-B*41:03','HLA-B*41:04','HLA-B*41:05','HLA-B*41:06','HLA-B*41:07','HLA-B*41:08','HLA-B*41:09','HLA-B*41:10','HLA-B*41:11','HLA-B*41:12','HLA-B*4201','HLA-B*4202','HLA-B*4204','HLA-B*4205','HLA-B*4206','HLA-B*4207','HLA-B*4208','HLA-B*4209','HLA-B*42:01','HLA-B*42:02','HLA-B*42:04','HLA-B*42:05','HLA-B*42:06','HLA-B*42:07','HLA-B*42:08','HLA-B*42:09','HLA-B*42:10','HLA-B*42:11','HLA-B*42:12','HLA-B*42:13','HLA-B*42:14','HLA-B*4402','HLA-B*4403','HLA-B*4404','HLA-B*4405','HLA-B*4406','HLA-B*4407','HLA-B*4408','HLA-B*4409','HLA-B*4410','HLA-B*4411','HLA-B*4412','HLA-B*4413','HLA-B*4414','HLA-B*4415','HLA-B*4416','HLA-B*4417','HLA-B*4418','HLA-B*4420','HLA-B*4421','HLA-B*4422','HLA-B*4424','HLA-B*4425','HLA-B*4426','HLA-B*4427','HLA-B*4428','HLA-B*4429','HLA-B*4430','HLA-B*4431','HLA-B*4432','HLA-B*4433','HLA-B*4434','HLA-B*4435','HLA-B*4436','HLA-B*4437','HLA-B*4438','HLA-B*4439','HLA-B*4440','HLA-B*4441','HLA-B*4442','HLA-B*4443','HLA-B*4444','HLA-B*4445','HLA-B*4446','HLA-B*4447','HLA-B*4448','HLA-B*4449','HLA-B*4450','HLA-B*4451','HLA-B*4453','HLA-B*4454','HLA-B*44:02','HLA-B*44:03','HLA-B*44:04','HLA-B*44:05','HLA-B*44:06','HLA-B*44:07','HLA-B*44:08','HLA-B*44:09','HLA-B*44:10','HLA-B*44:100','HLA-B*44:101','HLA-B*44:102','HLA-B*44:103','HLA-B*44:104','HLA-B*44:105','HLA-B*44:106','HLA-B*44:107','HLA-B*44:109','HLA-B*44:11','HLA-B*44:110','HLA-B*44:12','HLA-B*44:13','HLA-B*44:14','HLA-B*44:15','HLA-B*44:16','HLA-B*44:17','HLA-B*44:18','HLA-B*44:20','HLA-B*44:21','HLA-B*44:22','HLA-B*44:24','HLA-B*44:25','HLA-B*44:26','HLA-B*44:27','HLA-B*44:28','HLA-B*44:29','HLA-B*44:30','HLA-B*44:31','HLA-B*44:32','HLA-B*44:33','HLA-B*44:34','HLA-B*44:35','HLA-B*44:36','HLA-B*44:37','HLA-B*44:38','HLA-B*44:39','HLA-B*44:40','HLA-B*44:41','HLA-B*44:42','HLA-B*44:43','HLA-B*44:44','HLA-B*44:45','HLA-B*44:46','HLA-B*44:47','HLA-B*44:48','HLA-B*44:49','HLA-B*44:50','HLA-B*44:51','HLA-B*44:53','HLA-B*44:54','HLA-B*44:55','HLA-B*44:57','HLA-B*44:59','HLA-B*44:60','HLA-B*44:62','HLA-B*44:63','HLA-B*44:64','HLA-B*44:65','HLA-B*44:66','HLA-B*44:67','HLA-B*44:68','HLA-B*44:69','HLA-B*44:70','HLA-B*44:71','HLA-B*44:72','HLA-B*44:73','HLA-B*44:74','HLA-B*44:75','HLA-B*44:76','HLA-B*44:77','HLA-B*44:78','HLA-B*44:79','HLA-B*44:80','HLA-B*44:81','HLA-B*44:82','HLA-B*44:83','HLA-B*44:84','HLA-B*44:85','HLA-B*44:86','HLA-B*44:87','HLA-B*44:88','HLA-B*44:89','HLA-B*44:90','HLA-B*44:91','HLA-B*44:92','HLA-B*44:93','HLA-B*44:94','HLA-B*44:95','HLA-B*44:96','HLA-B*44:97','HLA-B*44:98','HLA-B*44:99','HLA-B*4501','HLA-B*4502','HLA-B*4503','HLA-B*4504','HLA-B*4505','HLA-B*4506','HLA-B*4507','HLA-B*45:01','HLA-B*45:02','HLA-B*45:03','HLA-B*45:04','HLA-B*45:05','HLA-B*45:06','HLA-B*45:07','HLA-B*45:08','HLA-B*45:09','HLA-B*45:10','HLA-B*45:11','HLA-B*45:12','HLA-B*4601','HLA-B*4602','HLA-B*4603','HLA-B*4604','HLA-B*4605','HLA-B*4606','HLA-B*4608','HLA-B*4609','HLA-B*4610','HLA-B*4611','HLA-B*46:01','HLA-B*46:02','HLA-B*46:03','HLA-B*46:04','HLA-B*46:05','HLA-B*46:06','HLA-B*46:08','HLA-B*46:09','HLA-B*46:10','HLA-B*46:11','HLA-B*46:12','HLA-B*46:13','HLA-B*46:14','HLA-B*46:16','HLA-B*46:17','HLA-B*46:18','HLA-B*46:19','HLA-B*46:20','HLA-B*46:21','HLA-B*46:22','HLA-B*46:23','HLA-B*46:24','HLA-B*4701','HLA-B*4702','HLA-B*4703','HLA-B*4704','HLA-B*4705','HLA-B*47:01','HLA-B*47:02','HLA-B*47:03','HLA-B*47:04','HLA-B*47:05','HLA-B*47:06','HLA-B*47:07','HLA-B*4801','HLA-B*4802','HLA-B*4803','HLA-B*4804','HLA-B*4805','HLA-B*4806','HLA-B*4807','HLA-B*4808','HLA-B*4809','HLA-B*4810','HLA-B*4811','HLA-B*4812','HLA-B*4813','HLA-B*4814','HLA-B*4815','HLA-B*4816','HLA-B*4817','HLA-B*4818','HLA-B*48:01','HLA-B*48:02','HLA-B*48:03','HLA-B*48:04','HLA-B*48:05','HLA-B*48:06','HLA-B*48:07','HLA-B*48:08','HLA-B*48:09','HLA-B*48:10','HLA-B*48:11','HLA-B*48:12','HLA-B*48:13','HLA-B*48:14','HLA-B*48:15','HLA-B*48:16','HLA-B*48:17','HLA-B*48:18','HLA-B*48:19','HLA-B*48:20','HLA-B*48:21','HLA-B*48:22','HLA-B*48:23','HLA-B*4901','HLA-B*4902','HLA-B*4903','HLA-B*4904','HLA-B*4905','HLA-B*49:01','HLA-B*49:02','HLA-B*49:03','HLA-B*49:04','HLA-B*49:05','HLA-B*49:06','HLA-B*49:07','HLA-B*49:08','HLA-B*49:09','HLA-B*49:10','HLA-B*5001','HLA-B*5002','HLA-B*5004','HLA-B*50:01','HLA-B*50:02','HLA-B*50:04','HLA-B*50:05','HLA-B*50:06','HLA-B*50:07','HLA-B*50:08','HLA-B*50:09','HLA-B*5101','HLA-B*5102','HLA-B*5103','HLA-B*5104','HLA-B*5105','HLA-B*5106','HLA-B*5107','HLA-B*5108','HLA-B*5109','HLA-B*5111','HLA-B*5112','HLA-B*5113','HLA-B*5114','HLA-B*5115','HLA-B*5116','HLA-B*5117','HLA-B*5118','HLA-B*5119','HLA-B*5120','HLA-B*5121','HLA-B*5122','HLA-B*5123','HLA-B*5124','HLA-B*5126','HLA-B*5128','HLA-B*5129','HLA-B*5130','HLA-B*5131','HLA-B*5132','HLA-B*5133','HLA-B*5134','HLA-B*5135','HLA-B*5136','HLA-B*5137','HLA-B*5138','HLA-B*5139','HLA-B*5140','HLA-B*5142','HLA-B*5143','HLA-B*5145','HLA-B*5146','HLA-B*5147','HLA-B*5148','HLA-B*5149','HLA-B*51:01','HLA-B*51:02','HLA-B*51:03','HLA-B*51:04','HLA-B*51:05','HLA-B*51:06','HLA-B*51:07','HLA-B*51:08','HLA-B*51:09','HLA-B*51:12','HLA-B*51:13','HLA-B*51:14','HLA-B*51:15','HLA-B*51:16','HLA-B*51:17','HLA-B*51:18','HLA-B*51:19','HLA-B*51:20','HLA-B*51:21','HLA-B*51:22','HLA-B*51:23','HLA-B*51:24','HLA-B*51:26','HLA-B*51:28','HLA-B*51:29','HLA-B*51:30','HLA-B*51:31','HLA-B*51:32','HLA-B*51:33','HLA-B*51:34','HLA-B*51:35','HLA-B*51:36','HLA-B*51:37','HLA-B*51:38','HLA-B*51:39','HLA-B*51:40','HLA-B*51:42','HLA-B*51:43','HLA-B*51:45','HLA-B*51:46','HLA-B*51:48','HLA-B*51:49','HLA-B*51:50','HLA-B*51:51','HLA-B*51:52','HLA-B*51:53','HLA-B*51:54','HLA-B*51:55','HLA-B*51:56','HLA-B*51:57','HLA-B*51:58','HLA-B*51:59','HLA-B*51:60','HLA-B*51:61','HLA-B*51:62','HLA-B*51:63','HLA-B*51:64','HLA-B*51:65','HLA-B*51:66','HLA-B*51:67','HLA-B*51:68','HLA-B*51:69','HLA-B*51:70','HLA-B*51:71','HLA-B*51:72','HLA-B*51:73','HLA-B*51:74','HLA-B*51:75','HLA-B*51:76','HLA-B*51:77','HLA-B*51:78','HLA-B*51:79','HLA-B*51:80','HLA-B*51:81','HLA-B*51:82','HLA-B*51:83','HLA-B*51:84','HLA-B*51:85','HLA-B*51:86','HLA-B*51:87','HLA-B*51:88','HLA-B*51:89','HLA-B*51:90','HLA-B*51:91','HLA-B*51:92','HLA-B*51:93','HLA-B*51:94','HLA-B*51:95','HLA-B*51:96','HLA-B*5201','HLA-B*5202','HLA-B*5203','HLA-B*5204','HLA-B*5205','HLA-B*5206','HLA-B*5207','HLA-B*5208','HLA-B*5209','HLA-B*5210','HLA-B*5211','HLA-B*52:01','HLA-B*52:02','HLA-B*52:03','HLA-B*52:04','HLA-B*52:05','HLA-B*52:06','HLA-B*52:07','HLA-B*52:08','HLA-B*52:09','HLA-B*52:10','HLA-B*52:11','HLA-B*52:12','HLA-B*52:13','HLA-B*52:14','HLA-B*52:15','HLA-B*52:16','HLA-B*52:17','HLA-B*52:18','HLA-B*52:19','HLA-B*52:20','HLA-B*52:21','HLA-B*5301','HLA-B*5302','HLA-B*5303','HLA-B*5304','HLA-B*5305','HLA-B*5306','HLA-B*5307','HLA-B*5308','HLA-B*5309','HLA-B*5310','HLA-B*5311','HLA-B*5312','HLA-B*5313','HLA-B*53:01','HLA-B*53:02','HLA-B*53:03','HLA-B*53:04','HLA-B*53:05','HLA-B*53:06','HLA-B*53:07','HLA-B*53:08','HLA-B*53:09','HLA-B*53:10','HLA-B*53:11','HLA-B*53:12','HLA-B*53:13','HLA-B*53:14','HLA-B*53:15','HLA-B*53:16','HLA-B*53:17','HLA-B*53:18','HLA-B*53:19','HLA-B*53:20','HLA-B*53:21','HLA-B*53:22','HLA-B*53:23','HLA-B*5401','HLA-B*5402','HLA-B*5403','HLA-B*5404','HLA-B*5405','HLA-B*5406','HLA-B*5407','HLA-B*5409','HLA-B*5410','HLA-B*5411','HLA-B*5412','HLA-B*5413','HLA-B*54:01','HLA-B*54:02','HLA-B*54:03','HLA-B*54:04','HLA-B*54:06','HLA-B*54:07','HLA-B*54:09','HLA-B*54:10','HLA-B*54:11','HLA-B*54:12','HLA-B*54:13','HLA-B*54:14','HLA-B*54:15','HLA-B*54:16','HLA-B*54:17','HLA-B*54:18','HLA-B*54:19','HLA-B*54:20','HLA-B*54:21','HLA-B*54:22','HLA-B*54:23','HLA-B*5501','HLA-B*5502','HLA-B*5503','HLA-B*5504','HLA-B*5505','HLA-B*5507','HLA-B*5508','HLA-B*5509','HLA-B*5510','HLA-B*5511','HLA-B*5512','HLA-B*5513','HLA-B*5514','HLA-B*5515','HLA-B*5516','HLA-B*5517','HLA-B*5518','HLA-B*5519','HLA-B*5520','HLA-B*5521','HLA-B*5522','HLA-B*5523','HLA-B*5524','HLA-B*5525','HLA-B*5526','HLA-B*5527','HLA-B*55:01','HLA-B*55:02','HLA-B*55:03','HLA-B*55:04','HLA-B*55:05','HLA-B*55:07','HLA-B*55:08','HLA-B*55:09','HLA-B*55:10','HLA-B*55:11','HLA-B*55:12','HLA-B*55:13','HLA-B*55:14','HLA-B*55:15','HLA-B*55:16','HLA-B*55:17','HLA-B*55:18','HLA-B*55:19','HLA-B*55:20','HLA-B*55:21','HLA-B*55:22','HLA-B*55:23','HLA-B*55:24','HLA-B*55:25','HLA-B*55:26','HLA-B*55:27','HLA-B*55:28','HLA-B*55:29','HLA-B*55:30','HLA-B*55:31','HLA-B*55:32','HLA-B*55:33','HLA-B*55:34','HLA-B*55:35','HLA-B*55:36','HLA-B*55:37','HLA-B*55:38','HLA-B*55:39','HLA-B*55:40','HLA-B*55:41','HLA-B*55:42','HLA-B*55:43','HLA-B*5601','HLA-B*5602','HLA-B*5603','HLA-B*5604','HLA-B*5605','HLA-B*5606','HLA-B*5607','HLA-B*5608','HLA-B*5609','HLA-B*5610','HLA-B*5611','HLA-B*5612','HLA-B*5613','HLA-B*5614','HLA-B*5615','HLA-B*5616','HLA-B*5617','HLA-B*5618','HLA-B*5620','HLA-B*56:01','HLA-B*56:02','HLA-B*56:03','HLA-B*56:04','HLA-B*56:05','HLA-B*56:06','HLA-B*56:07','HLA-B*56:08','HLA-B*56:09','HLA-B*56:10','HLA-B*56:11','HLA-B*56:12','HLA-B*56:13','HLA-B*56:14','HLA-B*56:15','HLA-B*56:16','HLA-B*56:17','HLA-B*56:18','HLA-B*56:20','HLA-B*56:21','HLA-B*56:22','HLA-B*56:23','HLA-B*56:24','HLA-B*56:25','HLA-B*56:26','HLA-B*56:27','HLA-B*56:29','HLA-B*5701','HLA-B*5702','HLA-B*5703','HLA-B*5704','HLA-B*5705','HLA-B*5706','HLA-B*5707','HLA-B*5708','HLA-B*5709','HLA-B*5710','HLA-B*5711','HLA-B*5712','HLA-B*5713','HLA-B*57:01','HLA-B*57:02','HLA-B*57:03','HLA-B*57:04','HLA-B*57:05','HLA-B*57:06','HLA-B*57:07','HLA-B*57:08','HLA-B*57:09','HLA-B*57:10','HLA-B*57:11','HLA-B*57:12','HLA-B*57:13','HLA-B*57:14','HLA-B*57:15','HLA-B*57:16','HLA-B*57:17','HLA-B*57:18','HLA-B*57:19','HLA-B*57:20','HLA-B*57:21','HLA-B*57:22','HLA-B*57:23','HLA-B*57:24','HLA-B*57:25','HLA-B*57:26','HLA-B*57:27','HLA-B*57:29','HLA-B*57:30','HLA-B*57:31','HLA-B*57:32','HLA-B*5801','HLA-B*5802','HLA-B*5804','HLA-B*5805','HLA-B*5806','HLA-B*5807','HLA-B*5808','HLA-B*5809','HLA-B*5811','HLA-B*5812','HLA-B*5813','HLA-B*5814','HLA-B*5815','HLA-B*58:01','HLA-B*58:02','HLA-B*58:04','HLA-B*58:05','HLA-B*58:06','HLA-B*58:07','HLA-B*58:08','HLA-B*58:09','HLA-B*58:11','HLA-B*58:12','HLA-B*58:13','HLA-B*58:14','HLA-B*58:15','HLA-B*58:16','HLA-B*58:18','HLA-B*58:19','HLA-B*58:20','HLA-B*58:21','HLA-B*58:22','HLA-B*58:23','HLA-B*58:24','HLA-B*58:25','HLA-B*58:26','HLA-B*58:27','HLA-B*58:28','HLA-B*58:29','HLA-B*58:30','HLA-B*5901','HLA-B*5902','HLA-B*59:01','HLA-B*59:02','HLA-B*59:03','HLA-B*59:04','HLA-B*59:05','HLA-B*6701','HLA-B*6702','HLA-B*67:01','HLA-B*67:02','HLA-B*7301','HLA-B*73:01','HLA-B*73:02','HLA-B*7801','HLA-B*7802','HLA-B*7803','HLA-B*7804','HLA-B*7805','HLA-B*78:01','HLA-B*78:02','HLA-B*78:03','HLA-B*78:04','HLA-B*78:05','HLA-B*78:06','HLA-B*78:07','HLA-B*8101','HLA-B*8102','HLA-B*81:01','HLA-B*81:02','HLA-B*81:03','HLA-B*81:05','HLA-B*8201','HLA-B*8202','HLA-B*82:01','HLA-B*82:02','HLA-B*82:03','HLA-B*8301','HLA-B*83:01','HLA-B*9501','HLA-B*9502','HLA-B*9503','HLA-B*9504','HLA-B*9505','HLA-B*9506','HLA-B*9507','HLA-B*9508','HLA-B*9509','HLA-B*9510','HLA-B*9512','HLA-B*9513','HLA-B*9514','HLA-B*9515','HLA-B*9516','HLA-B*9517','HLA-B*9518','HLA-B*9519','HLA-B*9520','HLA-B*9521','HLA-B*9522','HLA-B*9523','HLA-B*9524','HLA-B*9525','HLA-B*9526','HLA-B*9527','HLA-B*9528','HLA-B*9529','HLA-B*9530','HLA-B*9532','HLA-C*0102','HLA-C*0103','HLA-C*0104','HLA-C*0105','HLA-C*0106','HLA-C*0107','HLA-C*0108','HLA-C*0109','HLA-C*0110','HLA-C*0111','HLA-C*0112','HLA-C*0113','HLA-C*01:02','HLA-C*01:03','HLA-C*01:04','HLA-C*01:05','HLA-C*01:06','HLA-C*01:07','HLA-C*01:08','HLA-C*01:09','HLA-C*01:10','HLA-C*01:11','HLA-C*01:12','HLA-C*01:13','HLA-C*01:14','HLA-C*01:15','HLA-C*01:16','HLA-C*01:17','HLA-C*01:18','HLA-C*01:19','HLA-C*01:20','HLA-C*01:21','HLA-C*01:22','HLA-C*01:23','HLA-C*01:24','HLA-C*01:25','HLA-C*01:26','HLA-C*01:27','HLA-C*01:28','HLA-C*01:29','HLA-C*01:30','HLA-C*01:31','HLA-C*01:32','HLA-C*01:33','HLA-C*01:34','HLA-C*01:35','HLA-C*01:36','HLA-C*01:38','HLA-C*01:39','HLA-C*01:40','HLA-C*0202','HLA-C*0203','HLA-C*0204','HLA-C*0205','HLA-C*0206','HLA-C*0207','HLA-C*0208','HLA-C*0209','HLA-C*0210','HLA-C*0211','HLA-C*0212','HLA-C*0213','HLA-C*0214','HLA-C*02:02','HLA-C*02:03','HLA-C*02:04','HLA-C*02:05','HLA-C*02:06','HLA-C*02:07','HLA-C*02:08','HLA-C*02:09','HLA-C*02:10','HLA-C*02:11','HLA-C*02:12','HLA-C*02:13','HLA-C*02:14','HLA-C*02:15','HLA-C*02:16','HLA-C*02:17','HLA-C*02:18','HLA-C*02:19','HLA-C*02:20','HLA-C*02:21','HLA-C*02:22','HLA-C*02:23','HLA-C*02:24','HLA-C*02:26','HLA-C*02:27','HLA-C*02:28','HLA-C*02:29','HLA-C*02:30','HLA-C*02:31','HLA-C*02:32','HLA-C*02:33','HLA-C*02:34','HLA-C*02:35','HLA-C*02:36','HLA-C*02:37','HLA-C*02:39','HLA-C*02:40','HLA-C*0301','HLA-C*0302','HLA-C*0303','HLA-C*0304','HLA-C*0305','HLA-C*0306','HLA-C*0307','HLA-C*0308','HLA-C*0309','HLA-C*0310','HLA-C*0311','HLA-C*0312','HLA-C*0313','HLA-C*0314','HLA-C*0315','HLA-C*0316','HLA-C*0317','HLA-C*0318','HLA-C*0319','HLA-C*0321','HLA-C*0322','HLA-C*0323','HLA-C*0324','HLA-C*0325','HLA-C*03:01','HLA-C*03:02','HLA-C*03:03','HLA-C*03:04','HLA-C*03:05','HLA-C*03:06','HLA-C*03:07','HLA-C*03:08','HLA-C*03:09','HLA-C*03:10','HLA-C*03:11','HLA-C*03:12','HLA-C*03:13','HLA-C*03:14','HLA-C*03:15','HLA-C*03:16','HLA-C*03:17','HLA-C*03:18','HLA-C*03:19','HLA-C*03:21','HLA-C*03:23','HLA-C*03:24','HLA-C*03:25','HLA-C*03:26','HLA-C*03:27','HLA-C*03:28','HLA-C*03:29','HLA-C*03:30','HLA-C*03:31','HLA-C*03:32','HLA-C*03:33','HLA-C*03:34','HLA-C*03:35','HLA-C*03:36','HLA-C*03:37','HLA-C*03:38','HLA-C*03:39','HLA-C*03:40','HLA-C*03:41','HLA-C*03:42','HLA-C*03:43','HLA-C*03:44','HLA-C*03:45','HLA-C*03:46','HLA-C*03:47','HLA-C*03:48','HLA-C*03:49','HLA-C*03:50','HLA-C*03:51','HLA-C*03:52','HLA-C*03:53','HLA-C*03:54','HLA-C*03:55','HLA-C*03:56','HLA-C*03:57','HLA-C*03:58','HLA-C*03:59','HLA-C*03:60','HLA-C*03:61','HLA-C*03:62','HLA-C*03:63','HLA-C*03:64','HLA-C*03:65','HLA-C*03:66','HLA-C*03:67','HLA-C*03:68','HLA-C*03:69','HLA-C*03:70','HLA-C*03:71','HLA-C*03:72','HLA-C*03:73','HLA-C*03:74','HLA-C*03:75','HLA-C*03:76','HLA-C*03:77','HLA-C*03:78','HLA-C*03:79','HLA-C*03:80','HLA-C*03:81','HLA-C*03:82','HLA-C*03:83','HLA-C*03:84','HLA-C*03:85','HLA-C*03:86','HLA-C*03:87','HLA-C*03:88','HLA-C*03:89','HLA-C*03:90','HLA-C*03:91','HLA-C*03:92','HLA-C*03:93','HLA-C*03:94','HLA-C*0401','HLA-C*0403','HLA-C*0404','HLA-C*0405','HLA-C*0406','HLA-C*0407','HLA-C*0408','HLA-C*0409','HLA-C*0410','HLA-C*0411','HLA-C*0412','HLA-C*0413','HLA-C*0414','HLA-C*0415','HLA-C*0416','HLA-C*0417','HLA-C*0418','HLA-C*04:01','HLA-C*04:03','HLA-C*04:04','HLA-C*04:05','HLA-C*04:06','HLA-C*04:07','HLA-C*04:08','HLA-C*04:10','HLA-C*04:11','HLA-C*04:12','HLA-C*04:13','HLA-C*04:14','HLA-C*04:15','HLA-C*04:16','HLA-C*04:17','HLA-C*04:18','HLA-C*04:19','HLA-C*04:20','HLA-C*04:23','HLA-C*04:24','HLA-C*04:25','HLA-C*04:26','HLA-C*04:27','HLA-C*04:28','HLA-C*04:29','HLA-C*04:30','HLA-C*04:31','HLA-C*04:32','HLA-C*04:33','HLA-C*04:34','HLA-C*04:35','HLA-C*04:36','HLA-C*04:37','HLA-C*04:38','HLA-C*04:39','HLA-C*04:40','HLA-C*04:41','HLA-C*04:42','HLA-C*04:43','HLA-C*04:44','HLA-C*04:45','HLA-C*04:46','HLA-C*04:47','HLA-C*04:48','HLA-C*04:49','HLA-C*04:50','HLA-C*04:51','HLA-C*04:52','HLA-C*04:53','HLA-C*04:54','HLA-C*04:55','HLA-C*04:56','HLA-C*04:57','HLA-C*04:58','HLA-C*04:60','HLA-C*04:61','HLA-C*04:62','HLA-C*04:63','HLA-C*04:64','HLA-C*04:65','HLA-C*04:66','HLA-C*04:67','HLA-C*04:68','HLA-C*04:69','HLA-C*04:70','HLA-C*0501','HLA-C*0502','HLA-C*0503','HLA-C*0504','HLA-C*0505','HLA-C*0506','HLA-C*0508','HLA-C*0509','HLA-C*0510','HLA-C*0511','HLA-C*0512','HLA-C*0513','HLA-C*05:01','HLA-C*05:03','HLA-C*05:04','HLA-C*05:05','HLA-C*05:06','HLA-C*05:08','HLA-C*05:09','HLA-C*05:10','HLA-C*05:11','HLA-C*05:12','HLA-C*05:13','HLA-C*05:14','HLA-C*05:15','HLA-C*05:16','HLA-C*05:17','HLA-C*05:18','HLA-C*05:19','HLA-C*05:20','HLA-C*05:21','HLA-C*05:22','HLA-C*05:23','HLA-C*05:24','HLA-C*05:25','HLA-C*05:26','HLA-C*05:27','HLA-C*05:28','HLA-C*05:29','HLA-C*05:30','HLA-C*05:31','HLA-C*05:32','HLA-C*05:33','HLA-C*05:34','HLA-C*05:35','HLA-C*05:36','HLA-C*05:37','HLA-C*05:38','HLA-C*05:39','HLA-C*05:40','HLA-C*05:41','HLA-C*05:42','HLA-C*05:43','HLA-C*05:44','HLA-C*05:45','HLA-C*0602','HLA-C*0603','HLA-C*0604','HLA-C*0605','HLA-C*0606','HLA-C*0607','HLA-C*0608','HLA-C*0609','HLA-C*0610','HLA-C*0611','HLA-C*0612','HLA-C*0613','HLA-C*06:02','HLA-C*06:03','HLA-C*06:04','HLA-C*06:05','HLA-C*06:06','HLA-C*06:07','HLA-C*06:08','HLA-C*06:09','HLA-C*06:10','HLA-C*06:11','HLA-C*06:12','HLA-C*06:13','HLA-C*06:14','HLA-C*06:15','HLA-C*06:17','HLA-C*06:18','HLA-C*06:19','HLA-C*06:20','HLA-C*06:21','HLA-C*06:22','HLA-C*06:23','HLA-C*06:24','HLA-C*06:25','HLA-C*06:26','HLA-C*06:27','HLA-C*06:28','HLA-C*06:29','HLA-C*06:30','HLA-C*06:31','HLA-C*06:32','HLA-C*06:33','HLA-C*06:34','HLA-C*06:35','HLA-C*06:36','HLA-C*06:37','HLA-C*06:38','HLA-C*06:39','HLA-C*06:40','HLA-C*06:41','HLA-C*06:42','HLA-C*06:43','HLA-C*06:44','HLA-C*06:45','HLA-C*0701','HLA-C*0702','HLA-C*0703','HLA-C*0704','HLA-C*0705','HLA-C*0706','HLA-C*0707','HLA-C*0708','HLA-C*0709','HLA-C*0710','HLA-C*0711','HLA-C*0712','HLA-C*0713','HLA-C*0714','HLA-C*0715','HLA-C*0716','HLA-C*0717','HLA-C*0718','HLA-C*0719','HLA-C*0720','HLA-C*0721','HLA-C*0722','HLA-C*0723','HLA-C*0724','HLA-C*0725','HLA-C*0726','HLA-C*0727','HLA-C*0728','HLA-C*0729','HLA-C*0730','HLA-C*0731','HLA-C*0732','HLA-C*0734','HLA-C*0735','HLA-C*0736','HLA-C*0737','HLA-C*0738','HLA-C*07:01','HLA-C*07:02','HLA-C*07:03','HLA-C*07:04','HLA-C*07:05','HLA-C*07:06','HLA-C*07:07','HLA-C*07:08','HLA-C*07:09','HLA-C*07:10','HLA-C*07:100','HLA-C*07:101','HLA-C*07:102','HLA-C*07:103','HLA-C*07:105','HLA-C*07:106','HLA-C*07:107','HLA-C*07:108','HLA-C*07:109','HLA-C*07:11','HLA-C*07:110','HLA-C*07:111','HLA-C*07:112','HLA-C*07:113','HLA-C*07:114','HLA-C*07:115','HLA-C*07:116','HLA-C*07:117','HLA-C*07:118','HLA-C*07:119','HLA-C*07:12','HLA-C*07:120','HLA-C*07:122','HLA-C*07:123','HLA-C*07:124','HLA-C*07:125','HLA-C*07:126','HLA-C*07:127','HLA-C*07:128','HLA-C*07:129','HLA-C*07:13','HLA-C*07:130','HLA-C*07:131','HLA-C*07:132','HLA-C*07:133','HLA-C*07:134','HLA-C*07:135','HLA-C*07:136','HLA-C*07:137','HLA-C*07:138','HLA-C*07:139','HLA-C*07:14','HLA-C*07:140','HLA-C*07:141','HLA-C*07:142','HLA-C*07:143','HLA-C*07:144','HLA-C*07:145','HLA-C*07:146','HLA-C*07:147','HLA-C*07:148','HLA-C*07:149','HLA-C*07:15','HLA-C*07:16','HLA-C*07:17','HLA-C*07:18','HLA-C*07:19','HLA-C*07:20','HLA-C*07:21','HLA-C*07:22','HLA-C*07:23','HLA-C*07:24','HLA-C*07:25','HLA-C*07:26','HLA-C*07:27','HLA-C*07:28','HLA-C*07:29','HLA-C*07:30','HLA-C*07:31','HLA-C*07:35','HLA-C*07:36','HLA-C*07:37','HLA-C*07:38','HLA-C*07:39','HLA-C*07:40','HLA-C*07:41','HLA-C*07:42','HLA-C*07:43','HLA-C*07:44','HLA-C*07:45','HLA-C*07:46','HLA-C*07:47','HLA-C*07:48','HLA-C*07:49','HLA-C*07:50','HLA-C*07:51','HLA-C*07:52','HLA-C*07:53','HLA-C*07:54','HLA-C*07:56','HLA-C*07:57','HLA-C*07:58','HLA-C*07:59','HLA-C*07:60','HLA-C*07:62','HLA-C*07:63','HLA-C*07:64','HLA-C*07:65','HLA-C*07:66','HLA-C*07:67','HLA-C*07:68','HLA-C*07:69','HLA-C*07:70','HLA-C*07:71','HLA-C*07:72','HLA-C*07:73','HLA-C*07:74','HLA-C*07:75','HLA-C*07:76','HLA-C*07:77','HLA-C*07:78','HLA-C*07:79','HLA-C*07:80','HLA-C*07:81','HLA-C*07:82','HLA-C*07:83','HLA-C*07:84','HLA-C*07:85','HLA-C*07:86','HLA-C*07:87','HLA-C*07:88','HLA-C*07:89','HLA-C*07:90','HLA-C*07:91','HLA-C*07:92','HLA-C*07:93','HLA-C*07:94','HLA-C*07:95','HLA-C*07:96','HLA-C*07:97','HLA-C*07:99','HLA-C*0801','HLA-C*0802','HLA-C*0803','HLA-C*0804','HLA-C*0805','HLA-C*0806','HLA-C*0807','HLA-C*0808','HLA-C*0809','HLA-C*0810','HLA-C*0811','HLA-C*0812','HLA-C*0813','HLA-C*0814','HLA-C*08:01','HLA-C*08:02','HLA-C*08:03','HLA-C*08:04','HLA-C*08:05','HLA-C*08:06','HLA-C*08:07','HLA-C*08:08','HLA-C*08:09','HLA-C*08:10','HLA-C*08:11','HLA-C*08:12','HLA-C*08:13','HLA-C*08:14','HLA-C*08:15','HLA-C*08:16','HLA-C*08:17','HLA-C*08:18','HLA-C*08:19','HLA-C*08:20','HLA-C*08:21','HLA-C*08:22','HLA-C*08:23','HLA-C*08:24','HLA-C*08:25','HLA-C*08:27','HLA-C*08:28','HLA-C*08:29','HLA-C*08:30','HLA-C*08:31','HLA-C*08:32','HLA-C*08:33','HLA-C*08:34','HLA-C*08:35','HLA-C*1202','HLA-C*1203','HLA-C*1204','HLA-C*1205','HLA-C*1206','HLA-C*1207','HLA-C*1208','HLA-C*1209','HLA-C*1210','HLA-C*1211','HLA-C*1212','HLA-C*1213','HLA-C*1214','HLA-C*1215','HLA-C*1216','HLA-C*1217','HLA-C*12:02','HLA-C*12:03','HLA-C*12:04','HLA-C*12:05','HLA-C*12:06','HLA-C*12:07','HLA-C*12:08','HLA-C*12:09','HLA-C*12:10','HLA-C*12:11','HLA-C*12:12','HLA-C*12:13','HLA-C*12:14','HLA-C*12:15','HLA-C*12:16','HLA-C*12:17','HLA-C*12:18','HLA-C*12:19','HLA-C*12:20','HLA-C*12:21','HLA-C*12:22','HLA-C*12:23','HLA-C*12:24','HLA-C*12:25','HLA-C*12:26','HLA-C*12:27','HLA-C*12:28','HLA-C*12:29','HLA-C*12:30','HLA-C*12:31','HLA-C*12:32','HLA-C*12:33','HLA-C*12:34','HLA-C*12:35','HLA-C*12:36','HLA-C*12:37','HLA-C*12:38','HLA-C*12:40','HLA-C*12:41','HLA-C*12:43','HLA-C*12:44','HLA-C*1402','HLA-C*1403','HLA-C*1404','HLA-C*1405','HLA-C*1406','HLA-C*1407','HLA-C*14:02','HLA-C*14:03','HLA-C*14:04','HLA-C*14:05','HLA-C*14:06','HLA-C*14:08','HLA-C*14:09','HLA-C*14:10','HLA-C*14:11','HLA-C*14:12','HLA-C*14:13','HLA-C*14:14','HLA-C*14:15','HLA-C*14:16','HLA-C*14:17','HLA-C*14:18','HLA-C*14:19','HLA-C*14:20','HLA-C*1502','HLA-C*1503','HLA-C*1504','HLA-C*1505','HLA-C*1506','HLA-C*1507','HLA-C*1508','HLA-C*1509','HLA-C*1510','HLA-C*1511','HLA-C*1512','HLA-C*1513','HLA-C*1514','HLA-C*1515','HLA-C*1516','HLA-C*1517','HLA-C*15:02','HLA-C*15:03','HLA-C*15:04','HLA-C*15:05','HLA-C*15:06','HLA-C*15:07','HLA-C*15:08','HLA-C*15:09','HLA-C*15:10','HLA-C*15:11','HLA-C*15:12','HLA-C*15:13','HLA-C*15:15','HLA-C*15:16','HLA-C*15:17','HLA-C*15:18','HLA-C*15:19','HLA-C*15:20','HLA-C*15:21','HLA-C*15:22','HLA-C*15:23','HLA-C*15:24','HLA-C*15:25','HLA-C*15:26','HLA-C*15:27','HLA-C*15:28','HLA-C*15:29','HLA-C*15:30','HLA-C*15:31','HLA-C*15:33','HLA-C*15:34','HLA-C*15:35','HLA-C*1601','HLA-C*1602','HLA-C*1604','HLA-C*1606','HLA-C*1607','HLA-C*1608','HLA-C*16:01','HLA-C*16:02','HLA-C*16:04','HLA-C*16:06','HLA-C*16:07','HLA-C*16:08','HLA-C*16:09','HLA-C*16:10','HLA-C*16:11','HLA-C*16:12','HLA-C*16:13','HLA-C*16:14','HLA-C*16:15','HLA-C*16:17','HLA-C*16:18','HLA-C*16:19','HLA-C*16:20','HLA-C*16:21','HLA-C*16:22','HLA-C*16:23','HLA-C*16:24','HLA-C*16:25','HLA-C*16:26','HLA-C*1701','HLA-C*1702','HLA-C*1703','HLA-C*1704','HLA-C*17:01','HLA-C*17:02','HLA-C*17:03','HLA-C*17:04','HLA-C*17:05','HLA-C*17:06','HLA-C*17:07','HLA-C*1801','HLA-C*1802','HLA-C*18:01','HLA-C*18:02','HLA-C*18:03','HLA-E*0101','HLA-E*0103','HLA-E*01:01','HLA-E*01:03','HLA-G*0101','HLA-G*0102','HLA-G*0103','HLA-G*0104','HLA-G*0106','HLA-G*0107','HLA-G*0108','HLA-G*0109','HLA-G*01:01','HLA-G*01:02','HLA-G*01:03','HLA-G*01:04','HLA-G*01:06','HLA-G*01:07','HLA-G*01:08','HLA-G*01:09'])

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter = '\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in next(f) if x != ""]
        next(f, None) # Avoid header column, only access raw and rank scores
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCPAN_4_1]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCPAN_4_1 + i * Offset.NETMHCPAN_4_1])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCPAN_4_1 + i * Offset.NETMHCPAN_4_1])
        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
        return result


class NetMHCstabpan_1_0(AExternalEpitopePrediction):
    """
    Implements a wrapper to NetMHCstabpan 1.0

    .. note:

    Pan-specific prediction of peptide-MHC-I complex stability; a correlate of T cell immunogenicity
    M Rasmussen, E Fenoy, M Nielsen, Buus S, Accepted JI June, 2016
    """
    __name = "netMHCstabpan"
    __length = frozenset([8, 9, 10, 11])
    __version = "1.0"
    __command = "netMHCstabpan -p {peptides} -a {alleles} {options} -xls -xlsfile {out}"
    __alleles = frozenset(['HLA-A*01:01', 'HLA-A*01:02', 'HLA-A*01:03', 'HLA-A*01:06', 'HLA-A*01:07', 'HLA-A*01:08', 'HLA-A*01:09', 'HLA-A*01:10', 'HLA-A*01:12',
                            'HLA-A*01:13', 'HLA-A*01:14', 'HLA-A*01:17', 'HLA-A*01:19', 'HLA-A*01:20', 'HLA-A*01:21', 'HLA-A*01:23', 'HLA-A*01:24',
                            'HLA-A*01:25', 'HLA-A*01:26', 'HLA-A*01:28', 'HLA-A*01:29', 'HLA-A*01:30', 'HLA-A*01:32', 'HLA-A*01:33', 'HLA-A*01:35',
                            'HLA-A*01:36', 'HLA-A*01:37', 'HLA-A*01:38', 'HLA-A*01:39', 'HLA-A*01:40', 'HLA-A*01:41', 'HLA-A*01:42', 'HLA-A*01:43',
                            'HLA-A*01:44', 'HLA-A*01:45', 'HLA-A*01:46', 'HLA-A*01:47', 'HLA-A*01:48', 'HLA-A*01:49', 'HLA-A*01:50', 'HLA-A*01:51',
                            'HLA-A*01:54', 'HLA-A*01:55', 'HLA-A*01:58', 'HLA-A*01:59', 'HLA-A*01:60', 'HLA-A*01:61', 'HLA-A*01:62', 'HLA-A*01:63',
                            'HLA-A*01:64', 'HLA-A*01:65', 'HLA-A*01:66', 'HLA-A*02:01', 'HLA-A*02:02', 'HLA-A*02:03', 'HLA-A*02:04', 'HLA-A*02:05',
                            'HLA-A*02:06', 'HLA-A*02:07', 'HLA-A*02:08', 'HLA-A*02:09', 'HLA-A*02:10', 'HLA-A*02:11', 'HLA-A*02:12', 'HLA-A*02:13',
                            'HLA-A*02:14', 'HLA-A*02:16', 'HLA-A*02:17', 'HLA-A*02:18', 'HLA-A*02:19', 'HLA-A*02:20', 'HLA-A*02:21', 'HLA-A*02:22',
                            'HLA-A*02:24', 'HLA-A*02:25', 'HLA-A*02:26', 'HLA-A*02:27', 'HLA-A*02:28', 'HLA-A*02:29', 'HLA-A*02:30', 'HLA-A*02:31',
                            'HLA-A*02:33', 'HLA-A*02:34', 'HLA-A*02:35', 'HLA-A*02:36', 'HLA-A*02:37', 'HLA-A*02:38', 'HLA-A*02:39', 'HLA-A*02:40',
                            'HLA-A*02:41', 'HLA-A*02:42', 'HLA-A*02:44', 'HLA-A*02:45', 'HLA-A*02:46', 'HLA-A*02:47', 'HLA-A*02:48', 'HLA-A*02:49',
                            'HLA-A*02:50', 'HLA-A*02:51', 'HLA-A*02:52', 'HLA-A*02:54', 'HLA-A*02:55', 'HLA-A*02:56', 'HLA-A*02:57', 'HLA-A*02:58',
                            'HLA-A*02:59', 'HLA-A*02:60', 'HLA-A*02:61', 'HLA-A*02:62', 'HLA-A*02:63', 'HLA-A*02:64', 'HLA-A*02:65', 'HLA-A*02:66',
                            'HLA-A*02:67', 'HLA-A*02:68', 'HLA-A*02:69', 'HLA-A*02:70', 'HLA-A*02:71', 'HLA-A*02:72', 'HLA-A*02:73', 'HLA-A*02:74',
                            'HLA-A*02:75', 'HLA-A*02:76', 'HLA-A*02:77', 'HLA-A*02:78', 'HLA-A*02:79', 'HLA-A*02:80', 'HLA-A*02:81', 'HLA-A*02:84',
                            'HLA-A*02:85', 'HLA-A*02:86', 'HLA-A*02:87', 'HLA-A*02:89', 'HLA-A*02:90', 'HLA-A*02:91', 'HLA-A*02:92', 'HLA-A*02:93',
                            'HLA-A*02:95', 'HLA-A*02:96', 'HLA-A*02:97', 'HLA-A*02:99', 'HLA-A*02:101', 'HLA-A*02:102', 'HLA-A*02:103', 'HLA-A*02:104',
                            'HLA-A*02:105', 'HLA-A*02:106', 'HLA-A*02:107', 'HLA-A*02:108', 'HLA-A*02:109', 'HLA-A*02:110', 'HLA-A*02:111',
                            'HLA-A*02:112', 'HLA-A*02:114', 'HLA-A*02:115', 'HLA-A*02:116', 'HLA-A*02:117', 'HLA-A*02:118', 'HLA-A*02:119',
                            'HLA-A*02:120', 'HLA-A*02:121', 'HLA-A*02:122', 'HLA-A*02:123', 'HLA-A*02:124', 'HLA-A*02:126', 'HLA-A*02:127',
                            'HLA-A*02:128', 'HLA-A*02:129', 'HLA-A*02:130', 'HLA-A*02:131', 'HLA-A*02:132', 'HLA-A*02:133', 'HLA-A*02:134',
                            'HLA-A*02:135', 'HLA-A*02:136', 'HLA-A*02:137', 'HLA-A*02:138', 'HLA-A*02:139', 'HLA-A*02:140', 'HLA-A*02:141',
                            'HLA-A*02:142', 'HLA-A*02:143', 'HLA-A*02:144', 'HLA-A*02:145', 'HLA-A*02:146', 'HLA-A*02:147', 'HLA-A*02:148',
                            'HLA-A*02:149', 'HLA-A*02:150', 'HLA-A*02:151', 'HLA-A*02:152', 'HLA-A*02:153', 'HLA-A*02:154', 'HLA-A*02:155',
                            'HLA-A*02:156', 'HLA-A*02:157', 'HLA-A*02:158', 'HLA-A*02:159', 'HLA-A*02:160', 'HLA-A*02:161', 'HLA-A*02:162',
                            'HLA-A*02:163', 'HLA-A*02:164', 'HLA-A*02:165', 'HLA-A*02:166', 'HLA-A*02:167', 'HLA-A*02:168', 'HLA-A*02:169',
                            'HLA-A*02:170', 'HLA-A*02:171', 'HLA-A*02:172', 'HLA-A*02:173', 'HLA-A*02:174', 'HLA-A*02:175', 'HLA-A*02:176',
                            'HLA-A*02:177', 'HLA-A*02:178', 'HLA-A*02:179', 'HLA-A*02:180', 'HLA-A*02:181', 'HLA-A*02:182', 'HLA-A*02:183',
                            'HLA-A*02:184', 'HLA-A*02:185', 'HLA-A*02:186', 'HLA-A*02:187', 'HLA-A*02:188', 'HLA-A*02:189', 'HLA-A*02:190',
                            'HLA-A*02:191', 'HLA-A*02:192', 'HLA-A*02:193', 'HLA-A*02:194', 'HLA-A*02:195', 'HLA-A*02:196', 'HLA-A*02:197',
                            'HLA-A*02:198', 'HLA-A*02:199', 'HLA-A*02:200', 'HLA-A*02:201', 'HLA-A*02:202', 'HLA-A*02:203', 'HLA-A*02:204',
                            'HLA-A*02:205', 'HLA-A*02:206', 'HLA-A*02:207', 'HLA-A*02:208', 'HLA-A*02:209', 'HLA-A*02:210', 'HLA-A*02:211',
                            'HLA-A*02:212', 'HLA-A*02:213', 'HLA-A*02:214', 'HLA-A*02:215', 'HLA-A*02:216', 'HLA-A*02:217', 'HLA-A*02:218',
                            'HLA-A*02:219', 'HLA-A*02:220', 'HLA-A*02:221', 'HLA-A*02:224', 'HLA-A*02:228', 'HLA-A*02:229', 'HLA-A*02:230',
                            'HLA-A*02:231', 'HLA-A*02:232', 'HLA-A*02:233', 'HLA-A*02:234', 'HLA-A*02:235', 'HLA-A*02:236', 'HLA-A*02:237',
                            'HLA-A*02:238', 'HLA-A*02:239', 'HLA-A*02:240', 'HLA-A*02:241', 'HLA-A*02:242', 'HLA-A*02:243', 'HLA-A*02:244',
                            'HLA-A*02:245', 'HLA-A*02:246', 'HLA-A*02:247', 'HLA-A*02:248', 'HLA-A*02:249', 'HLA-A*02:251', 'HLA-A*02:252',
                            'HLA-A*02:253', 'HLA-A*02:254', 'HLA-A*02:255', 'HLA-A*02:256', 'HLA-A*02:257', 'HLA-A*02:258', 'HLA-A*02:259',
                            'HLA-A*02:260', 'HLA-A*02:261', 'HLA-A*02:262', 'HLA-A*02:263', 'HLA-A*02:264', 'HLA-A*02:265', 'HLA-A*02:266',
                            'HLA-A*03:01', 'HLA-A*03:02', 'HLA-A*03:04', 'HLA-A*03:05', 'HLA-A*03:06', 'HLA-A*03:07', 'HLA-A*03:08', 'HLA-A*03:09',
                            'HLA-A*03:10', 'HLA-A*03:12', 'HLA-A*03:13', 'HLA-A*03:14', 'HLA-A*03:15', 'HLA-A*03:16', 'HLA-A*03:17', 'HLA-A*03:18',
                            'HLA-A*03:19', 'HLA-A*03:20', 'HLA-A*03:22', 'HLA-A*03:23', 'HLA-A*03:24', 'HLA-A*03:25', 'HLA-A*03:26', 'HLA-A*03:27',
                            'HLA-A*03:28', 'HLA-A*03:29', 'HLA-A*03:30', 'HLA-A*03:31', 'HLA-A*03:32', 'HLA-A*03:33', 'HLA-A*03:34', 'HLA-A*03:35',
                            'HLA-A*03:37', 'HLA-A*03:38', 'HLA-A*03:39', 'HLA-A*03:40', 'HLA-A*03:41', 'HLA-A*03:42', 'HLA-A*03:43', 'HLA-A*03:44',
                            'HLA-A*03:45', 'HLA-A*03:46', 'HLA-A*03:47', 'HLA-A*03:48', 'HLA-A*03:49', 'HLA-A*03:50', 'HLA-A*03:51', 'HLA-A*03:52',
                            'HLA-A*03:53', 'HLA-A*03:54', 'HLA-A*03:55', 'HLA-A*03:56', 'HLA-A*03:57', 'HLA-A*03:58', 'HLA-A*03:59', 'HLA-A*03:60',
                            'HLA-A*03:61', 'HLA-A*03:62', 'HLA-A*03:63', 'HLA-A*03:64', 'HLA-A*03:65', 'HLA-A*03:66', 'HLA-A*03:67', 'HLA-A*03:70',
                            'HLA-A*03:71', 'HLA-A*03:72', 'HLA-A*03:73', 'HLA-A*03:74', 'HLA-A*03:75', 'HLA-A*03:76', 'HLA-A*03:77', 'HLA-A*03:78',
                            'HLA-A*03:79', 'HLA-A*03:80', 'HLA-A*03:81', 'HLA-A*03:82', 'HLA-A*11:01', 'HLA-A*11:02', 'HLA-A*11:03', 'HLA-A*11:04',
                            'HLA-A*11:05', 'HLA-A*11:06', 'HLA-A*11:07', 'HLA-A*11:08', 'HLA-A*11:09', 'HLA-A*11:10', 'HLA-A*11:11', 'HLA-A*11:12',
                            'HLA-A*11:13', 'HLA-A*11:14', 'HLA-A*11:15', 'HLA-A*11:16', 'HLA-A*11:17', 'HLA-A*11:18', 'HLA-A*11:19', 'HLA-A*11:20',
                            'HLA-A*11:22', 'HLA-A*11:23', 'HLA-A*11:24', 'HLA-A*11:25', 'HLA-A*11:26', 'HLA-A*11:27', 'HLA-A*11:29', 'HLA-A*11:30',
                            'HLA-A*11:31', 'HLA-A*11:32', 'HLA-A*11:33', 'HLA-A*11:34', 'HLA-A*11:35', 'HLA-A*11:36', 'HLA-A*11:37', 'HLA-A*11:38',
                            'HLA-A*11:39', 'HLA-A*11:40', 'HLA-A*11:41', 'HLA-A*11:42', 'HLA-A*11:43', 'HLA-A*11:44', 'HLA-A*11:45', 'HLA-A*11:46',
                            'HLA-A*11:47', 'HLA-A*11:48', 'HLA-A*11:49', 'HLA-A*11:51', 'HLA-A*11:53', 'HLA-A*11:54', 'HLA-A*11:55', 'HLA-A*11:56',
                            'HLA-A*11:57', 'HLA-A*11:58', 'HLA-A*11:59', 'HLA-A*11:60', 'HLA-A*11:61', 'HLA-A*11:62', 'HLA-A*11:63', 'HLA-A*11:64',
                            'HLA-A*23:01', 'HLA-A*23:02', 'HLA-A*23:03', 'HLA-A*23:04', 'HLA-A*23:05', 'HLA-A*23:06', 'HLA-A*23:09', 'HLA-A*23:10',
                            'HLA-A*23:12', 'HLA-A*23:13', 'HLA-A*23:14', 'HLA-A*23:15', 'HLA-A*23:16', 'HLA-A*23:17', 'HLA-A*23:18', 'HLA-A*23:20',
                            'HLA-A*23:21', 'HLA-A*23:22', 'HLA-A*23:23', 'HLA-A*23:24', 'HLA-A*23:25', 'HLA-A*23:26', 'HLA-A*24:02', 'HLA-A*24:03',
                            'HLA-A*24:04', 'HLA-A*24:05', 'HLA-A*24:06', 'HLA-A*24:07', 'HLA-A*24:08', 'HLA-A*24:10', 'HLA-A*24:13', 'HLA-A*24:14',
                            'HLA-A*24:15', 'HLA-A*24:17', 'HLA-A*24:18', 'HLA-A*24:19', 'HLA-A*24:20', 'HLA-A*24:21', 'HLA-A*24:22', 'HLA-A*24:23',
                            'HLA-A*24:24', 'HLA-A*24:25', 'HLA-A*24:26', 'HLA-A*24:27', 'HLA-A*24:28', 'HLA-A*24:29', 'HLA-A*24:30', 'HLA-A*24:31',
                            'HLA-A*24:32', 'HLA-A*24:33', 'HLA-A*24:34', 'HLA-A*24:35', 'HLA-A*24:37', 'HLA-A*24:38', 'HLA-A*24:39', 'HLA-A*24:41',
                            'HLA-A*24:42', 'HLA-A*24:43', 'HLA-A*24:44', 'HLA-A*24:46', 'HLA-A*24:47', 'HLA-A*24:49', 'HLA-A*24:50', 'HLA-A*24:51',
                            'HLA-A*24:52', 'HLA-A*24:53', 'HLA-A*24:54', 'HLA-A*24:55', 'HLA-A*24:56', 'HLA-A*24:57', 'HLA-A*24:58', 'HLA-A*24:59',
                            'HLA-A*24:61', 'HLA-A*24:62', 'HLA-A*24:63', 'HLA-A*24:64', 'HLA-A*24:66', 'HLA-A*24:67', 'HLA-A*24:68', 'HLA-A*24:69',
                            'HLA-A*24:70', 'HLA-A*24:71', 'HLA-A*24:72', 'HLA-A*24:73', 'HLA-A*24:74', 'HLA-A*24:75', 'HLA-A*24:76', 'HLA-A*24:77',
                            'HLA-A*24:78', 'HLA-A*24:79', 'HLA-A*24:80', 'HLA-A*24:81', 'HLA-A*24:82', 'HLA-A*24:85', 'HLA-A*24:87', 'HLA-A*24:88',
                            'HLA-A*24:89', 'HLA-A*24:91', 'HLA-A*24:92', 'HLA-A*24:93', 'HLA-A*24:94', 'HLA-A*24:95', 'HLA-A*24:96', 'HLA-A*24:97',
                            'HLA-A*24:98', 'HLA-A*24:99', 'HLA-A*24:100', 'HLA-A*24:101', 'HLA-A*24:102', 'HLA-A*24:103', 'HLA-A*24:104',
                            'HLA-A*24:105', 'HLA-A*24:106', 'HLA-A*24:107', 'HLA-A*24:108', 'HLA-A*24:109', 'HLA-A*24:110', 'HLA-A*24:111',
                            'HLA-A*24:112', 'HLA-A*24:113', 'HLA-A*24:114', 'HLA-A*24:115', 'HLA-A*24:116', 'HLA-A*24:117', 'HLA-A*24:118',
                            'HLA-A*24:119', 'HLA-A*24:120', 'HLA-A*24:121', 'HLA-A*24:122', 'HLA-A*24:123', 'HLA-A*24:124', 'HLA-A*24:125',
                            'HLA-A*24:126', 'HLA-A*24:127', 'HLA-A*24:128', 'HLA-A*24:129', 'HLA-A*24:130', 'HLA-A*24:131', 'HLA-A*24:133',
                            'HLA-A*24:134', 'HLA-A*24:135', 'HLA-A*24:136', 'HLA-A*24:137', 'HLA-A*24:138', 'HLA-A*24:139', 'HLA-A*24:140',
                            'HLA-A*24:141', 'HLA-A*24:142', 'HLA-A*24:143', 'HLA-A*24:144', 'HLA-A*25:01', 'HLA-A*25:02', 'HLA-A*25:03', 'HLA-A*25:04',
                            'HLA-A*25:05', 'HLA-A*25:06', 'HLA-A*25:07', 'HLA-A*25:08', 'HLA-A*25:09', 'HLA-A*25:10', 'HLA-A*25:11', 'HLA-A*25:13',
                            'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*26:04', 'HLA-A*26:05', 'HLA-A*26:06', 'HLA-A*26:07', 'HLA-A*26:08',
                            'HLA-A*26:09', 'HLA-A*26:10', 'HLA-A*26:12', 'HLA-A*26:13', 'HLA-A*26:14', 'HLA-A*26:15', 'HLA-A*26:16', 'HLA-A*26:17',
                            'HLA-A*26:18', 'HLA-A*26:19', 'HLA-A*26:20', 'HLA-A*26:21', 'HLA-A*26:22', 'HLA-A*26:23', 'HLA-A*26:24', 'HLA-A*26:26',
                            'HLA-A*26:27', 'HLA-A*26:28', 'HLA-A*26:29', 'HLA-A*26:30', 'HLA-A*26:31', 'HLA-A*26:32', 'HLA-A*26:33', 'HLA-A*26:34',
                            'HLA-A*26:35', 'HLA-A*26:36', 'HLA-A*26:37', 'HLA-A*26:38', 'HLA-A*26:39', 'HLA-A*26:40', 'HLA-A*26:41', 'HLA-A*26:42',
                            'HLA-A*26:43', 'HLA-A*26:45', 'HLA-A*26:46', 'HLA-A*26:47', 'HLA-A*26:48', 'HLA-A*26:49', 'HLA-A*26:50', 'HLA-A*29:01',
                            'HLA-A*29:02', 'HLA-A*29:03', 'HLA-A*29:04', 'HLA-A*29:05', 'HLA-A*29:06', 'HLA-A*29:07', 'HLA-A*29:09', 'HLA-A*29:10',
                            'HLA-A*29:11', 'HLA-A*29:12', 'HLA-A*29:13', 'HLA-A*29:14', 'HLA-A*29:15', 'HLA-A*29:16', 'HLA-A*29:17', 'HLA-A*29:18',
                            'HLA-A*29:19', 'HLA-A*29:20', 'HLA-A*29:21', 'HLA-A*29:22', 'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*30:03', 'HLA-A*30:04',
                            'HLA-A*30:06', 'HLA-A*30:07', 'HLA-A*30:08', 'HLA-A*30:09', 'HLA-A*30:10', 'HLA-A*30:11', 'HLA-A*30:12', 'HLA-A*30:13',
                            'HLA-A*30:15', 'HLA-A*30:16', 'HLA-A*30:17', 'HLA-A*30:18', 'HLA-A*30:19', 'HLA-A*30:20', 'HLA-A*30:22', 'HLA-A*30:23',
                            'HLA-A*30:24', 'HLA-A*30:25', 'HLA-A*30:26', 'HLA-A*30:28', 'HLA-A*30:29', 'HLA-A*30:30', 'HLA-A*30:31', 'HLA-A*30:32',
                            'HLA-A*30:33', 'HLA-A*30:34', 'HLA-A*30:35', 'HLA-A*30:36', 'HLA-A*30:37', 'HLA-A*30:38', 'HLA-A*30:39', 'HLA-A*30:40',
                            'HLA-A*30:41', 'HLA-A*31:01', 'HLA-A*31:02', 'HLA-A*31:03', 'HLA-A*31:04', 'HLA-A*31:05', 'HLA-A*31:06', 'HLA-A*31:07',
                            'HLA-A*31:08', 'HLA-A*31:09', 'HLA-A*31:10', 'HLA-A*31:11', 'HLA-A*31:12', 'HLA-A*31:13', 'HLA-A*31:15', 'HLA-A*31:16',
                            'HLA-A*31:17', 'HLA-A*31:18', 'HLA-A*31:19', 'HLA-A*31:20', 'HLA-A*31:21', 'HLA-A*31:22', 'HLA-A*31:23', 'HLA-A*31:24',
                            'HLA-A*31:25', 'HLA-A*31:26', 'HLA-A*31:27', 'HLA-A*31:28', 'HLA-A*31:29', 'HLA-A*31:30', 'HLA-A*31:31', 'HLA-A*31:32',
                            'HLA-A*31:33', 'HLA-A*31:34', 'HLA-A*31:35', 'HLA-A*31:36', 'HLA-A*31:37', 'HLA-A*32:01', 'HLA-A*32:02', 'HLA-A*32:03',
                            'HLA-A*32:04', 'HLA-A*32:05', 'HLA-A*32:06', 'HLA-A*32:07', 'HLA-A*32:08', 'HLA-A*32:09', 'HLA-A*32:10', 'HLA-A*32:12',
                            'HLA-A*32:13', 'HLA-A*32:14', 'HLA-A*32:15', 'HLA-A*32:16', 'HLA-A*32:17', 'HLA-A*32:18', 'HLA-A*32:20', 'HLA-A*32:21',
                            'HLA-A*32:22', 'HLA-A*32:23', 'HLA-A*32:24', 'HLA-A*32:25', 'HLA-A*33:01', 'HLA-A*33:03', 'HLA-A*33:04', 'HLA-A*33:05',
                            'HLA-A*33:06', 'HLA-A*33:07', 'HLA-A*33:08', 'HLA-A*33:09', 'HLA-A*33:10', 'HLA-A*33:11', 'HLA-A*33:12', 'HLA-A*33:13',
                            'HLA-A*33:14', 'HLA-A*33:15', 'HLA-A*33:16', 'HLA-A*33:17', 'HLA-A*33:18', 'HLA-A*33:19', 'HLA-A*33:20', 'HLA-A*33:21',
                            'HLA-A*33:22', 'HLA-A*33:23', 'HLA-A*33:24', 'HLA-A*33:25', 'HLA-A*33:26', 'HLA-A*33:27', 'HLA-A*33:28', 'HLA-A*33:29',
                            'HLA-A*33:30', 'HLA-A*33:31', 'HLA-A*34:01', 'HLA-A*34:02', 'HLA-A*34:03', 'HLA-A*34:04', 'HLA-A*34:05', 'HLA-A*34:06',
                            'HLA-A*34:07', 'HLA-A*34:08', 'HLA-A*36:01', 'HLA-A*36:02', 'HLA-A*36:03', 'HLA-A*36:04', 'HLA-A*36:05', 'HLA-A*43:01',
                            'HLA-A*66:01', 'HLA-A*66:02', 'HLA-A*66:03', 'HLA-A*66:04', 'HLA-A*66:05', 'HLA-A*66:06', 'HLA-A*66:07', 'HLA-A*66:08',
                            'HLA-A*66:09', 'HLA-A*66:10', 'HLA-A*66:11', 'HLA-A*66:12', 'HLA-A*66:13', 'HLA-A*66:14', 'HLA-A*66:15', 'HLA-A*68:01',
                            'HLA-A*68:02', 'HLA-A*68:03', 'HLA-A*68:04', 'HLA-A*68:05', 'HLA-A*68:06', 'HLA-A*68:07', 'HLA-A*68:08', 'HLA-A*68:09',
                            'HLA-A*68:10', 'HLA-A*68:12', 'HLA-A*68:13', 'HLA-A*68:14', 'HLA-A*68:15', 'HLA-A*68:16', 'HLA-A*68:17', 'HLA-A*68:19',
                            'HLA-A*68:20', 'HLA-A*68:21', 'HLA-A*68:22', 'HLA-A*68:23', 'HLA-A*68:24', 'HLA-A*68:25', 'HLA-A*68:26', 'HLA-A*68:27',
                            'HLA-A*68:28', 'HLA-A*68:29', 'HLA-A*68:30', 'HLA-A*68:31', 'HLA-A*68:32', 'HLA-A*68:33', 'HLA-A*68:34', 'HLA-A*68:35',
                            'HLA-A*68:36', 'HLA-A*68:37', 'HLA-A*68:38', 'HLA-A*68:39', 'HLA-A*68:40', 'HLA-A*68:41', 'HLA-A*68:42', 'HLA-A*68:43',
                            'HLA-A*68:44', 'HLA-A*68:45', 'HLA-A*68:46', 'HLA-A*68:47', 'HLA-A*68:48', 'HLA-A*68:50', 'HLA-A*68:51', 'HLA-A*68:52',
                            'HLA-A*68:53', 'HLA-A*68:54', 'HLA-A*69:01', 'HLA-A*74:01', 'HLA-A*74:02', 'HLA-A*74:03', 'HLA-A*74:04', 'HLA-A*74:05',
                            'HLA-A*74:06', 'HLA-A*74:07', 'HLA-A*74:08', 'HLA-A*74:09', 'HLA-A*74:10', 'HLA-A*74:11', 'HLA-A*74:13', 'HLA-A*80:01',
                            'HLA-A*80:02', 'HLA-B*07:02', 'HLA-B*07:03', 'HLA-B*07:04', 'HLA-B*07:05', 'HLA-B*07:06', 'HLA-B*07:07', 'HLA-B*07:08',
                            'HLA-B*07:09', 'HLA-B*07:10', 'HLA-B*07:11', 'HLA-B*07:12', 'HLA-B*07:13', 'HLA-B*07:14', 'HLA-B*07:15', 'HLA-B*07:16',
                            'HLA-B*07:17', 'HLA-B*07:18', 'HLA-B*07:19', 'HLA-B*07:20', 'HLA-B*07:21', 'HLA-B*07:22', 'HLA-B*07:23', 'HLA-B*07:24',
                            'HLA-B*07:25', 'HLA-B*07:26', 'HLA-B*07:27', 'HLA-B*07:28', 'HLA-B*07:29', 'HLA-B*07:30', 'HLA-B*07:31', 'HLA-B*07:32',
                            'HLA-B*07:33', 'HLA-B*07:34', 'HLA-B*07:35', 'HLA-B*07:36', 'HLA-B*07:37', 'HLA-B*07:38', 'HLA-B*07:39', 'HLA-B*07:40',
                            'HLA-B*07:41', 'HLA-B*07:42', 'HLA-B*07:43', 'HLA-B*07:44', 'HLA-B*07:45', 'HLA-B*07:46', 'HLA-B*07:47', 'HLA-B*07:48',
                            'HLA-B*07:50', 'HLA-B*07:51', 'HLA-B*07:52', 'HLA-B*07:53', 'HLA-B*07:54', 'HLA-B*07:55', 'HLA-B*07:56', 'HLA-B*07:57',
                            'HLA-B*07:58', 'HLA-B*07:59', 'HLA-B*07:60', 'HLA-B*07:61', 'HLA-B*07:62', 'HLA-B*07:63', 'HLA-B*07:64', 'HLA-B*07:65',
                            'HLA-B*07:66', 'HLA-B*07:68', 'HLA-B*07:69', 'HLA-B*07:70', 'HLA-B*07:71', 'HLA-B*07:72', 'HLA-B*07:73', 'HLA-B*07:74',
                            'HLA-B*07:75', 'HLA-B*07:76', 'HLA-B*07:77', 'HLA-B*07:78', 'HLA-B*07:79', 'HLA-B*07:80', 'HLA-B*07:81', 'HLA-B*07:82',
                            'HLA-B*07:83', 'HLA-B*07:84', 'HLA-B*07:85', 'HLA-B*07:86', 'HLA-B*07:87', 'HLA-B*07:88', 'HLA-B*07:89', 'HLA-B*07:90',
                            'HLA-B*07:91', 'HLA-B*07:92', 'HLA-B*07:93', 'HLA-B*07:94', 'HLA-B*07:95', 'HLA-B*07:96', 'HLA-B*07:97', 'HLA-B*07:98',
                            'HLA-B*07:99', 'HLA-B*07:100', 'HLA-B*07:101', 'HLA-B*07:102', 'HLA-B*07:103', 'HLA-B*07:104', 'HLA-B*07:105',
                            'HLA-B*07:106', 'HLA-B*07:107', 'HLA-B*07:108', 'HLA-B*07:109', 'HLA-B*07:110', 'HLA-B*07:112', 'HLA-B*07:113',
                            'HLA-B*07:114', 'HLA-B*07:115', 'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*08:04', 'HLA-B*08:05', 'HLA-B*08:07',
                            'HLA-B*08:09', 'HLA-B*08:10', 'HLA-B*08:11', 'HLA-B*08:12', 'HLA-B*08:13', 'HLA-B*08:14', 'HLA-B*08:15', 'HLA-B*08:16',
                            'HLA-B*08:17', 'HLA-B*08:18', 'HLA-B*08:20', 'HLA-B*08:21', 'HLA-B*08:22', 'HLA-B*08:23', 'HLA-B*08:24', 'HLA-B*08:25',
                            'HLA-B*08:26', 'HLA-B*08:27', 'HLA-B*08:28', 'HLA-B*08:29', 'HLA-B*08:31', 'HLA-B*08:32', 'HLA-B*08:33', 'HLA-B*08:34',
                            'HLA-B*08:35', 'HLA-B*08:36', 'HLA-B*08:37', 'HLA-B*08:38', 'HLA-B*08:39', 'HLA-B*08:40', 'HLA-B*08:41', 'HLA-B*08:42',
                            'HLA-B*08:43', 'HLA-B*08:44', 'HLA-B*08:45', 'HLA-B*08:46', 'HLA-B*08:47', 'HLA-B*08:48', 'HLA-B*08:49', 'HLA-B*08:50',
                            'HLA-B*08:51', 'HLA-B*08:52', 'HLA-B*08:53', 'HLA-B*08:54', 'HLA-B*08:55', 'HLA-B*08:56', 'HLA-B*08:57', 'HLA-B*08:58',
                            'HLA-B*08:59', 'HLA-B*08:60', 'HLA-B*08:61', 'HLA-B*08:62', 'HLA-B*13:01', 'HLA-B*13:02', 'HLA-B*13:03', 'HLA-B*13:04',
                            'HLA-B*13:06', 'HLA-B*13:09', 'HLA-B*13:10', 'HLA-B*13:11', 'HLA-B*13:12', 'HLA-B*13:13', 'HLA-B*13:14', 'HLA-B*13:15',
                            'HLA-B*13:16', 'HLA-B*13:17', 'HLA-B*13:18', 'HLA-B*13:19', 'HLA-B*13:20', 'HLA-B*13:21', 'HLA-B*13:22', 'HLA-B*13:23',
                            'HLA-B*13:25', 'HLA-B*13:26', 'HLA-B*13:27', 'HLA-B*13:28', 'HLA-B*13:29', 'HLA-B*13:30', 'HLA-B*13:31', 'HLA-B*13:32',
                            'HLA-B*13:33', 'HLA-B*13:34', 'HLA-B*13:35', 'HLA-B*13:36', 'HLA-B*13:37', 'HLA-B*13:38', 'HLA-B*13:39', 'HLA-B*14:01',
                            'HLA-B*14:02', 'HLA-B*14:03', 'HLA-B*14:04', 'HLA-B*14:05', 'HLA-B*14:06', 'HLA-B*14:08', 'HLA-B*14:09', 'HLA-B*14:10',
                            'HLA-B*14:11', 'HLA-B*14:12', 'HLA-B*14:13', 'HLA-B*14:14', 'HLA-B*14:15', 'HLA-B*14:16', 'HLA-B*14:17', 'HLA-B*14:18',
                            'HLA-B*15:01', 'HLA-B*15:02', 'HLA-B*15:03', 'HLA-B*15:04', 'HLA-B*15:05', 'HLA-B*15:06', 'HLA-B*15:07', 'HLA-B*15:08',
                            'HLA-B*15:09', 'HLA-B*15:10', 'HLA-B*15:11', 'HLA-B*15:12', 'HLA-B*15:13', 'HLA-B*15:14', 'HLA-B*15:15', 'HLA-B*15:16',
                            'HLA-B*15:17', 'HLA-B*15:18', 'HLA-B*15:19', 'HLA-B*15:20', 'HLA-B*15:21', 'HLA-B*15:23', 'HLA-B*15:24', 'HLA-B*15:25',
                            'HLA-B*15:27', 'HLA-B*15:28', 'HLA-B*15:29', 'HLA-B*15:30', 'HLA-B*15:31', 'HLA-B*15:32', 'HLA-B*15:33', 'HLA-B*15:34',
                            'HLA-B*15:35', 'HLA-B*15:36', 'HLA-B*15:37', 'HLA-B*15:38', 'HLA-B*15:39', 'HLA-B*15:40', 'HLA-B*15:42', 'HLA-B*15:43',
                            'HLA-B*15:44', 'HLA-B*15:45', 'HLA-B*15:46', 'HLA-B*15:47', 'HLA-B*15:48', 'HLA-B*15:49', 'HLA-B*15:50', 'HLA-B*15:51',
                            'HLA-B*15:52', 'HLA-B*15:53', 'HLA-B*15:54', 'HLA-B*15:55', 'HLA-B*15:56', 'HLA-B*15:57', 'HLA-B*15:58', 'HLA-B*15:60',
                            'HLA-B*15:61', 'HLA-B*15:62', 'HLA-B*15:63', 'HLA-B*15:64', 'HLA-B*15:65', 'HLA-B*15:66', 'HLA-B*15:67', 'HLA-B*15:68',
                            'HLA-B*15:69', 'HLA-B*15:70', 'HLA-B*15:71', 'HLA-B*15:72', 'HLA-B*15:73', 'HLA-B*15:74', 'HLA-B*15:75', 'HLA-B*15:76',
                            'HLA-B*15:77', 'HLA-B*15:78', 'HLA-B*15:80', 'HLA-B*15:81', 'HLA-B*15:82', 'HLA-B*15:83', 'HLA-B*15:84', 'HLA-B*15:85',
                            'HLA-B*15:86', 'HLA-B*15:87', 'HLA-B*15:88', 'HLA-B*15:89', 'HLA-B*15:90', 'HLA-B*15:91', 'HLA-B*15:92', 'HLA-B*15:93',
                            'HLA-B*15:95', 'HLA-B*15:96', 'HLA-B*15:97', 'HLA-B*15:98', 'HLA-B*15:99', 'HLA-B*15:101', 'HLA-B*15:102', 'HLA-B*15:103',
                            'HLA-B*15:104', 'HLA-B*15:105', 'HLA-B*15:106', 'HLA-B*15:107', 'HLA-B*15:108', 'HLA-B*15:109', 'HLA-B*15:110',
                            'HLA-B*15:112', 'HLA-B*15:113', 'HLA-B*15:114', 'HLA-B*15:115', 'HLA-B*15:116', 'HLA-B*15:117', 'HLA-B*15:118',
                            'HLA-B*15:119', 'HLA-B*15:120', 'HLA-B*15:121', 'HLA-B*15:122', 'HLA-B*15:123', 'HLA-B*15:124', 'HLA-B*15:125',
                            'HLA-B*15:126', 'HLA-B*15:127', 'HLA-B*15:128', 'HLA-B*15:129', 'HLA-B*15:131', 'HLA-B*15:132', 'HLA-B*15:133',
                            'HLA-B*15:134', 'HLA-B*15:135', 'HLA-B*15:136', 'HLA-B*15:137', 'HLA-B*15:138', 'HLA-B*15:139', 'HLA-B*15:140',
                            'HLA-B*15:141', 'HLA-B*15:142', 'HLA-B*15:143', 'HLA-B*15:144', 'HLA-B*15:145', 'HLA-B*15:146', 'HLA-B*15:147',
                            'HLA-B*15:148', 'HLA-B*15:150', 'HLA-B*15:151', 'HLA-B*15:152', 'HLA-B*15:153', 'HLA-B*15:154', 'HLA-B*15:155',
                            'HLA-B*15:156', 'HLA-B*15:157', 'HLA-B*15:158', 'HLA-B*15:159', 'HLA-B*15:160', 'HLA-B*15:161', 'HLA-B*15:162',
                            'HLA-B*15:163', 'HLA-B*15:164', 'HLA-B*15:165', 'HLA-B*15:166', 'HLA-B*15:167', 'HLA-B*15:168', 'HLA-B*15:169',
                            'HLA-B*15:170', 'HLA-B*15:171', 'HLA-B*15:172', 'HLA-B*15:173', 'HLA-B*15:174', 'HLA-B*15:175', 'HLA-B*15:176',
                            'HLA-B*15:177', 'HLA-B*15:178', 'HLA-B*15:179', 'HLA-B*15:180', 'HLA-B*15:183', 'HLA-B*15:184', 'HLA-B*15:185',
                            'HLA-B*15:186', 'HLA-B*15:187', 'HLA-B*15:188', 'HLA-B*15:189', 'HLA-B*15:191', 'HLA-B*15:192', 'HLA-B*15:193',
                            'HLA-B*15:194', 'HLA-B*15:195', 'HLA-B*15:196', 'HLA-B*15:197', 'HLA-B*15:198', 'HLA-B*15:199', 'HLA-B*15:200',
                            'HLA-B*15:201', 'HLA-B*15:202', 'HLA-B*18:01', 'HLA-B*18:02', 'HLA-B*18:03', 'HLA-B*18:04', 'HLA-B*18:05', 'HLA-B*18:06',
                            'HLA-B*18:07', 'HLA-B*18:08', 'HLA-B*18:09', 'HLA-B*18:10', 'HLA-B*18:11', 'HLA-B*18:12', 'HLA-B*18:13', 'HLA-B*18:14',
                            'HLA-B*18:15', 'HLA-B*18:18', 'HLA-B*18:19', 'HLA-B*18:20', 'HLA-B*18:21', 'HLA-B*18:22', 'HLA-B*18:24', 'HLA-B*18:25',
                            'HLA-B*18:26', 'HLA-B*18:27', 'HLA-B*18:28', 'HLA-B*18:29', 'HLA-B*18:30', 'HLA-B*18:31', 'HLA-B*18:32', 'HLA-B*18:33',
                            'HLA-B*18:34', 'HLA-B*18:35', 'HLA-B*18:36', 'HLA-B*18:37', 'HLA-B*18:38', 'HLA-B*18:39', 'HLA-B*18:40', 'HLA-B*18:41',
                            'HLA-B*18:42', 'HLA-B*18:43', 'HLA-B*18:44', 'HLA-B*18:45', 'HLA-B*18:46', 'HLA-B*18:47', 'HLA-B*18:48', 'HLA-B*18:49',
                            'HLA-B*18:50', 'HLA-B*27:01', 'HLA-B*27:02', 'HLA-B*27:03', 'HLA-B*27:04', 'HLA-B*27:05', 'HLA-B*27:06', 'HLA-B*27:07',
                            'HLA-B*27:08', 'HLA-B*27:09', 'HLA-B*27:10', 'HLA-B*27:11', 'HLA-B*27:12', 'HLA-B*27:13', 'HLA-B*27:14', 'HLA-B*27:15',
                            'HLA-B*27:16', 'HLA-B*27:17', 'HLA-B*27:18', 'HLA-B*27:19', 'HLA-B*27:20', 'HLA-B*27:21', 'HLA-B*27:23', 'HLA-B*27:24',
                            'HLA-B*27:25', 'HLA-B*27:26', 'HLA-B*27:27', 'HLA-B*27:28', 'HLA-B*27:29', 'HLA-B*27:30', 'HLA-B*27:31', 'HLA-B*27:32',
                            'HLA-B*27:33', 'HLA-B*27:34', 'HLA-B*27:35', 'HLA-B*27:36', 'HLA-B*27:37', 'HLA-B*27:38', 'HLA-B*27:39', 'HLA-B*27:40',
                            'HLA-B*27:41', 'HLA-B*27:42', 'HLA-B*27:43', 'HLA-B*27:44', 'HLA-B*27:45', 'HLA-B*27:46', 'HLA-B*27:47', 'HLA-B*27:48',
                            'HLA-B*27:49', 'HLA-B*27:50', 'HLA-B*27:51', 'HLA-B*27:52', 'HLA-B*27:53', 'HLA-B*27:54', 'HLA-B*27:55', 'HLA-B*27:56',
                            'HLA-B*27:57', 'HLA-B*27:58', 'HLA-B*27:60', 'HLA-B*27:61', 'HLA-B*27:62', 'HLA-B*27:63', 'HLA-B*27:67', 'HLA-B*27:68',
                            'HLA-B*27:69', 'HLA-B*35:01', 'HLA-B*35:02', 'HLA-B*35:03', 'HLA-B*35:04', 'HLA-B*35:05', 'HLA-B*35:06', 'HLA-B*35:07',
                            'HLA-B*35:08', 'HLA-B*35:09', 'HLA-B*35:10', 'HLA-B*35:11', 'HLA-B*35:12', 'HLA-B*35:13', 'HLA-B*35:14', 'HLA-B*35:15',
                            'HLA-B*35:16', 'HLA-B*35:17', 'HLA-B*35:18', 'HLA-B*35:19', 'HLA-B*35:20', 'HLA-B*35:21', 'HLA-B*35:22', 'HLA-B*35:23',
                            'HLA-B*35:24', 'HLA-B*35:25', 'HLA-B*35:26', 'HLA-B*35:27', 'HLA-B*35:28', 'HLA-B*35:29', 'HLA-B*35:30', 'HLA-B*35:31',
                            'HLA-B*35:32', 'HLA-B*35:33', 'HLA-B*35:34', 'HLA-B*35:35', 'HLA-B*35:36', 'HLA-B*35:37', 'HLA-B*35:38', 'HLA-B*35:39',
                            'HLA-B*35:41', 'HLA-B*35:42', 'HLA-B*35:43', 'HLA-B*35:44', 'HLA-B*35:45', 'HLA-B*35:46', 'HLA-B*35:47', 'HLA-B*35:48',
                            'HLA-B*35:49', 'HLA-B*35:50', 'HLA-B*35:51', 'HLA-B*35:52', 'HLA-B*35:54', 'HLA-B*35:55', 'HLA-B*35:56', 'HLA-B*35:57',
                            'HLA-B*35:58', 'HLA-B*35:59', 'HLA-B*35:60', 'HLA-B*35:61', 'HLA-B*35:62', 'HLA-B*35:63', 'HLA-B*35:64', 'HLA-B*35:66',
                            'HLA-B*35:67', 'HLA-B*35:68', 'HLA-B*35:69', 'HLA-B*35:70', 'HLA-B*35:71', 'HLA-B*35:72', 'HLA-B*35:74', 'HLA-B*35:75',
                            'HLA-B*35:76', 'HLA-B*35:77', 'HLA-B*35:78', 'HLA-B*35:79', 'HLA-B*35:80', 'HLA-B*35:81', 'HLA-B*35:82', 'HLA-B*35:83',
                            'HLA-B*35:84', 'HLA-B*35:85', 'HLA-B*35:86', 'HLA-B*35:87', 'HLA-B*35:88', 'HLA-B*35:89', 'HLA-B*35:90', 'HLA-B*35:91',
                            'HLA-B*35:92', 'HLA-B*35:93', 'HLA-B*35:94', 'HLA-B*35:95', 'HLA-B*35:96', 'HLA-B*35:97', 'HLA-B*35:98', 'HLA-B*35:99',
                            'HLA-B*35:100', 'HLA-B*35:101', 'HLA-B*35:102', 'HLA-B*35:103', 'HLA-B*35:104', 'HLA-B*35:105', 'HLA-B*35:106',
                            'HLA-B*35:107', 'HLA-B*35:108', 'HLA-B*35:109', 'HLA-B*35:110', 'HLA-B*35:111', 'HLA-B*35:112', 'HLA-B*35:113',
                            'HLA-B*35:114', 'HLA-B*35:115', 'HLA-B*35:116', 'HLA-B*35:117', 'HLA-B*35:118', 'HLA-B*35:119', 'HLA-B*35:120',
                            'HLA-B*35:121', 'HLA-B*35:122', 'HLA-B*35:123', 'HLA-B*35:124', 'HLA-B*35:125', 'HLA-B*35:126', 'HLA-B*35:127',
                            'HLA-B*35:128', 'HLA-B*35:131', 'HLA-B*35:132', 'HLA-B*35:133', 'HLA-B*35:135', 'HLA-B*35:136', 'HLA-B*35:137',
                            'HLA-B*35:138', 'HLA-B*35:139', 'HLA-B*35:140', 'HLA-B*35:141', 'HLA-B*35:142', 'HLA-B*35:143', 'HLA-B*35:144',
                            'HLA-B*37:01', 'HLA-B*37:02', 'HLA-B*37:04', 'HLA-B*37:05', 'HLA-B*37:06', 'HLA-B*37:07', 'HLA-B*37:08', 'HLA-B*37:09',
                            'HLA-B*37:10', 'HLA-B*37:11', 'HLA-B*37:12', 'HLA-B*37:13', 'HLA-B*37:14', 'HLA-B*37:15', 'HLA-B*37:17', 'HLA-B*37:18',
                            'HLA-B*37:19', 'HLA-B*37:20', 'HLA-B*37:21', 'HLA-B*37:22', 'HLA-B*37:23', 'HLA-B*38:01', 'HLA-B*38:02', 'HLA-B*38:03',
                            'HLA-B*38:04', 'HLA-B*38:05', 'HLA-B*38:06', 'HLA-B*38:07', 'HLA-B*38:08', 'HLA-B*38:09', 'HLA-B*38:10', 'HLA-B*38:11',
                            'HLA-B*38:12', 'HLA-B*38:13', 'HLA-B*38:14', 'HLA-B*38:15', 'HLA-B*38:16', 'HLA-B*38:17', 'HLA-B*38:18', 'HLA-B*38:19',
                            'HLA-B*38:20', 'HLA-B*38:21', 'HLA-B*38:22', 'HLA-B*38:23', 'HLA-B*39:01', 'HLA-B*39:02', 'HLA-B*39:03', 'HLA-B*39:04',
                            'HLA-B*39:05', 'HLA-B*39:06', 'HLA-B*39:07', 'HLA-B*39:08', 'HLA-B*39:09', 'HLA-B*39:10', 'HLA-B*39:11', 'HLA-B*39:12',
                            'HLA-B*39:13', 'HLA-B*39:14', 'HLA-B*39:15', 'HLA-B*39:16', 'HLA-B*39:17', 'HLA-B*39:18', 'HLA-B*39:19', 'HLA-B*39:20',
                            'HLA-B*39:22', 'HLA-B*39:23', 'HLA-B*39:24', 'HLA-B*39:26', 'HLA-B*39:27', 'HLA-B*39:28', 'HLA-B*39:29', 'HLA-B*39:30',
                            'HLA-B*39:31', 'HLA-B*39:32', 'HLA-B*39:33', 'HLA-B*39:34', 'HLA-B*39:35', 'HLA-B*39:36', 'HLA-B*39:37', 'HLA-B*39:39',
                            'HLA-B*39:41', 'HLA-B*39:42', 'HLA-B*39:43', 'HLA-B*39:44', 'HLA-B*39:45', 'HLA-B*39:46', 'HLA-B*39:47', 'HLA-B*39:48',
                            'HLA-B*39:49', 'HLA-B*39:50', 'HLA-B*39:51', 'HLA-B*39:52', 'HLA-B*39:53', 'HLA-B*39:54', 'HLA-B*39:55', 'HLA-B*39:56',
                            'HLA-B*39:57', 'HLA-B*39:58', 'HLA-B*39:59', 'HLA-B*39:60', 'HLA-B*40:01', 'HLA-B*40:02', 'HLA-B*40:03', 'HLA-B*40:04',
                            'HLA-B*40:05', 'HLA-B*40:06', 'HLA-B*40:07', 'HLA-B*40:08', 'HLA-B*40:09', 'HLA-B*40:10', 'HLA-B*40:11', 'HLA-B*40:12',
                            'HLA-B*40:13', 'HLA-B*40:14', 'HLA-B*40:15', 'HLA-B*40:16', 'HLA-B*40:18', 'HLA-B*40:19', 'HLA-B*40:20', 'HLA-B*40:21',
                            'HLA-B*40:23', 'HLA-B*40:24', 'HLA-B*40:25', 'HLA-B*40:26', 'HLA-B*40:27', 'HLA-B*40:28', 'HLA-B*40:29', 'HLA-B*40:30',
                            'HLA-B*40:31', 'HLA-B*40:32', 'HLA-B*40:33', 'HLA-B*40:34', 'HLA-B*40:35', 'HLA-B*40:36', 'HLA-B*40:37', 'HLA-B*40:38',
                            'HLA-B*40:39', 'HLA-B*40:40', 'HLA-B*40:42', 'HLA-B*40:43', 'HLA-B*40:44', 'HLA-B*40:45', 'HLA-B*40:46', 'HLA-B*40:47',
                            'HLA-B*40:48', 'HLA-B*40:49', 'HLA-B*40:50', 'HLA-B*40:51', 'HLA-B*40:52', 'HLA-B*40:53', 'HLA-B*40:54', 'HLA-B*40:55',
                            'HLA-B*40:56', 'HLA-B*40:57', 'HLA-B*40:58', 'HLA-B*40:59', 'HLA-B*40:60', 'HLA-B*40:61', 'HLA-B*40:62', 'HLA-B*40:63',
                            'HLA-B*40:64', 'HLA-B*40:65', 'HLA-B*40:66', 'HLA-B*40:67', 'HLA-B*40:68', 'HLA-B*40:69', 'HLA-B*40:70', 'HLA-B*40:71',
                            'HLA-B*40:72', 'HLA-B*40:73', 'HLA-B*40:74', 'HLA-B*40:75', 'HLA-B*40:76', 'HLA-B*40:77', 'HLA-B*40:78', 'HLA-B*40:79',
                            'HLA-B*40:80', 'HLA-B*40:81', 'HLA-B*40:82', 'HLA-B*40:83', 'HLA-B*40:84', 'HLA-B*40:85', 'HLA-B*40:86', 'HLA-B*40:87',
                            'HLA-B*40:88', 'HLA-B*40:89', 'HLA-B*40:90', 'HLA-B*40:91', 'HLA-B*40:92', 'HLA-B*40:93', 'HLA-B*40:94', 'HLA-B*40:95',
                            'HLA-B*40:96', 'HLA-B*40:97', 'HLA-B*40:98', 'HLA-B*40:99', 'HLA-B*40:100', 'HLA-B*40:101', 'HLA-B*40:102', 'HLA-B*40:103',
                            'HLA-B*40:104', 'HLA-B*40:105', 'HLA-B*40:106', 'HLA-B*40:107', 'HLA-B*40:108', 'HLA-B*40:109', 'HLA-B*40:110',
                            'HLA-B*40:111', 'HLA-B*40:112', 'HLA-B*40:113', 'HLA-B*40:114', 'HLA-B*40:115', 'HLA-B*40:116', 'HLA-B*40:117',
                            'HLA-B*40:119', 'HLA-B*40:120', 'HLA-B*40:121', 'HLA-B*40:122', 'HLA-B*40:123', 'HLA-B*40:124', 'HLA-B*40:125',
                            'HLA-B*40:126', 'HLA-B*40:127', 'HLA-B*40:128', 'HLA-B*40:129', 'HLA-B*40:130', 'HLA-B*40:131', 'HLA-B*40:132',
                            'HLA-B*40:134', 'HLA-B*40:135', 'HLA-B*40:136', 'HLA-B*40:137', 'HLA-B*40:138', 'HLA-B*40:139', 'HLA-B*40:140',
                            'HLA-B*40:141', 'HLA-B*40:143', 'HLA-B*40:145', 'HLA-B*40:146', 'HLA-B*40:147', 'HLA-B*41:01', 'HLA-B*41:02', 'HLA-B*41:03',
                            'HLA-B*41:04', 'HLA-B*41:05', 'HLA-B*41:06', 'HLA-B*41:07', 'HLA-B*41:08', 'HLA-B*41:09', 'HLA-B*41:10', 'HLA-B*41:11',
                            'HLA-B*41:12', 'HLA-B*42:01', 'HLA-B*42:02', 'HLA-B*42:04', 'HLA-B*42:05', 'HLA-B*42:06', 'HLA-B*42:07', 'HLA-B*42:08',
                            'HLA-B*42:09', 'HLA-B*42:10', 'HLA-B*42:11', 'HLA-B*42:12', 'HLA-B*42:13', 'HLA-B*42:14', 'HLA-B*44:02', 'HLA-B*44:03',
                            'HLA-B*44:04', 'HLA-B*44:05', 'HLA-B*44:06', 'HLA-B*44:07', 'HLA-B*44:08', 'HLA-B*44:09', 'HLA-B*44:10', 'HLA-B*44:11',
                            'HLA-B*44:12', 'HLA-B*44:13', 'HLA-B*44:14', 'HLA-B*44:15', 'HLA-B*44:16', 'HLA-B*44:17', 'HLA-B*44:18', 'HLA-B*44:20',
                            'HLA-B*44:21', 'HLA-B*44:22', 'HLA-B*44:24', 'HLA-B*44:25', 'HLA-B*44:26', 'HLA-B*44:27', 'HLA-B*44:28', 'HLA-B*44:29',
                            'HLA-B*44:30', 'HLA-B*44:31', 'HLA-B*44:32', 'HLA-B*44:33', 'HLA-B*44:34', 'HLA-B*44:35', 'HLA-B*44:36', 'HLA-B*44:37',
                            'HLA-B*44:38', 'HLA-B*44:39', 'HLA-B*44:40', 'HLA-B*44:41', 'HLA-B*44:42', 'HLA-B*44:43', 'HLA-B*44:44', 'HLA-B*44:45',
                            'HLA-B*44:46', 'HLA-B*44:47', 'HLA-B*44:48', 'HLA-B*44:49', 'HLA-B*44:50', 'HLA-B*44:51', 'HLA-B*44:53', 'HLA-B*44:54',
                            'HLA-B*44:55', 'HLA-B*44:57', 'HLA-B*44:59', 'HLA-B*44:60', 'HLA-B*44:62', 'HLA-B*44:63', 'HLA-B*44:64', 'HLA-B*44:65',
                            'HLA-B*44:66', 'HLA-B*44:67', 'HLA-B*44:68', 'HLA-B*44:69', 'HLA-B*44:70', 'HLA-B*44:71', 'HLA-B*44:72', 'HLA-B*44:73',
                            'HLA-B*44:74', 'HLA-B*44:75', 'HLA-B*44:76', 'HLA-B*44:77', 'HLA-B*44:78', 'HLA-B*44:79', 'HLA-B*44:80', 'HLA-B*44:81',
                            'HLA-B*44:82', 'HLA-B*44:83', 'HLA-B*44:84', 'HLA-B*44:85', 'HLA-B*44:86', 'HLA-B*44:87', 'HLA-B*44:88', 'HLA-B*44:89',
                            'HLA-B*44:90', 'HLA-B*44:91', 'HLA-B*44:92', 'HLA-B*44:93', 'HLA-B*44:94', 'HLA-B*44:95', 'HLA-B*44:96', 'HLA-B*44:97',
                            'HLA-B*44:98', 'HLA-B*44:99', 'HLA-B*44:100', 'HLA-B*44:101', 'HLA-B*44:102', 'HLA-B*44:103', 'HLA-B*44:104',
                            'HLA-B*44:105', 'HLA-B*44:106', 'HLA-B*44:107', 'HLA-B*44:109', 'HLA-B*44:110', 'HLA-B*45:01', 'HLA-B*45:02', 'HLA-B*45:03',
                            'HLA-B*45:04', 'HLA-B*45:05', 'HLA-B*45:06', 'HLA-B*45:07', 'HLA-B*45:08', 'HLA-B*45:09', 'HLA-B*45:10', 'HLA-B*45:11',
                            'HLA-B*45:12', 'HLA-B*46:01', 'HLA-B*46:02', 'HLA-B*46:03', 'HLA-B*46:04', 'HLA-B*46:05', 'HLA-B*46:06', 'HLA-B*46:08',
                            'HLA-B*46:09', 'HLA-B*46:10', 'HLA-B*46:11', 'HLA-B*46:12', 'HLA-B*46:13', 'HLA-B*46:14', 'HLA-B*46:16', 'HLA-B*46:17',
                            'HLA-B*46:18', 'HLA-B*46:19', 'HLA-B*46:20', 'HLA-B*46:21', 'HLA-B*46:22', 'HLA-B*46:23', 'HLA-B*46:24', 'HLA-B*47:01',
                            'HLA-B*47:02', 'HLA-B*47:03', 'HLA-B*47:04', 'HLA-B*47:05', 'HLA-B*47:06', 'HLA-B*47:07', 'HLA-B*48:01', 'HLA-B*48:02',
                            'HLA-B*48:03', 'HLA-B*48:04', 'HLA-B*48:05', 'HLA-B*48:06', 'HLA-B*48:07', 'HLA-B*48:08', 'HLA-B*48:09', 'HLA-B*48:10',
                            'HLA-B*48:11', 'HLA-B*48:12', 'HLA-B*48:13', 'HLA-B*48:14', 'HLA-B*48:15', 'HLA-B*48:16', 'HLA-B*48:17', 'HLA-B*48:18',
                            'HLA-B*48:19', 'HLA-B*48:20', 'HLA-B*48:21', 'HLA-B*48:22', 'HLA-B*48:23', 'HLA-B*49:01', 'HLA-B*49:02', 'HLA-B*49:03',
                            'HLA-B*49:04', 'HLA-B*49:05', 'HLA-B*49:06', 'HLA-B*49:07', 'HLA-B*49:08', 'HLA-B*49:09', 'HLA-B*49:10', 'HLA-B*50:01',
                            'HLA-B*50:02', 'HLA-B*50:04', 'HLA-B*50:05', 'HLA-B*50:06', 'HLA-B*50:07', 'HLA-B*50:08', 'HLA-B*50:09', 'HLA-B*51:01',
                            'HLA-B*51:02', 'HLA-B*51:03', 'HLA-B*51:04', 'HLA-B*51:05', 'HLA-B*51:06', 'HLA-B*51:07', 'HLA-B*51:08', 'HLA-B*51:09',
                            'HLA-B*51:12', 'HLA-B*51:13', 'HLA-B*51:14', 'HLA-B*51:15', 'HLA-B*51:16', 'HLA-B*51:17', 'HLA-B*51:18', 'HLA-B*51:19',
                            'HLA-B*51:20', 'HLA-B*51:21', 'HLA-B*51:22', 'HLA-B*51:23', 'HLA-B*51:24', 'HLA-B*51:26', 'HLA-B*51:28', 'HLA-B*51:29',
                            'HLA-B*51:30', 'HLA-B*51:31', 'HLA-B*51:32', 'HLA-B*51:33', 'HLA-B*51:34', 'HLA-B*51:35', 'HLA-B*51:36', 'HLA-B*51:37',
                            'HLA-B*51:38', 'HLA-B*51:39', 'HLA-B*51:40', 'HLA-B*51:42', 'HLA-B*51:43', 'HLA-B*51:45', 'HLA-B*51:46', 'HLA-B*51:48',
                            'HLA-B*51:49', 'HLA-B*51:50', 'HLA-B*51:51', 'HLA-B*51:52', 'HLA-B*51:53', 'HLA-B*51:54', 'HLA-B*51:55', 'HLA-B*51:56',
                            'HLA-B*51:57', 'HLA-B*51:58', 'HLA-B*51:59', 'HLA-B*51:60', 'HLA-B*51:61', 'HLA-B*51:62', 'HLA-B*51:63', 'HLA-B*51:64',
                            'HLA-B*51:65', 'HLA-B*51:66', 'HLA-B*51:67', 'HLA-B*51:68', 'HLA-B*51:69', 'HLA-B*51:70', 'HLA-B*51:71', 'HLA-B*51:72',
                            'HLA-B*51:73', 'HLA-B*51:74', 'HLA-B*51:75', 'HLA-B*51:76', 'HLA-B*51:77', 'HLA-B*51:78', 'HLA-B*51:79', 'HLA-B*51:80',
                            'HLA-B*51:81', 'HLA-B*51:82', 'HLA-B*51:83', 'HLA-B*51:84', 'HLA-B*51:85', 'HLA-B*51:86', 'HLA-B*51:87', 'HLA-B*51:88',
                            'HLA-B*51:89', 'HLA-B*51:90', 'HLA-B*51:91', 'HLA-B*51:92', 'HLA-B*51:93', 'HLA-B*51:94', 'HLA-B*51:95', 'HLA-B*51:96',
                            'HLA-B*52:01', 'HLA-B*52:02', 'HLA-B*52:03', 'HLA-B*52:04', 'HLA-B*52:05', 'HLA-B*52:06', 'HLA-B*52:07', 'HLA-B*52:08',
                            'HLA-B*52:09', 'HLA-B*52:10', 'HLA-B*52:11', 'HLA-B*52:12', 'HLA-B*52:13', 'HLA-B*52:14', 'HLA-B*52:15', 'HLA-B*52:16',
                            'HLA-B*52:17', 'HLA-B*52:18', 'HLA-B*52:19', 'HLA-B*52:20', 'HLA-B*52:21', 'HLA-B*53:01', 'HLA-B*53:02', 'HLA-B*53:03',
                            'HLA-B*53:04', 'HLA-B*53:05', 'HLA-B*53:06', 'HLA-B*53:07', 'HLA-B*53:08', 'HLA-B*53:09', 'HLA-B*53:10', 'HLA-B*53:11',
                            'HLA-B*53:12', 'HLA-B*53:13', 'HLA-B*53:14', 'HLA-B*53:15', 'HLA-B*53:16', 'HLA-B*53:17', 'HLA-B*53:18', 'HLA-B*53:19',
                            'HLA-B*53:20', 'HLA-B*53:21', 'HLA-B*53:22', 'HLA-B*53:23', 'HLA-B*54:01', 'HLA-B*54:02', 'HLA-B*54:03', 'HLA-B*54:04',
                            'HLA-B*54:06', 'HLA-B*54:07', 'HLA-B*54:09', 'HLA-B*54:10', 'HLA-B*54:11', 'HLA-B*54:12', 'HLA-B*54:13', 'HLA-B*54:14',
                            'HLA-B*54:15', 'HLA-B*54:16', 'HLA-B*54:17', 'HLA-B*54:18', 'HLA-B*54:19', 'HLA-B*54:20', 'HLA-B*54:21', 'HLA-B*54:22',
                            'HLA-B*54:23', 'HLA-B*55:01', 'HLA-B*55:02', 'HLA-B*55:03', 'HLA-B*55:04', 'HLA-B*55:05', 'HLA-B*55:07', 'HLA-B*55:08',
                            'HLA-B*55:09', 'HLA-B*55:10', 'HLA-B*55:11', 'HLA-B*55:12', 'HLA-B*55:13', 'HLA-B*55:14', 'HLA-B*55:15', 'HLA-B*55:16',
                            'HLA-B*55:17', 'HLA-B*55:18', 'HLA-B*55:19', 'HLA-B*55:20', 'HLA-B*55:21', 'HLA-B*55:22', 'HLA-B*55:23', 'HLA-B*55:24',
                            'HLA-B*55:25', 'HLA-B*55:26', 'HLA-B*55:27', 'HLA-B*55:28', 'HLA-B*55:29', 'HLA-B*55:30', 'HLA-B*55:31', 'HLA-B*55:32',
                            'HLA-B*55:33', 'HLA-B*55:34', 'HLA-B*55:35', 'HLA-B*55:36', 'HLA-B*55:37', 'HLA-B*55:38', 'HLA-B*55:39', 'HLA-B*55:40',
                            'HLA-B*55:41', 'HLA-B*55:42', 'HLA-B*55:43', 'HLA-B*56:01', 'HLA-B*56:02', 'HLA-B*56:03', 'HLA-B*56:04', 'HLA-B*56:05',
                            'HLA-B*56:06', 'HLA-B*56:07', 'HLA-B*56:08', 'HLA-B*56:09', 'HLA-B*56:10', 'HLA-B*56:11', 'HLA-B*56:12', 'HLA-B*56:13',
                            'HLA-B*56:14', 'HLA-B*56:15', 'HLA-B*56:16', 'HLA-B*56:17', 'HLA-B*56:18', 'HLA-B*56:20', 'HLA-B*56:21', 'HLA-B*56:22',
                            'HLA-B*56:23', 'HLA-B*56:24', 'HLA-B*56:25', 'HLA-B*56:26', 'HLA-B*56:27', 'HLA-B*56:29', 'HLA-B*57:01', 'HLA-B*57:02',
                            'HLA-B*57:03', 'HLA-B*57:04', 'HLA-B*57:05', 'HLA-B*57:06', 'HLA-B*57:07', 'HLA-B*57:08', 'HLA-B*57:09', 'HLA-B*57:10',
                            'HLA-B*57:11', 'HLA-B*57:12', 'HLA-B*57:13', 'HLA-B*57:14', 'HLA-B*57:15', 'HLA-B*57:16', 'HLA-B*57:17', 'HLA-B*57:18',
                            'HLA-B*57:19', 'HLA-B*57:20', 'HLA-B*57:21', 'HLA-B*57:22', 'HLA-B*57:23', 'HLA-B*57:24', 'HLA-B*57:25', 'HLA-B*57:26',
                            'HLA-B*57:27', 'HLA-B*57:29', 'HLA-B*57:30', 'HLA-B*57:31', 'HLA-B*57:32', 'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*58:04',
                            'HLA-B*58:05', 'HLA-B*58:06', 'HLA-B*58:07', 'HLA-B*58:08', 'HLA-B*58:09', 'HLA-B*58:11', 'HLA-B*58:12', 'HLA-B*58:13',
                            'HLA-B*58:14', 'HLA-B*58:15', 'HLA-B*58:16', 'HLA-B*58:18', 'HLA-B*58:19', 'HLA-B*58:20', 'HLA-B*58:21', 'HLA-B*58:22',
                            'HLA-B*58:23', 'HLA-B*58:24', 'HLA-B*58:25', 'HLA-B*58:26', 'HLA-B*58:27', 'HLA-B*58:28', 'HLA-B*58:29', 'HLA-B*58:30',
                            'HLA-B*59:01', 'HLA-B*59:02', 'HLA-B*59:03', 'HLA-B*59:04', 'HLA-B*59:05', 'HLA-B*67:01', 'HLA-B*67:02', 'HLA-B*73:01',
                            'HLA-B*73:02', 'HLA-B*78:01', 'HLA-B*78:02', 'HLA-B*78:03', 'HLA-B*78:04', 'HLA-B*78:05', 'HLA-B*78:06', 'HLA-B*78:07',
                            'HLA-B*81:01', 'HLA-B*81:02', 'HLA-B*81:03', 'HLA-B*81:05', 'HLA-B*82:01', 'HLA-B*82:02', 'HLA-B*82:03', 'HLA-B*83:01',
                            'HLA-C*01:02', 'HLA-C*01:03', 'HLA-C*01:04', 'HLA-C*01:05', 'HLA-C*01:06', 'HLA-C*01:07', 'HLA-C*01:08', 'HLA-C*01:09',
                            'HLA-C*01:10', 'HLA-C*01:11', 'HLA-C*01:12', 'HLA-C*01:13', 'HLA-C*01:14', 'HLA-C*01:15', 'HLA-C*01:16', 'HLA-C*01:17',
                            'HLA-C*01:18', 'HLA-C*01:19', 'HLA-C*01:20', 'HLA-C*01:21', 'HLA-C*01:22', 'HLA-C*01:23', 'HLA-C*01:24', 'HLA-C*01:25',
                            'HLA-C*01:26', 'HLA-C*01:27', 'HLA-C*01:28', 'HLA-C*01:29', 'HLA-C*01:30', 'HLA-C*01:31', 'HLA-C*01:32', 'HLA-C*01:33',
                            'HLA-C*01:34', 'HLA-C*01:35', 'HLA-C*01:36', 'HLA-C*01:38', 'HLA-C*01:39', 'HLA-C*01:40', 'HLA-C*02:02', 'HLA-C*02:03',
                            'HLA-C*02:04', 'HLA-C*02:05', 'HLA-C*02:06', 'HLA-C*02:07', 'HLA-C*02:08', 'HLA-C*02:09', 'HLA-C*02:10', 'HLA-C*02:11',
                            'HLA-C*02:12', 'HLA-C*02:13', 'HLA-C*02:14', 'HLA-C*02:15', 'HLA-C*02:16', 'HLA-C*02:17', 'HLA-C*02:18', 'HLA-C*02:19',
                            'HLA-C*02:20', 'HLA-C*02:21', 'HLA-C*02:22', 'HLA-C*02:23', 'HLA-C*02:24', 'HLA-C*02:26', 'HLA-C*02:27', 'HLA-C*02:28',
                            'HLA-C*02:29', 'HLA-C*02:30', 'HLA-C*02:31', 'HLA-C*02:32', 'HLA-C*02:33', 'HLA-C*02:34', 'HLA-C*02:35', 'HLA-C*02:36',
                            'HLA-C*02:37', 'HLA-C*02:39', 'HLA-C*02:40', 'HLA-C*03:01', 'HLA-C*03:02', 'HLA-C*03:03', 'HLA-C*03:04', 'HLA-C*03:05',
                            'HLA-C*03:06', 'HLA-C*03:07', 'HLA-C*03:08', 'HLA-C*03:09', 'HLA-C*03:10', 'HLA-C*03:11', 'HLA-C*03:12', 'HLA-C*03:13',
                            'HLA-C*03:14', 'HLA-C*03:15', 'HLA-C*03:16', 'HLA-C*03:17', 'HLA-C*03:18', 'HLA-C*03:19', 'HLA-C*03:21', 'HLA-C*03:23',
                            'HLA-C*03:24', 'HLA-C*03:25', 'HLA-C*03:26', 'HLA-C*03:27', 'HLA-C*03:28', 'HLA-C*03:29', 'HLA-C*03:30', 'HLA-C*03:31',
                            'HLA-C*03:32', 'HLA-C*03:33', 'HLA-C*03:34', 'HLA-C*03:35', 'HLA-C*03:36', 'HLA-C*03:37', 'HLA-C*03:38', 'HLA-C*03:39',
                            'HLA-C*03:40', 'HLA-C*03:41', 'HLA-C*03:42', 'HLA-C*03:43', 'HLA-C*03:44', 'HLA-C*03:45', 'HLA-C*03:46', 'HLA-C*03:47',
                            'HLA-C*03:48', 'HLA-C*03:49', 'HLA-C*03:50', 'HLA-C*03:51', 'HLA-C*03:52', 'HLA-C*03:53', 'HLA-C*03:54', 'HLA-C*03:55',
                            'HLA-C*03:56', 'HLA-C*03:57', 'HLA-C*03:58', 'HLA-C*03:59', 'HLA-C*03:60', 'HLA-C*03:61', 'HLA-C*03:62', 'HLA-C*03:63',
                            'HLA-C*03:64', 'HLA-C*03:65', 'HLA-C*03:66', 'HLA-C*03:67', 'HLA-C*03:68', 'HLA-C*03:69', 'HLA-C*03:70', 'HLA-C*03:71',
                            'HLA-C*03:72', 'HLA-C*03:73', 'HLA-C*03:74', 'HLA-C*03:75', 'HLA-C*03:76', 'HLA-C*03:77', 'HLA-C*03:78', 'HLA-C*03:79',
                            'HLA-C*03:80', 'HLA-C*03:81', 'HLA-C*03:82', 'HLA-C*03:83', 'HLA-C*03:84', 'HLA-C*03:85', 'HLA-C*03:86', 'HLA-C*03:87',
                            'HLA-C*03:88', 'HLA-C*03:89', 'HLA-C*03:90', 'HLA-C*03:91', 'HLA-C*03:92', 'HLA-C*03:93', 'HLA-C*03:94', 'HLA-C*04:01',
                            'HLA-C*04:03', 'HLA-C*04:04', 'HLA-C*04:05', 'HLA-C*04:06', 'HLA-C*04:07', 'HLA-C*04:08', 'HLA-C*04:10', 'HLA-C*04:11',
                            'HLA-C*04:12', 'HLA-C*04:13', 'HLA-C*04:14', 'HLA-C*04:15', 'HLA-C*04:16', 'HLA-C*04:17', 'HLA-C*04:18', 'HLA-C*04:19',
                            'HLA-C*04:20', 'HLA-C*04:23', 'HLA-C*04:24', 'HLA-C*04:25', 'HLA-C*04:26', 'HLA-C*04:27', 'HLA-C*04:28', 'HLA-C*04:29',
                            'HLA-C*04:30', 'HLA-C*04:31', 'HLA-C*04:32', 'HLA-C*04:33', 'HLA-C*04:34', 'HLA-C*04:35', 'HLA-C*04:36', 'HLA-C*04:37',
                            'HLA-C*04:38', 'HLA-C*04:39', 'HLA-C*04:40', 'HLA-C*04:41', 'HLA-C*04:42', 'HLA-C*04:43', 'HLA-C*04:44', 'HLA-C*04:45',
                            'HLA-C*04:46', 'HLA-C*04:47', 'HLA-C*04:48', 'HLA-C*04:49', 'HLA-C*04:50', 'HLA-C*04:51', 'HLA-C*04:52', 'HLA-C*04:53',
                            'HLA-C*04:54', 'HLA-C*04:55', 'HLA-C*04:56', 'HLA-C*04:57', 'HLA-C*04:58', 'HLA-C*04:60', 'HLA-C*04:61', 'HLA-C*04:62',
                            'HLA-C*04:63', 'HLA-C*04:64', 'HLA-C*04:65', 'HLA-C*04:66', 'HLA-C*04:67', 'HLA-C*04:68', 'HLA-C*04:69', 'HLA-C*04:70',
                            'HLA-C*05:01', 'HLA-C*05:03', 'HLA-C*05:04', 'HLA-C*05:05', 'HLA-C*05:06', 'HLA-C*05:08', 'HLA-C*05:09', 'HLA-C*05:10',
                            'HLA-C*05:11', 'HLA-C*05:12', 'HLA-C*05:13', 'HLA-C*05:14', 'HLA-C*05:15', 'HLA-C*05:16', 'HLA-C*05:17', 'HLA-C*05:18',
                            'HLA-C*05:19', 'HLA-C*05:20', 'HLA-C*05:21', 'HLA-C*05:22', 'HLA-C*05:23', 'HLA-C*05:24', 'HLA-C*05:25', 'HLA-C*05:26',
                            'HLA-C*05:27', 'HLA-C*05:28', 'HLA-C*05:29', 'HLA-C*05:30', 'HLA-C*05:31', 'HLA-C*05:32', 'HLA-C*05:33', 'HLA-C*05:34',
                            'HLA-C*05:35', 'HLA-C*05:36', 'HLA-C*05:37', 'HLA-C*05:38', 'HLA-C*05:39', 'HLA-C*05:40', 'HLA-C*05:41', 'HLA-C*05:42',
                            'HLA-C*05:43', 'HLA-C*05:44', 'HLA-C*05:45', 'HLA-C*06:02', 'HLA-C*06:03', 'HLA-C*06:04', 'HLA-C*06:05', 'HLA-C*06:06',
                            'HLA-C*06:07', 'HLA-C*06:08', 'HLA-C*06:09', 'HLA-C*06:10', 'HLA-C*06:11', 'HLA-C*06:12', 'HLA-C*06:13', 'HLA-C*06:14',
                            'HLA-C*06:15', 'HLA-C*06:17', 'HLA-C*06:18', 'HLA-C*06:19', 'HLA-C*06:20', 'HLA-C*06:21', 'HLA-C*06:22', 'HLA-C*06:23',
                            'HLA-C*06:24', 'HLA-C*06:25', 'HLA-C*06:26', 'HLA-C*06:27', 'HLA-C*06:28', 'HLA-C*06:29', 'HLA-C*06:30', 'HLA-C*06:31',
                            'HLA-C*06:32', 'HLA-C*06:33', 'HLA-C*06:34', 'HLA-C*06:35', 'HLA-C*06:36', 'HLA-C*06:37', 'HLA-C*06:38', 'HLA-C*06:39',
                            'HLA-C*06:40', 'HLA-C*06:41', 'HLA-C*06:42', 'HLA-C*06:43', 'HLA-C*06:44', 'HLA-C*06:45', 'HLA-C*07:01', 'HLA-C*07:02',
                            'HLA-C*07:03', 'HLA-C*07:04', 'HLA-C*07:05', 'HLA-C*07:06', 'HLA-C*07:07', 'HLA-C*07:08', 'HLA-C*07:09', 'HLA-C*07:10',
                            'HLA-C*07:11', 'HLA-C*07:12', 'HLA-C*07:13', 'HLA-C*07:14', 'HLA-C*07:15', 'HLA-C*07:16', 'HLA-C*07:17', 'HLA-C*07:18',
                            'HLA-C*07:19', 'HLA-C*07:20', 'HLA-C*07:21', 'HLA-C*07:22', 'HLA-C*07:23', 'HLA-C*07:24', 'HLA-C*07:25', 'HLA-C*07:26',
                            'HLA-C*07:27', 'HLA-C*07:28', 'HLA-C*07:29', 'HLA-C*07:30', 'HLA-C*07:31', 'HLA-C*07:35', 'HLA-C*07:36', 'HLA-C*07:37',
                            'HLA-C*07:38', 'HLA-C*07:39', 'HLA-C*07:40', 'HLA-C*07:41', 'HLA-C*07:42', 'HLA-C*07:43', 'HLA-C*07:44', 'HLA-C*07:45',
                            'HLA-C*07:46', 'HLA-C*07:47', 'HLA-C*07:48', 'HLA-C*07:49', 'HLA-C*07:50', 'HLA-C*07:51', 'HLA-C*07:52', 'HLA-C*07:53',
                            'HLA-C*07:54', 'HLA-C*07:56', 'HLA-C*07:57', 'HLA-C*07:58', 'HLA-C*07:59', 'HLA-C*07:60', 'HLA-C*07:62', 'HLA-C*07:63',
                            'HLA-C*07:64', 'HLA-C*07:65', 'HLA-C*07:66', 'HLA-C*07:67', 'HLA-C*07:68', 'HLA-C*07:69', 'HLA-C*07:70', 'HLA-C*07:71',
                            'HLA-C*07:72', 'HLA-C*07:73', 'HLA-C*07:74', 'HLA-C*07:75', 'HLA-C*07:76', 'HLA-C*07:77', 'HLA-C*07:78', 'HLA-C*07:79',
                            'HLA-C*07:80', 'HLA-C*07:81', 'HLA-C*07:82', 'HLA-C*07:83', 'HLA-C*07:84', 'HLA-C*07:85', 'HLA-C*07:86', 'HLA-C*07:87',
                            'HLA-C*07:88', 'HLA-C*07:89', 'HLA-C*07:90', 'HLA-C*07:91', 'HLA-C*07:92', 'HLA-C*07:93', 'HLA-C*07:94', 'HLA-C*07:95',
                            'HLA-C*07:96', 'HLA-C*07:97', 'HLA-C*07:99', 'HLA-C*07:100', 'HLA-C*07:101', 'HLA-C*07:102', 'HLA-C*07:103', 'HLA-C*07:105',
                            'HLA-C*07:106', 'HLA-C*07:107', 'HLA-C*07:108', 'HLA-C*07:109', 'HLA-C*07:110', 'HLA-C*07:111', 'HLA-C*07:112',
                            'HLA-C*07:113', 'HLA-C*07:114', 'HLA-C*07:115', 'HLA-C*07:116', 'HLA-C*07:117', 'HLA-C*07:118', 'HLA-C*07:119',
                            'HLA-C*07:120', 'HLA-C*07:122', 'HLA-C*07:123', 'HLA-C*07:124', 'HLA-C*07:125', 'HLA-C*07:126', 'HLA-C*07:127',
                            'HLA-C*07:128', 'HLA-C*07:129', 'HLA-C*07:130', 'HLA-C*07:131', 'HLA-C*07:132', 'HLA-C*07:133', 'HLA-C*07:134',
                            'HLA-C*07:135', 'HLA-C*07:136', 'HLA-C*07:137', 'HLA-C*07:138', 'HLA-C*07:139', 'HLA-C*07:140', 'HLA-C*07:141',
                            'HLA-C*07:142', 'HLA-C*07:143', 'HLA-C*07:144', 'HLA-C*07:145', 'HLA-C*07:146', 'HLA-C*07:147', 'HLA-C*07:148',
                            'HLA-C*07:149', 'HLA-C*08:01', 'HLA-C*08:02', 'HLA-C*08:03', 'HLA-C*08:04', 'HLA-C*08:05', 'HLA-C*08:06', 'HLA-C*08:07',
                            'HLA-C*08:08', 'HLA-C*08:09', 'HLA-C*08:10', 'HLA-C*08:11', 'HLA-C*08:12', 'HLA-C*08:13', 'HLA-C*08:14', 'HLA-C*08:15',
                            'HLA-C*08:16', 'HLA-C*08:17', 'HLA-C*08:18', 'HLA-C*08:19', 'HLA-C*08:20', 'HLA-C*08:21', 'HLA-C*08:22', 'HLA-C*08:23',
                            'HLA-C*08:24', 'HLA-C*08:25', 'HLA-C*08:27', 'HLA-C*08:28', 'HLA-C*08:29', 'HLA-C*08:30', 'HLA-C*08:31', 'HLA-C*08:32',
                            'HLA-C*08:33', 'HLA-C*08:34', 'HLA-C*08:35', 'HLA-C*12:02', 'HLA-C*12:03', 'HLA-C*12:04', 'HLA-C*12:05', 'HLA-C*12:06',
                            'HLA-C*12:07', 'HLA-C*12:08', 'HLA-C*12:09', 'HLA-C*12:10', 'HLA-C*12:11', 'HLA-C*12:12', 'HLA-C*12:13', 'HLA-C*12:14',
                            'HLA-C*12:15', 'HLA-C*12:16', 'HLA-C*12:17', 'HLA-C*12:18', 'HLA-C*12:19', 'HLA-C*12:20', 'HLA-C*12:21', 'HLA-C*12:22',
                            'HLA-C*12:23', 'HLA-C*12:24', 'HLA-C*12:25', 'HLA-C*12:26', 'HLA-C*12:27', 'HLA-C*12:28', 'HLA-C*12:29', 'HLA-C*12:30',
                            'HLA-C*12:31', 'HLA-C*12:32', 'HLA-C*12:33', 'HLA-C*12:34', 'HLA-C*12:35', 'HLA-C*12:36', 'HLA-C*12:37', 'HLA-C*12:38',
                            'HLA-C*12:40', 'HLA-C*12:41', 'HLA-C*12:43', 'HLA-C*12:44', 'HLA-C*14:02', 'HLA-C*14:03', 'HLA-C*14:04', 'HLA-C*14:05',
                            'HLA-C*14:06', 'HLA-C*14:08', 'HLA-C*14:09', 'HLA-C*14:10', 'HLA-C*14:11', 'HLA-C*14:12', 'HLA-C*14:13', 'HLA-C*14:14',
                            'HLA-C*14:15', 'HLA-C*14:16', 'HLA-C*14:17', 'HLA-C*14:18', 'HLA-C*14:19', 'HLA-C*14:20', 'HLA-C*15:02', 'HLA-C*15:03',
                            'HLA-C*15:04', 'HLA-C*15:05', 'HLA-C*15:06', 'HLA-C*15:07', 'HLA-C*15:08', 'HLA-C*15:09', 'HLA-C*15:10', 'HLA-C*15:11',
                            'HLA-C*15:12', 'HLA-C*15:13', 'HLA-C*15:15', 'HLA-C*15:16', 'HLA-C*15:17', 'HLA-C*15:18', 'HLA-C*15:19', 'HLA-C*15:20',
                            'HLA-C*15:21', 'HLA-C*15:22', 'HLA-C*15:23', 'HLA-C*15:24', 'HLA-C*15:25', 'HLA-C*15:26', 'HLA-C*15:27', 'HLA-C*15:28',
                            'HLA-C*15:29', 'HLA-C*15:30', 'HLA-C*15:31', 'HLA-C*15:33', 'HLA-C*15:34', 'HLA-C*15:35', 'HLA-C*16:01', 'HLA-C*16:02',
                            'HLA-C*16:04', 'HLA-C*16:06', 'HLA-C*16:07', 'HLA-C*16:08', 'HLA-C*16:09', 'HLA-C*16:10', 'HLA-C*16:11', 'HLA-C*16:12',
                            'HLA-C*16:13', 'HLA-C*16:14', 'HLA-C*16:15', 'HLA-C*16:17', 'HLA-C*16:18', 'HLA-C*16:19', 'HLA-C*16:20', 'HLA-C*16:21',
                            'HLA-C*16:22', 'HLA-C*16:23', 'HLA-C*16:24', 'HLA-C*16:25', 'HLA-C*16:26', 'HLA-C*17:01', 'HLA-C*17:02', 'HLA-C*17:03',
                            'HLA-C*17:04', 'HLA-C*17:05', 'HLA-C*17:06', 'HLA-C*17:07', 'HLA-C*18:01', 'HLA-C*18:02', 'HLA-C*18:03', 'HLA-G*01:01',
                            'HLA-G*01:02', 'HLA-G*01:03', 'HLA-G*01:04', 'HLA-G*01:06', 'HLA-G*01:07', 'HLA-G*01:08', 'HLA-G*01:09', 'HLA-E*01:01'])

    @property
    def command(self):
        return self.__command

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def supportedAlleles(self):
        return self.__alleles

    @property
    def supportedLength(self):
        return self.__length

    def convert_alleles(self, alleles):
        """
        Converts :class:`~epytope.Core.Allele.Allele` into the internal allele representation of the predictor
        and returns a string representation

        :param  alleles: The :class:`~epytope.Core.Allele.Allele` for which the
                         internal predictor representation is needed
        :type alleles: list(:class:`~epytope.Core.Allele.Allele`)
        :return: Returns a string representation of the input :class:`~epytope.Core.Allele.Allele`
        :rtype: list(str)
        """
        return ["HLA-%s%s:%s" % (a.locus, a.supertype, a.subtype) for a in alleles]

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        result = defaultdict(dict)
        with open(file, "r") as f:
            f = csv.reader(f, delimiter='\t')
            alleles = [x for x in next(f) if x != ""]
            ranks = defaultdict(defaultdict)
            rank_pos = 5
            offset = 3
            header = next(f)
            if "Aff(nM)" in header:  # With option command line option '-ia', which includes prediction score in output file
                scores = defaultdict(defaultdict)
                for row in f:
                    pep_seq = row[PeptideIndex.NETMHCSTABPAN_1_0]
                    for i, a in enumerate(alleles):
                        scores[a][pep_seq] = float(row[ScoreIndex.NETMHCSTABPAN_1_0 + i * Offset.NETMHCSTABPAN_1_0_W_SCORE])
                        ranks[a][pep_seq] = float(row[RankIndex.NETMHCSTABPAN_1_0 + i * Offset.NETMHCSTABPAN_1_0_W_SCORE])
                        # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
                result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}
            else:
                for row in f:
                    pep_seq = row[PeptideIndex.NETMHCSTABPAN_1_0]
                    for i, a in enumerate(alleles):
                        ranks[a][pep_seq] = float(row[RankIndex.NETMHCSTABPAN_1_0 + i * Offset.NETMHCSTABPAN_1_0_WO_SCORE])
                        # Create dictionary with hierarchy: {'Allele1':{'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
                result = {allele: {"Rank":list(ranks.values())[j]} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        # can not be determined netmhcpan does not support --version or similar
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        NO return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))



class NetMHCII_2_2(AExternalEpitopePrediction):
    """
    Implements a wrapper for NetMHCII

    .. note::

        Nielsen, M., & Lund, O. (2009). NN-align. An artificial neural network-based alignment algorithm for MHC class
        II peptide binding prediction. BMC Bioinformatics, 10(1), 296.

        Nielsen, M., Lundegaard, C., & Lund, O. (2007). Prediction of MHC class II binding affinity using SMM-align,
        a novel stabilization matrix alignment method. BMC Bioinformatics, 8(1), 238.
    """
    __supported_length = frozenset([15])
    __name = "netmhcII"
    __command = 'netMHCII {peptides} -a {alleles} {options} | grep -v "#" > {out}'
    __alleles = frozenset(
        ['HLA-DRB1*01:01', 'HLA-DRB1*03:01', 'HLA-DRB1*04:01', 'HLA-DRB1*04:04', 'HLA-DRB1*04:05', 'HLA-DRB1*07:01', 'HLA-DRB1*08:02', 'HLA-DRB1*09:01',
         'HLA-DRB1*11:01', 'HLA-DRB1*13:02', 'HLA-DRB1*15:01', 'HLA-DRB3*01:01', 'HLA-DRB4*01:01', 'HLA-DRB5*01:01',
         'H-2-Iab', 'H-2-Iad'])
    __version = "2.2"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        for r in f:
            if not r:
                continue
            
            row = r[0].split()
            if not len(row):
                continue
            
            if "HLA" not in row[HLAIndex.NETMHCII_2_2]:
                continue
            allele = row[HLAIndex.NETMHCII_2_2]
            pep = row[PeptideIndex.NETMHCII_2_2]
            scores[allele][pep] = float(row[ScoreIndex.NETMHCII_2_2])

        result = {allele: {"Score":list(scores.values())[j]} for j, allele in enumerate(scores.keys())}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to _file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(">pepe_%i\n%s" % (i, p) for i, p in enumerate(input)))


class NetMHCII_2_3(NetMHCII_2_2):
    """
    Implements a wrapper for NetMHCII 2.3

    .. note::

        Jensen KK, Andreatta M, Marcatili P, Buus S, Greenbaum JA, Yan Z, Sette A, Peters B, and Nielsen M. (2018)
        Improved methods for predicting peptide binding affinity to MHC class II molecules. 
    """
    __supported_length = frozenset([15])
    __name = "netmhcII"
    __command = 'netMHCII {peptides} -a {alleles} {options} | grep -v "#" > {out}'
    __alleles = frozenset(
        ['HLA-DRB1*01:01', 'HLA-DRB1*01:03', 'HLA-DRB1*03:01', 'HLA-DRB1*04:01', 'HLA-DRB1*04:02',
'HLA-DRB1*04:03', 'HLA-DRB1*04:04', 'HLA-DRB1*04:05', 'HLA-DRB1*07:01', 'HLA-DRB1*08:01',
'HLA-DRB1*08:02', 'HLA-DRB1*09:01', 'HLA-DRB1*10:01', 'HLA-DRB1*11:01', 'HLA-DRB1*12:01',
'HLA-DRB1*13:01', 'HLA-DRB1*13:02', 'HLA-DRB1*15:01', 'HLA-DRB1*16:02', 'HLA-DRB3*01:01',
'HLA-DRB3*02:02', 'HLA-DRB3*03:01', 'HLA-DRB4*01:01', 'HLA-DRB4*01:03', 'HLA-DRB5*01:01',
'HLA-DPA1*01:03-DPB1*02:01', 'HLA-DPA1*01:03-DPB1*03:01', 'HLA-DPA1*01:03-DPB1*04:01',
'HLA-DPA1*01:03-DPB1*04:02', 'HLA-DPA1*01:03-DPB1*06:01', 'HLA-DPA1*02:01-DPB1*01:01', 'HLA-DPA1*02:01-DPB1*05:01', 'HLA-DPA1*02:01-DPB1*14:01',
'HLA-DPA1*03:01-DPB1*04:02', 'HLA-DQA1*01:01-DQB1*05:01', 'HLA-DQA1*01:02-DQB1*05:01', 'HLA-DQA1*01:02-DQB1*05:02', 'HLA-DQA1*01:02-DQB1*06:02',
'HLA-DQA1*01:03-DQB1*06:03', 'HLA-DQA1*01:04-DQB1*05:03', 'HLA-DQA1*02:01-DQB1*02:02', 'HLA-DQA1*02:01-DQB1*03:01', 'HLA-DQA1*02:01-DQB1*03:03',
'HLA-DQA1*02:01-DQB1*04:02', 'HLA-DQA1*03:01-DQB1*03:01', 'HLA-DQA1*03:01-DQB1*03:02', 'HLA-DQA1*03:03-DQB1*04:02', 'HLA-DQA1*04:01-DQB1*04:02',
'HLA-DQA1*05:01-DQB1*02:01', 'HLA-DQA1*05:01-DQB1*03:01', 'HLA-DQA1*05:01-DQB1*03:02', 'HLA-DQA1*05:01-DQB1*03:03', 'HLA-DQA1*05:01-DQB1*04:02',
'HLA-DQA1*06:01-DQB1*04:02', 'H-2-Iab', 'H-2-Iad', 'H-2-Iak', 'H-2-Ias', 'H-2-Iau', 'H-2-Iad', 'H-2-Iak'])
    __version = "2.3"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        elif isinstance(allele, CombinedAllele):
            return '%s-%s%s%s-%s%s%s' % (allele.organism, allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                          allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
        else:
            return "%s_%s%s" % (allele.locus, allele.supertype, allele.subtype)


    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)

        for r in f:
            if not r:
                continue
            
            row = r[0].split()
            if not len(row):
                continue
            
            if all(prefix not in row[HLAIndex.NETMHCII_2_3] for prefix in ['HLA-', 'H-2', 'D']):
                continue

            allele = row[HLAIndex.NETMHCII_2_3]
            
            pep = row[PeptideIndex.NETMHCII_2_3]
            scores[allele][pep] = float(row[ScoreIndex.NETMHCII_2_3])
            ranks[allele][pep] = float(row[ScoreIndex.NETMHCII_2_3])
            

        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(scores.keys())}

        return result


class NetMHCIIpan_3_0(AExternalEpitopePrediction):
    """
    Implements a wrapper for NetMHCIIpan.

    .. note::

        Andreatta, M., Karosiene, E., Rasmussen, M., Stryhn, A., Buus, S., & Nielsen, M. (2015).
        Accurate pan-specific prediction of peptide-MHC class II binding affinity with improved binding
        core identification. Immunogenetics, 1-10.
    """

    __supported_length = frozenset([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    __name = "netmhcIIpan"
    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"
    __alleles = frozenset(
        ['HLA-DRB1*01:01', 'HLA-DRB1*01:02', 'HLA-DRB1*01:03', 'HLA-DRB1*01:04', 'HLA-DRB1*01:05', 'HLA-DRB1*01:06',
         'HLA-DRB1*01:07', 'HLA-DRB1*01:08', 'HLA-DRB1*01:09', 'HLA-DRB1*01:10', 'HLA-DRB1*01:11', 'HLA-DRB1*01:12',
         'HLA-DRB1*01:13', 'HLA-DRB1*01:14', 'HLA-DRB1*01:15', 'HLA-DRB1*01:16', 'HLA-DRB1*01:17', 'HLA-DRB1*01:18',
         'HLA-DRB1*01:19', 'HLA-DRB1*01:20', 'HLA-DRB1*01:21', 'HLA-DRB1*01:22', 'HLA-DRB1*01:23', 'HLA-DRB1*01:24',
         'HLA-DRB1*01:25', 'HLA-DRB1*01:26', 'HLA-DRB1*01:27', 'HLA-DRB1*01:28', 'HLA-DRB1*01:29', 'HLA-DRB1*01:30',
         'HLA-DRB1*01:31', 'HLA-DRB1*01:32', 'HLA-DRB1*03:01', 'HLA-DRB1*03:02', 'HLA-DRB1*03:03', 'HLA-DRB1*03:04',
         'HLA-DRB1*03:05', 'HLA-DRB1*03:06', 'HLA-DRB1*03:07', 'HLA-DRB1*03:08', 'HLA-DRB1*03:10', 'HLA-DRB1*03:11',
         'HLA-DRB1*03:13', 'HLA-DRB1*03:14', 'HLA-DRB1*03:15', 'HLA-DRB1*03:17', 'HLA-DRB1*03:18', 'HLA-DRB1*03:19',
         'HLA-DRB1*03:20', 'HLA-DRB1*03:21', 'HLA-DRB1*03:22', 'HLA-DRB1*03:23', 'HLA-DRB1*03:24', 'HLA-DRB1*03:25',
         'HLA-DRB1*03:26', 'HLA-DRB1*03:27', 'HLA-DRB1*03:28', 'HLA-DRB1*03:29', 'HLA-DRB1*03:30', 'HLA-DRB1*03:31',
         'HLA-DRB1*03:32', 'HLA-DRB1*03:33', 'HLA-DRB1*03:34', 'HLA-DRB1*03:35', 'HLA-DRB1*03:36', 'HLA-DRB1*03:37',
         'HLA-DRB1*03:38', 'HLA-DRB1*03:39', 'HLA-DRB1*03:40', 'HLA-DRB1*03:41', 'HLA-DRB1*03:42', 'HLA-DRB1*03:43',
         'HLA-DRB1*03:44', 'HLA-DRB1*03:45', 'HLA-DRB1*03:46', 'HLA-DRB1*03:47', 'HLA-DRB1*03:48', 'HLA-DRB1*03:49',
         'HLA-DRB1*03:50', 'HLA-DRB1*03:51', 'HLA-DRB1*03:52', 'HLA-DRB1*03:53', 'HLA-DRB1*03:54', 'HLA-DRB1*03:55',
         'HLA-DRB1*04:01', 'HLA-DRB1*04:02', 'HLA-DRB1*04:03', 'HLA-DRB1*04:04', 'HLA-DRB1*04:05', 'HLA-DRB1*04:06',
         'HLA-DRB1*04:07', 'HLA-DRB1*04:08', 'HLA-DRB1*04:09', 'HLA-DRB1*04:10', 'HLA-DRB1*04:11', 'HLA-DRB1*04:12',
         'HLA-DRB1*04:13', 'HLA-DRB1*04:14', 'HLA-DRB1*04:15', 'HLA-DRB1*04:16', 'HLA-DRB1*04:17', 'HLA-DRB1*04:18',
         'HLA-DRB1*04:19', 'HLA-DRB1*04:21', 'HLA-DRB1*04:22', 'HLA-DRB1*04:23', 'HLA-DRB1*04:24', 'HLA-DRB1*04:26',
         'HLA-DRB1*04:27', 'HLA-DRB1*04:28', 'HLA-DRB1*04:29', 'HLA-DRB1*04:30', 'HLA-DRB1*04:31', 'HLA-DRB1*04:33',
         'HLA-DRB1*04:34', 'HLA-DRB1*04:35', 'HLA-DRB1*04:36', 'HLA-DRB1*04:37', 'HLA-DRB1*04:38', 'HLA-DRB1*04:39',
         'HLA-DRB1*04:40', 'HLA-DRB1*04:41', 'HLA-DRB1*04:42', 'HLA-DRB1*04:43', 'HLA-DRB1*04:44', 'HLA-DRB1*04:45',
         'HLA-DRB1*04:46', 'HLA-DRB1*04:47', 'HLA-DRB1*04:48', 'HLA-DRB1*04:49', 'HLA-DRB1*04:50', 'HLA-DRB1*04:51',
         'HLA-DRB1*04:52', 'HLA-DRB1*04:53', 'HLA-DRB1*04:54', 'HLA-DRB1*04:55', 'HLA-DRB1*04:56', 'HLA-DRB1*04:57',
         'HLA-DRB1*04:58', 'HLA-DRB1*04:59', 'HLA-DRB1*04:60', 'HLA-DRB1*04:61', 'HLA-DRB1*04:62', 'HLA-DRB1*04:63',
         'HLA-DRB1*04:64', 'HLA-DRB1*04:65', 'HLA-DRB1*04:66', 'HLA-DRB1*04:67', 'HLA-DRB1*04:68', 'HLA-DRB1*04:69',
         'HLA-DRB1*04:70', 'HLA-DRB1*04:71', 'HLA-DRB1*04:72', 'HLA-DRB1*04:73', 'HLA-DRB1*04:74', 'HLA-DRB1*04:75',
         'HLA-DRB1*04:76', 'HLA-DRB1*04:77', 'HLA-DRB1*04:78', 'HLA-DRB1*04:79', 'HLA-DRB1*04:80', 'HLA-DRB1*04:82',
         'HLA-DRB1*04:83', 'HLA-DRB1*04:84', 'HLA-DRB1*04:85', 'HLA-DRB1*04:86', 'HLA-DRB1*04:87', 'HLA-DRB1*04:88',
         'HLA-DRB1*04:89', 'HLA-DRB1*04:91', 'HLA-DRB1*07:01', 'HLA-DRB1*07:03', 'HLA-DRB1*07:04', 'HLA-DRB1*07:05',
         'HLA-DRB1*07:06', 'HLA-DRB1*07:07', 'HLA-DRB1*07:08', 'HLA-DRB1*07:09', 'HLA-DRB1*07:11', 'HLA-DRB1*07:12',
         'HLA-DRB1*07:13', 'HLA-DRB1*07:14', 'HLA-DRB1*07:15', 'HLA-DRB1*07:16', 'HLA-DRB1*07:17', 'HLA-DRB1*07:19',
         'HLA-DRB1*08:01', 'HLA-DRB1*08:02', 'HLA-DRB1*08:03', 'HLA-DRB1*08:04', 'HLA-DRB1*08:05', 'HLA-DRB1*08:06',
         'HLA-DRB1*08:07', 'HLA-DRB1*08:08', 'HLA-DRB1*08:09', 'HLA-DRB1*08:10', 'HLA-DRB1*08:11', 'HLA-DRB1*08:12',
         'HLA-DRB1*08:13', 'HLA-DRB1*08:14', 'HLA-DRB1*08:15', 'HLA-DRB1*08:16', 'HLA-DRB1*08:18', 'HLA-DRB1*08:19',
         'HLA-DRB1*08:20', 'HLA-DRB1*08:21', 'HLA-DRB1*08:22', 'HLA-DRB1*08:23', 'HLA-DRB1*08:24', 'HLA-DRB1*08:25',
         'HLA-DRB1*08:26', 'HLA-DRB1*08:27', 'HLA-DRB1*08:28', 'HLA-DRB1*08:29', 'HLA-DRB1*08:30', 'HLA-DRB1*08:31',
         'HLA-DRB1*08:32', 'HLA-DRB1*08:33', 'HLA-DRB1*08:34', 'HLA-DRB1*08:35', 'HLA-DRB1*08:36', 'HLA-DRB1*08:37',
         'HLA-DRB1*08:38', 'HLA-DRB1*08:39', 'HLA-DRB1*08:40', 'HLA-DRB1*09:01', 'HLA-DRB1*09:02', 'HLA-DRB1*09:03',
         'HLA-DRB1*09:04', 'HLA-DRB1*09:05', 'HLA-DRB1*09:06', 'HLA-DRB1*09:07', 'HLA-DRB1*09:08', 'HLA-DRB1*09:09',
         'HLA-DRB1*10:01', 'HLA-DRB1*10:02', 'HLA-DRB1*10:03', 'HLA-DRB1*11:01', 'HLA-DRB1*11:02', 'HLA-DRB1*11:03',
         'HLA-DRB1*11:04', 'HLA-DRB1*11:05', 'HLA-DRB1*11:06', 'HLA-DRB1*11:07', 'HLA-DRB1*11:08', 'HLA-DRB1*11:09',
         'HLA-DRB1*11:10', 'HLA-DRB1*11:11', 'HLA-DRB1*11:12', 'HLA-DRB1*11:13', 'HLA-DRB1*11:14', 'HLA-DRB1*11:15',
         'HLA-DRB1*11:16', 'HLA-DRB1*11:17', 'HLA-DRB1*11:18', 'HLA-DRB1*11:19', 'HLA-DRB1*11:20', 'HLA-DRB1*11:21',
         'HLA-DRB1*11:24', 'HLA-DRB1*11:25', 'HLA-DRB1*11:27', 'HLA-DRB1*11:28', 'HLA-DRB1*11:29', 'HLA-DRB1*11:30',
         'HLA-DRB1*11:31', 'HLA-DRB1*11:32', 'HLA-DRB1*11:33', 'HLA-DRB1*11:34', 'HLA-DRB1*11:35', 'HLA-DRB1*11:36',
         'HLA-DRB1*11:37', 'HLA-DRB1*11:38', 'HLA-DRB1*11:39', 'HLA-DRB1*11:41', 'HLA-DRB1*11:42', 'HLA-DRB1*11:43',
         'HLA-DRB1*11:44', 'HLA-DRB1*11:45', 'HLA-DRB1*11:46', 'HLA-DRB1*11:47', 'HLA-DRB1*11:48', 'HLA-DRB1*11:49',
         'HLA-DRB1*11:50', 'HLA-DRB1*11:51', 'HLA-DRB1*11:52', 'HLA-DRB1*11:53', 'HLA-DRB1*11:54', 'HLA-DRB1*11:55',
         'HLA-DRB1*11:56', 'HLA-DRB1*11:57', 'HLA-DRB1*11:58', 'HLA-DRB1*11:59', 'HLA-DRB1*11:60', 'HLA-DRB1*11:61',
         'HLA-DRB1*11:62', 'HLA-DRB1*11:63', 'HLA-DRB1*11:64', 'HLA-DRB1*11:65', 'HLA-DRB1*11:66', 'HLA-DRB1*11:67',
         'HLA-DRB1*11:68', 'HLA-DRB1*11:69', 'HLA-DRB1*11:70', 'HLA-DRB1*11:72', 'HLA-DRB1*11:73', 'HLA-DRB1*11:74',
         'HLA-DRB1*11:75', 'HLA-DRB1*11:76', 'HLA-DRB1*11:77', 'HLA-DRB1*11:78', 'HLA-DRB1*11:79', 'HLA-DRB1*11:80',
         'HLA-DRB1*11:81', 'HLA-DRB1*11:82', 'HLA-DRB1*11:83', 'HLA-DRB1*11:84', 'HLA-DRB1*11:85', 'HLA-DRB1*11:86',
         'HLA-DRB1*11:87', 'HLA-DRB1*11:88', 'HLA-DRB1*11:89', 'HLA-DRB1*11:90', 'HLA-DRB1*11:91', 'HLA-DRB1*11:92',
         'HLA-DRB1*11:93', 'HLA-DRB1*11:94', 'HLA-DRB1*11:95', 'HLA-DRB1*11:96', 'HLA-DRB1*12:01', 'HLA-DRB1*12:02',
         'HLA-DRB1*12:03', 'HLA-DRB1*12:04', 'HLA-DRB1*12:05', 'HLA-DRB1*12:06', 'HLA-DRB1*12:07', 'HLA-DRB1*12:08',
         'HLA-DRB1*12:09', 'HLA-DRB1*12:10', 'HLA-DRB1*12:11', 'HLA-DRB1*12:12', 'HLA-DRB1*12:13', 'HLA-DRB1*12:14',
         'HLA-DRB1*12:15', 'HLA-DRB1*12:16', 'HLA-DRB1*12:17', 'HLA-DRB1*12:18', 'HLA-DRB1*12:19', 'HLA-DRB1*12:20',
         'HLA-DRB1*12:21', 'HLA-DRB1*12:22', 'HLA-DRB1*12:23', 'HLA-DRB1*13:01', 'HLA-DRB1*13:02', 'HLA-DRB1*13:03',
         'HLA-DRB1*13:04', 'HLA-DRB1*13:05', 'HLA-DRB1*13:06', 'HLA-DRB1*13:07', 'HLA-DRB1*13:08', 'HLA-DRB1*13:09',
         'HLA-DRB1*13:10', 'HLA-DRB1*13:100', 'HLA-DRB1*13:101', 'HLA-DRB1*13:11', 'HLA-DRB1*13:12', 'HLA-DRB1*13:13',
         'HLA-DRB1*13:14', 'HLA-DRB1*13:15', 'HLA-DRB1*13:16', 'HLA-DRB1*13:17', 'HLA-DRB1*13:18', 'HLA-DRB1*13:19',
         'HLA-DRB1*13:20', 'HLA-DRB1*13:21', 'HLA-DRB1*13:22', 'HLA-DRB1*13:23', 'HLA-DRB1*13:24', 'HLA-DRB1*13:26',
         'HLA-DRB1*13:27', 'HLA-DRB1*13:29', 'HLA-DRB1*13:30', 'HLA-DRB1*13:31', 'HLA-DRB1*13:32', 'HLA-DRB1*13:33',
         'HLA-DRB1*13:34', 'HLA-DRB1*13:35', 'HLA-DRB1*13:36', 'HLA-DRB1*13:37', 'HLA-DRB1*13:38', 'HLA-DRB1*13:39',
         'HLA-DRB1*13:41', 'HLA-DRB1*13:42', 'HLA-DRB1*13:43', 'HLA-DRB1*13:44', 'HLA-DRB1*13:46', 'HLA-DRB1*13:47',
         'HLA-DRB1*13:48', 'HLA-DRB1*13:49', 'HLA-DRB1*13:50', 'HLA-DRB1*13:51', 'HLA-DRB1*13:52', 'HLA-DRB1*13:53',
         'HLA-DRB1*13:54', 'HLA-DRB1*13:55', 'HLA-DRB1*13:56', 'HLA-DRB1*13:57', 'HLA-DRB1*13:58', 'HLA-DRB1*13:59',
         'HLA-DRB1*13:60', 'HLA-DRB1*13:61', 'HLA-DRB1*13:62', 'HLA-DRB1*13:63', 'HLA-DRB1*13:64', 'HLA-DRB1*13:65',
         'HLA-DRB1*13:66', 'HLA-DRB1*13:67', 'HLA-DRB1*13:68', 'HLA-DRB1*13:69', 'HLA-DRB1*13:70', 'HLA-DRB1*13:71',
         'HLA-DRB1*13:72', 'HLA-DRB1*13:73', 'HLA-DRB1*13:74', 'HLA-DRB1*13:75', 'HLA-DRB1*13:76', 'HLA-DRB1*13:77',
         'HLA-DRB1*13:78', 'HLA-DRB1*13:79', 'HLA-DRB1*13:80', 'HLA-DRB1*13:81', 'HLA-DRB1*13:82', 'HLA-DRB1*13:83',
         'HLA-DRB1*13:84', 'HLA-DRB1*13:85', 'HLA-DRB1*13:86', 'HLA-DRB1*13:87', 'HLA-DRB1*13:88', 'HLA-DRB1*13:89',
         'HLA-DRB1*13:90', 'HLA-DRB1*13:91', 'HLA-DRB1*13:92', 'HLA-DRB1*13:93', 'HLA-DRB1*13:94', 'HLA-DRB1*13:95',
         'HLA-DRB1*13:96', 'HLA-DRB1*13:97', 'HLA-DRB1*13:98', 'HLA-DRB1*13:99', 'HLA-DRB1*14:01', 'HLA-DRB1*14:02',
         'HLA-DRB1*14:03', 'HLA-DRB1*14:04', 'HLA-DRB1*14:05', 'HLA-DRB1*14:06', 'HLA-DRB1*14:07', 'HLA-DRB1*14:08',
         'HLA-DRB1*14:09', 'HLA-DRB1*14:10', 'HLA-DRB1*14:11', 'HLA-DRB1*14:12', 'HLA-DRB1*14:13', 'HLA-DRB1*14:14',
         'HLA-DRB1*14:15', 'HLA-DRB1*14:16', 'HLA-DRB1*14:17', 'HLA-DRB1*14:18', 'HLA-DRB1*14:19', 'HLA-DRB1*14:20',
         'HLA-DRB1*14:21', 'HLA-DRB1*14:22', 'HLA-DRB1*14:23', 'HLA-DRB1*14:24', 'HLA-DRB1*14:25', 'HLA-DRB1*14:26',
         'HLA-DRB1*14:27', 'HLA-DRB1*14:28', 'HLA-DRB1*14:29', 'HLA-DRB1*14:30', 'HLA-DRB1*14:31', 'HLA-DRB1*14:32',
         'HLA-DRB1*14:33', 'HLA-DRB1*14:34', 'HLA-DRB1*14:35', 'HLA-DRB1*14:36', 'HLA-DRB1*14:37', 'HLA-DRB1*14:38',
         'HLA-DRB1*14:39', 'HLA-DRB1*14:40', 'HLA-DRB1*14:41', 'HLA-DRB1*14:42', 'HLA-DRB1*14:43', 'HLA-DRB1*14:44',
         'HLA-DRB1*14:45', 'HLA-DRB1*14:46', 'HLA-DRB1*14:47', 'HLA-DRB1*14:48', 'HLA-DRB1*14:49', 'HLA-DRB1*14:50',
         'HLA-DRB1*14:51', 'HLA-DRB1*14:52', 'HLA-DRB1*14:53', 'HLA-DRB1*14:54', 'HLA-DRB1*14:55', 'HLA-DRB1*14:56',
         'HLA-DRB1*14:57', 'HLA-DRB1*14:58', 'HLA-DRB1*14:59', 'HLA-DRB1*14:60', 'HLA-DRB1*14:61', 'HLA-DRB1*14:62',
         'HLA-DRB1*14:63', 'HLA-DRB1*14:64', 'HLA-DRB1*14:65', 'HLA-DRB1*14:67', 'HLA-DRB1*14:68', 'HLA-DRB1*14:69',
         'HLA-DRB1*14:70', 'HLA-DRB1*14:71', 'HLA-DRB1*14:72', 'HLA-DRB1*14:73', 'HLA-DRB1*14:74', 'HLA-DRB1*14:75',
         'HLA-DRB1*14:76', 'HLA-DRB1*14:77', 'HLA-DRB1*14:78', 'HLA-DRB1*14:79', 'HLA-DRB1*14:80', 'HLA-DRB1*14:81',
         'HLA-DRB1*14:82', 'HLA-DRB1*14:83', 'HLA-DRB1*14:84', 'HLA-DRB1*14:85', 'HLA-DRB1*14:86', 'HLA-DRB1*14:87',
         'HLA-DRB1*14:88', 'HLA-DRB1*14:89', 'HLA-DRB1*14:90', 'HLA-DRB1*14:91', 'HLA-DRB1*14:93', 'HLA-DRB1*14:94',
         'HLA-DRB1*14:95', 'HLA-DRB1*14:96', 'HLA-DRB1*14:97', 'HLA-DRB1*14:98', 'HLA-DRB1*14:99', 'HLA-DRB1*15:01',
         'HLA-DRB1*15:02', 'HLA-DRB1*15:03', 'HLA-DRB1*15:04', 'HLA-DRB1*15:05', 'HLA-DRB1*15:06', 'HLA-DRB1*15:07',
         'HLA-DRB1*15:08', 'HLA-DRB1*15:09', 'HLA-DRB1*15:10', 'HLA-DRB1*15:11', 'HLA-DRB1*15:12', 'HLA-DRB1*15:13',
         'HLA-DRB1*15:14', 'HLA-DRB1*15:15', 'HLA-DRB1*15:16', 'HLA-DRB1*15:18', 'HLA-DRB1*15:19', 'HLA-DRB1*15:20',
         'HLA-DRB1*15:21', 'HLA-DRB1*15:22', 'HLA-DRB1*15:23', 'HLA-DRB1*15:24', 'HLA-DRB1*15:25', 'HLA-DRB1*15:26',
         'HLA-DRB1*15:27', 'HLA-DRB1*15:28', 'HLA-DRB1*15:29', 'HLA-DRB1*15:30', 'HLA-DRB1*15:31', 'HLA-DRB1*15:32',
         'HLA-DRB1*15:33', 'HLA-DRB1*15:34', 'HLA-DRB1*15:35', 'HLA-DRB1*15:36', 'HLA-DRB1*15:37', 'HLA-DRB1*15:38',
         'HLA-DRB1*15:39', 'HLA-DRB1*15:40', 'HLA-DRB1*15:41', 'HLA-DRB1*15:42', 'HLA-DRB1*15:43', 'HLA-DRB1*15:44',
         'HLA-DRB1*15:45', 'HLA-DRB1*15:46', 'HLA-DRB1*15:47', 'HLA-DRB1*15:48', 'HLA-DRB1*15:49', 'HLA-DRB1*16:01',
         'HLA-DRB1*16:02', 'HLA-DRB1*16:03', 'HLA-DRB1*16:04', 'HLA-DRB1*16:05', 'HLA-DRB1*16:07', 'HLA-DRB1*16:08',
         'HLA-DRB1*16:09', 'HLA-DRB1*16:10', 'HLA-DRB1*16:11', 'HLA-DRB1*16:12', 'HLA-DRB1*16:14', 'HLA-DRB1*16:15',
         'HLA-DRB1*16:16', 'HLA-DRB3*01:01', 'HLA-DRB3*01:04', 'HLA-DRB3*01:05', 'HLA-DRB3*01:08', 'HLA-DRB3*01:09',
         'HLA-DRB3*01:11', 'HLA-DRB3*01:12', 'HLA-DRB3*01:13', 'HLA-DRB3*01:14', 'HLA-DRB3*02:01', 'HLA-DRB3*02:02',
         'HLA-DRB3*02:04', 'HLA-DRB3*02:05', 'HLA-DRB3*02:09', 'HLA-DRB3*02:10', 'HLA-DRB3*02:11', 'HLA-DRB3*02:12',
         'HLA-DRB3*02:13', 'HLA-DRB3*02:14', 'HLA-DRB3*02:15', 'HLA-DRB3*02:16', 'HLA-DRB3*02:17', 'HLA-DRB3*02:18',
         'HLA-DRB3*02:19', 'HLA-DRB3*02:20', 'HLA-DRB3*02:21', 'HLA-DRB3*02:22', 'HLA-DRB3*02:23', 'HLA-DRB3*02:24',
         'HLA-DRB3*02:25', 'HLA-DRB3*03:01', 'HLA-DRB3*03:03', 'HLA-DRB4*01:01', 'HLA-DRB4*01:03', 'HLA-DRB4*01:04',
         'HLA-DRB4*01:06', 'HLA-DRB4*01:07', 'HLA-DRB4*01:08', 'HLA-DRB5*01:01', 'HLA-DRB5*01:02', 'HLA-DRB5*01:03',
         'HLA-DRB5*01:04', 'HLA-DRB5*01:05', 'HLA-DRB5*01:06', 'HLA-DRB5*01:08N', 'HLA-DRB5*01:11', 'HLA-DRB5*01:12',
         'HLA-DRB5*01:13', 'HLA-DRB5*01:14', 'HLA-DRB5*02:02', 'HLA-DRB5*02:03', 'HLA-DRB5*02:04', 'HLA-DRB5*02:05',
         'HLA-DPA1*01:03-DPB1*01:01', 'HLA-DPA1*01:03-DPB1*02:01', 'HLA-DPA1*01:03-DPB1*02:02', 'HLA-DPA1*01:03-DPB1*03:01',
         'HLA-DPA1*01:03-DPB1*04:01', 'HLA-DPA1*01:03-DPB1*04:02', 'HLA-DPA1*01:03-DPB1*05:01', 'HLA-DPA1*01:03-DPB1*06:01',
         'HLA-DPA1*01:03-DPB1*08:01', 'HLA-DPA1*01:03-DPB1*09:01', 'HLA-DPA1*01:03-DPB1*10:001', 'HLA-DPA1*01:03-DPB1*10:01',
         'HLA-DPA1*01:03-DPB1*10:101', 'HLA-DPA1*01:03-DPB1*10:201',
         'HLA-DPA1*01:03-DPB1*10:301', 'HLA-DPA1*01:03-DPB1*10:401',
         'HLA-DPA1*01:03-DPB1*10:501', 'HLA-DPA1*01:03-DPB1*10:601', 'HLA-DPA1*01:03-DPB1*10:701', 'HLA-DPA1*01:03-DPB1*10:801',
         'HLA-DPA1*01:03-DPB1*10:901', 'HLA-DPA1*01:03-DPB1*11:001',
         'HLA-DPA1*01:03-DPB1*11:01', 'HLA-DPA1*01:03-DPB1*11:101', 'HLA-DPA1*01:03-DPB1*11:201', 'HLA-DPA1*01:03-DPB1*11:301',
         'HLA-DPA1*01:03-DPB1*11:401', 'HLA-DPA1*01:03-DPB1*11:501',
         'HLA-DPA1*01:03-DPB1*11:601', 'HLA-DPA1*01:03-DPB1*11:701', 'HLA-DPA1*01:03-DPB1*11:801', 'HLA-DPA1*01:03-DPB1*11:901',
         'HLA-DPA1*01:03-DPB1*12:101', 'HLA-DPA1*01:03-DPB1*12:201',
         'HLA-DPA1*01:03-DPB1*12:301', 'HLA-DPA1*01:03-DPB1*12:401', 'HLA-DPA1*01:03-DPB1*12:501', 'HLA-DPA1*01:03-DPB1*12:601',
         'HLA-DPA1*01:03-DPB1*12:701', 'HLA-DPA1*01:03-DPB1*12:801',
         'HLA-DPA1*01:03-DPB1*12:901', 'HLA-DPA1*01:03-DPB1*13:001', 'HLA-DPA1*01:03-DPB1*13:01', 'HLA-DPA1*01:03-DPB1*13:101',
         'HLA-DPA1*01:03-DPB1*13:201', 'HLA-DPA1*01:03-DPB1*13:301',
         'HLA-DPA1*01:03-DPB1*13:401', 'HLA-DPA1*01:03-DPB1*14:01', 'HLA-DPA1*01:03-DPB1*15:01', 'HLA-DPA1*01:03-DPB1*16:01',
         'HLA-DPA1*01:03-DPB1*17:01', 'HLA-DPA1*01:03-DPB1*18:01',
         'HLA-DPA1*01:03-DPB1*19:01', 'HLA-DPA1*01:03-DPB1*20:01', 'HLA-DPA1*01:03-DPB1*21:01', 'HLA-DPA1*01:03-DPB1*22:01',
         'HLA-DPA1*01:03-DPB1*23:01', 'HLA-DPA1*01:03-DPB1*24:01',
         'HLA-DPA1*01:03-DPB1*25:01', 'HLA-DPA1*01:03-DPB1*26:01', 'HLA-DPA1*01:03-DPB1*27:01', 'HLA-DPA1*01:03-DPB1*28:01',
         'HLA-DPA1*01:03-DPB1*29:01', 'HLA-DPA1*01:03-DPB1*30:01',
         'HLA-DPA1*01:03-DPB1*31:01', 'HLA-DPA1*01:03-DPB1*32:01', 'HLA-DPA1*01:03-DPB1*33:01', 'HLA-DPA1*01:03-DPB1*34:01',
         'HLA-DPA1*01:03-DPB1*35:01', 'HLA-DPA1*01:03-DPB1*36:01',
         'HLA-DPA1*01:03-DPB1*37:01', 'HLA-DPA1*01:03-DPB1*38:01', 'HLA-DPA1*01:03-DPB1*39:01', 'HLA-DPA1*01:03-DPB1*40:01',
         'HLA-DPA1*01:03-DPB1*41:01', 'HLA-DPA1*01:03-DPB1*44:01',
         'HLA-DPA1*01:03-DPB1*45:01', 'HLA-DPA1*01:03-DPB1*46:01', 'HLA-DPA1*01:03-DPB1*47:01', 'HLA-DPA1*01:03-DPB1*48:01',
         'HLA-DPA1*01:03-DPB1*49:01', 'HLA-DPA1*01:03-DPB1*50:01',
         'HLA-DPA1*01:03-DPB1*51:01', 'HLA-DPA1*01:03-DPB1*52:01', 'HLA-DPA1*01:03-DPB1*53:01', 'HLA-DPA1*01:03-DPB1*54:01',
         'HLA-DPA1*01:03-DPB1*55:01', 'HLA-DPA1*01:03-DPB1*56:01',
         'HLA-DPA1*01:03-DPB1*58:01', 'HLA-DPA1*01:03-DPB1*59:01', 'HLA-DPA1*01:03-DPB1*60:01', 'HLA-DPA1*01:03-DPB1*62:01',
         'HLA-DPA1*01:03-DPB1*63:01', 'HLA-DPA1*01:03-DPB1*65:01',
         'HLA-DPA1*01:03-DPB1*66:01', 'HLA-DPA1*01:03-DPB1*67:01', 'HLA-DPA1*01:03-DPB1*68:01', 'HLA-DPA1*01:03-DPB1*69:01',
         'HLA-DPA1*01:03-DPB1*70:01', 'HLA-DPA1*01:03-DPB1*71:01',
         'HLA-DPA1*01:03-DPB1*72:01', 'HLA-DPA1*01:03-DPB1*73:01', 'HLA-DPA1*01:03-DPB1*74:01', 'HLA-DPA1*01:03-DPB1*75:01',
         'HLA-DPA1*01:03-DPB1*76:01', 'HLA-DPA1*01:03-DPB1*77:01',
         'HLA-DPA1*01:03-DPB1*78:01', 'HLA-DPA1*01:03-DPB1*79:01', 'HLA-DPA1*01:03-DPB1*80:01', 'HLA-DPA1*01:03-DPB1*81:01',
         'HLA-DPA1*01:03-DPB1*82:01', 'HLA-DPA1*01:03-DPB1*83:01',
         'HLA-DPA1*01:03-DPB1*84:01', 'HLA-DPA1*01:03-DPB1*85:01', 'HLA-DPA1*01:03-DPB1*86:01', 'HLA-DPA1*01:03-DPB1*87:01',
         'HLA-DPA1*01:03-DPB1*88:01', 'HLA-DPA1*01:03-DPB1*89:01',
         'HLA-DPA1*01:03-DPB1*90:01', 'HLA-DPA1*01:03-DPB1*91:01', 'HLA-DPA1*01:03-DPB1*92:01', 'HLA-DPA1*01:03-DPB1*93:01',
         'HLA-DPA1*01:03-DPB1*94:01', 'HLA-DPA1*01:03-DPB1*95:01',
         'HLA-DPA1*01:03-DPB1*96:01', 'HLA-DPA1*01:03-DPB1*97:01', 'HLA-DPA1*01:03-DPB1*98:01', 'HLA-DPA1*01:03-DPB1*99:01',
         'HLA-DPA1*01:04-DPB1*01:01', 'HLA-DPA1*01:04-DPB1*02:01',
         'HLA-DPA1*01:04-DPB1*02:02', 'HLA-DPA1*01:04-DPB1*03:01', 'HLA-DPA1*01:04-DPB1*04:01', 'HLA-DPA1*01:04-DPB1*04:02',
         'HLA-DPA1*01:04-DPB1*05:01', 'HLA-DPA1*01:04-DPB1*06:01',
         'HLA-DPA1*01:04-DPB1*08:01', 'HLA-DPA1*01:04-DPB1*09:01', 'HLA-DPA1*01:04-DPB1*10:001', 'HLA-DPA1*01:04-DPB1*10:01',
         'HLA-DPA1*01:04-DPB1*10:101', 'HLA-DPA1*01:04-DPB1*10:201',
         'HLA-DPA1*01:04-DPB1*10:301', 'HLA-DPA1*01:04-DPB1*10:401', 'HLA-DPA1*01:04-DPB1*10:501', 'HLA-DPA1*01:04-DPB1*10:601',
         'HLA-DPA1*01:04-DPB1*10:701', 'HLA-DPA1*01:04-DPB1*10:801',
         'HLA-DPA1*01:04-DPB1*10:901', 'HLA-DPA1*01:04-DPB1*11:001', 'HLA-DPA1*01:04-DPB1*11:01', 'HLA-DPA1*01:04-DPB1*11:101',
         'HLA-DPA1*01:04-DPB1*11:201', 'HLA-DPA1*01:04-DPB1*11:301',
         'HLA-DPA1*01:04-DPB1*11:401', 'HLA-DPA1*01:04-DPB1*11:501', 'HLA-DPA1*01:04-DPB1*11:601', 'HLA-DPA1*01:04-DPB1*11:701',
         'HLA-DPA1*01:04-DPB1*11:801', 'HLA-DPA1*01:04-DPB1*11:901',
         'HLA-DPA1*01:04-DPB1*12:101', 'HLA-DPA1*01:04-DPB1*12:201', 'HLA-DPA1*01:04-DPB1*12:301', 'HLA-DPA1*01:04-DPB1*12:401',
         'HLA-DPA1*01:04-DPB1*12:501', 'HLA-DPA1*01:04-DPB1*12:601',
         'HLA-DPA1*01:04-DPB1*12:701', 'HLA-DPA1*01:04-DPB1*12:801', 'HLA-DPA1*01:04-DPB1*12:901', 'HLA-DPA1*01:04-DPB1*13:001',
         'HLA-DPA1*01:04-DPB1*13:01', 'HLA-DPA1*01:04-DPB1*13:101',
         'HLA-DPA1*01:04-DPB1*13:201', 'HLA-DPA1*01:04-DPB1*13:301', 'HLA-DPA1*01:04-DPB1*13:401', 'HLA-DPA1*01:04-DPB1*14:01',
         'HLA-DPA1*01:04-DPB1*15:01', 'HLA-DPA1*01:04-DPB1*16:01',
         'HLA-DPA1*01:04-DPB1*17:01', 'HLA-DPA1*01:04-DPB1*18:01', 'HLA-DPA1*01:04-DPB1*19:01', 'HLA-DPA1*01:04-DPB1*20:01',
         'HLA-DPA1*01:04-DPB1*21:01', 'HLA-DPA1*01:04-DPB1*22:01',
         'HLA-DPA1*01:04-DPB1*23:01', 'HLA-DPA1*01:04-DPB1*24:01', 'HLA-DPA1*01:04-DPB1*25:01', 'HLA-DPA1*01:04-DPB1*26:01',
         'HLA-DPA1*01:04-DPB1*27:01', 'HLA-DPA1*01:04-DPB1*28:01',
         'HLA-DPA1*01:04-DPB1*29:01', 'HLA-DPA1*01:04-DPB1*30:01', 'HLA-DPA1*01:04-DPB1*31:01', 'HLA-DPA1*01:04-DPB1*32:01',
         'HLA-DPA1*01:04-DPB1*33:01', 'HLA-DPA1*01:04-DPB1*34:01',
         'HLA-DPA1*01:04-DPB1*35:01', 'HLA-DPA1*01:04-DPB1*36:01', 'HLA-DPA1*01:04-DPB1*37:01', 'HLA-DPA1*01:04-DPB1*38:01',
         'HLA-DPA1*01:04-DPB1*39:01', 'HLA-DPA1*01:04-DPB1*40:01',
         'HLA-DPA1*01:04-DPB1*41:01', 'HLA-DPA1*01:04-DPB1*44:01', 'HLA-DPA1*01:04-DPB1*45:01', 'HLA-DPA1*01:04-DPB1*46:01',
         'HLA-DPA1*01:04-DPB1*47:01', 'HLA-DPA1*01:04-DPB1*48:01',
         'HLA-DPA1*01:04-DPB1*49:01', 'HLA-DPA1*01:04-DPB1*50:01', 'HLA-DPA1*01:04-DPB1*51:01', 'HLA-DPA1*01:04-DPB1*52:01',
         'HLA-DPA1*01:04-DPB1*53:01', 'HLA-DPA1*01:04-DPB1*54:01',
         'HLA-DPA1*01:04-DPB1*55:01', 'HLA-DPA1*01:04-DPB1*56:01', 'HLA-DPA1*01:04-DPB1*58:01', 'HLA-DPA1*01:04-DPB1*59:01',
         'HLA-DPA1*01:04-DPB1*60:01', 'HLA-DPA1*01:04-DPB1*62:01',
         'HLA-DPA1*01:04-DPB1*63:01', 'HLA-DPA1*01:04-DPB1*65:01', 'HLA-DPA1*01:04-DPB1*66:01', 'HLA-DPA1*01:04-DPB1*67:01',
         'HLA-DPA1*01:04-DPB1*68:01', 'HLA-DPA1*01:04-DPB1*69:01',
         'HLA-DPA1*01:04-DPB1*70:01', 'HLA-DPA1*01:04-DPB1*71:01', 'HLA-DPA1*01:04-DPB1*72:01', 'HLA-DPA1*01:04-DPB1*73:01',
         'HLA-DPA1*01:04-DPB1*74:01', 'HLA-DPA1*01:04-DPB1*75:01',
         'HLA-DPA1*01:04-DPB1*76:01', 'HLA-DPA1*01:04-DPB1*77:01', 'HLA-DPA1*01:04-DPB1*78:01', 'HLA-DPA1*01:04-DPB1*79:01',
         'HLA-DPA1*01:04-DPB1*80:01', 'HLA-DPA1*01:04-DPB1*81:01',
         'HLA-DPA1*01:04-DPB1*82:01', 'HLA-DPA1*01:04-DPB1*83:01', 'HLA-DPA1*01:04-DPB1*84:01', 'HLA-DPA1*01:04-DPB1*85:01',
         'HLA-DPA1*01:04-DPB1*86:01', 'HLA-DPA1*01:04-DPB1*87:01',
         'HLA-DPA1*01:04-DPB1*88:01', 'HLA-DPA1*01:04-DPB1*89:01', 'HLA-DPA1*01:04-DPB1*90:01', 'HLA-DPA1*01:04-DPB1*91:01',
         'HLA-DPA1*01:04-DPB1*92:01', 'HLA-DPA1*01:04-DPB1*93:01',
         'HLA-DPA1*01:04-DPB1*94:01', 'HLA-DPA1*01:04-DPB1*95:01', 'HLA-DPA1*01:04-DPB1*96:01', 'HLA-DPA1*01:04-DPB1*97:01',
         'HLA-DPA1*01:04-DPB1*98:01', 'HLA-DPA1*01:04-DPB1*99:01',
         'HLA-DPA1*01:05-DPB1*01:01', 'HLA-DPA1*01:05-DPB1*02:01', 'HLA-DPA1*01:05-DPB1*02:02', 'HLA-DPA1*01:05-DPB1*03:01',
         'HLA-DPA1*01:05-DPB1*04:01', 'HLA-DPA1*01:05-DPB1*04:02',
         'HLA-DPA1*01:05-DPB1*05:01', 'HLA-DPA1*01:05-DPB1*06:01', 'HLA-DPA1*01:05-DPB1*08:01', 'HLA-DPA1*01:05-DPB1*09:01',
         'HLA-DPA1*01:05-DPB1*10:001', 'HLA-DPA1*01:05-DPB1*10:01',
         'HLA-DPA1*01:05-DPB1*10:101', 'HLA-DPA1*01:05-DPB1*10:201', 'HLA-DPA1*01:05-DPB1*10:301', 'HLA-DPA1*01:05-DPB1*10:401',
         'HLA-DPA1*01:05-DPB1*10:501', 'HLA-DPA1*01:05-DPB1*10:601',
         'HLA-DPA1*01:05-DPB1*10:701', 'HLA-DPA1*01:05-DPB1*10:801', 'HLA-DPA1*01:05-DPB1*10:901', 'HLA-DPA1*01:05-DPB1*11:001',
         'HLA-DPA1*01:05-DPB1*11:01', 'HLA-DPA1*01:05-DPB1*11:101',
         'HLA-DPA1*01:05-DPB1*11:201', 'HLA-DPA1*01:05-DPB1*11:301', 'HLA-DPA1*01:05-DPB1*11:401', 'HLA-DPA1*01:05-DPB1*11:501',
         'HLA-DPA1*01:05-DPB1*11:601', 'HLA-DPA1*01:05-DPB1*11:701',
         'HLA-DPA1*01:05-DPB1*11:801', 'HLA-DPA1*01:05-DPB1*11:901', 'HLA-DPA1*01:05-DPB1*12:101', 'HLA-DPA1*01:05-DPB1*12:201',
         'HLA-DPA1*01:05-DPB1*12:301', 'HLA-DPA1*01:05-DPB1*12:401',
         'HLA-DPA1*01:05-DPB1*12:501', 'HLA-DPA1*01:05-DPB1*12:601', 'HLA-DPA1*01:05-DPB1*12:701', 'HLA-DPA1*01:05-DPB1*12:801',
         'HLA-DPA1*01:05-DPB1*12:901', 'HLA-DPA1*01:05-DPB1*13:001',
         'HLA-DPA1*01:05-DPB1*13:01', 'HLA-DPA1*01:05-DPB1*13:101', 'HLA-DPA1*01:05-DPB1*13:201', 'HLA-DPA1*01:05-DPB1*13:301',
         'HLA-DPA1*01:05-DPB1*13:401', 'HLA-DPA1*01:05-DPB1*14:01',
         'HLA-DPA1*01:05-DPB1*15:01', 'HLA-DPA1*01:05-DPB1*16:01', 'HLA-DPA1*01:05-DPB1*17:01', 'HLA-DPA1*01:05-DPB1*18:01',
         'HLA-DPA1*01:05-DPB1*19:01', 'HLA-DPA1*01:05-DPB1*20:01',
         'HLA-DPA1*01:05-DPB1*21:01', 'HLA-DPA1*01:05-DPB1*22:01', 'HLA-DPA1*01:05-DPB1*23:01', 'HLA-DPA1*01:05-DPB1*24:01',
         'HLA-DPA1*01:05-DPB1*25:01', 'HLA-DPA1*01:05-DPB1*26:01',
         'HLA-DPA1*01:05-DPB1*27:01', 'HLA-DPA1*01:05-DPB1*28:01', 'HLA-DPA1*01:05-DPB1*29:01', 'HLA-DPA1*01:05-DPB1*30:01',
         'HLA-DPA1*01:05-DPB1*31:01', 'HLA-DPA1*01:05-DPB1*32:01',
         'HLA-DPA1*01:05-DPB1*33:01', 'HLA-DPA1*01:05-DPB1*34:01', 'HLA-DPA1*01:05-DPB1*35:01', 'HLA-DPA1*01:05-DPB1*36:01',
         'HLA-DPA1*01:05-DPB1*37:01', 'HLA-DPA1*01:05-DPB1*38:01',
         'HLA-DPA1*01:05-DPB1*39:01', 'HLA-DPA1*01:05-DPB1*40:01', 'HLA-DPA1*01:05-DPB1*41:01', 'HLA-DPA1*01:05-DPB1*44:01',
         'HLA-DPA1*01:05-DPB1*45:01', 'HLA-DPA1*01:05-DPB1*46:01',
         'HLA-DPA1*01:05-DPB1*47:01', 'HLA-DPA1*01:05-DPB1*48:01', 'HLA-DPA1*01:05-DPB1*49:01', 'HLA-DPA1*01:05-DPB1*50:01',
         'HLA-DPA1*01:05-DPB1*51:01', 'HLA-DPA1*01:05-DPB1*52:01',
         'HLA-DPA1*01:05-DPB1*53:01', 'HLA-DPA1*01:05-DPB1*54:01', 'HLA-DPA1*01:05-DPB1*55:01', 'HLA-DPA1*01:05-DPB1*56:01',
         'HLA-DPA1*01:05-DPB1*58:01', 'HLA-DPA1*01:05-DPB1*59:01',
         'HLA-DPA1*01:05-DPB1*60:01', 'HLA-DPA1*01:05-DPB1*62:01', 'HLA-DPA1*01:05-DPB1*63:01', 'HLA-DPA1*01:05-DPB1*65:01',
         'HLA-DPA1*01:05-DPB1*66:01', 'HLA-DPA1*01:05-DPB1*67:01',
         'HLA-DPA1*01:05-DPB1*68:01', 'HLA-DPA1*01:05-DPB1*69:01', 'HLA-DPA1*01:05-DPB1*70:01', 'HLA-DPA1*01:05-DPB1*71:01',
         'HLA-DPA1*01:05-DPB1*72:01', 'HLA-DPA1*01:05-DPB1*73:01',
         'HLA-DPA1*01:05-DPB1*74:01', 'HLA-DPA1*01:05-DPB1*75:01', 'HLA-DPA1*01:05-DPB1*76:01', 'HLA-DPA1*01:05-DPB1*77:01',
         'HLA-DPA1*01:05-DPB1*78:01', 'HLA-DPA1*01:05-DPB1*79:01',
         'HLA-DPA1*01:05-DPB1*80:01', 'HLA-DPA1*01:05-DPB1*81:01', 'HLA-DPA1*01:05-DPB1*82:01', 'HLA-DPA1*01:05-DPB1*83:01',
         'HLA-DPA1*01:05-DPB1*84:01', 'HLA-DPA1*01:05-DPB1*85:01',
         'HLA-DPA1*01:05-DPB1*86:01', 'HLA-DPA1*01:05-DPB1*87:01', 'HLA-DPA1*01:05-DPB1*88:01', 'HLA-DPA1*01:05-DPB1*89:01',
         'HLA-DPA1*01:05-DPB1*90:01', 'HLA-DPA1*01:05-DPB1*91:01',
         'HLA-DPA1*01:05-DPB1*92:01', 'HLA-DPA1*01:05-DPB1*93:01', 'HLA-DPA1*01:05-DPB1*94:01', 'HLA-DPA1*01:05-DPB1*95:01',
         'HLA-DPA1*01:05-DPB1*96:01', 'HLA-DPA1*01:05-DPB1*97:01',
         'HLA-DPA1*01:05-DPB1*98:01', 'HLA-DPA1*01:05-DPB1*99:01', 'HLA-DPA1*01:06-DPB1*01:01', 'HLA-DPA1*01:06-DPB1*02:01',
         'HLA-DPA1*01:06-DPB1*02:02', 'HLA-DPA1*01:06-DPB1*03:01',
         'HLA-DPA1*01:06-DPB1*04:01', 'HLA-DPA1*01:06-DPB1*04:02', 'HLA-DPA1*01:06-DPB1*05:01', 'HLA-DPA1*01:06-DPB1*06:01',
         'HLA-DPA1*01:06-DPB1*08:01', 'HLA-DPA1*01:06-DPB1*09:01',
         'HLA-DPA1*01:06-DPB1*10:001', 'HLA-DPA1*01:06-DPB1*10:01', 'HLA-DPA1*01:06-DPB1*10:101', 'HLA-DPA1*01:06-DPB1*10:201',
         'HLA-DPA1*01:06-DPB1*10:301', 'HLA-DPA1*01:06-DPB1*10:401',
         'HLA-DPA1*01:06-DPB1*10:501', 'HLA-DPA1*01:06-DPB1*10:601', 'HLA-DPA1*01:06-DPB1*10:701', 'HLA-DPA1*01:06-DPB1*10:801',
         'HLA-DPA1*01:06-DPB1*10:901', 'HLA-DPA1*01:06-DPB1*11:001',
         'HLA-DPA1*01:06-DPB1*11:01', 'HLA-DPA1*01:06-DPB1*11:101', 'HLA-DPA1*01:06-DPB1*11:201', 'HLA-DPA1*01:06-DPB1*11:301',
         'HLA-DPA1*01:06-DPB1*11:401', 'HLA-DPA1*01:06-DPB1*11:501',
         'HLA-DPA1*01:06-DPB1*11:601', 'HLA-DPA1*01:06-DPB1*11:701', 'HLA-DPA1*01:06-DPB1*11:801', 'HLA-DPA1*01:06-DPB1*11:901',
         'HLA-DPA1*01:06-DPB1*12:101', 'HLA-DPA1*01:06-DPB1*12:201',
         'HLA-DPA1*01:06-DPB1*12:301', 'HLA-DPA1*01:06-DPB1*12:401', 'HLA-DPA1*01:06-DPB1*12:501', 'HLA-DPA1*01:06-DPB1*12:601',
         'HLA-DPA1*01:06-DPB1*12:701', 'HLA-DPA1*01:06-DPB1*12:801',
         'HLA-DPA1*01:06-DPB1*12:901', 'HLA-DPA1*01:06-DPB1*13:001', 'HLA-DPA1*01:06-DPB1*13:01', 'HLA-DPA1*01:06-DPB1*13:101',
         'HLA-DPA1*01:06-DPB1*13:201', 'HLA-DPA1*01:06-DPB1*13:301',
         'HLA-DPA1*01:06-DPB1*13:401', 'HLA-DPA1*01:06-DPB1*14:01', 'HLA-DPA1*01:06-DPB1*15:01', 'HLA-DPA1*01:06-DPB1*16:01',
         'HLA-DPA1*01:06-DPB1*17:01', 'HLA-DPA1*01:06-DPB1*18:01',
         'HLA-DPA1*01:06-DPB1*19:01', 'HLA-DPA1*01:06-DPB1*20:01', 'HLA-DPA1*01:06-DPB1*21:01', 'HLA-DPA1*01:06-DPB1*22:01',
         'HLA-DPA1*01:06-DPB1*23:01', 'HLA-DPA1*01:06-DPB1*24:01',
         'HLA-DPA1*01:06-DPB1*25:01', 'HLA-DPA1*01:06-DPB1*26:01', 'HLA-DPA1*01:06-DPB1*27:01', 'HLA-DPA1*01:06-DPB1*28:01',
         'HLA-DPA1*01:06-DPB1*29:01', 'HLA-DPA1*01:06-DPB1*30:01',
         'HLA-DPA1*01:06-DPB1*31:01', 'HLA-DPA1*01:06-DPB1*32:01', 'HLA-DPA1*01:06-DPB1*33:01', 'HLA-DPA1*01:06-DPB1*34:01',
         'HLA-DPA1*01:06-DPB1*35:01', 'HLA-DPA1*01:06-DPB1*36:01',
         'HLA-DPA1*01:06-DPB1*37:01', 'HLA-DPA1*01:06-DPB1*38:01', 'HLA-DPA1*01:06-DPB1*39:01', 'HLA-DPA1*01:06-DPB1*40:01',
         'HLA-DPA1*01:06-DPB1*41:01', 'HLA-DPA1*01:06-DPB1*44:01',
         'HLA-DPA1*01:06-DPB1*45:01', 'HLA-DPA1*01:06-DPB1*46:01', 'HLA-DPA1*01:06-DPB1*47:01', 'HLA-DPA1*01:06-DPB1*48:01',
         'HLA-DPA1*01:06-DPB1*49:01', 'HLA-DPA1*01:06-DPB1*50:01',
         'HLA-DPA1*01:06-DPB1*51:01', 'HLA-DPA1*01:06-DPB1*52:01', 'HLA-DPA1*01:06-DPB1*53:01', 'HLA-DPA1*01:06-DPB1*54:01',
         'HLA-DPA1*01:06-DPB1*55:01', 'HLA-DPA1*01:06-DPB1*56:01',
         'HLA-DPA1*01:06-DPB1*58:01', 'HLA-DPA1*01:06-DPB1*59:01', 'HLA-DPA1*01:06-DPB1*60:01', 'HLA-DPA1*01:06-DPB1*62:01',
         'HLA-DPA1*01:06-DPB1*63:01', 'HLA-DPA1*01:06-DPB1*65:01',
         'HLA-DPA1*01:06-DPB1*66:01', 'HLA-DPA1*01:06-DPB1*67:01', 'HLA-DPA1*01:06-DPB1*68:01', 'HLA-DPA1*01:06-DPB1*69:01',
         'HLA-DPA1*01:06-DPB1*70:01', 'HLA-DPA1*01:06-DPB1*71:01',
         'HLA-DPA1*01:06-DPB1*72:01', 'HLA-DPA1*01:06-DPB1*73:01', 'HLA-DPA1*01:06-DPB1*74:01', 'HLA-DPA1*01:06-DPB1*75:01',
         'HLA-DPA1*01:06-DPB1*76:01', 'HLA-DPA1*01:06-DPB1*77:01',
         'HLA-DPA1*01:06-DPB1*78:01', 'HLA-DPA1*01:06-DPB1*79:01', 'HLA-DPA1*01:06-DPB1*80:01', 'HLA-DPA1*01:06-DPB1*81:01',
         'HLA-DPA1*01:06-DPB1*82:01', 'HLA-DPA1*01:06-DPB1*83:01',
         'HLA-DPA1*01:06-DPB1*84:01', 'HLA-DPA1*01:06-DPB1*85:01', 'HLA-DPA1*01:06-DPB1*86:01', 'HLA-DPA1*01:06-DPB1*87:01',
         'HLA-DPA1*01:06-DPB1*88:01', 'HLA-DPA1*01:06-DPB1*89:01',
         'HLA-DPA1*01:06-DPB1*90:01', 'HLA-DPA1*01:06-DPB1*91:01', 'HLA-DPA1*01:06-DPB1*92:01', 'HLA-DPA1*01:06-DPB1*93:01',
         'HLA-DPA1*01:06-DPB1*94:01', 'HLA-DPA1*01:06-DPB1*95:01',
         'HLA-DPA1*01:06-DPB1*96:01', 'HLA-DPA1*01:06-DPB1*97:01', 'HLA-DPA1*01:06-DPB1*98:01', 'HLA-DPA1*01:06-DPB1*99:01',
         'HLA-DPA1*01:07-DPB1*01:01', 'HLA-DPA1*01:07-DPB1*02:01',
         'HLA-DPA1*01:07-DPB1*02:02', 'HLA-DPA1*01:07-DPB1*03:01', 'HLA-DPA1*01:07-DPB1*04:01', 'HLA-DPA1*01:07-DPB1*04:02',
         'HLA-DPA1*01:07-DPB1*05:01', 'HLA-DPA1*01:07-DPB1*06:01',
         'HLA-DPA1*01:07-DPB1*08:01', 'HLA-DPA1*01:07-DPB1*09:01', 'HLA-DPA1*01:07-DPB1*10:001', 'HLA-DPA1*01:07-DPB1*10:01',
         'HLA-DPA1*01:07-DPB1*10:101', 'HLA-DPA1*01:07-DPB1*10:201',
         'HLA-DPA1*01:07-DPB1*10:301', 'HLA-DPA1*01:07-DPB1*10:401', 'HLA-DPA1*01:07-DPB1*10:501', 'HLA-DPA1*01:07-DPB1*10:601',
         'HLA-DPA1*01:07-DPB1*10:701', 'HLA-DPA1*01:07-DPB1*10:801',
         'HLA-DPA1*01:07-DPB1*10:901', 'HLA-DPA1*01:07-DPB1*11:001', 'HLA-DPA1*01:07-DPB1*11:01', 'HLA-DPA1*01:07-DPB1*11:101',
         'HLA-DPA1*01:07-DPB1*11:201', 'HLA-DPA1*01:07-DPB1*11:301',
         'HLA-DPA1*01:07-DPB1*11:401', 'HLA-DPA1*01:07-DPB1*11:501', 'HLA-DPA1*01:07-DPB1*11:601', 'HLA-DPA1*01:07-DPB1*11:701',
         'HLA-DPA1*01:07-DPB1*11:801', 'HLA-DPA1*01:07-DPB1*11:901',
         'HLA-DPA1*01:07-DPB1*12:101', 'HLA-DPA1*01:07-DPB1*12:201', 'HLA-DPA1*01:07-DPB1*12:301', 'HLA-DPA1*01:07-DPB1*12:401',
         'HLA-DPA1*01:07-DPB1*12:501', 'HLA-DPA1*01:07-DPB1*12:601',
         'HLA-DPA1*01:07-DPB1*12:701', 'HLA-DPA1*01:07-DPB1*12:801', 'HLA-DPA1*01:07-DPB1*12:901', 'HLA-DPA1*01:07-DPB1*13:001',
         'HLA-DPA1*01:07-DPB1*13:01', 'HLA-DPA1*01:07-DPB1*13:101',
         'HLA-DPA1*01:07-DPB1*13:201', 'HLA-DPA1*01:07-DPB1*13:301', 'HLA-DPA1*01:07-DPB1*13:401', 'HLA-DPA1*01:07-DPB1*14:01',
         'HLA-DPA1*01:07-DPB1*15:01', 'HLA-DPA1*01:07-DPB1*16:01',
         'HLA-DPA1*01:07-DPB1*17:01', 'HLA-DPA1*01:07-DPB1*18:01', 'HLA-DPA1*01:07-DPB1*19:01', 'HLA-DPA1*01:07-DPB1*20:01',
         'HLA-DPA1*01:07-DPB1*21:01', 'HLA-DPA1*01:07-DPB1*22:01',
         'HLA-DPA1*01:07-DPB1*23:01', 'HLA-DPA1*01:07-DPB1*24:01', 'HLA-DPA1*01:07-DPB1*25:01', 'HLA-DPA1*01:07-DPB1*26:01',
         'HLA-DPA1*01:07-DPB1*27:01', 'HLA-DPA1*01:07-DPB1*28:01',
         'HLA-DPA1*01:07-DPB1*29:01', 'HLA-DPA1*01:07-DPB1*30:01', 'HLA-DPA1*01:07-DPB1*31:01', 'HLA-DPA1*01:07-DPB1*32:01',
         'HLA-DPA1*01:07-DPB1*33:01', 'HLA-DPA1*01:07-DPB1*34:01',
         'HLA-DPA1*01:07-DPB1*35:01', 'HLA-DPA1*01:07-DPB1*36:01', 'HLA-DPA1*01:07-DPB1*37:01', 'HLA-DPA1*01:07-DPB1*38:01',
         'HLA-DPA1*01:07-DPB1*39:01', 'HLA-DPA1*01:07-DPB1*40:01',
         'HLA-DPA1*01:07-DPB1*41:01', 'HLA-DPA1*01:07-DPB1*44:01', 'HLA-DPA1*01:07-DPB1*45:01', 'HLA-DPA1*01:07-DPB1*46:01',
         'HLA-DPA1*01:07-DPB1*47:01', 'HLA-DPA1*01:07-DPB1*48:01',
         'HLA-DPA1*01:07-DPB1*49:01', 'HLA-DPA1*01:07-DPB1*50:01', 'HLA-DPA1*01:07-DPB1*51:01', 'HLA-DPA1*01:07-DPB1*52:01',
         'HLA-DPA1*01:07-DPB1*53:01', 'HLA-DPA1*01:07-DPB1*54:01',
         'HLA-DPA1*01:07-DPB1*55:01', 'HLA-DPA1*01:07-DPB1*56:01', 'HLA-DPA1*01:07-DPB1*58:01', 'HLA-DPA1*01:07-DPB1*59:01',
         'HLA-DPA1*01:07-DPB1*60:01', 'HLA-DPA1*01:07-DPB1*62:01',
         'HLA-DPA1*01:07-DPB1*63:01', 'HLA-DPA1*01:07-DPB1*65:01', 'HLA-DPA1*01:07-DPB1*66:01', 'HLA-DPA1*01:07-DPB1*67:01',
         'HLA-DPA1*01:07-DPB1*68:01', 'HLA-DPA1*01:07-DPB1*69:01',
         'HLA-DPA1*01:07-DPB1*70:01', 'HLA-DPA1*01:07-DPB1*71:01', 'HLA-DPA1*01:07-DPB1*72:01', 'HLA-DPA1*01:07-DPB1*73:01',
         'HLA-DPA1*01:07-DPB1*74:01', 'HLA-DPA1*01:07-DPB1*75:01',
         'HLA-DPA1*01:07-DPB1*76:01', 'HLA-DPA1*01:07-DPB1*77:01', 'HLA-DPA1*01:07-DPB1*78:01', 'HLA-DPA1*01:07-DPB1*79:01',
         'HLA-DPA1*01:07-DPB1*80:01', 'HLA-DPA1*01:07-DPB1*81:01',
         'HLA-DPA1*01:07-DPB1*82:01', 'HLA-DPA1*01:07-DPB1*83:01', 'HLA-DPA1*01:07-DPB1*84:01', 'HLA-DPA1*01:07-DPB1*85:01',
         'HLA-DPA1*01:07-DPB1*86:01', 'HLA-DPA1*01:07-DPB1*87:01',
         'HLA-DPA1*01:07-DPB1*88:01', 'HLA-DPA1*01:07-DPB1*89:01', 'HLA-DPA1*01:07-DPB1*90:01', 'HLA-DPA1*01:07-DPB1*91:01',
         'HLA-DPA1*01:07-DPB1*92:01', 'HLA-DPA1*01:07-DPB1*93:01',
         'HLA-DPA1*01:07-DPB1*94:01', 'HLA-DPA1*01:07-DPB1*95:01', 'HLA-DPA1*01:07-DPB1*96:01', 'HLA-DPA1*01:07-DPB1*97:01',
         'HLA-DPA1*01:07-DPB1*98:01', 'HLA-DPA1*01:07-DPB1*99:01',
         'HLA-DPA1*01:08-DPB1*01:01', 'HLA-DPA1*01:08-DPB1*02:01', 'HLA-DPA1*01:08-DPB1*02:02', 'HLA-DPA1*01:08-DPB1*03:01',
         'HLA-DPA1*01:08-DPB1*04:01', 'HLA-DPA1*01:08-DPB1*04:02',
         'HLA-DPA1*01:08-DPB1*05:01', 'HLA-DPA1*01:08-DPB1*06:01', 'HLA-DPA1*01:08-DPB1*08:01', 'HLA-DPA1*01:08-DPB1*09:01',
         'HLA-DPA1*01:08-DPB1*10:001', 'HLA-DPA1*01:08-DPB1*10:01',
         'HLA-DPA1*01:08-DPB1*10:101', 'HLA-DPA1*01:08-DPB1*10:201', 'HLA-DPA1*01:08-DPB1*10:301', 'HLA-DPA1*01:08-DPB1*10:401',
         'HLA-DPA1*01:08-DPB1*10:501', 'HLA-DPA1*01:08-DPB1*10:601',
         'HLA-DPA1*01:08-DPB1*10:701', 'HLA-DPA1*01:08-DPB1*10:801', 'HLA-DPA1*01:08-DPB1*10:901', 'HLA-DPA1*01:08-DPB1*11:001',
         'HLA-DPA1*01:08-DPB1*11:01', 'HLA-DPA1*01:08-DPB1*11:101',
         'HLA-DPA1*01:08-DPB1*11:201', 'HLA-DPA1*01:08-DPB1*11:301', 'HLA-DPA1*01:08-DPB1*11:401', 'HLA-DPA1*01:08-DPB1*11:501',
         'HLA-DPA1*01:08-DPB1*11:601', 'HLA-DPA1*01:08-DPB1*11:701',
         'HLA-DPA1*01:08-DPB1*11:801', 'HLA-DPA1*01:08-DPB1*11:901', 'HLA-DPA1*01:08-DPB1*12:101', 'HLA-DPA1*01:08-DPB1*12:201',
         'HLA-DPA1*01:08-DPB1*12:301', 'HLA-DPA1*01:08-DPB1*12:401',
         'HLA-DPA1*01:08-DPB1*12:501', 'HLA-DPA1*01:08-DPB1*12:601', 'HLA-DPA1*01:08-DPB1*12:701', 'HLA-DPA1*01:08-DPB1*12:801',
         'HLA-DPA1*01:08-DPB1*12:901', 'HLA-DPA1*01:08-DPB1*13:001',
         'HLA-DPA1*01:08-DPB1*13:01', 'HLA-DPA1*01:08-DPB1*13:101', 'HLA-DPA1*01:08-DPB1*13:201', 'HLA-DPA1*01:08-DPB1*13:301',
         'HLA-DPA1*01:08-DPB1*13:401', 'HLA-DPA1*01:08-DPB1*14:01',
         'HLA-DPA1*01:08-DPB1*15:01', 'HLA-DPA1*01:08-DPB1*16:01', 'HLA-DPA1*01:08-DPB1*17:01', 'HLA-DPA1*01:08-DPB1*18:01',
         'HLA-DPA1*01:08-DPB1*19:01', 'HLA-DPA1*01:08-DPB1*20:01',
         'HLA-DPA1*01:08-DPB1*21:01', 'HLA-DPA1*01:08-DPB1*22:01', 'HLA-DPA1*01:08-DPB1*23:01', 'HLA-DPA1*01:08-DPB1*24:01',
         'HLA-DPA1*01:08-DPB1*25:01', 'HLA-DPA1*01:08-DPB1*26:01',
         'HLA-DPA1*01:08-DPB1*27:01', 'HLA-DPA1*01:08-DPB1*28:01', 'HLA-DPA1*01:08-DPB1*29:01', 'HLA-DPA1*01:08-DPB1*30:01',
         'HLA-DPA1*01:08-DPB1*31:01', 'HLA-DPA1*01:08-DPB1*32:01',
         'HLA-DPA1*01:08-DPB1*33:01', 'HLA-DPA1*01:08-DPB1*34:01', 'HLA-DPA1*01:08-DPB1*35:01', 'HLA-DPA1*01:08-DPB1*36:01',
         'HLA-DPA1*01:08-DPB1*37:01', 'HLA-DPA1*01:08-DPB1*38:01',
         'HLA-DPA1*01:08-DPB1*39:01', 'HLA-DPA1*01:08-DPB1*40:01', 'HLA-DPA1*01:08-DPB1*41:01', 'HLA-DPA1*01:08-DPB1*44:01',
         'HLA-DPA1*01:08-DPB1*45:01', 'HLA-DPA1*01:08-DPB1*46:01',
         'HLA-DPA1*01:08-DPB1*47:01', 'HLA-DPA1*01:08-DPB1*48:01', 'HLA-DPA1*01:08-DPB1*49:01', 'HLA-DPA1*01:08-DPB1*50:01',
         'HLA-DPA1*01:08-DPB1*51:01', 'HLA-DPA1*01:08-DPB1*52:01',
         'HLA-DPA1*01:08-DPB1*53:01', 'HLA-DPA1*01:08-DPB1*54:01', 'HLA-DPA1*01:08-DPB1*55:01', 'HLA-DPA1*01:08-DPB1*56:01',
         'HLA-DPA1*01:08-DPB1*58:01', 'HLA-DPA1*01:08-DPB1*59:01',
         'HLA-DPA1*01:08-DPB1*60:01', 'HLA-DPA1*01:08-DPB1*62:01', 'HLA-DPA1*01:08-DPB1*63:01', 'HLA-DPA1*01:08-DPB1*65:01',
         'HLA-DPA1*01:08-DPB1*66:01', 'HLA-DPA1*01:08-DPB1*67:01',
         'HLA-DPA1*01:08-DPB1*68:01', 'HLA-DPA1*01:08-DPB1*69:01', 'HLA-DPA1*01:08-DPB1*70:01', 'HLA-DPA1*01:08-DPB1*71:01',
         'HLA-DPA1*01:08-DPB1*72:01', 'HLA-DPA1*01:08-DPB1*73:01',
         'HLA-DPA1*01:08-DPB1*74:01', 'HLA-DPA1*01:08-DPB1*75:01', 'HLA-DPA1*01:08-DPB1*76:01', 'HLA-DPA1*01:08-DPB1*77:01',
         'HLA-DPA1*01:08-DPB1*78:01', 'HLA-DPA1*01:08-DPB1*79:01',
         'HLA-DPA1*01:08-DPB1*80:01', 'HLA-DPA1*01:08-DPB1*81:01', 'HLA-DPA1*01:08-DPB1*82:01', 'HLA-DPA1*01:08-DPB1*83:01',
         'HLA-DPA1*01:08-DPB1*84:01', 'HLA-DPA1*01:08-DPB1*85:01',
         'HLA-DPA1*01:08-DPB1*86:01', 'HLA-DPA1*01:08-DPB1*87:01', 'HLA-DPA1*01:08-DPB1*88:01', 'HLA-DPA1*01:08-DPB1*89:01',
         'HLA-DPA1*01:08-DPB1*90:01', 'HLA-DPA1*01:08-DPB1*91:01',
         'HLA-DPA1*01:08-DPB1*92:01', 'HLA-DPA1*01:08-DPB1*93:01', 'HLA-DPA1*01:08-DPB1*94:01', 'HLA-DPA1*01:08-DPB1*95:01',
         'HLA-DPA1*01:08-DPB1*96:01', 'HLA-DPA1*01:08-DPB1*97:01',
         'HLA-DPA1*01:08-DPB1*98:01', 'HLA-DPA1*01:08-DPB1*99:01', 'HLA-DPA1*01:09-DPB1*01:01', 'HLA-DPA1*01:09-DPB1*02:01',
         'HLA-DPA1*01:09-DPB1*02:02', 'HLA-DPA1*01:09-DPB1*03:01',
         'HLA-DPA1*01:09-DPB1*04:01', 'HLA-DPA1*01:09-DPB1*04:02', 'HLA-DPA1*01:09-DPB1*05:01', 'HLA-DPA1*01:09-DPB1*06:01',
         'HLA-DPA1*01:09-DPB1*08:01', 'HLA-DPA1*01:09-DPB1*09:01',
         'HLA-DPA1*01:09-DPB1*10:001', 'HLA-DPA1*01:09-DPB1*10:01', 'HLA-DPA1*01:09-DPB1*10:101', 'HLA-DPA1*01:09-DPB1*10:201',
         'HLA-DPA1*01:09-DPB1*10:301', 'HLA-DPA1*01:09-DPB1*10:401',
         'HLA-DPA1*01:09-DPB1*10:501', 'HLA-DPA1*01:09-DPB1*10:601', 'HLA-DPA1*01:09-DPB1*10:701', 'HLA-DPA1*01:09-DPB1*10:801',
         'HLA-DPA1*01:09-DPB1*10:901', 'HLA-DPA1*01:09-DPB1*11:001',
         'HLA-DPA1*01:09-DPB1*11:01', 'HLA-DPA1*01:09-DPB1*11:101', 'HLA-DPA1*01:09-DPB1*11:201', 'HLA-DPA1*01:09-DPB1*11:301',
         'HLA-DPA1*01:09-DPB1*11:401', 'HLA-DPA1*01:09-DPB1*11:501',
         'HLA-DPA1*01:09-DPB1*11:601', 'HLA-DPA1*01:09-DPB1*11:701', 'HLA-DPA1*01:09-DPB1*11:801', 'HLA-DPA1*01:09-DPB1*11:901',
         'HLA-DPA1*01:09-DPB1*12:101', 'HLA-DPA1*01:09-DPB1*12:201',
         'HLA-DPA1*01:09-DPB1*12:301', 'HLA-DPA1*01:09-DPB1*12:401', 'HLA-DPA1*01:09-DPB1*12:501', 'HLA-DPA1*01:09-DPB1*12:601',
         'HLA-DPA1*01:09-DPB1*12:701', 'HLA-DPA1*01:09-DPB1*12:801',
         'HLA-DPA1*01:09-DPB1*12:901', 'HLA-DPA1*01:09-DPB1*13:001', 'HLA-DPA1*01:09-DPB1*13:01', 'HLA-DPA1*01:09-DPB1*13:101',
         'HLA-DPA1*01:09-DPB1*13:201', 'HLA-DPA1*01:09-DPB1*13:301',
         'HLA-DPA1*01:09-DPB1*13:401', 'HLA-DPA1*01:09-DPB1*14:01', 'HLA-DPA1*01:09-DPB1*15:01', 'HLA-DPA1*01:09-DPB1*16:01',
         'HLA-DPA1*01:09-DPB1*17:01', 'HLA-DPA1*01:09-DPB1*18:01',
         'HLA-DPA1*01:09-DPB1*19:01', 'HLA-DPA1*01:09-DPB1*20:01', 'HLA-DPA1*01:09-DPB1*21:01', 'HLA-DPA1*01:09-DPB1*22:01',
         'HLA-DPA1*01:09-DPB1*23:01', 'HLA-DPA1*01:09-DPB1*24:01',
         'HLA-DPA1*01:09-DPB1*25:01', 'HLA-DPA1*01:09-DPB1*26:01', 'HLA-DPA1*01:09-DPB1*27:01', 'HLA-DPA1*01:09-DPB1*28:01',
         'HLA-DPA1*01:09-DPB1*29:01', 'HLA-DPA1*01:09-DPB1*30:01',
         'HLA-DPA1*01:09-DPB1*31:01', 'HLA-DPA1*01:09-DPB1*32:01', 'HLA-DPA1*01:09-DPB1*33:01', 'HLA-DPA1*01:09-DPB1*34:01',
         'HLA-DPA1*01:09-DPB1*35:01', 'HLA-DPA1*01:09-DPB1*36:01',
         'HLA-DPA1*01:09-DPB1*37:01', 'HLA-DPA1*01:09-DPB1*38:01', 'HLA-DPA1*01:09-DPB1*39:01', 'HLA-DPA1*01:09-DPB1*40:01',
         'HLA-DPA1*01:09-DPB1*41:01', 'HLA-DPA1*01:09-DPB1*44:01',
         'HLA-DPA1*01:09-DPB1*45:01', 'HLA-DPA1*01:09-DPB1*46:01', 'HLA-DPA1*01:09-DPB1*47:01', 'HLA-DPA1*01:09-DPB1*48:01',
         'HLA-DPA1*01:09-DPB1*49:01', 'HLA-DPA1*01:09-DPB1*50:01',
         'HLA-DPA1*01:09-DPB1*51:01', 'HLA-DPA1*01:09-DPB1*52:01', 'HLA-DPA1*01:09-DPB1*53:01', 'HLA-DPA1*01:09-DPB1*54:01',
         'HLA-DPA1*01:09-DPB1*55:01', 'HLA-DPA1*01:09-DPB1*56:01',
         'HLA-DPA1*01:09-DPB1*58:01', 'HLA-DPA1*01:09-DPB1*59:01', 'HLA-DPA1*01:09-DPB1*60:01', 'HLA-DPA1*01:09-DPB1*62:01',
         'HLA-DPA1*01:09-DPB1*63:01', 'HLA-DPA1*01:09-DPB1*65:01',
         'HLA-DPA1*01:09-DPB1*66:01', 'HLA-DPA1*01:09-DPB1*67:01', 'HLA-DPA1*01:09-DPB1*68:01', 'HLA-DPA1*01:09-DPB1*69:01',
         'HLA-DPA1*01:09-DPB1*70:01', 'HLA-DPA1*01:09-DPB1*71:01',
         'HLA-DPA1*01:09-DPB1*72:01', 'HLA-DPA1*01:09-DPB1*73:01', 'HLA-DPA1*01:09-DPB1*74:01', 'HLA-DPA1*01:09-DPB1*75:01',
         'HLA-DPA1*01:09-DPB1*76:01', 'HLA-DPA1*01:09-DPB1*77:01',
         'HLA-DPA1*01:09-DPB1*78:01', 'HLA-DPA1*01:09-DPB1*79:01', 'HLA-DPA1*01:09-DPB1*80:01', 'HLA-DPA1*01:09-DPB1*81:01',
         'HLA-DPA1*01:09-DPB1*82:01', 'HLA-DPA1*01:09-DPB1*83:01',
         'HLA-DPA1*01:09-DPB1*84:01', 'HLA-DPA1*01:09-DPB1*85:01', 'HLA-DPA1*01:09-DPB1*86:01', 'HLA-DPA1*01:09-DPB1*87:01',
         'HLA-DPA1*01:09-DPB1*88:01', 'HLA-DPA1*01:09-DPB1*89:01',
         'HLA-DPA1*01:09-DPB1*90:01', 'HLA-DPA1*01:09-DPB1*91:01', 'HLA-DPA1*01:09-DPB1*92:01', 'HLA-DPA1*01:09-DPB1*93:01',
         'HLA-DPA1*01:09-DPB1*94:01', 'HLA-DPA1*01:09-DPB1*95:01',
         'HLA-DPA1*01:09-DPB1*96:01', 'HLA-DPA1*01:09-DPB1*97:01', 'HLA-DPA1*01:09-DPB1*98:01', 'HLA-DPA1*01:09-DPB1*99:01',
         'HLA-DPA1*01:10-DPB1*01:01', 'HLA-DPA1*01:10-DPB1*02:01',
         'HLA-DPA1*01:10-DPB1*02:02', 'HLA-DPA1*01:10-DPB1*03:01', 'HLA-DPA1*01:10-DPB1*04:01', 'HLA-DPA1*01:10-DPB1*04:02',
         'HLA-DPA1*01:10-DPB1*05:01', 'HLA-DPA1*01:10-DPB1*06:01',
         'HLA-DPA1*01:10-DPB1*08:01', 'HLA-DPA1*01:10-DPB1*09:01', 'HLA-DPA1*01:10-DPB1*10:001', 'HLA-DPA1*01:10-DPB1*10:01',
         'HLA-DPA1*01:10-DPB1*10:101', 'HLA-DPA1*01:10-DPB1*10:201',
         'HLA-DPA1*01:10-DPB1*10:301', 'HLA-DPA1*01:10-DPB1*10:401', 'HLA-DPA1*01:10-DPB1*10:501', 'HLA-DPA1*01:10-DPB1*10:601',
         'HLA-DPA1*01:10-DPB1*10:701', 'HLA-DPA1*01:10-DPB1*10:801',
         'HLA-DPA1*01:10-DPB1*10:901', 'HLA-DPA1*01:10-DPB1*11:001', 'HLA-DPA1*01:10-DPB1*11:01', 'HLA-DPA1*01:10-DPB1*11:101',
         'HLA-DPA1*01:10-DPB1*11:201', 'HLA-DPA1*01:10-DPB1*11:301',
         'HLA-DPA1*01:10-DPB1*11:401', 'HLA-DPA1*01:10-DPB1*11:501', 'HLA-DPA1*01:10-DPB1*11:601', 'HLA-DPA1*01:10-DPB1*11:701',
         'HLA-DPA1*01:10-DPB1*11:801', 'HLA-DPA1*01:10-DPB1*11:901',
         'HLA-DPA1*01:10-DPB1*12:101', 'HLA-DPA1*01:10-DPB1*12:201', 'HLA-DPA1*01:10-DPB1*12:301', 'HLA-DPA1*01:10-DPB1*12:401',
         'HLA-DPA1*01:10-DPB1*12:501', 'HLA-DPA1*01:10-DPB1*12:601',
         'HLA-DPA1*01:10-DPB1*12:701', 'HLA-DPA1*01:10-DPB1*12:801', 'HLA-DPA1*01:10-DPB1*12:901', 'HLA-DPA1*01:10-DPB1*13:001',
         'HLA-DPA1*01:10-DPB1*13:01', 'HLA-DPA1*01:10-DPB1*13:101',
         'HLA-DPA1*01:10-DPB1*13:201', 'HLA-DPA1*01:10-DPB1*13:301', 'HLA-DPA1*01:10-DPB1*13:401', 'HLA-DPA1*01:10-DPB1*14:01',
         'HLA-DPA1*01:10-DPB1*15:01', 'HLA-DPA1*01:10-DPB1*16:01',
         'HLA-DPA1*01:10-DPB1*17:01', 'HLA-DPA1*01:10-DPB1*18:01', 'HLA-DPA1*01:10-DPB1*19:01', 'HLA-DPA1*01:10-DPB1*20:01',
         'HLA-DPA1*01:10-DPB1*21:01', 'HLA-DPA1*01:10-DPB1*22:01',
         'HLA-DPA1*01:10-DPB1*23:01', 'HLA-DPA1*01:10-DPB1*24:01', 'HLA-DPA1*01:10-DPB1*25:01', 'HLA-DPA1*01:10-DPB1*26:01',
         'HLA-DPA1*01:10-DPB1*27:01', 'HLA-DPA1*01:10-DPB1*28:01',
         'HLA-DPA1*01:10-DPB1*29:01', 'HLA-DPA1*01:10-DPB1*30:01', 'HLA-DPA1*01:10-DPB1*31:01', 'HLA-DPA1*01:10-DPB1*32:01',
         'HLA-DPA1*01:10-DPB1*33:01', 'HLA-DPA1*01:10-DPB1*34:01',
         'HLA-DPA1*01:10-DPB1*35:01', 'HLA-DPA1*01:10-DPB1*36:01', 'HLA-DPA1*01:10-DPB1*37:01', 'HLA-DPA1*01:10-DPB1*38:01',
         'HLA-DPA1*01:10-DPB1*39:01', 'HLA-DPA1*01:10-DPB1*40:01',
         'HLA-DPA1*01:10-DPB1*41:01', 'HLA-DPA1*01:10-DPB1*44:01', 'HLA-DPA1*01:10-DPB1*45:01', 'HLA-DPA1*01:10-DPB1*46:01',
         'HLA-DPA1*01:10-DPB1*47:01', 'HLA-DPA1*01:10-DPB1*48:01',
         'HLA-DPA1*01:10-DPB1*49:01', 'HLA-DPA1*01:10-DPB1*50:01', 'HLA-DPA1*01:10-DPB1*51:01', 'HLA-DPA1*01:10-DPB1*52:01',
         'HLA-DPA1*01:10-DPB1*53:01', 'HLA-DPA1*01:10-DPB1*54:01',
         'HLA-DPA1*01:10-DPB1*55:01', 'HLA-DPA1*01:10-DPB1*56:01', 'HLA-DPA1*01:10-DPB1*58:01', 'HLA-DPA1*01:10-DPB1*59:01',
         'HLA-DPA1*01:10-DPB1*60:01', 'HLA-DPA1*01:10-DPB1*62:01',
         'HLA-DPA1*01:10-DPB1*63:01', 'HLA-DPA1*01:10-DPB1*65:01', 'HLA-DPA1*01:10-DPB1*66:01', 'HLA-DPA1*01:10-DPB1*67:01',
         'HLA-DPA1*01:10-DPB1*68:01', 'HLA-DPA1*01:10-DPB1*69:01',
         'HLA-DPA1*01:10-DPB1*70:01', 'HLA-DPA1*01:10-DPB1*71:01', 'HLA-DPA1*01:10-DPB1*72:01', 'HLA-DPA1*01:10-DPB1*73:01',
         'HLA-DPA1*01:10-DPB1*74:01', 'HLA-DPA1*01:10-DPB1*75:01',
         'HLA-DPA1*01:10-DPB1*76:01', 'HLA-DPA1*01:10-DPB1*77:01', 'HLA-DPA1*01:10-DPB1*78:01', 'HLA-DPA1*01:10-DPB1*79:01',
         'HLA-DPA1*01:10-DPB1*80:01', 'HLA-DPA1*01:10-DPB1*81:01',
         'HLA-DPA1*01:10-DPB1*82:01', 'HLA-DPA1*01:10-DPB1*83:01', 'HLA-DPA1*01:10-DPB1*84:01', 'HLA-DPA1*01:10-DPB1*85:01',
         'HLA-DPA1*01:10-DPB1*86:01', 'HLA-DPA1*01:10-DPB1*87:01',
         'HLA-DPA1*01:10-DPB1*88:01', 'HLA-DPA1*01:10-DPB1*89:01', 'HLA-DPA1*01:10-DPB1*90:01', 'HLA-DPA1*01:10-DPB1*91:01',
         'HLA-DPA1*01:10-DPB1*92:01', 'HLA-DPA1*01:10-DPB1*93:01',
         'HLA-DPA1*01:10-DPB1*94:01', 'HLA-DPA1*01:10-DPB1*95:01', 'HLA-DPA1*01:10-DPB1*96:01', 'HLA-DPA1*01:10-DPB1*97:01',
         'HLA-DPA1*01:10-DPB1*98:01', 'HLA-DPA1*01:10-DPB1*99:01',
         'HLA-DPA1*02:01-DPB1*01:01', 'HLA-DPA1*02:01-DPB1*02:01', 'HLA-DPA1*02:01-DPB1*02:02', 'HLA-DPA1*02:01-DPB1*03:01',
         'HLA-DPA1*02:01-DPB1*04:01', 'HLA-DPA1*02:01-DPB1*04:02',
         'HLA-DPA1*02:01-DPB1*05:01', 'HLA-DPA1*02:01-DPB1*06:01', 'HLA-DPA1*02:01-DPB1*08:01', 'HLA-DPA1*02:01-DPB1*09:01',
         'HLA-DPA1*02:01-DPB1*10:001', 'HLA-DPA1*02:01-DPB1*10:01',
         'HLA-DPA1*02:01-DPB1*10:101', 'HLA-DPA1*02:01-DPB1*10:201', 'HLA-DPA1*02:01-DPB1*10:301', 'HLA-DPA1*02:01-DPB1*10:401',
         'HLA-DPA1*02:01-DPB1*10:501', 'HLA-DPA1*02:01-DPB1*10:601',
         'HLA-DPA1*02:01-DPB1*10:701', 'HLA-DPA1*02:01-DPB1*10:801', 'HLA-DPA1*02:01-DPB1*10:901', 'HLA-DPA1*02:01-DPB1*11:001',
         'HLA-DPA1*02:01-DPB1*11:01', 'HLA-DPA1*02:01-DPB1*11:101',
         'HLA-DPA1*02:01-DPB1*11:201', 'HLA-DPA1*02:01-DPB1*11:301', 'HLA-DPA1*02:01-DPB1*11:401', 'HLA-DPA1*02:01-DPB1*11:501',
         'HLA-DPA1*02:01-DPB1*11:601', 'HLA-DPA1*02:01-DPB1*11:701',
         'HLA-DPA1*02:01-DPB1*11:801', 'HLA-DPA1*02:01-DPB1*11:901', 'HLA-DPA1*02:01-DPB1*12:101', 'HLA-DPA1*02:01-DPB1*12:201',
         'HLA-DPA1*02:01-DPB1*12:301', 'HLA-DPA1*02:01-DPB1*12:401',
         'HLA-DPA1*02:01-DPB1*12:501', 'HLA-DPA1*02:01-DPB1*12:601', 'HLA-DPA1*02:01-DPB1*12:701', 'HLA-DPA1*02:01-DPB1*12:801',
         'HLA-DPA1*02:01-DPB1*12:901', 'HLA-DPA1*02:01-DPB1*13:001',
         'HLA-DPA1*02:01-DPB1*13:01', 'HLA-DPA1*02:01-DPB1*13:101', 'HLA-DPA1*02:01-DPB1*13:201', 'HLA-DPA1*02:01-DPB1*13:301',
         'HLA-DPA1*02:01-DPB1*13:401', 'HLA-DPA1*02:01-DPB1*14:01',
         'HLA-DPA1*02:01-DPB1*15:01', 'HLA-DPA1*02:01-DPB1*16:01', 'HLA-DPA1*02:01-DPB1*17:01', 'HLA-DPA1*02:01-DPB1*18:01',
         'HLA-DPA1*02:01-DPB1*19:01', 'HLA-DPA1*02:01-DPB1*20:01',
         'HLA-DPA1*02:01-DPB1*21:01', 'HLA-DPA1*02:01-DPB1*22:01', 'HLA-DPA1*02:01-DPB1*23:01', 'HLA-DPA1*02:01-DPB1*24:01',
         'HLA-DPA1*02:01-DPB1*25:01', 'HLA-DPA1*02:01-DPB1*26:01',
         'HLA-DPA1*02:01-DPB1*27:01', 'HLA-DPA1*02:01-DPB1*28:01', 'HLA-DPA1*02:01-DPB1*29:01', 'HLA-DPA1*02:01-DPB1*30:01',
         'HLA-DPA1*02:01-DPB1*31:01', 'HLA-DPA1*02:01-DPB1*32:01',
         'HLA-DPA1*02:01-DPB1*33:01', 'HLA-DPA1*02:01-DPB1*34:01', 'HLA-DPA1*02:01-DPB1*35:01', 'HLA-DPA1*02:01-DPB1*36:01',
         'HLA-DPA1*02:01-DPB1*37:01', 'HLA-DPA1*02:01-DPB1*38:01',
         'HLA-DPA1*02:01-DPB1*39:01', 'HLA-DPA1*02:01-DPB1*40:01', 'HLA-DPA1*02:01-DPB1*41:01', 'HLA-DPA1*02:01-DPB1*44:01',
         'HLA-DPA1*02:01-DPB1*45:01', 'HLA-DPA1*02:01-DPB1*46:01',
         'HLA-DPA1*02:01-DPB1*47:01', 'HLA-DPA1*02:01-DPB1*48:01', 'HLA-DPA1*02:01-DPB1*49:01', 'HLA-DPA1*02:01-DPB1*50:01',
         'HLA-DPA1*02:01-DPB1*51:01', 'HLA-DPA1*02:01-DPB1*52:01',
         'HLA-DPA1*02:01-DPB1*53:01', 'HLA-DPA1*02:01-DPB1*54:01', 'HLA-DPA1*02:01-DPB1*55:01', 'HLA-DPA1*02:01-DPB1*56:01',
         'HLA-DPA1*02:01-DPB1*58:01', 'HLA-DPA1*02:01-DPB1*59:01',
         'HLA-DPA1*02:01-DPB1*60:01', 'HLA-DPA1*02:01-DPB1*62:01', 'HLA-DPA1*02:01-DPB1*63:01', 'HLA-DPA1*02:01-DPB1*65:01',
         'HLA-DPA1*02:01-DPB1*66:01', 'HLA-DPA1*02:01-DPB1*67:01',
         'HLA-DPA1*02:01-DPB1*68:01', 'HLA-DPA1*02:01-DPB1*69:01', 'HLA-DPA1*02:01-DPB1*70:01', 'HLA-DPA1*02:01-DPB1*71:01',
         'HLA-DPA1*02:01-DPB1*72:01', 'HLA-DPA1*02:01-DPB1*73:01',
         'HLA-DPA1*02:01-DPB1*74:01', 'HLA-DPA1*02:01-DPB1*75:01', 'HLA-DPA1*02:01-DPB1*76:01', 'HLA-DPA1*02:01-DPB1*77:01',
         'HLA-DPA1*02:01-DPB1*78:01', 'HLA-DPA1*02:01-DPB1*79:01',
         'HLA-DPA1*02:01-DPB1*80:01', 'HLA-DPA1*02:01-DPB1*81:01', 'HLA-DPA1*02:01-DPB1*82:01', 'HLA-DPA1*02:01-DPB1*83:01',
         'HLA-DPA1*02:01-DPB1*84:01', 'HLA-DPA1*02:01-DPB1*85:01',
         'HLA-DPA1*02:01-DPB1*86:01', 'HLA-DPA1*02:01-DPB1*87:01', 'HLA-DPA1*02:01-DPB1*88:01', 'HLA-DPA1*02:01-DPB1*89:01',
         'HLA-DPA1*02:01-DPB1*90:01', 'HLA-DPA1*02:01-DPB1*91:01',
         'HLA-DPA1*02:01-DPB1*92:01', 'HLA-DPA1*02:01-DPB1*93:01', 'HLA-DPA1*02:01-DPB1*94:01', 'HLA-DPA1*02:01-DPB1*95:01',
         'HLA-DPA1*02:01-DPB1*96:01', 'HLA-DPA1*02:01-DPB1*97:01',
         'HLA-DPA1*02:01-DPB1*98:01', 'HLA-DPA1*02:01-DPB1*99:01', 'HLA-DPA1*02:02-DPB1*01:01', 'HLA-DPA1*02:02-DPB1*02:01',
         'HLA-DPA1*02:02-DPB1*02:02', 'HLA-DPA1*02:02-DPB1*03:01',
         'HLA-DPA1*02:02-DPB1*04:01', 'HLA-DPA1*02:02-DPB1*04:02', 'HLA-DPA1*02:02-DPB1*05:01', 'HLA-DPA1*02:02-DPB1*06:01',
         'HLA-DPA1*02:02-DPB1*08:01', 'HLA-DPA1*02:02-DPB1*09:01',
         'HLA-DPA1*02:02-DPB1*10:001', 'HLA-DPA1*02:02-DPB1*10:01', 'HLA-DPA1*02:02-DPB1*10:101', 'HLA-DPA1*02:02-DPB1*10:201',
         'HLA-DPA1*02:02-DPB1*10:301', 'HLA-DPA1*02:02-DPB1*10:401',
         'HLA-DPA1*02:02-DPB1*10:501', 'HLA-DPA1*02:02-DPB1*10:601', 'HLA-DPA1*02:02-DPB1*10:701', 'HLA-DPA1*02:02-DPB1*10:801',
         'HLA-DPA1*02:02-DPB1*10:901', 'HLA-DPA1*02:02-DPB1*11:001',
         'HLA-DPA1*02:02-DPB1*11:01', 'HLA-DPA1*02:02-DPB1*11:101', 'HLA-DPA1*02:02-DPB1*11:201', 'HLA-DPA1*02:02-DPB1*11:301',
         'HLA-DPA1*02:02-DPB1*11:401', 'HLA-DPA1*02:02-DPB1*11:501',
         'HLA-DPA1*02:02-DPB1*11:601', 'HLA-DPA1*02:02-DPB1*11:701', 'HLA-DPA1*02:02-DPB1*11:801', 'HLA-DPA1*02:02-DPB1*11:901',
         'HLA-DPA1*02:02-DPB1*12:101', 'HLA-DPA1*02:02-DPB1*12:201',
         'HLA-DPA1*02:02-DPB1*12:301', 'HLA-DPA1*02:02-DPB1*12:401', 'HLA-DPA1*02:02-DPB1*12:501', 'HLA-DPA1*02:02-DPB1*12:601',
         'HLA-DPA1*02:02-DPB1*12:701', 'HLA-DPA1*02:02-DPB1*12:801',
         'HLA-DPA1*02:02-DPB1*12:901', 'HLA-DPA1*02:02-DPB1*13:001', 'HLA-DPA1*02:02-DPB1*13:01', 'HLA-DPA1*02:02-DPB1*13:101',
         'HLA-DPA1*02:02-DPB1*13:201', 'HLA-DPA1*02:02-DPB1*13:301',
         'HLA-DPA1*02:02-DPB1*13:401', 'HLA-DPA1*02:02-DPB1*14:01', 'HLA-DPA1*02:02-DPB1*15:01', 'HLA-DPA1*02:02-DPB1*16:01',
         'HLA-DPA1*02:02-DPB1*17:01', 'HLA-DPA1*02:02-DPB1*18:01',
         'HLA-DPA1*02:02-DPB1*19:01', 'HLA-DPA1*02:02-DPB1*20:01', 'HLA-DPA1*02:02-DPB1*21:01', 'HLA-DPA1*02:02-DPB1*22:01',
         'HLA-DPA1*02:02-DPB1*23:01', 'HLA-DPA1*02:02-DPB1*24:01',
         'HLA-DPA1*02:02-DPB1*25:01', 'HLA-DPA1*02:02-DPB1*26:01', 'HLA-DPA1*02:02-DPB1*27:01', 'HLA-DPA1*02:02-DPB1*28:01',
         'HLA-DPA1*02:02-DPB1*29:01', 'HLA-DPA1*02:02-DPB1*30:01',
         'HLA-DPA1*02:02-DPB1*31:01', 'HLA-DPA1*02:02-DPB1*32:01', 'HLA-DPA1*02:02-DPB1*33:01', 'HLA-DPA1*02:02-DPB1*34:01',
         'HLA-DPA1*02:02-DPB1*35:01', 'HLA-DPA1*02:02-DPB1*36:01',
         'HLA-DPA1*02:02-DPB1*37:01', 'HLA-DPA1*02:02-DPB1*38:01', 'HLA-DPA1*02:02-DPB1*39:01', 'HLA-DPA1*02:02-DPB1*40:01',
         'HLA-DPA1*02:02-DPB1*41:01', 'HLA-DPA1*02:02-DPB1*44:01',
         'HLA-DPA1*02:02-DPB1*45:01', 'HLA-DPA1*02:02-DPB1*46:01', 'HLA-DPA1*02:02-DPB1*47:01', 'HLA-DPA1*02:02-DPB1*48:01',
         'HLA-DPA1*02:02-DPB1*49:01', 'HLA-DPA1*02:02-DPB1*50:01',
         'HLA-DPA1*02:02-DPB1*51:01', 'HLA-DPA1*02:02-DPB1*52:01', 'HLA-DPA1*02:02-DPB1*53:01', 'HLA-DPA1*02:02-DPB1*54:01',
         'HLA-DPA1*02:02-DPB1*55:01', 'HLA-DPA1*02:02-DPB1*56:01',
         'HLA-DPA1*02:02-DPB1*58:01', 'HLA-DPA1*02:02-DPB1*59:01', 'HLA-DPA1*02:02-DPB1*60:01', 'HLA-DPA1*02:02-DPB1*62:01',
         'HLA-DPA1*02:02-DPB1*63:01', 'HLA-DPA1*02:02-DPB1*65:01',
         'HLA-DPA1*02:02-DPB1*66:01', 'HLA-DPA1*02:02-DPB1*67:01', 'HLA-DPA1*02:02-DPB1*68:01', 'HLA-DPA1*02:02-DPB1*69:01',
         'HLA-DPA1*02:02-DPB1*70:01', 'HLA-DPA1*02:02-DPB1*71:01',
         'HLA-DPA1*02:02-DPB1*72:01', 'HLA-DPA1*02:02-DPB1*73:01', 'HLA-DPA1*02:02-DPB1*74:01', 'HLA-DPA1*02:02-DPB1*75:01',
         'HLA-DPA1*02:02-DPB1*76:01', 'HLA-DPA1*02:02-DPB1*77:01',
         'HLA-DPA1*02:02-DPB1*78:01', 'HLA-DPA1*02:02-DPB1*79:01', 'HLA-DPA1*02:02-DPB1*80:01', 'HLA-DPA1*02:02-DPB1*81:01',
         'HLA-DPA1*02:02-DPB1*82:01', 'HLA-DPA1*02:02-DPB1*83:01',
         'HLA-DPA1*02:02-DPB1*84:01', 'HLA-DPA1*02:02-DPB1*85:01', 'HLA-DPA1*02:02-DPB1*86:01', 'HLA-DPA1*02:02-DPB1*87:01',
         'HLA-DPA1*02:02-DPB1*88:01', 'HLA-DPA1*02:02-DPB1*89:01',
         'HLA-DPA1*02:02-DPB1*90:01', 'HLA-DPA1*02:02-DPB1*91:01', 'HLA-DPA1*02:02-DPB1*92:01', 'HLA-DPA1*02:02-DPB1*93:01',
         'HLA-DPA1*02:02-DPB1*94:01', 'HLA-DPA1*02:02-DPB1*95:01',
         'HLA-DPA1*02:02-DPB1*96:01', 'HLA-DPA1*02:02-DPB1*97:01', 'HLA-DPA1*02:02-DPB1*98:01', 'HLA-DPA1*02:02-DPB1*99:01',
         'HLA-DPA1*02:03-DPB1*01:01', 'HLA-DPA1*02:03-DPB1*02:01',
         'HLA-DPA1*02:03-DPB1*02:02', 'HLA-DPA1*02:03-DPB1*03:01', 'HLA-DPA1*02:03-DPB1*04:01', 'HLA-DPA1*02:03-DPB1*04:02',
         'HLA-DPA1*02:03-DPB1*05:01', 'HLA-DPA1*02:03-DPB1*06:01',
         'HLA-DPA1*02:03-DPB1*08:01', 'HLA-DPA1*02:03-DPB1*09:01', 'HLA-DPA1*02:03-DPB1*10:001', 'HLA-DPA1*02:03-DPB1*10:01',
         'HLA-DPA1*02:03-DPB1*10:101', 'HLA-DPA1*02:03-DPB1*10:201',
         'HLA-DPA1*02:03-DPB1*10:301', 'HLA-DPA1*02:03-DPB1*10:401', 'HLA-DPA1*02:03-DPB1*10:501', 'HLA-DPA1*02:03-DPB1*10:601',
         'HLA-DPA1*02:03-DPB1*10:701', 'HLA-DPA1*02:03-DPB1*10:801',
         'HLA-DPA1*02:03-DPB1*10:901', 'HLA-DPA1*02:03-DPB1*11:001', 'HLA-DPA1*02:03-DPB1*11:01', 'HLA-DPA1*02:03-DPB1*11:101',
         'HLA-DPA1*02:03-DPB1*11:201', 'HLA-DPA1*02:03-DPB1*11:301',
         'HLA-DPA1*02:03-DPB1*11:401', 'HLA-DPA1*02:03-DPB1*11:501', 'HLA-DPA1*02:03-DPB1*11:601', 'HLA-DPA1*02:03-DPB1*11:701',
         'HLA-DPA1*02:03-DPB1*11:801', 'HLA-DPA1*02:03-DPB1*11:901',
         'HLA-DPA1*02:03-DPB1*12:101', 'HLA-DPA1*02:03-DPB1*12:201', 'HLA-DPA1*02:03-DPB1*12:301', 'HLA-DPA1*02:03-DPB1*12:401',
         'HLA-DPA1*02:03-DPB1*12:501', 'HLA-DPA1*02:03-DPB1*12:601',
         'HLA-DPA1*02:03-DPB1*12:701', 'HLA-DPA1*02:03-DPB1*12:801', 'HLA-DPA1*02:03-DPB1*12:901', 'HLA-DPA1*02:03-DPB1*13:001',
         'HLA-DPA1*02:03-DPB1*13:01', 'HLA-DPA1*02:03-DPB1*13:101',
         'HLA-DPA1*02:03-DPB1*13:201', 'HLA-DPA1*02:03-DPB1*13:301', 'HLA-DPA1*02:03-DPB1*13:401', 'HLA-DPA1*02:03-DPB1*14:01',
         'HLA-DPA1*02:03-DPB1*15:01', 'HLA-DPA1*02:03-DPB1*16:01',
         'HLA-DPA1*02:03-DPB1*17:01', 'HLA-DPA1*02:03-DPB1*18:01', 'HLA-DPA1*02:03-DPB1*19:01', 'HLA-DPA1*02:03-DPB1*20:01',
         'HLA-DPA1*02:03-DPB1*21:01', 'HLA-DPA1*02:03-DPB1*22:01',
         'HLA-DPA1*02:03-DPB1*23:01', 'HLA-DPA1*02:03-DPB1*24:01', 'HLA-DPA1*02:03-DPB1*25:01', 'HLA-DPA1*02:03-DPB1*26:01',
         'HLA-DPA1*02:03-DPB1*27:01', 'HLA-DPA1*02:03-DPB1*28:01',
         'HLA-DPA1*02:03-DPB1*29:01', 'HLA-DPA1*02:03-DPB1*30:01', 'HLA-DPA1*02:03-DPB1*31:01', 'HLA-DPA1*02:03-DPB1*32:01',
         'HLA-DPA1*02:03-DPB1*33:01', 'HLA-DPA1*02:03-DPB1*34:01',
         'HLA-DPA1*02:03-DPB1*35:01', 'HLA-DPA1*02:03-DPB1*36:01', 'HLA-DPA1*02:03-DPB1*37:01', 'HLA-DPA1*02:03-DPB1*38:01',
         'HLA-DPA1*02:03-DPB1*39:01', 'HLA-DPA1*02:03-DPB1*40:01',
         'HLA-DPA1*02:03-DPB1*41:01', 'HLA-DPA1*02:03-DPB1*44:01', 'HLA-DPA1*02:03-DPB1*45:01', 'HLA-DPA1*02:03-DPB1*46:01',
         'HLA-DPA1*02:03-DPB1*47:01', 'HLA-DPA1*02:03-DPB1*48:01',
         'HLA-DPA1*02:03-DPB1*49:01', 'HLA-DPA1*02:03-DPB1*50:01', 'HLA-DPA1*02:03-DPB1*51:01', 'HLA-DPA1*02:03-DPB1*52:01',
         'HLA-DPA1*02:03-DPB1*53:01', 'HLA-DPA1*02:03-DPB1*54:01',
         'HLA-DPA1*02:03-DPB1*55:01', 'HLA-DPA1*02:03-DPB1*56:01', 'HLA-DPA1*02:03-DPB1*58:01', 'HLA-DPA1*02:03-DPB1*59:01',
         'HLA-DPA1*02:03-DPB1*60:01', 'HLA-DPA1*02:03-DPB1*62:01',
         'HLA-DPA1*02:03-DPB1*63:01', 'HLA-DPA1*02:03-DPB1*65:01', 'HLA-DPA1*02:03-DPB1*66:01', 'HLA-DPA1*02:03-DPB1*67:01',
         'HLA-DPA1*02:03-DPB1*68:01', 'HLA-DPA1*02:03-DPB1*69:01',
         'HLA-DPA1*02:03-DPB1*70:01', 'HLA-DPA1*02:03-DPB1*71:01', 'HLA-DPA1*02:03-DPB1*72:01', 'HLA-DPA1*02:03-DPB1*73:01',
         'HLA-DPA1*02:03-DPB1*74:01', 'HLA-DPA1*02:03-DPB1*75:01',
         'HLA-DPA1*02:03-DPB1*76:01', 'HLA-DPA1*02:03-DPB1*77:01', 'HLA-DPA1*02:03-DPB1*78:01', 'HLA-DPA1*02:03-DPB1*79:01',
         'HLA-DPA1*02:03-DPB1*80:01', 'HLA-DPA1*02:03-DPB1*81:01',
         'HLA-DPA1*02:03-DPB1*82:01', 'HLA-DPA1*02:03-DPB1*83:01', 'HLA-DPA1*02:03-DPB1*84:01', 'HLA-DPA1*02:03-DPB1*85:01',
         'HLA-DPA1*02:03-DPB1*86:01', 'HLA-DPA1*02:03-DPB1*87:01',
         'HLA-DPA1*02:03-DPB1*88:01', 'HLA-DPA1*02:03-DPB1*89:01', 'HLA-DPA1*02:03-DPB1*90:01', 'HLA-DPA1*02:03-DPB1*91:01',
         'HLA-DPA1*02:03-DPB1*92:01', 'HLA-DPA1*02:03-DPB1*93:01',
         'HLA-DPA1*02:03-DPB1*94:01', 'HLA-DPA1*02:03-DPB1*95:01', 'HLA-DPA1*02:03-DPB1*96:01', 'HLA-DPA1*02:03-DPB1*97:01',
         'HLA-DPA1*02:03-DPB1*98:01', 'HLA-DPA1*02:03-DPB1*99:01',
         'HLA-DPA1*02:04-DPB1*01:01', 'HLA-DPA1*02:04-DPB1*02:01', 'HLA-DPA1*02:04-DPB1*02:02', 'HLA-DPA1*02:04-DPB1*03:01',
         'HLA-DPA1*02:04-DPB1*04:01', 'HLA-DPA1*02:04-DPB1*04:02',
         'HLA-DPA1*02:04-DPB1*05:01', 'HLA-DPA1*02:04-DPB1*06:01', 'HLA-DPA1*02:04-DPB1*08:01', 'HLA-DPA1*02:04-DPB1*09:01',
         'HLA-DPA1*02:04-DPB1*10:001', 'HLA-DPA1*02:04-DPB1*10:01',
         'HLA-DPA1*02:04-DPB1*10:101', 'HLA-DPA1*02:04-DPB1*10:201', 'HLA-DPA1*02:04-DPB1*10:301', 'HLA-DPA1*02:04-DPB1*10:401',
         'HLA-DPA1*02:04-DPB1*10:501', 'HLA-DPA1*02:04-DPB1*10:601',
         'HLA-DPA1*02:04-DPB1*10:701', 'HLA-DPA1*02:04-DPB1*10:801', 'HLA-DPA1*02:04-DPB1*10:901', 'HLA-DPA1*02:04-DPB1*11:001',
         'HLA-DPA1*02:04-DPB1*11:01', 'HLA-DPA1*02:04-DPB1*11:101',
         'HLA-DPA1*02:04-DPB1*11:201', 'HLA-DPA1*02:04-DPB1*11:301', 'HLA-DPA1*02:04-DPB1*11:401', 'HLA-DPA1*02:04-DPB1*11:501',
         'HLA-DPA1*02:04-DPB1*11:601', 'HLA-DPA1*02:04-DPB1*11:701',
         'HLA-DPA1*02:04-DPB1*11:801', 'HLA-DPA1*02:04-DPB1*11:901', 'HLA-DPA1*02:04-DPB1*12:101', 'HLA-DPA1*02:04-DPB1*12:201',
         'HLA-DPA1*02:04-DPB1*12:301', 'HLA-DPA1*02:04-DPB1*12:401',
         'HLA-DPA1*02:04-DPB1*12:501', 'HLA-DPA1*02:04-DPB1*12:601', 'HLA-DPA1*02:04-DPB1*12:701', 'HLA-DPA1*02:04-DPB1*12:801',
         'HLA-DPA1*02:04-DPB1*12:901', 'HLA-DPA1*02:04-DPB1*13:001',
         'HLA-DPA1*02:04-DPB1*13:01', 'HLA-DPA1*02:04-DPB1*13:101', 'HLA-DPA1*02:04-DPB1*13:201', 'HLA-DPA1*02:04-DPB1*13:301',
         'HLA-DPA1*02:04-DPB1*13:401', 'HLA-DPA1*02:04-DPB1*14:01',
         'HLA-DPA1*02:04-DPB1*15:01', 'HLA-DPA1*02:04-DPB1*16:01', 'HLA-DPA1*02:04-DPB1*17:01', 'HLA-DPA1*02:04-DPB1*18:01',
         'HLA-DPA1*02:04-DPB1*19:01', 'HLA-DPA1*02:04-DPB1*20:01',
         'HLA-DPA1*02:04-DPB1*21:01', 'HLA-DPA1*02:04-DPB1*22:01', 'HLA-DPA1*02:04-DPB1*23:01', 'HLA-DPA1*02:04-DPB1*24:01',
         'HLA-DPA1*02:04-DPB1*25:01', 'HLA-DPA1*02:04-DPB1*26:01',
         'HLA-DPA1*02:04-DPB1*27:01', 'HLA-DPA1*02:04-DPB1*28:01', 'HLA-DPA1*02:04-DPB1*29:01', 'HLA-DPA1*02:04-DPB1*30:01',
         'HLA-DPA1*02:04-DPB1*31:01', 'HLA-DPA1*02:04-DPB1*32:01',
         'HLA-DPA1*02:04-DPB1*33:01', 'HLA-DPA1*02:04-DPB1*34:01', 'HLA-DPA1*02:04-DPB1*35:01', 'HLA-DPA1*02:04-DPB1*36:01',
         'HLA-DPA1*02:04-DPB1*37:01', 'HLA-DPA1*02:04-DPB1*38:01',
         'HLA-DPA1*02:04-DPB1*39:01', 'HLA-DPA1*02:04-DPB1*40:01', 'HLA-DPA1*02:04-DPB1*41:01', 'HLA-DPA1*02:04-DPB1*44:01',
         'HLA-DPA1*02:04-DPB1*45:01', 'HLA-DPA1*02:04-DPB1*46:01',
         'HLA-DPA1*02:04-DPB1*47:01', 'HLA-DPA1*02:04-DPB1*48:01', 'HLA-DPA1*02:04-DPB1*49:01', 'HLA-DPA1*02:04-DPB1*50:01',
         'HLA-DPA1*02:04-DPB1*51:01', 'HLA-DPA1*02:04-DPB1*52:01',
         'HLA-DPA1*02:04-DPB1*53:01', 'HLA-DPA1*02:04-DPB1*54:01', 'HLA-DPA1*02:04-DPB1*55:01', 'HLA-DPA1*02:04-DPB1*56:01',
         'HLA-DPA1*02:04-DPB1*58:01', 'HLA-DPA1*02:04-DPB1*59:01',
         'HLA-DPA1*02:04-DPB1*60:01', 'HLA-DPA1*02:04-DPB1*62:01', 'HLA-DPA1*02:04-DPB1*63:01', 'HLA-DPA1*02:04-DPB1*65:01',
         'HLA-DPA1*02:04-DPB1*66:01', 'HLA-DPA1*02:04-DPB1*67:01',
         'HLA-DPA1*02:04-DPB1*68:01', 'HLA-DPA1*02:04-DPB1*69:01', 'HLA-DPA1*02:04-DPB1*70:01', 'HLA-DPA1*02:04-DPB1*71:01',
         'HLA-DPA1*02:04-DPB1*72:01', 'HLA-DPA1*02:04-DPB1*73:01',
         'HLA-DPA1*02:04-DPB1*74:01', 'HLA-DPA1*02:04-DPB1*75:01', 'HLA-DPA1*02:04-DPB1*76:01', 'HLA-DPA1*02:04-DPB1*77:01',
         'HLA-DPA1*02:04-DPB1*78:01', 'HLA-DPA1*02:04-DPB1*79:01',
         'HLA-DPA1*02:04-DPB1*80:01', 'HLA-DPA1*02:04-DPB1*81:01', 'HLA-DPA1*02:04-DPB1*82:01', 'HLA-DPA1*02:04-DPB1*83:01',
         'HLA-DPA1*02:04-DPB1*84:01', 'HLA-DPA1*02:04-DPB1*85:01',
         'HLA-DPA1*02:04-DPB1*86:01', 'HLA-DPA1*02:04-DPB1*87:01', 'HLA-DPA1*02:04-DPB1*88:01', 'HLA-DPA1*02:04-DPB1*89:01',
         'HLA-DPA1*02:04-DPB1*90:01', 'HLA-DPA1*02:04-DPB1*91:01',
         'HLA-DPA1*02:04-DPB1*92:01', 'HLA-DPA1*02:04-DPB1*93:01', 'HLA-DPA1*02:04-DPB1*94:01', 'HLA-DPA1*02:04-DPB1*95:01',
         'HLA-DPA1*02:04-DPB1*96:01', 'HLA-DPA1*02:04-DPB1*97:01',
         'HLA-DPA1*02:04-DPB1*98:01', 'HLA-DPA1*02:04-DPB1*99:01', 'HLA-DPA1*03:01-DPB1*01:01', 'HLA-DPA1*03:01-DPB1*02:01',
         'HLA-DPA1*03:01-DPB1*02:02', 'HLA-DPA1*03:01-DPB1*03:01',
         'HLA-DPA1*03:01-DPB1*04:01', 'HLA-DPA1*03:01-DPB1*04:02', 'HLA-DPA1*03:01-DPB1*05:01', 'HLA-DPA1*03:01-DPB1*06:01',
         'HLA-DPA1*03:01-DPB1*08:01', 'HLA-DPA1*03:01-DPB1*09:01',
         'HLA-DPA1*03:01-DPB1*10:001', 'HLA-DPA1*03:01-DPB1*10:01', 'HLA-DPA1*03:01-DPB1*10:101', 'HLA-DPA1*03:01-DPB1*10:201',
         'HLA-DPA1*03:01-DPB1*10:301', 'HLA-DPA1*03:01-DPB1*10:401',
         'HLA-DPA1*03:01-DPB1*10:501', 'HLA-DPA1*03:01-DPB1*10:601', 'HLA-DPA1*03:01-DPB1*10:701', 'HLA-DPA1*03:01-DPB1*10:801',
         'HLA-DPA1*03:01-DPB1*10:901', 'HLA-DPA1*03:01-DPB1*11:001',
         'HLA-DPA1*03:01-DPB1*11:01', 'HLA-DPA1*03:01-DPB1*11:101', 'HLA-DPA1*03:01-DPB1*11:201', 'HLA-DPA1*03:01-DPB1*11:301',
         'HLA-DPA1*03:01-DPB1*11:401', 'HLA-DPA1*03:01-DPB1*11:501',
         'HLA-DPA1*03:01-DPB1*11:601', 'HLA-DPA1*03:01-DPB1*11:701', 'HLA-DPA1*03:01-DPB1*11:801', 'HLA-DPA1*03:01-DPB1*11:901',
         'HLA-DPA1*03:01-DPB1*12:101', 'HLA-DPA1*03:01-DPB1*12:201',
         'HLA-DPA1*03:01-DPB1*12:301', 'HLA-DPA1*03:01-DPB1*12:401', 'HLA-DPA1*03:01-DPB1*12:501', 'HLA-DPA1*03:01-DPB1*12:601',
         'HLA-DPA1*03:01-DPB1*12:701', 'HLA-DPA1*03:01-DPB1*12:801',
         'HLA-DPA1*03:01-DPB1*12:901', 'HLA-DPA1*03:01-DPB1*13:001', 'HLA-DPA1*03:01-DPB1*13:01', 'HLA-DPA1*03:01-DPB1*13:101',
         'HLA-DPA1*03:01-DPB1*13:201', 'HLA-DPA1*03:01-DPB1*13:301',
         'HLA-DPA1*03:01-DPB1*13:401', 'HLA-DPA1*03:01-DPB1*14:01', 'HLA-DPA1*03:01-DPB1*15:01', 'HLA-DPA1*03:01-DPB1*16:01',
         'HLA-DPA1*03:01-DPB1*17:01', 'HLA-DPA1*03:01-DPB1*18:01',
         'HLA-DPA1*03:01-DPB1*19:01', 'HLA-DPA1*03:01-DPB1*20:01', 'HLA-DPA1*03:01-DPB1*21:01', 'HLA-DPA1*03:01-DPB1*22:01',
         'HLA-DPA1*03:01-DPB1*23:01', 'HLA-DPA1*03:01-DPB1*24:01',
         'HLA-DPA1*03:01-DPB1*25:01', 'HLA-DPA1*03:01-DPB1*26:01', 'HLA-DPA1*03:01-DPB1*27:01', 'HLA-DPA1*03:01-DPB1*28:01',
         'HLA-DPA1*03:01-DPB1*29:01', 'HLA-DPA1*03:01-DPB1*30:01',
         'HLA-DPA1*03:01-DPB1*31:01', 'HLA-DPA1*03:01-DPB1*32:01', 'HLA-DPA1*03:01-DPB1*33:01', 'HLA-DPA1*03:01-DPB1*34:01',
         'HLA-DPA1*03:01-DPB1*35:01', 'HLA-DPA1*03:01-DPB1*36:01',
         'HLA-DPA1*03:01-DPB1*37:01', 'HLA-DPA1*03:01-DPB1*38:01', 'HLA-DPA1*03:01-DPB1*39:01', 'HLA-DPA1*03:01-DPB1*40:01',
         'HLA-DPA1*03:01-DPB1*41:01', 'HLA-DPA1*03:01-DPB1*44:01',
         'HLA-DPA1*03:01-DPB1*45:01', 'HLA-DPA1*03:01-DPB1*46:01', 'HLA-DPA1*03:01-DPB1*47:01', 'HLA-DPA1*03:01-DPB1*48:01',
         'HLA-DPA1*03:01-DPB1*49:01', 'HLA-DPA1*03:01-DPB1*50:01',
         'HLA-DPA1*03:01-DPB1*51:01', 'HLA-DPA1*03:01-DPB1*52:01', 'HLA-DPA1*03:01-DPB1*53:01', 'HLA-DPA1*03:01-DPB1*54:01',
         'HLA-DPA1*03:01-DPB1*55:01', 'HLA-DPA1*03:01-DPB1*56:01',
         'HLA-DPA1*03:01-DPB1*58:01', 'HLA-DPA1*03:01-DPB1*59:01', 'HLA-DPA1*03:01-DPB1*60:01', 'HLA-DPA1*03:01-DPB1*62:01',
         'HLA-DPA1*03:01-DPB1*63:01', 'HLA-DPA1*03:01-DPB1*65:01',
         'HLA-DPA1*03:01-DPB1*66:01', 'HLA-DPA1*03:01-DPB1*67:01', 'HLA-DPA1*03:01-DPB1*68:01', 'HLA-DPA1*03:01-DPB1*69:01',
         'HLA-DPA1*03:01-DPB1*70:01', 'HLA-DPA1*03:01-DPB1*71:01',
         'HLA-DPA1*03:01-DPB1*72:01', 'HLA-DPA1*03:01-DPB1*73:01', 'HLA-DPA1*03:01-DPB1*74:01', 'HLA-DPA1*03:01-DPB1*75:01',
         'HLA-DPA1*03:01-DPB1*76:01', 'HLA-DPA1*03:01-DPB1*77:01',
         'HLA-DPA1*03:01-DPB1*78:01', 'HLA-DPA1*03:01-DPB1*79:01', 'HLA-DPA1*03:01-DPB1*80:01', 'HLA-DPA1*03:01-DPB1*81:01',
         'HLA-DPA1*03:01-DPB1*82:01', 'HLA-DPA1*03:01-DPB1*83:01',
         'HLA-DPA1*03:01-DPB1*84:01', 'HLA-DPA1*03:01-DPB1*85:01', 'HLA-DPA1*03:01-DPB1*86:01', 'HLA-DPA1*03:01-DPB1*87:01',
         'HLA-DPA1*03:01-DPB1*88:01', 'HLA-DPA1*03:01-DPB1*89:01',
         'HLA-DPA1*03:01-DPB1*90:01', 'HLA-DPA1*03:01-DPB1*91:01', 'HLA-DPA1*03:01-DPB1*92:01', 'HLA-DPA1*03:01-DPB1*93:01',
         'HLA-DPA1*03:01-DPB1*94:01', 'HLA-DPA1*03:01-DPB1*95:01',
         'HLA-DPA1*03:01-DPB1*96:01', 'HLA-DPA1*03:01-DPB1*97:01', 'HLA-DPA1*03:01-DPB1*98:01', 'HLA-DPA1*03:01-DPB1*99:01',
         'HLA-DPA1*03:02-DPB1*01:01', 'HLA-DPA1*03:02-DPB1*02:01',
         'HLA-DPA1*03:02-DPB1*02:02', 'HLA-DPA1*03:02-DPB1*03:01', 'HLA-DPA1*03:02-DPB1*04:01', 'HLA-DPA1*03:02-DPB1*04:02',
         'HLA-DPA1*03:02-DPB1*05:01', 'HLA-DPA1*03:02-DPB1*06:01',
         'HLA-DPA1*03:02-DPB1*08:01', 'HLA-DPA1*03:02-DPB1*09:01', 'HLA-DPA1*03:02-DPB1*10:001', 'HLA-DPA1*03:02-DPB1*10:01',
         'HLA-DPA1*03:02-DPB1*10:101', 'HLA-DPA1*03:02-DPB1*10:201',
         'HLA-DPA1*03:02-DPB1*10:301', 'HLA-DPA1*03:02-DPB1*10:401', 'HLA-DPA1*03:02-DPB1*10:501', 'HLA-DPA1*03:02-DPB1*10:601',
         'HLA-DPA1*03:02-DPB1*10:701', 'HLA-DPA1*03:02-DPB1*10:801',
         'HLA-DPA1*03:02-DPB1*10:901', 'HLA-DPA1*03:02-DPB1*11:001', 'HLA-DPA1*03:02-DPB1*11:01', 'HLA-DPA1*03:02-DPB1*11:101',
         'HLA-DPA1*03:02-DPB1*11:201', 'HLA-DPA1*03:02-DPB1*11:301',
         'HLA-DPA1*03:02-DPB1*11:401', 'HLA-DPA1*03:02-DPB1*11:501', 'HLA-DPA1*03:02-DPB1*11:601', 'HLA-DPA1*03:02-DPB1*11:701',
         'HLA-DPA1*03:02-DPB1*11:801', 'HLA-DPA1*03:02-DPB1*11:901',
         'HLA-DPA1*03:02-DPB1*12:101', 'HLA-DPA1*03:02-DPB1*12:201', 'HLA-DPA1*03:02-DPB1*12:301', 'HLA-DPA1*03:02-DPB1*12:401',
         'HLA-DPA1*03:02-DPB1*12:501', 'HLA-DPA1*03:02-DPB1*12:601',
         'HLA-DPA1*03:02-DPB1*12:701', 'HLA-DPA1*03:02-DPB1*12:801', 'HLA-DPA1*03:02-DPB1*12:901', 'HLA-DPA1*03:02-DPB1*13:001',
         'HLA-DPA1*03:02-DPB1*13:01', 'HLA-DPA1*03:02-DPB1*13:101',
         'HLA-DPA1*03:02-DPB1*13:201', 'HLA-DPA1*03:02-DPB1*13:301', 'HLA-DPA1*03:02-DPB1*13:401', 'HLA-DPA1*03:02-DPB1*14:01',
         'HLA-DPA1*03:02-DPB1*15:01', 'HLA-DPA1*03:02-DPB1*16:01',
         'HLA-DPA1*03:02-DPB1*17:01', 'HLA-DPA1*03:02-DPB1*18:01', 'HLA-DPA1*03:02-DPB1*19:01', 'HLA-DPA1*03:02-DPB1*20:01',
         'HLA-DPA1*03:02-DPB1*21:01', 'HLA-DPA1*03:02-DPB1*22:01',
         'HLA-DPA1*03:02-DPB1*23:01', 'HLA-DPA1*03:02-DPB1*24:01', 'HLA-DPA1*03:02-DPB1*25:01', 'HLA-DPA1*03:02-DPB1*26:01',
         'HLA-DPA1*03:02-DPB1*27:01', 'HLA-DPA1*03:02-DPB1*28:01',
         'HLA-DPA1*03:02-DPB1*29:01', 'HLA-DPA1*03:02-DPB1*30:01', 'HLA-DPA1*03:02-DPB1*31:01', 'HLA-DPA1*03:02-DPB1*32:01',
         'HLA-DPA1*03:02-DPB1*33:01', 'HLA-DPA1*03:02-DPB1*34:01',
         'HLA-DPA1*03:02-DPB1*35:01', 'HLA-DPA1*03:02-DPB1*36:01', 'HLA-DPA1*03:02-DPB1*37:01', 'HLA-DPA1*03:02-DPB1*38:01',
         'HLA-DPA1*03:02-DPB1*39:01', 'HLA-DPA1*03:02-DPB1*40:01',
         'HLA-DPA1*03:02-DPB1*41:01', 'HLA-DPA1*03:02-DPB1*44:01', 'HLA-DPA1*03:02-DPB1*45:01', 'HLA-DPA1*03:02-DPB1*46:01',
         'HLA-DPA1*03:02-DPB1*47:01', 'HLA-DPA1*03:02-DPB1*48:01',
         'HLA-DPA1*03:02-DPB1*49:01', 'HLA-DPA1*03:02-DPB1*50:01', 'HLA-DPA1*03:02-DPB1*51:01', 'HLA-DPA1*03:02-DPB1*52:01',
         'HLA-DPA1*03:02-DPB1*53:01', 'HLA-DPA1*03:02-DPB1*54:01',
         'HLA-DPA1*03:02-DPB1*55:01', 'HLA-DPA1*03:02-DPB1*56:01', 'HLA-DPA1*03:02-DPB1*58:01', 'HLA-DPA1*03:02-DPB1*59:01',
         'HLA-DPA1*03:02-DPB1*60:01', 'HLA-DPA1*03:02-DPB1*62:01',
         'HLA-DPA1*03:02-DPB1*63:01', 'HLA-DPA1*03:02-DPB1*65:01', 'HLA-DPA1*03:02-DPB1*66:01', 'HLA-DPA1*03:02-DPB1*67:01',
         'HLA-DPA1*03:02-DPB1*68:01', 'HLA-DPA1*03:02-DPB1*69:01',
         'HLA-DPA1*03:02-DPB1*70:01', 'HLA-DPA1*03:02-DPB1*71:01', 'HLA-DPA1*03:02-DPB1*72:01', 'HLA-DPA1*03:02-DPB1*73:01',
         'HLA-DPA1*03:02-DPB1*74:01', 'HLA-DPA1*03:02-DPB1*75:01',
         'HLA-DPA1*03:02-DPB1*76:01', 'HLA-DPA1*03:02-DPB1*77:01', 'HLA-DPA1*03:02-DPB1*78:01', 'HLA-DPA1*03:02-DPB1*79:01',
         'HLA-DPA1*03:02-DPB1*80:01', 'HLA-DPA1*03:02-DPB1*81:01',
         'HLA-DPA1*03:02-DPB1*82:01', 'HLA-DPA1*03:02-DPB1*83:01', 'HLA-DPA1*03:02-DPB1*84:01', 'HLA-DPA1*03:02-DPB1*85:01',
         'HLA-DPA1*03:02-DPB1*86:01', 'HLA-DPA1*03:02-DPB1*87:01',
         'HLA-DPA1*03:02-DPB1*88:01', 'HLA-DPA1*03:02-DPB1*89:01', 'HLA-DPA1*03:02-DPB1*90:01', 'HLA-DPA1*03:02-DPB1*91:01',
         'HLA-DPA1*03:02-DPB1*92:01', 'HLA-DPA1*03:02-DPB1*93:01',
         'HLA-DPA1*03:02-DPB1*94:01', 'HLA-DPA1*03:02-DPB1*95:01', 'HLA-DPA1*03:02-DPB1*96:01', 'HLA-DPA1*03:02-DPB1*97:01',
         'HLA-DPA1*03:02-DPB1*98:01', 'HLA-DPA1*03:02-DPB1*99:01',
         'HLA-DPA1*03:03-DPB1*01:01', 'HLA-DPA1*03:03-DPB1*02:01', 'HLA-DPA1*03:03-DPB1*02:02', 'HLA-DPA1*03:03-DPB1*03:01',
         'HLA-DPA1*03:03-DPB1*04:01', 'HLA-DPA1*03:03-DPB1*04:02',
         'HLA-DPA1*03:03-DPB1*05:01', 'HLA-DPA1*03:03-DPB1*06:01', 'HLA-DPA1*03:03-DPB1*08:01', 'HLA-DPA1*03:03-DPB1*09:01',
         'HLA-DPA1*03:03-DPB1*10:001', 'HLA-DPA1*03:03-DPB1*10:01',
         'HLA-DPA1*03:03-DPB1*10:101', 'HLA-DPA1*03:03-DPB1*10:201', 'HLA-DPA1*03:03-DPB1*10:301', 'HLA-DPA1*03:03-DPB1*10:401',
         'HLA-DPA1*03:03-DPB1*10:501', 'HLA-DPA1*03:03-DPB1*10:601',
         'HLA-DPA1*03:03-DPB1*10:701', 'HLA-DPA1*03:03-DPB1*10:801', 'HLA-DPA1*03:03-DPB1*10:901', 'HLA-DPA1*03:03-DPB1*11:001',
         'HLA-DPA1*03:03-DPB1*11:01', 'HLA-DPA1*03:03-DPB1*11:101',
         'HLA-DPA1*03:03-DPB1*11:201', 'HLA-DPA1*03:03-DPB1*11:301', 'HLA-DPA1*03:03-DPB1*11:401', 'HLA-DPA1*03:03-DPB1*11:501',
         'HLA-DPA1*03:03-DPB1*11:601', 'HLA-DPA1*03:03-DPB1*11:701',
         'HLA-DPA1*03:03-DPB1*11:801', 'HLA-DPA1*03:03-DPB1*11:901', 'HLA-DPA1*03:03-DPB1*12:101', 'HLA-DPA1*03:03-DPB1*12:201',
         'HLA-DPA1*03:03-DPB1*12:301', 'HLA-DPA1*03:03-DPB1*12:401',
         'HLA-DPA1*03:03-DPB1*12:501', 'HLA-DPA1*03:03-DPB1*12:601', 'HLA-DPA1*03:03-DPB1*12:701', 'HLA-DPA1*03:03-DPB1*12:801',
         'HLA-DPA1*03:03-DPB1*12:901', 'HLA-DPA1*03:03-DPB1*13:001',
         'HLA-DPA1*03:03-DPB1*13:01', 'HLA-DPA1*03:03-DPB1*13:101', 'HLA-DPA1*03:03-DPB1*13:201', 'HLA-DPA1*03:03-DPB1*13:301',
         'HLA-DPA1*03:03-DPB1*13:401', 'HLA-DPA1*03:03-DPB1*14:01',
         'HLA-DPA1*03:03-DPB1*15:01', 'HLA-DPA1*03:03-DPB1*16:01', 'HLA-DPA1*03:03-DPB1*17:01', 'HLA-DPA1*03:03-DPB1*18:01',
         'HLA-DPA1*03:03-DPB1*19:01', 'HLA-DPA1*03:03-DPB1*20:01',
         'HLA-DPA1*03:03-DPB1*21:01', 'HLA-DPA1*03:03-DPB1*22:01', 'HLA-DPA1*03:03-DPB1*23:01', 'HLA-DPA1*03:03-DPB1*24:01',
         'HLA-DPA1*03:03-DPB1*25:01', 'HLA-DPA1*03:03-DPB1*26:01',
         'HLA-DPA1*03:03-DPB1*27:01', 'HLA-DPA1*03:03-DPB1*28:01', 'HLA-DPA1*03:03-DPB1*29:01', 'HLA-DPA1*03:03-DPB1*30:01',
         'HLA-DPA1*03:03-DPB1*31:01', 'HLA-DPA1*03:03-DPB1*32:01',
         'HLA-DPA1*03:03-DPB1*33:01', 'HLA-DPA1*03:03-DPB1*34:01', 'HLA-DPA1*03:03-DPB1*35:01', 'HLA-DPA1*03:03-DPB1*36:01',
         'HLA-DPA1*03:03-DPB1*37:01', 'HLA-DPA1*03:03-DPB1*38:01',
         'HLA-DPA1*03:03-DPB1*39:01', 'HLA-DPA1*03:03-DPB1*40:01', 'HLA-DPA1*03:03-DPB1*41:01', 'HLA-DPA1*03:03-DPB1*44:01',
         'HLA-DPA1*03:03-DPB1*45:01', 'HLA-DPA1*03:03-DPB1*46:01',
         'HLA-DPA1*03:03-DPB1*47:01', 'HLA-DPA1*03:03-DPB1*48:01', 'HLA-DPA1*03:03-DPB1*49:01', 'HLA-DPA1*03:03-DPB1*50:01',
         'HLA-DPA1*03:03-DPB1*51:01', 'HLA-DPA1*03:03-DPB1*52:01',
         'HLA-DPA1*03:03-DPB1*53:01', 'HLA-DPA1*03:03-DPB1*54:01', 'HLA-DPA1*03:03-DPB1*55:01', 'HLA-DPA1*03:03-DPB1*56:01',
         'HLA-DPA1*03:03-DPB1*58:01', 'HLA-DPA1*03:03-DPB1*59:01',
         'HLA-DPA1*03:03-DPB1*60:01', 'HLA-DPA1*03:03-DPB1*62:01', 'HLA-DPA1*03:03-DPB1*63:01', 'HLA-DPA1*03:03-DPB1*65:01',
         'HLA-DPA1*03:03-DPB1*66:01', 'HLA-DPA1*03:03-DPB1*67:01',
         'HLA-DPA1*03:03-DPB1*68:01', 'HLA-DPA1*03:03-DPB1*69:01', 'HLA-DPA1*03:03-DPB1*70:01', 'HLA-DPA1*03:03-DPB1*71:01',
         'HLA-DPA1*03:03-DPB1*72:01', 'HLA-DPA1*03:03-DPB1*73:01',
         'HLA-DPA1*03:03-DPB1*74:01', 'HLA-DPA1*03:03-DPB1*75:01', 'HLA-DPA1*03:03-DPB1*76:01', 'HLA-DPA1*03:03-DPB1*77:01',
         'HLA-DPA1*03:03-DPB1*78:01', 'HLA-DPA1*03:03-DPB1*79:01',
         'HLA-DPA1*03:03-DPB1*80:01', 'HLA-DPA1*03:03-DPB1*81:01', 'HLA-DPA1*03:03-DPB1*82:01', 'HLA-DPA1*03:03-DPB1*83:01',
         'HLA-DPA1*03:03-DPB1*84:01', 'HLA-DPA1*03:03-DPB1*85:01',
         'HLA-DPA1*03:03-DPB1*86:01', 'HLA-DPA1*03:03-DPB1*87:01', 'HLA-DPA1*03:03-DPB1*88:01', 'HLA-DPA1*03:03-DPB1*89:01',
         'HLA-DPA1*03:03-DPB1*90:01', 'HLA-DPA1*03:03-DPB1*91:01',
         'HLA-DPA1*03:03-DPB1*92:01', 'HLA-DPA1*03:03-DPB1*93:01', 'HLA-DPA1*03:03-DPB1*94:01', 'HLA-DPA1*03:03-DPB1*95:01',
         'HLA-DPA1*03:03-DPB1*96:01', 'HLA-DPA1*03:03-DPB1*97:01',
         'HLA-DPA1*03:03-DPB1*98:01', 'HLA-DPA1*03:03-DPB1*99:01', 'HLA-DPA1*04:01-DPB1*01:01', 'HLA-DPA1*04:01-DPB1*02:01',
         'HLA-DPA1*04:01-DPB1*02:02', 'HLA-DPA1*04:01-DPB1*03:01',
         'HLA-DPA1*04:01-DPB1*04:01', 'HLA-DPA1*04:01-DPB1*04:02', 'HLA-DPA1*04:01-DPB1*05:01', 'HLA-DPA1*04:01-DPB1*06:01',
         'HLA-DPA1*04:01-DPB1*08:01', 'HLA-DPA1*04:01-DPB1*09:01',
         'HLA-DPA1*04:01-DPB1*10:001', 'HLA-DPA1*04:01-DPB1*10:01', 'HLA-DPA1*04:01-DPB1*10:101', 'HLA-DPA1*04:01-DPB1*10:201',
         'HLA-DPA1*04:01-DPB1*10:301', 'HLA-DPA1*04:01-DPB1*10:401',
         'HLA-DPA1*04:01-DPB1*10:501', 'HLA-DPA1*04:01-DPB1*10:601', 'HLA-DPA1*04:01-DPB1*10:701', 'HLA-DPA1*04:01-DPB1*10:801',
         'HLA-DPA1*04:01-DPB1*10:901', 'HLA-DPA1*04:01-DPB1*11:001',
         'HLA-DPA1*04:01-DPB1*11:01', 'HLA-DPA1*04:01-DPB1*11:101', 'HLA-DPA1*04:01-DPB1*11:201', 'HLA-DPA1*04:01-DPB1*11:301',
         'HLA-DPA1*04:01-DPB1*11:401', 'HLA-DPA1*04:01-DPB1*11:501',
         'HLA-DPA1*04:01-DPB1*11:601', 'HLA-DPA1*04:01-DPB1*11:701', 'HLA-DPA1*04:01-DPB1*11:801', 'HLA-DPA1*04:01-DPB1*11:901',
         'HLA-DPA1*04:01-DPB1*12:101', 'HLA-DPA1*04:01-DPB1*12:201',
         'HLA-DPA1*04:01-DPB1*12:301', 'HLA-DPA1*04:01-DPB1*12:401', 'HLA-DPA1*04:01-DPB1*12:501', 'HLA-DPA1*04:01-DPB1*12:601',
         'HLA-DPA1*04:01-DPB1*12:701', 'HLA-DPA1*04:01-DPB1*12:801',
         'HLA-DPA1*04:01-DPB1*12:901', 'HLA-DPA1*04:01-DPB1*13:001', 'HLA-DPA1*04:01-DPB1*13:01', 'HLA-DPA1*04:01-DPB1*13:101',
         'HLA-DPA1*04:01-DPB1*13:201', 'HLA-DPA1*04:01-DPB1*13:301',
         'HLA-DPA1*04:01-DPB1*13:401', 'HLA-DPA1*04:01-DPB1*14:01', 'HLA-DPA1*04:01-DPB1*15:01', 'HLA-DPA1*04:01-DPB1*16:01',
         'HLA-DPA1*04:01-DPB1*17:01', 'HLA-DPA1*04:01-DPB1*18:01',
         'HLA-DPA1*04:01-DPB1*19:01', 'HLA-DPA1*04:01-DPB1*20:01', 'HLA-DPA1*04:01-DPB1*21:01', 'HLA-DPA1*04:01-DPB1*22:01',
         'HLA-DPA1*04:01-DPB1*23:01', 'HLA-DPA1*04:01-DPB1*24:01',
         'HLA-DPA1*04:01-DPB1*25:01', 'HLA-DPA1*04:01-DPB1*26:01', 'HLA-DPA1*04:01-DPB1*27:01', 'HLA-DPA1*04:01-DPB1*28:01',
         'HLA-DPA1*04:01-DPB1*29:01', 'HLA-DPA1*04:01-DPB1*30:01',
         'HLA-DPA1*04:01-DPB1*31:01', 'HLA-DPA1*04:01-DPB1*32:01', 'HLA-DPA1*04:01-DPB1*33:01', 'HLA-DPA1*04:01-DPB1*34:01',
         'HLA-DPA1*04:01-DPB1*35:01', 'HLA-DPA1*04:01-DPB1*36:01',
         'HLA-DPA1*04:01-DPB1*37:01', 'HLA-DPA1*04:01-DPB1*38:01', 'HLA-DPA1*04:01-DPB1*39:01', 'HLA-DPA1*04:01-DPB1*40:01',
         'HLA-DPA1*04:01-DPB1*41:01', 'HLA-DPA1*04:01-DPB1*44:01',
         'HLA-DPA1*04:01-DPB1*45:01', 'HLA-DPA1*04:01-DPB1*46:01', 'HLA-DPA1*04:01-DPB1*47:01', 'HLA-DPA1*04:01-DPB1*48:01',
         'HLA-DPA1*04:01-DPB1*49:01', 'HLA-DPA1*04:01-DPB1*50:01',
         'HLA-DPA1*04:01-DPB1*51:01', 'HLA-DPA1*04:01-DPB1*52:01', 'HLA-DPA1*04:01-DPB1*53:01', 'HLA-DPA1*04:01-DPB1*54:01',
         'HLA-DPA1*04:01-DPB1*55:01', 'HLA-DPA1*04:01-DPB1*56:01',
         'HLA-DPA1*04:01-DPB1*58:01', 'HLA-DPA1*04:01-DPB1*59:01', 'HLA-DPA1*04:01-DPB1*60:01', 'HLA-DPA1*04:01-DPB1*62:01',
         'HLA-DPA1*04:01-DPB1*63:01', 'HLA-DPA1*04:01-DPB1*65:01',
         'HLA-DPA1*04:01-DPB1*66:01', 'HLA-DPA1*04:01-DPB1*67:01', 'HLA-DPA1*04:01-DPB1*68:01', 'HLA-DPA1*04:01-DPB1*69:01',
         'HLA-DPA1*04:01-DPB1*70:01', 'HLA-DPA1*04:01-DPB1*71:01',
         'HLA-DPA1*04:01-DPB1*72:01', 'HLA-DPA1*04:01-DPB1*73:01', 'HLA-DPA1*04:01-DPB1*74:01', 'HLA-DPA1*04:01-DPB1*75:01',
         'HLA-DPA1*04:01-DPB1*76:01', 'HLA-DPA1*04:01-DPB1*77:01',
         'HLA-DPA1*04:01-DPB1*78:01', 'HLA-DPA1*04:01-DPB1*79:01', 'HLA-DPA1*04:01-DPB1*80:01', 'HLA-DPA1*04:01-DPB1*81:01',
         'HLA-DPA1*04:01-DPB1*82:01', 'HLA-DPA1*04:01-DPB1*83:01',
         'HLA-DPA1*04:01-DPB1*84:01', 'HLA-DPA1*04:01-DPB1*85:01', 'HLA-DPA1*04:01-DPB1*86:01', 'HLA-DPA1*04:01-DPB1*87:01',
         'HLA-DPA1*04:01-DPB1*88:01', 'HLA-DPA1*04:01-DPB1*89:01',
         'HLA-DPA1*04:01-DPB1*90:01', 'HLA-DPA1*04:01-DPB1*91:01', 'HLA-DPA1*04:01-DPB1*92:01', 'HLA-DPA1*04:01-DPB1*93:01',
         'HLA-DPA1*04:01-DPB1*94:01', 'HLA-DPA1*04:01-DPB1*95:01',
         'HLA-DPA1*04:01-DPB1*96:01', 'HLA-DPA1*04:01-DPB1*97:01', 'HLA-DPA1*04:01-DPB1*98:01', 'HLA-DPA1*04:01-DPB1*99:01',
         'HLA-DQA1*01:01-DQB1*02:01', 'HLA-DQA1*01:01-DQB1*02:02',
         'HLA-DQA1*01:01-DQB1*02:03', 'HLA-DQA1*01:01-DQB1*02:04', 'HLA-DQA1*01:01-DQB1*02:05', 'HLA-DQA1*01:01-DQB1*02:06',
         'HLA-DQA1*01:01-DQB1*03:01', 'HLA-DQA1*01:01-DQB1*03:02',
         'HLA-DQA1*01:01-DQB1*03:03', 'HLA-DQA1*01:01-DQB1*03:04', 'HLA-DQA1*01:01-DQB1*03:05', 'HLA-DQA1*01:01-DQB1*03:06',
         'HLA-DQA1*01:01-DQB1*03:07', 'HLA-DQA1*01:01-DQB1*03:08',
         'HLA-DQA1*01:01-DQB1*03:09', 'HLA-DQA1*01:01-DQB1*03:10', 'HLA-DQA1*01:01-DQB1*03:11', 'HLA-DQA1*01:01-DQB1*03:12',
         'HLA-DQA1*01:01-DQB1*03:13', 'HLA-DQA1*01:01-DQB1*03:14',
         'HLA-DQA1*01:01-DQB1*03:15', 'HLA-DQA1*01:01-DQB1*03:16', 'HLA-DQA1*01:01-DQB1*03:17', 'HLA-DQA1*01:01-DQB1*03:18',
         'HLA-DQA1*01:01-DQB1*03:19', 'HLA-DQA1*01:01-DQB1*03:20',
         'HLA-DQA1*01:01-DQB1*03:21', 'HLA-DQA1*01:01-DQB1*03:22', 'HLA-DQA1*01:01-DQB1*03:23', 'HLA-DQA1*01:01-DQB1*03:24',
         'HLA-DQA1*01:01-DQB1*03:25', 'HLA-DQA1*01:01-DQB1*03:26',
         'HLA-DQA1*01:01-DQB1*03:27', 'HLA-DQA1*01:01-DQB1*03:28', 'HLA-DQA1*01:01-DQB1*03:29', 'HLA-DQA1*01:01-DQB1*03:30',
         'HLA-DQA1*01:01-DQB1*03:31', 'HLA-DQA1*01:01-DQB1*03:32',
         'HLA-DQA1*01:01-DQB1*03:33', 'HLA-DQA1*01:01-DQB1*03:34', 'HLA-DQA1*01:01-DQB1*03:35', 'HLA-DQA1*01:01-DQB1*03:36',
         'HLA-DQA1*01:01-DQB1*03:37', 'HLA-DQA1*01:01-DQB1*03:38',
         'HLA-DQA1*01:01-DQB1*04:01', 'HLA-DQA1*01:01-DQB1*04:02', 'HLA-DQA1*01:01-DQB1*04:03', 'HLA-DQA1*01:01-DQB1*04:04',
         'HLA-DQA1*01:01-DQB1*04:05', 'HLA-DQA1*01:01-DQB1*04:06',
         'HLA-DQA1*01:01-DQB1*04:07', 'HLA-DQA1*01:01-DQB1*04:08', 'HLA-DQA1*01:01-DQB1*05:01', 'HLA-DQA1*01:01-DQB1*05:02',
         'HLA-DQA1*01:01-DQB1*05:03', 'HLA-DQA1*01:01-DQB1*05:05',
         'HLA-DQA1*01:01-DQB1*05:06', 'HLA-DQA1*01:01-DQB1*05:07', 'HLA-DQA1*01:01-DQB1*05:08', 'HLA-DQA1*01:01-DQB1*05:09',
         'HLA-DQA1*01:01-DQB1*05:10', 'HLA-DQA1*01:01-DQB1*05:11',
         'HLA-DQA1*01:01-DQB1*05:12', 'HLA-DQA1*01:01-DQB1*05:13', 'HLA-DQA1*01:01-DQB1*05:14', 'HLA-DQA1*01:01-DQB1*06:01',
         'HLA-DQA1*01:01-DQB1*06:02', 'HLA-DQA1*01:01-DQB1*06:03',
         'HLA-DQA1*01:01-DQB1*06:04', 'HLA-DQA1*01:01-DQB1*06:07', 'HLA-DQA1*01:01-DQB1*06:08', 'HLA-DQA1*01:01-DQB1*06:09',
         'HLA-DQA1*01:01-DQB1*06:10', 'HLA-DQA1*01:01-DQB1*06:11',
         'HLA-DQA1*01:01-DQB1*06:12', 'HLA-DQA1*01:01-DQB1*06:14', 'HLA-DQA1*01:01-DQB1*06:15', 'HLA-DQA1*01:01-DQB1*06:16',
         'HLA-DQA1*01:01-DQB1*06:17', 'HLA-DQA1*01:01-DQB1*06:18',
         'HLA-DQA1*01:01-DQB1*06:19', 'HLA-DQA1*01:01-DQB1*06:21', 'HLA-DQA1*01:01-DQB1*06:22', 'HLA-DQA1*01:01-DQB1*06:23',
         'HLA-DQA1*01:01-DQB1*06:24', 'HLA-DQA1*01:01-DQB1*06:25',
         'HLA-DQA1*01:01-DQB1*06:27', 'HLA-DQA1*01:01-DQB1*06:28', 'HLA-DQA1*01:01-DQB1*06:29', 'HLA-DQA1*01:01-DQB1*06:30',
         'HLA-DQA1*01:01-DQB1*06:31', 'HLA-DQA1*01:01-DQB1*06:32',
         'HLA-DQA1*01:01-DQB1*06:33', 'HLA-DQA1*01:01-DQB1*06:34', 'HLA-DQA1*01:01-DQB1*06:35', 'HLA-DQA1*01:01-DQB1*06:36',
         'HLA-DQA1*01:01-DQB1*06:37', 'HLA-DQA1*01:01-DQB1*06:38',
         'HLA-DQA1*01:01-DQB1*06:39', 'HLA-DQA1*01:01-DQB1*06:40', 'HLA-DQA1*01:01-DQB1*06:41', 'HLA-DQA1*01:01-DQB1*06:42',
         'HLA-DQA1*01:01-DQB1*06:43', 'HLA-DQA1*01:01-DQB1*06:44',
         'HLA-DQA1*01:02-DQB1*02:01', 'HLA-DQA1*01:02-DQB1*02:02', 'HLA-DQA1*01:02-DQB1*02:03', 'HLA-DQA1*01:02-DQB1*02:04',
         'HLA-DQA1*01:02-DQB1*02:05', 'HLA-DQA1*01:02-DQB1*02:06',
         'HLA-DQA1*01:02-DQB1*03:01', 'HLA-DQA1*01:02-DQB1*03:02', 'HLA-DQA1*01:02-DQB1*03:03', 'HLA-DQA1*01:02-DQB1*03:04',
         'HLA-DQA1*01:02-DQB1*03:05', 'HLA-DQA1*01:02-DQB1*03:06',
         'HLA-DQA1*01:02-DQB1*03:07', 'HLA-DQA1*01:02-DQB1*03:08', 'HLA-DQA1*01:02-DQB1*03:09', 'HLA-DQA1*01:02-DQB1*03:10',
         'HLA-DQA1*01:02-DQB1*03:11', 'HLA-DQA1*01:02-DQB1*03:12',
         'HLA-DQA1*01:02-DQB1*03:13', 'HLA-DQA1*01:02-DQB1*03:14', 'HLA-DQA1*01:02-DQB1*03:15', 'HLA-DQA1*01:02-DQB1*03:16',
         'HLA-DQA1*01:02-DQB1*03:17', 'HLA-DQA1*01:02-DQB1*03:18',
         'HLA-DQA1*01:02-DQB1*03:19', 'HLA-DQA1*01:02-DQB1*03:20', 'HLA-DQA1*01:02-DQB1*03:21', 'HLA-DQA1*01:02-DQB1*03:22',
         'HLA-DQA1*01:02-DQB1*03:23', 'HLA-DQA1*01:02-DQB1*03:24',
         'HLA-DQA1*01:02-DQB1*03:25', 'HLA-DQA1*01:02-DQB1*03:26', 'HLA-DQA1*01:02-DQB1*03:27', 'HLA-DQA1*01:02-DQB1*03:28',
         'HLA-DQA1*01:02-DQB1*03:29', 'HLA-DQA1*01:02-DQB1*03:30',
         'HLA-DQA1*01:02-DQB1*03:31', 'HLA-DQA1*01:02-DQB1*03:32', 'HLA-DQA1*01:02-DQB1*03:33', 'HLA-DQA1*01:02-DQB1*03:34',
         'HLA-DQA1*01:02-DQB1*03:35', 'HLA-DQA1*01:02-DQB1*03:36',
         'HLA-DQA1*01:02-DQB1*03:37', 'HLA-DQA1*01:02-DQB1*03:38', 'HLA-DQA1*01:02-DQB1*04:01', 'HLA-DQA1*01:02-DQB1*04:02',
         'HLA-DQA1*01:02-DQB1*04:03', 'HLA-DQA1*01:02-DQB1*04:04',
         'HLA-DQA1*01:02-DQB1*04:05', 'HLA-DQA1*01:02-DQB1*04:06', 'HLA-DQA1*01:02-DQB1*04:07', 'HLA-DQA1*01:02-DQB1*04:08',
         'HLA-DQA1*01:02-DQB1*05:01', 'HLA-DQA1*01:02-DQB1*05:02',
         'HLA-DQA1*01:02-DQB1*05:03', 'HLA-DQA1*01:02-DQB1*05:05', 'HLA-DQA1*01:02-DQB1*05:06', 'HLA-DQA1*01:02-DQB1*05:07',
         'HLA-DQA1*01:02-DQB1*05:08', 'HLA-DQA1*01:02-DQB1*05:09',
         'HLA-DQA1*01:02-DQB1*05:10', 'HLA-DQA1*01:02-DQB1*05:11', 'HLA-DQA1*01:02-DQB1*05:12', 'HLA-DQA1*01:02-DQB1*05:13',
         'HLA-DQA1*01:02-DQB1*05:14', 'HLA-DQA1*01:02-DQB1*06:01',
         'HLA-DQA1*01:02-DQB1*06:02', 'HLA-DQA1*01:02-DQB1*06:03', 'HLA-DQA1*01:02-DQB1*06:04', 'HLA-DQA1*01:02-DQB1*06:07',
         'HLA-DQA1*01:02-DQB1*06:08', 'HLA-DQA1*01:02-DQB1*06:09',
         'HLA-DQA1*01:02-DQB1*06:10', 'HLA-DQA1*01:02-DQB1*06:11', 'HLA-DQA1*01:02-DQB1*06:12', 'HLA-DQA1*01:02-DQB1*06:14',
         'HLA-DQA1*01:02-DQB1*06:15', 'HLA-DQA1*01:02-DQB1*06:16',
         'HLA-DQA1*01:02-DQB1*06:17', 'HLA-DQA1*01:02-DQB1*06:18', 'HLA-DQA1*01:02-DQB1*06:19', 'HLA-DQA1*01:02-DQB1*06:21',
         'HLA-DQA1*01:02-DQB1*06:22', 'HLA-DQA1*01:02-DQB1*06:23',
         'HLA-DQA1*01:02-DQB1*06:24', 'HLA-DQA1*01:02-DQB1*06:25', 'HLA-DQA1*01:02-DQB1*06:27', 'HLA-DQA1*01:02-DQB1*06:28',
         'HLA-DQA1*01:02-DQB1*06:29', 'HLA-DQA1*01:02-DQB1*06:30',
         'HLA-DQA1*01:02-DQB1*06:31', 'HLA-DQA1*01:02-DQB1*06:32', 'HLA-DQA1*01:02-DQB1*06:33', 'HLA-DQA1*01:02-DQB1*06:34',
         'HLA-DQA1*01:02-DQB1*06:35', 'HLA-DQA1*01:02-DQB1*06:36',
         'HLA-DQA1*01:02-DQB1*06:37', 'HLA-DQA1*01:02-DQB1*06:38', 'HLA-DQA1*01:02-DQB1*06:39', 'HLA-DQA1*01:02-DQB1*06:40',
         'HLA-DQA1*01:02-DQB1*06:41', 'HLA-DQA1*01:02-DQB1*06:42',
         'HLA-DQA1*01:02-DQB1*06:43', 'HLA-DQA1*01:02-DQB1*06:44', 'HLA-DQA1*01:03-DQB1*02:01', 'HLA-DQA1*01:03-DQB1*02:02',
         'HLA-DQA1*01:03-DQB1*02:03', 'HLA-DQA1*01:03-DQB1*02:04',
         'HLA-DQA1*01:03-DQB1*02:05', 'HLA-DQA1*01:03-DQB1*02:06', 'HLA-DQA1*01:03-DQB1*03:01', 'HLA-DQA1*01:03-DQB1*03:02',
         'HLA-DQA1*01:03-DQB1*03:03', 'HLA-DQA1*01:03-DQB1*03:04',
         'HLA-DQA1*01:03-DQB1*03:05', 'HLA-DQA1*01:03-DQB1*03:06', 'HLA-DQA1*01:03-DQB1*03:07', 'HLA-DQA1*01:03-DQB1*03:08',
         'HLA-DQA1*01:03-DQB1*03:09', 'HLA-DQA1*01:03-DQB1*03:10',
         'HLA-DQA1*01:03-DQB1*03:11', 'HLA-DQA1*01:03-DQB1*03:12', 'HLA-DQA1*01:03-DQB1*03:13', 'HLA-DQA1*01:03-DQB1*03:14',
         'HLA-DQA1*01:03-DQB1*03:15', 'HLA-DQA1*01:03-DQB1*03:16',
         'HLA-DQA1*01:03-DQB1*03:17', 'HLA-DQA1*01:03-DQB1*03:18', 'HLA-DQA1*01:03-DQB1*03:19', 'HLA-DQA1*01:03-DQB1*03:20',
         'HLA-DQA1*01:03-DQB1*03:21', 'HLA-DQA1*01:03-DQB1*03:22',
         'HLA-DQA1*01:03-DQB1*03:23', 'HLA-DQA1*01:03-DQB1*03:24', 'HLA-DQA1*01:03-DQB1*03:25', 'HLA-DQA1*01:03-DQB1*03:26',
         'HLA-DQA1*01:03-DQB1*03:27', 'HLA-DQA1*01:03-DQB1*03:28',
         'HLA-DQA1*01:03-DQB1*03:29', 'HLA-DQA1*01:03-DQB1*03:30', 'HLA-DQA1*01:03-DQB1*03:31', 'HLA-DQA1*01:03-DQB1*03:32',
         'HLA-DQA1*01:03-DQB1*03:33', 'HLA-DQA1*01:03-DQB1*03:34',
         'HLA-DQA1*01:03-DQB1*03:35', 'HLA-DQA1*01:03-DQB1*03:36', 'HLA-DQA1*01:03-DQB1*03:37', 'HLA-DQA1*01:03-DQB1*03:38',
         'HLA-DQA1*01:03-DQB1*04:01', 'HLA-DQA1*01:03-DQB1*04:02',
         'HLA-DQA1*01:03-DQB1*04:03', 'HLA-DQA1*01:03-DQB1*04:04', 'HLA-DQA1*01:03-DQB1*04:05', 'HLA-DQA1*01:03-DQB1*04:06',
         'HLA-DQA1*01:03-DQB1*04:07', 'HLA-DQA1*01:03-DQB1*04:08',
         'HLA-DQA1*01:03-DQB1*05:01', 'HLA-DQA1*01:03-DQB1*05:02', 'HLA-DQA1*01:03-DQB1*05:03', 'HLA-DQA1*01:03-DQB1*05:05',
         'HLA-DQA1*01:03-DQB1*05:06', 'HLA-DQA1*01:03-DQB1*05:07',
         'HLA-DQA1*01:03-DQB1*05:08', 'HLA-DQA1*01:03-DQB1*05:09', 'HLA-DQA1*01:03-DQB1*05:10', 'HLA-DQA1*01:03-DQB1*05:11',
         'HLA-DQA1*01:03-DQB1*05:12', 'HLA-DQA1*01:03-DQB1*05:13',
         'HLA-DQA1*01:03-DQB1*05:14', 'HLA-DQA1*01:03-DQB1*06:01', 'HLA-DQA1*01:03-DQB1*06:02', 'HLA-DQA1*01:03-DQB1*06:03',
         'HLA-DQA1*01:03-DQB1*06:04', 'HLA-DQA1*01:03-DQB1*06:07',
         'HLA-DQA1*01:03-DQB1*06:08', 'HLA-DQA1*01:03-DQB1*06:09', 'HLA-DQA1*01:03-DQB1*06:10', 'HLA-DQA1*01:03-DQB1*06:11',
         'HLA-DQA1*01:03-DQB1*06:12', 'HLA-DQA1*01:03-DQB1*06:14',
         'HLA-DQA1*01:03-DQB1*06:15', 'HLA-DQA1*01:03-DQB1*06:16', 'HLA-DQA1*01:03-DQB1*06:17', 'HLA-DQA1*01:03-DQB1*06:18',
         'HLA-DQA1*01:03-DQB1*06:19', 'HLA-DQA1*01:03-DQB1*06:21',
         'HLA-DQA1*01:03-DQB1*06:22', 'HLA-DQA1*01:03-DQB1*06:23', 'HLA-DQA1*01:03-DQB1*06:24', 'HLA-DQA1*01:03-DQB1*06:25',
         'HLA-DQA1*01:03-DQB1*06:27', 'HLA-DQA1*01:03-DQB1*06:28',
         'HLA-DQA1*01:03-DQB1*06:29', 'HLA-DQA1*01:03-DQB1*06:30', 'HLA-DQA1*01:03-DQB1*06:31', 'HLA-DQA1*01:03-DQB1*06:32',
         'HLA-DQA1*01:03-DQB1*06:33', 'HLA-DQA1*01:03-DQB1*06:34',
         'HLA-DQA1*01:03-DQB1*06:35', 'HLA-DQA1*01:03-DQB1*06:36', 'HLA-DQA1*01:03-DQB1*06:37', 'HLA-DQA1*01:03-DQB1*06:38',
         'HLA-DQA1*01:03-DQB1*06:39', 'HLA-DQA1*01:03-DQB1*06:40',
         'HLA-DQA1*01:03-DQB1*06:41', 'HLA-DQA1*01:03-DQB1*06:42', 'HLA-DQA1*01:03-DQB1*06:43', 'HLA-DQA1*01:03-DQB1*06:44',
         'HLA-DQA1*01:04-DQB1*02:01', 'HLA-DQA1*01:04-DQB1*02:02',
         'HLA-DQA1*01:04-DQB1*02:03', 'HLA-DQA1*01:04-DQB1*02:04', 'HLA-DQA1*01:04-DQB1*02:05', 'HLA-DQA1*01:04-DQB1*02:06',
         'HLA-DQA1*01:04-DQB1*03:01', 'HLA-DQA1*01:04-DQB1*03:02',
         'HLA-DQA1*01:04-DQB1*03:03', 'HLA-DQA1*01:04-DQB1*03:04', 'HLA-DQA1*01:04-DQB1*03:05', 'HLA-DQA1*01:04-DQB1*03:06',
         'HLA-DQA1*01:04-DQB1*03:07', 'HLA-DQA1*01:04-DQB1*03:08',
         'HLA-DQA1*01:04-DQB1*03:09', 'HLA-DQA1*01:04-DQB1*03:10', 'HLA-DQA1*01:04-DQB1*03:11', 'HLA-DQA1*01:04-DQB1*03:12',
         'HLA-DQA1*01:04-DQB1*03:13', 'HLA-DQA1*01:04-DQB1*03:14',
         'HLA-DQA1*01:04-DQB1*03:15', 'HLA-DQA1*01:04-DQB1*03:16', 'HLA-DQA1*01:04-DQB1*03:17', 'HLA-DQA1*01:04-DQB1*03:18',
         'HLA-DQA1*01:04-DQB1*03:19', 'HLA-DQA1*01:04-DQB1*03:20',
         'HLA-DQA1*01:04-DQB1*03:21', 'HLA-DQA1*01:04-DQB1*03:22', 'HLA-DQA1*01:04-DQB1*03:23', 'HLA-DQA1*01:04-DQB1*03:24',
         'HLA-DQA1*01:04-DQB1*03:25', 'HLA-DQA1*01:04-DQB1*03:26',
         'HLA-DQA1*01:04-DQB1*03:27', 'HLA-DQA1*01:04-DQB1*03:28', 'HLA-DQA1*01:04-DQB1*03:29', 'HLA-DQA1*01:04-DQB1*03:30',
         'HLA-DQA1*01:04-DQB1*03:31', 'HLA-DQA1*01:04-DQB1*03:32',
         'HLA-DQA1*01:04-DQB1*03:33', 'HLA-DQA1*01:04-DQB1*03:34', 'HLA-DQA1*01:04-DQB1*03:35', 'HLA-DQA1*01:04-DQB1*03:36',
         'HLA-DQA1*01:04-DQB1*03:37', 'HLA-DQA1*01:04-DQB1*03:38',
         'HLA-DQA1*01:04-DQB1*04:01', 'HLA-DQA1*01:04-DQB1*04:02', 'HLA-DQA1*01:04-DQB1*04:03', 'HLA-DQA1*01:04-DQB1*04:04',
         'HLA-DQA1*01:04-DQB1*04:05', 'HLA-DQA1*01:04-DQB1*04:06',
         'HLA-DQA1*01:04-DQB1*04:07', 'HLA-DQA1*01:04-DQB1*04:08', 'HLA-DQA1*01:04-DQB1*05:01', 'HLA-DQA1*01:04-DQB1*05:02',
         'HLA-DQA1*01:04-DQB1*05:03', 'HLA-DQA1*01:04-DQB1*05:05',
         'HLA-DQA1*01:04-DQB1*05:06', 'HLA-DQA1*01:04-DQB1*05:07', 'HLA-DQA1*01:04-DQB1*05:08', 'HLA-DQA1*01:04-DQB1*05:09',
         'HLA-DQA1*01:04-DQB1*05:10', 'HLA-DQA1*01:04-DQB1*05:11',
         'HLA-DQA1*01:04-DQB1*05:12', 'HLA-DQA1*01:04-DQB1*05:13', 'HLA-DQA1*01:04-DQB1*05:14', 'HLA-DQA1*01:04-DQB1*06:01',
         'HLA-DQA1*01:04-DQB1*06:02', 'HLA-DQA1*01:04-DQB1*06:03',
         'HLA-DQA1*01:04-DQB1*06:04', 'HLA-DQA1*01:04-DQB1*06:07', 'HLA-DQA1*01:04-DQB1*06:08', 'HLA-DQA1*01:04-DQB1*06:09',
         'HLA-DQA1*01:04-DQB1*06:10', 'HLA-DQA1*01:04-DQB1*06:11',
         'HLA-DQA1*01:04-DQB1*06:12', 'HLA-DQA1*01:04-DQB1*06:14', 'HLA-DQA1*01:04-DQB1*06:15', 'HLA-DQA1*01:04-DQB1*06:16',
         'HLA-DQA1*01:04-DQB1*06:17', 'HLA-DQA1*01:04-DQB1*06:18',
         'HLA-DQA1*01:04-DQB1*06:19', 'HLA-DQA1*01:04-DQB1*06:21', 'HLA-DQA1*01:04-DQB1*06:22', 'HLA-DQA1*01:04-DQB1*06:23',
         'HLA-DQA1*01:04-DQB1*06:24', 'HLA-DQA1*01:04-DQB1*06:25',
         'HLA-DQA1*01:04-DQB1*06:27', 'HLA-DQA1*01:04-DQB1*06:28', 'HLA-DQA1*01:04-DQB1*06:29', 'HLA-DQA1*01:04-DQB1*06:30',
         'HLA-DQA1*01:04-DQB1*06:31', 'HLA-DQA1*01:04-DQB1*06:32',
         'HLA-DQA1*01:04-DQB1*06:33', 'HLA-DQA1*01:04-DQB1*06:34', 'HLA-DQA1*01:04-DQB1*06:35', 'HLA-DQA1*01:04-DQB1*06:36',
         'HLA-DQA1*01:04-DQB1*06:37', 'HLA-DQA1*01:04-DQB1*06:38',
         'HLA-DQA1*01:04-DQB1*06:39', 'HLA-DQA1*01:04-DQB1*06:40', 'HLA-DQA1*01:04-DQB1*06:41', 'HLA-DQA1*01:04-DQB1*06:42',
         'HLA-DQA1*01:04-DQB1*06:43', 'HLA-DQA1*01:04-DQB1*06:44',
         'HLA-DQA1*01:05-DQB1*02:01', 'HLA-DQA1*01:05-DQB1*02:02', 'HLA-DQA1*01:05-DQB1*02:03', 'HLA-DQA1*01:05-DQB1*02:04',
         'HLA-DQA1*01:05-DQB1*02:05', 'HLA-DQA1*01:05-DQB1*02:06',
         'HLA-DQA1*01:05-DQB1*03:01', 'HLA-DQA1*01:05-DQB1*03:02', 'HLA-DQA1*01:05-DQB1*03:03', 'HLA-DQA1*01:05-DQB1*03:04',
         'HLA-DQA1*01:05-DQB1*03:05', 'HLA-DQA1*01:05-DQB1*03:06',
         'HLA-DQA1*01:05-DQB1*03:07', 'HLA-DQA1*01:05-DQB1*03:08', 'HLA-DQA1*01:05-DQB1*03:09', 'HLA-DQA1*01:05-DQB1*03:10',
         'HLA-DQA1*01:05-DQB1*03:11', 'HLA-DQA1*01:05-DQB1*03:12',
         'HLA-DQA1*01:05-DQB1*03:13', 'HLA-DQA1*01:05-DQB1*03:14', 'HLA-DQA1*01:05-DQB1*03:15', 'HLA-DQA1*01:05-DQB1*03:16',
         'HLA-DQA1*01:05-DQB1*03:17', 'HLA-DQA1*01:05-DQB1*03:18',
         'HLA-DQA1*01:05-DQB1*03:19', 'HLA-DQA1*01:05-DQB1*03:20', 'HLA-DQA1*01:05-DQB1*03:21', 'HLA-DQA1*01:05-DQB1*03:22',
         'HLA-DQA1*01:05-DQB1*03:23', 'HLA-DQA1*01:05-DQB1*03:24',
         'HLA-DQA1*01:05-DQB1*03:25', 'HLA-DQA1*01:05-DQB1*03:26', 'HLA-DQA1*01:05-DQB1*03:27', 'HLA-DQA1*01:05-DQB1*03:28',
         'HLA-DQA1*01:05-DQB1*03:29', 'HLA-DQA1*01:05-DQB1*03:30',
         'HLA-DQA1*01:05-DQB1*03:31', 'HLA-DQA1*01:05-DQB1*03:32', 'HLA-DQA1*01:05-DQB1*03:33', 'HLA-DQA1*01:05-DQB1*03:34',
         'HLA-DQA1*01:05-DQB1*03:35', 'HLA-DQA1*01:05-DQB1*03:36',
         'HLA-DQA1*01:05-DQB1*03:37', 'HLA-DQA1*01:05-DQB1*03:38', 'HLA-DQA1*01:05-DQB1*04:01', 'HLA-DQA1*01:05-DQB1*04:02',
         'HLA-DQA1*01:05-DQB1*04:03', 'HLA-DQA1*01:05-DQB1*04:04',
         'HLA-DQA1*01:05-DQB1*04:05', 'HLA-DQA1*01:05-DQB1*04:06', 'HLA-DQA1*01:05-DQB1*04:07', 'HLA-DQA1*01:05-DQB1*04:08',
         'HLA-DQA1*01:05-DQB1*05:01', 'HLA-DQA1*01:05-DQB1*05:02',
         'HLA-DQA1*01:05-DQB1*05:03', 'HLA-DQA1*01:05-DQB1*05:05', 'HLA-DQA1*01:05-DQB1*05:06', 'HLA-DQA1*01:05-DQB1*05:07',
         'HLA-DQA1*01:05-DQB1*05:08', 'HLA-DQA1*01:05-DQB1*05:09',
         'HLA-DQA1*01:05-DQB1*05:10', 'HLA-DQA1*01:05-DQB1*05:11', 'HLA-DQA1*01:05-DQB1*05:12', 'HLA-DQA1*01:05-DQB1*05:13',
         'HLA-DQA1*01:05-DQB1*05:14', 'HLA-DQA1*01:05-DQB1*06:01',
         'HLA-DQA1*01:05-DQB1*06:02', 'HLA-DQA1*01:05-DQB1*06:03', 'HLA-DQA1*01:05-DQB1*06:04', 'HLA-DQA1*01:05-DQB1*06:07',
         'HLA-DQA1*01:05-DQB1*06:08', 'HLA-DQA1*01:05-DQB1*06:09',
         'HLA-DQA1*01:05-DQB1*06:10', 'HLA-DQA1*01:05-DQB1*06:11', 'HLA-DQA1*01:05-DQB1*06:12', 'HLA-DQA1*01:05-DQB1*06:14',
         'HLA-DQA1*01:05-DQB1*06:15', 'HLA-DQA1*01:05-DQB1*06:16',
         'HLA-DQA1*01:05-DQB1*06:17', 'HLA-DQA1*01:05-DQB1*06:18', 'HLA-DQA1*01:05-DQB1*06:19', 'HLA-DQA1*01:05-DQB1*06:21',
         'HLA-DQA1*01:05-DQB1*06:22', 'HLA-DQA1*01:05-DQB1*06:23',
         'HLA-DQA1*01:05-DQB1*06:24', 'HLA-DQA1*01:05-DQB1*06:25', 'HLA-DQA1*01:05-DQB1*06:27', 'HLA-DQA1*01:05-DQB1*06:28',
         'HLA-DQA1*01:05-DQB1*06:29', 'HLA-DQA1*01:05-DQB1*06:30',
         'HLA-DQA1*01:05-DQB1*06:31', 'HLA-DQA1*01:05-DQB1*06:32', 'HLA-DQA1*01:05-DQB1*06:33', 'HLA-DQA1*01:05-DQB1*06:34',
         'HLA-DQA1*01:05-DQB1*06:35', 'HLA-DQA1*01:05-DQB1*06:36',
         'HLA-DQA1*01:05-DQB1*06:37', 'HLA-DQA1*01:05-DQB1*06:38', 'HLA-DQA1*01:05-DQB1*06:39', 'HLA-DQA1*01:05-DQB1*06:40',
         'HLA-DQA1*01:05-DQB1*06:41', 'HLA-DQA1*01:05-DQB1*06:42',
         'HLA-DQA1*01:05-DQB1*06:43', 'HLA-DQA1*01:05-DQB1*06:44', 'HLA-DQA1*01:06-DQB1*02:01', 'HLA-DQA1*01:06-DQB1*02:02',
         'HLA-DQA1*01:06-DQB1*02:03', 'HLA-DQA1*01:06-DQB1*02:04',
         'HLA-DQA1*01:06-DQB1*02:05', 'HLA-DQA1*01:06-DQB1*02:06', 'HLA-DQA1*01:06-DQB1*03:01', 'HLA-DQA1*01:06-DQB1*03:02',
         'HLA-DQA1*01:06-DQB1*03:03', 'HLA-DQA1*01:06-DQB1*03:04',
         'HLA-DQA1*01:06-DQB1*03:05', 'HLA-DQA1*01:06-DQB1*03:06', 'HLA-DQA1*01:06-DQB1*03:07', 'HLA-DQA1*01:06-DQB1*03:08',
         'HLA-DQA1*01:06-DQB1*03:09', 'HLA-DQA1*01:06-DQB1*03:10',
         'HLA-DQA1*01:06-DQB1*03:11', 'HLA-DQA1*01:06-DQB1*03:12', 'HLA-DQA1*01:06-DQB1*03:13', 'HLA-DQA1*01:06-DQB1*03:14',
         'HLA-DQA1*01:06-DQB1*03:15', 'HLA-DQA1*01:06-DQB1*03:16',
         'HLA-DQA1*01:06-DQB1*03:17', 'HLA-DQA1*01:06-DQB1*03:18', 'HLA-DQA1*01:06-DQB1*03:19', 'HLA-DQA1*01:06-DQB1*03:20',
         'HLA-DQA1*01:06-DQB1*03:21', 'HLA-DQA1*01:06-DQB1*03:22',
         'HLA-DQA1*01:06-DQB1*03:23', 'HLA-DQA1*01:06-DQB1*03:24', 'HLA-DQA1*01:06-DQB1*03:25', 'HLA-DQA1*01:06-DQB1*03:26',
         'HLA-DQA1*01:06-DQB1*03:27', 'HLA-DQA1*01:06-DQB1*03:28',
         'HLA-DQA1*01:06-DQB1*03:29', 'HLA-DQA1*01:06-DQB1*03:30', 'HLA-DQA1*01:06-DQB1*03:31', 'HLA-DQA1*01:06-DQB1*03:32',
         'HLA-DQA1*01:06-DQB1*03:33', 'HLA-DQA1*01:06-DQB1*03:34',
         'HLA-DQA1*01:06-DQB1*03:35', 'HLA-DQA1*01:06-DQB1*03:36', 'HLA-DQA1*01:06-DQB1*03:37', 'HLA-DQA1*01:06-DQB1*03:38',
         'HLA-DQA1*01:06-DQB1*04:01', 'HLA-DQA1*01:06-DQB1*04:02',
         'HLA-DQA1*01:06-DQB1*04:03', 'HLA-DQA1*01:06-DQB1*04:04', 'HLA-DQA1*01:06-DQB1*04:05', 'HLA-DQA1*01:06-DQB1*04:06',
         'HLA-DQA1*01:06-DQB1*04:07', 'HLA-DQA1*01:06-DQB1*04:08',
         'HLA-DQA1*01:06-DQB1*05:01', 'HLA-DQA1*01:06-DQB1*05:02', 'HLA-DQA1*01:06-DQB1*05:03', 'HLA-DQA1*01:06-DQB1*05:05',
         'HLA-DQA1*01:06-DQB1*05:06', 'HLA-DQA1*01:06-DQB1*05:07',
         'HLA-DQA1*01:06-DQB1*05:08', 'HLA-DQA1*01:06-DQB1*05:09', 'HLA-DQA1*01:06-DQB1*05:10', 'HLA-DQA1*01:06-DQB1*05:11',
         'HLA-DQA1*01:06-DQB1*05:12', 'HLA-DQA1*01:06-DQB1*05:13',
         'HLA-DQA1*01:06-DQB1*05:14', 'HLA-DQA1*01:06-DQB1*06:01', 'HLA-DQA1*01:06-DQB1*06:02', 'HLA-DQA1*01:06-DQB1*06:03',
         'HLA-DQA1*01:06-DQB1*06:04', 'HLA-DQA1*01:06-DQB1*06:07',
         'HLA-DQA1*01:06-DQB1*06:08', 'HLA-DQA1*01:06-DQB1*06:09', 'HLA-DQA1*01:06-DQB1*06:10', 'HLA-DQA1*01:06-DQB1*06:11',
         'HLA-DQA1*01:06-DQB1*06:12', 'HLA-DQA1*01:06-DQB1*06:14',
         'HLA-DQA1*01:06-DQB1*06:15', 'HLA-DQA1*01:06-DQB1*06:16', 'HLA-DQA1*01:06-DQB1*06:17', 'HLA-DQA1*01:06-DQB1*06:18',
         'HLA-DQA1*01:06-DQB1*06:19', 'HLA-DQA1*01:06-DQB1*06:21',
         'HLA-DQA1*01:06-DQB1*06:22', 'HLA-DQA1*01:06-DQB1*06:23', 'HLA-DQA1*01:06-DQB1*06:24', 'HLA-DQA1*01:06-DQB1*06:25',
         'HLA-DQA1*01:06-DQB1*06:27', 'HLA-DQA1*01:06-DQB1*06:28',
         'HLA-DQA1*01:06-DQB1*06:29', 'HLA-DQA1*01:06-DQB1*06:30', 'HLA-DQA1*01:06-DQB1*06:31', 'HLA-DQA1*01:06-DQB1*06:32',
         'HLA-DQA1*01:06-DQB1*06:33', 'HLA-DQA1*01:06-DQB1*06:34',
         'HLA-DQA1*01:06-DQB1*06:35', 'HLA-DQA1*01:06-DQB1*06:36', 'HLA-DQA1*01:06-DQB1*06:37', 'HLA-DQA1*01:06-DQB1*06:38',
         'HLA-DQA1*01:06-DQB1*06:39', 'HLA-DQA1*01:06-DQB1*06:40',
         'HLA-DQA1*01:06-DQB1*06:41', 'HLA-DQA1*01:06-DQB1*06:42', 'HLA-DQA1*01:06-DQB1*06:43', 'HLA-DQA1*01:06-DQB1*06:44',
         'HLA-DQA1*01:07-DQB1*02:01', 'HLA-DQA1*01:07-DQB1*02:02',
         'HLA-DQA1*01:07-DQB1*02:03', 'HLA-DQA1*01:07-DQB1*02:04', 'HLA-DQA1*01:07-DQB1*02:05', 'HLA-DQA1*01:07-DQB1*02:06',
         'HLA-DQA1*01:07-DQB1*03:01', 'HLA-DQA1*01:07-DQB1*03:02',
         'HLA-DQA1*01:07-DQB1*03:03', 'HLA-DQA1*01:07-DQB1*03:04', 'HLA-DQA1*01:07-DQB1*03:05', 'HLA-DQA1*01:07-DQB1*03:06',
         'HLA-DQA1*01:07-DQB1*03:07', 'HLA-DQA1*01:07-DQB1*03:08',
         'HLA-DQA1*01:07-DQB1*03:09', 'HLA-DQA1*01:07-DQB1*03:10', 'HLA-DQA1*01:07-DQB1*03:11', 'HLA-DQA1*01:07-DQB1*03:12',
         'HLA-DQA1*01:07-DQB1*03:13', 'HLA-DQA1*01:07-DQB1*03:14',
         'HLA-DQA1*01:07-DQB1*03:15', 'HLA-DQA1*01:07-DQB1*03:16', 'HLA-DQA1*01:07-DQB1*03:17', 'HLA-DQA1*01:07-DQB1*03:18',
         'HLA-DQA1*01:07-DQB1*03:19', 'HLA-DQA1*01:07-DQB1*03:20',
         'HLA-DQA1*01:07-DQB1*03:21', 'HLA-DQA1*01:07-DQB1*03:22', 'HLA-DQA1*01:07-DQB1*03:23', 'HLA-DQA1*01:07-DQB1*03:24',
         'HLA-DQA1*01:07-DQB1*03:25', 'HLA-DQA1*01:07-DQB1*03:26',
         'HLA-DQA1*01:07-DQB1*03:27', 'HLA-DQA1*01:07-DQB1*03:28', 'HLA-DQA1*01:07-DQB1*03:29', 'HLA-DQA1*01:07-DQB1*03:30',
         'HLA-DQA1*01:07-DQB1*03:31', 'HLA-DQA1*01:07-DQB1*03:32',
         'HLA-DQA1*01:07-DQB1*03:33', 'HLA-DQA1*01:07-DQB1*03:34', 'HLA-DQA1*01:07-DQB1*03:35', 'HLA-DQA1*01:07-DQB1*03:36',
         'HLA-DQA1*01:07-DQB1*03:37', 'HLA-DQA1*01:07-DQB1*03:38',
         'HLA-DQA1*01:07-DQB1*04:01', 'HLA-DQA1*01:07-DQB1*04:02', 'HLA-DQA1*01:07-DQB1*04:03', 'HLA-DQA1*01:07-DQB1*04:04',
         'HLA-DQA1*01:07-DQB1*04:05', 'HLA-DQA1*01:07-DQB1*04:06',
         'HLA-DQA1*01:07-DQB1*04:07', 'HLA-DQA1*01:07-DQB1*04:08', 'HLA-DQA1*01:07-DQB1*05:01', 'HLA-DQA1*01:07-DQB1*05:02',
         'HLA-DQA1*01:07-DQB1*05:03', 'HLA-DQA1*01:07-DQB1*05:05',
         'HLA-DQA1*01:07-DQB1*05:06', 'HLA-DQA1*01:07-DQB1*05:07', 'HLA-DQA1*01:07-DQB1*05:08', 'HLA-DQA1*01:07-DQB1*05:09',
         'HLA-DQA1*01:07-DQB1*05:10', 'HLA-DQA1*01:07-DQB1*05:11',
         'HLA-DQA1*01:07-DQB1*05:12', 'HLA-DQA1*01:07-DQB1*05:13', 'HLA-DQA1*01:07-DQB1*05:14', 'HLA-DQA1*01:07-DQB1*06:01',
         'HLA-DQA1*01:07-DQB1*06:02', 'HLA-DQA1*01:07-DQB1*06:03',
         'HLA-DQA1*01:07-DQB1*06:04', 'HLA-DQA1*01:07-DQB1*06:07', 'HLA-DQA1*01:07-DQB1*06:08', 'HLA-DQA1*01:07-DQB1*06:09',
         'HLA-DQA1*01:07-DQB1*06:10', 'HLA-DQA1*01:07-DQB1*06:11',
         'HLA-DQA1*01:07-DQB1*06:12', 'HLA-DQA1*01:07-DQB1*06:14', 'HLA-DQA1*01:07-DQB1*06:15', 'HLA-DQA1*01:07-DQB1*06:16',
         'HLA-DQA1*01:07-DQB1*06:17', 'HLA-DQA1*01:07-DQB1*06:18',
         'HLA-DQA1*01:07-DQB1*06:19', 'HLA-DQA1*01:07-DQB1*06:21', 'HLA-DQA1*01:07-DQB1*06:22', 'HLA-DQA1*01:07-DQB1*06:23',
         'HLA-DQA1*01:07-DQB1*06:24', 'HLA-DQA1*01:07-DQB1*06:25',
         'HLA-DQA1*01:07-DQB1*06:27', 'HLA-DQA1*01:07-DQB1*06:28', 'HLA-DQA1*01:07-DQB1*06:29', 'HLA-DQA1*01:07-DQB1*06:30',
         'HLA-DQA1*01:07-DQB1*06:31', 'HLA-DQA1*01:07-DQB1*06:32',
         'HLA-DQA1*01:07-DQB1*06:33', 'HLA-DQA1*01:07-DQB1*06:34', 'HLA-DQA1*01:07-DQB1*06:35', 'HLA-DQA1*01:07-DQB1*06:36',
         'HLA-DQA1*01:07-DQB1*06:37', 'HLA-DQA1*01:07-DQB1*06:38',
         'HLA-DQA1*01:07-DQB1*06:39', 'HLA-DQA1*01:07-DQB1*06:40', 'HLA-DQA1*01:07-DQB1*06:41', 'HLA-DQA1*01:07-DQB1*06:42',
         'HLA-DQA1*01:07-DQB1*06:43', 'HLA-DQA1*01:07-DQB1*06:44',
         'HLA-DQA1*01:08-DQB1*02:01', 'HLA-DQA1*01:08-DQB1*02:02', 'HLA-DQA1*01:08-DQB1*02:03', 'HLA-DQA1*01:08-DQB1*02:04',
         'HLA-DQA1*01:08-DQB1*02:05', 'HLA-DQA1*01:08-DQB1*02:06',
         'HLA-DQA1*01:08-DQB1*03:01', 'HLA-DQA1*01:08-DQB1*03:02', 'HLA-DQA1*01:08-DQB1*03:03', 'HLA-DQA1*01:08-DQB1*03:04',
         'HLA-DQA1*01:08-DQB1*03:05', 'HLA-DQA1*01:08-DQB1*03:06',
         'HLA-DQA1*01:08-DQB1*03:07', 'HLA-DQA1*01:08-DQB1*03:08', 'HLA-DQA1*01:08-DQB1*03:09', 'HLA-DQA1*01:08-DQB1*03:10',
         'HLA-DQA1*01:08-DQB1*03:11', 'HLA-DQA1*01:08-DQB1*03:12',
         'HLA-DQA1*01:08-DQB1*03:13', 'HLA-DQA1*01:08-DQB1*03:14', 'HLA-DQA1*01:08-DQB1*03:15', 'HLA-DQA1*01:08-DQB1*03:16',
         'HLA-DQA1*01:08-DQB1*03:17', 'HLA-DQA1*01:08-DQB1*03:18',
         'HLA-DQA1*01:08-DQB1*03:19', 'HLA-DQA1*01:08-DQB1*03:20', 'HLA-DQA1*01:08-DQB1*03:21', 'HLA-DQA1*01:08-DQB1*03:22',
         'HLA-DQA1*01:08-DQB1*03:23', 'HLA-DQA1*01:08-DQB1*03:24',
         'HLA-DQA1*01:08-DQB1*03:25', 'HLA-DQA1*01:08-DQB1*03:26', 'HLA-DQA1*01:08-DQB1*03:27', 'HLA-DQA1*01:08-DQB1*03:28',
         'HLA-DQA1*01:08-DQB1*03:29', 'HLA-DQA1*01:08-DQB1*03:30',
         'HLA-DQA1*01:08-DQB1*03:31', 'HLA-DQA1*01:08-DQB1*03:32', 'HLA-DQA1*01:08-DQB1*03:33', 'HLA-DQA1*01:08-DQB1*03:34',
         'HLA-DQA1*01:08-DQB1*03:35', 'HLA-DQA1*01:08-DQB1*03:36',
         'HLA-DQA1*01:08-DQB1*03:37', 'HLA-DQA1*01:08-DQB1*03:38', 'HLA-DQA1*01:08-DQB1*04:01', 'HLA-DQA1*01:08-DQB1*04:02',
         'HLA-DQA1*01:08-DQB1*04:03', 'HLA-DQA1*01:08-DQB1*04:04',
         'HLA-DQA1*01:08-DQB1*04:05', 'HLA-DQA1*01:08-DQB1*04:06', 'HLA-DQA1*01:08-DQB1*04:07', 'HLA-DQA1*01:08-DQB1*04:08',
         'HLA-DQA1*01:08-DQB1*05:01', 'HLA-DQA1*01:08-DQB1*05:02',
         'HLA-DQA1*01:08-DQB1*05:03', 'HLA-DQA1*01:08-DQB1*05:05', 'HLA-DQA1*01:08-DQB1*05:06', 'HLA-DQA1*01:08-DQB1*05:07',
         'HLA-DQA1*01:08-DQB1*05:08', 'HLA-DQA1*01:08-DQB1*05:09',
         'HLA-DQA1*01:08-DQB1*05:10', 'HLA-DQA1*01:08-DQB1*05:11', 'HLA-DQA1*01:08-DQB1*05:12', 'HLA-DQA1*01:08-DQB1*05:13',
         'HLA-DQA1*01:08-DQB1*05:14', 'HLA-DQA1*01:08-DQB1*06:01',
         'HLA-DQA1*01:08-DQB1*06:02', 'HLA-DQA1*01:08-DQB1*06:03', 'HLA-DQA1*01:08-DQB1*06:04', 'HLA-DQA1*01:08-DQB1*06:07',
         'HLA-DQA1*01:08-DQB1*06:08', 'HLA-DQA1*01:08-DQB1*06:09',
         'HLA-DQA1*01:08-DQB1*06:10', 'HLA-DQA1*01:08-DQB1*06:11', 'HLA-DQA1*01:08-DQB1*06:12', 'HLA-DQA1*01:08-DQB1*06:14',
         'HLA-DQA1*01:08-DQB1*06:15', 'HLA-DQA1*01:08-DQB1*06:16',
         'HLA-DQA1*01:08-DQB1*06:17', 'HLA-DQA1*01:08-DQB1*06:18', 'HLA-DQA1*01:08-DQB1*06:19', 'HLA-DQA1*01:08-DQB1*06:21',
         'HLA-DQA1*01:08-DQB1*06:22', 'HLA-DQA1*01:08-DQB1*06:23',
         'HLA-DQA1*01:08-DQB1*06:24', 'HLA-DQA1*01:08-DQB1*06:25', 'HLA-DQA1*01:08-DQB1*06:27', 'HLA-DQA1*01:08-DQB1*06:28',
         'HLA-DQA1*01:08-DQB1*06:29', 'HLA-DQA1*01:08-DQB1*06:30',
         'HLA-DQA1*01:08-DQB1*06:31', 'HLA-DQA1*01:08-DQB1*06:32', 'HLA-DQA1*01:08-DQB1*06:33', 'HLA-DQA1*01:08-DQB1*06:34',
         'HLA-DQA1*01:08-DQB1*06:35', 'HLA-DQA1*01:08-DQB1*06:36',
         'HLA-DQA1*01:08-DQB1*06:37', 'HLA-DQA1*01:08-DQB1*06:38', 'HLA-DQA1*01:08-DQB1*06:39', 'HLA-DQA1*01:08-DQB1*06:40',
         'HLA-DQA1*01:08-DQB1*06:41', 'HLA-DQA1*01:08-DQB1*06:42',
         'HLA-DQA1*01:08-DQB1*06:43', 'HLA-DQA1*01:08-DQB1*06:44', 'HLA-DQA1*01:09-DQB1*02:01', 'HLA-DQA1*01:09-DQB1*02:02',
         'HLA-DQA1*01:09-DQB1*02:03', 'HLA-DQA1*01:09-DQB1*02:04',
         'HLA-DQA1*01:09-DQB1*02:05', 'HLA-DQA1*01:09-DQB1*02:06', 'HLA-DQA1*01:09-DQB1*03:01', 'HLA-DQA1*01:09-DQB1*03:02',
         'HLA-DQA1*01:09-DQB1*03:03', 'HLA-DQA1*01:09-DQB1*03:04',
         'HLA-DQA1*01:09-DQB1*03:05', 'HLA-DQA1*01:09-DQB1*03:06', 'HLA-DQA1*01:09-DQB1*03:07', 'HLA-DQA1*01:09-DQB1*03:08',
         'HLA-DQA1*01:09-DQB1*03:09', 'HLA-DQA1*01:09-DQB1*03:10',
         'HLA-DQA1*01:09-DQB1*03:11', 'HLA-DQA1*01:09-DQB1*03:12', 'HLA-DQA1*01:09-DQB1*03:13', 'HLA-DQA1*01:09-DQB1*03:14',
         'HLA-DQA1*01:09-DQB1*03:15', 'HLA-DQA1*01:09-DQB1*03:16',
         'HLA-DQA1*01:09-DQB1*03:17', 'HLA-DQA1*01:09-DQB1*03:18', 'HLA-DQA1*01:09-DQB1*03:19', 'HLA-DQA1*01:09-DQB1*03:20',
         'HLA-DQA1*01:09-DQB1*03:21', 'HLA-DQA1*01:09-DQB1*03:22',
         'HLA-DQA1*01:09-DQB1*03:23', 'HLA-DQA1*01:09-DQB1*03:24', 'HLA-DQA1*01:09-DQB1*03:25', 'HLA-DQA1*01:09-DQB1*03:26',
         'HLA-DQA1*01:09-DQB1*03:27', 'HLA-DQA1*01:09-DQB1*03:28',
         'HLA-DQA1*01:09-DQB1*03:29', 'HLA-DQA1*01:09-DQB1*03:30', 'HLA-DQA1*01:09-DQB1*03:31', 'HLA-DQA1*01:09-DQB1*03:32',
         'HLA-DQA1*01:09-DQB1*03:33', 'HLA-DQA1*01:09-DQB1*03:34',
         'HLA-DQA1*01:09-DQB1*03:35', 'HLA-DQA1*01:09-DQB1*03:36', 'HLA-DQA1*01:09-DQB1*03:37', 'HLA-DQA1*01:09-DQB1*03:38',
         'HLA-DQA1*01:09-DQB1*04:01', 'HLA-DQA1*01:09-DQB1*04:02',
         'HLA-DQA1*01:09-DQB1*04:03', 'HLA-DQA1*01:09-DQB1*04:04', 'HLA-DQA1*01:09-DQB1*04:05', 'HLA-DQA1*01:09-DQB1*04:06',
         'HLA-DQA1*01:09-DQB1*04:07', 'HLA-DQA1*01:09-DQB1*04:08',
         'HLA-DQA1*01:09-DQB1*05:01', 'HLA-DQA1*01:09-DQB1*05:02', 'HLA-DQA1*01:09-DQB1*05:03', 'HLA-DQA1*01:09-DQB1*05:05',
         'HLA-DQA1*01:09-DQB1*05:06', 'HLA-DQA1*01:09-DQB1*05:07',
         'HLA-DQA1*01:09-DQB1*05:08', 'HLA-DQA1*01:09-DQB1*05:09', 'HLA-DQA1*01:09-DQB1*05:10', 'HLA-DQA1*01:09-DQB1*05:11',
         'HLA-DQA1*01:09-DQB1*05:12', 'HLA-DQA1*01:09-DQB1*05:13',
         'HLA-DQA1*01:09-DQB1*05:14', 'HLA-DQA1*01:09-DQB1*06:01', 'HLA-DQA1*01:09-DQB1*06:02', 'HLA-DQA1*01:09-DQB1*06:03',
         'HLA-DQA1*01:09-DQB1*06:04', 'HLA-DQA1*01:09-DQB1*06:07',
         'HLA-DQA1*01:09-DQB1*06:08', 'HLA-DQA1*01:09-DQB1*06:09', 'HLA-DQA1*01:09-DQB1*06:10', 'HLA-DQA1*01:09-DQB1*06:11',
         'HLA-DQA1*01:09-DQB1*06:12', 'HLA-DQA1*01:09-DQB1*06:14',
         'HLA-DQA1*01:09-DQB1*06:15', 'HLA-DQA1*01:09-DQB1*06:16', 'HLA-DQA1*01:09-DQB1*06:17', 'HLA-DQA1*01:09-DQB1*06:18',
         'HLA-DQA1*01:09-DQB1*06:19', 'HLA-DQA1*01:09-DQB1*06:21',
         'HLA-DQA1*01:09-DQB1*06:22', 'HLA-DQA1*01:09-DQB1*06:23', 'HLA-DQA1*01:09-DQB1*06:24', 'HLA-DQA1*01:09-DQB1*06:25',
         'HLA-DQA1*01:09-DQB1*06:27', 'HLA-DQA1*01:09-DQB1*06:28',
         'HLA-DQA1*01:09-DQB1*06:29', 'HLA-DQA1*01:09-DQB1*06:30', 'HLA-DQA1*01:09-DQB1*06:31', 'HLA-DQA1*01:09-DQB1*06:32',
         'HLA-DQA1*01:09-DQB1*06:33', 'HLA-DQA1*01:09-DQB1*06:34',
         'HLA-DQA1*01:09-DQB1*06:35', 'HLA-DQA1*01:09-DQB1*06:36', 'HLA-DQA1*01:09-DQB1*06:37', 'HLA-DQA1*01:09-DQB1*06:38',
         'HLA-DQA1*01:09-DQB1*06:39', 'HLA-DQA1*01:09-DQB1*06:40',
         'HLA-DQA1*01:09-DQB1*06:41', 'HLA-DQA1*01:09-DQB1*06:42', 'HLA-DQA1*01:09-DQB1*06:43', 'HLA-DQA1*01:09-DQB1*06:44',
         'HLA-DQA1*02:01-DQB1*02:01', 'HLA-DQA1*02:01-DQB1*02:02',
         'HLA-DQA1*02:01-DQB1*02:03', 'HLA-DQA1*02:01-DQB1*02:04', 'HLA-DQA1*02:01-DQB1*02:05', 'HLA-DQA1*02:01-DQB1*02:06',
         'HLA-DQA1*02:01-DQB1*03:01', 'HLA-DQA1*02:01-DQB1*03:02',
         'HLA-DQA1*02:01-DQB1*03:03', 'HLA-DQA1*02:01-DQB1*03:04', 'HLA-DQA1*02:01-DQB1*03:05', 'HLA-DQA1*02:01-DQB1*03:06',
         'HLA-DQA1*02:01-DQB1*03:07', 'HLA-DQA1*02:01-DQB1*03:08',
         'HLA-DQA1*02:01-DQB1*03:09', 'HLA-DQA1*02:01-DQB1*03:10', 'HLA-DQA1*02:01-DQB1*03:11', 'HLA-DQA1*02:01-DQB1*03:12',
         'HLA-DQA1*02:01-DQB1*03:13', 'HLA-DQA1*02:01-DQB1*03:14',
         'HLA-DQA1*02:01-DQB1*03:15', 'HLA-DQA1*02:01-DQB1*03:16', 'HLA-DQA1*02:01-DQB1*03:17', 'HLA-DQA1*02:01-DQB1*03:18',
         'HLA-DQA1*02:01-DQB1*03:19', 'HLA-DQA1*02:01-DQB1*03:20',
         'HLA-DQA1*02:01-DQB1*03:21', 'HLA-DQA1*02:01-DQB1*03:22', 'HLA-DQA1*02:01-DQB1*03:23', 'HLA-DQA1*02:01-DQB1*03:24',
         'HLA-DQA1*02:01-DQB1*03:25', 'HLA-DQA1*02:01-DQB1*03:26',
         'HLA-DQA1*02:01-DQB1*03:27', 'HLA-DQA1*02:01-DQB1*03:28', 'HLA-DQA1*02:01-DQB1*03:29', 'HLA-DQA1*02:01-DQB1*03:30',
         'HLA-DQA1*02:01-DQB1*03:31', 'HLA-DQA1*02:01-DQB1*03:32',
         'HLA-DQA1*02:01-DQB1*03:33', 'HLA-DQA1*02:01-DQB1*03:34', 'HLA-DQA1*02:01-DQB1*03:35', 'HLA-DQA1*02:01-DQB1*03:36',
         'HLA-DQA1*02:01-DQB1*03:37', 'HLA-DQA1*02:01-DQB1*03:38',
         'HLA-DQA1*02:01-DQB1*04:01', 'HLA-DQA1*02:01-DQB1*04:02', 'HLA-DQA1*02:01-DQB1*04:03', 'HLA-DQA1*02:01-DQB1*04:04',
         'HLA-DQA1*02:01-DQB1*04:05', 'HLA-DQA1*02:01-DQB1*04:06',
         'HLA-DQA1*02:01-DQB1*04:07', 'HLA-DQA1*02:01-DQB1*04:08', 'HLA-DQA1*02:01-DQB1*05:01', 'HLA-DQA1*02:01-DQB1*05:02',
         'HLA-DQA1*02:01-DQB1*05:03', 'HLA-DQA1*02:01-DQB1*05:05',
         'HLA-DQA1*02:01-DQB1*05:06', 'HLA-DQA1*02:01-DQB1*05:07', 'HLA-DQA1*02:01-DQB1*05:08', 'HLA-DQA1*02:01-DQB1*05:09',
         'HLA-DQA1*02:01-DQB1*05:10', 'HLA-DQA1*02:01-DQB1*05:11',
         'HLA-DQA1*02:01-DQB1*05:12', 'HLA-DQA1*02:01-DQB1*05:13', 'HLA-DQA1*02:01-DQB1*05:14', 'HLA-DQA1*02:01-DQB1*06:01',
         'HLA-DQA1*02:01-DQB1*06:02', 'HLA-DQA1*02:01-DQB1*06:03',
         'HLA-DQA1*02:01-DQB1*06:04', 'HLA-DQA1*02:01-DQB1*06:07', 'HLA-DQA1*02:01-DQB1*06:08', 'HLA-DQA1*02:01-DQB1*06:09',
         'HLA-DQA1*02:01-DQB1*06:10', 'HLA-DQA1*02:01-DQB1*06:11',
         'HLA-DQA1*02:01-DQB1*06:12', 'HLA-DQA1*02:01-DQB1*06:14', 'HLA-DQA1*02:01-DQB1*06:15', 'HLA-DQA1*02:01-DQB1*06:16',
         'HLA-DQA1*02:01-DQB1*06:17', 'HLA-DQA1*02:01-DQB1*06:18',
         'HLA-DQA1*02:01-DQB1*06:19', 'HLA-DQA1*02:01-DQB1*06:21', 'HLA-DQA1*02:01-DQB1*06:22', 'HLA-DQA1*02:01-DQB1*06:23',
         'HLA-DQA1*02:01-DQB1*06:24', 'HLA-DQA1*02:01-DQB1*06:25',
         'HLA-DQA1*02:01-DQB1*06:27', 'HLA-DQA1*02:01-DQB1*06:28', 'HLA-DQA1*02:01-DQB1*06:29', 'HLA-DQA1*02:01-DQB1*06:30',
         'HLA-DQA1*02:01-DQB1*06:31', 'HLA-DQA1*02:01-DQB1*06:32',
         'HLA-DQA1*02:01-DQB1*06:33', 'HLA-DQA1*02:01-DQB1*06:34', 'HLA-DQA1*02:01-DQB1*06:35', 'HLA-DQA1*02:01-DQB1*06:36',
         'HLA-DQA1*02:01-DQB1*06:37', 'HLA-DQA1*02:01-DQB1*06:38',
         'HLA-DQA1*02:01-DQB1*06:39', 'HLA-DQA1*02:01-DQB1*06:40', 'HLA-DQA1*02:01-DQB1*06:41', 'HLA-DQA1*02:01-DQB1*06:42',
         'HLA-DQA1*02:01-DQB1*06:43', 'HLA-DQA1*02:01-DQB1*06:44',
         'HLA-DQA1*03:01-DQB1*02:01', 'HLA-DQA1*03:01-DQB1*02:02', 'HLA-DQA1*03:01-DQB1*02:03', 'HLA-DQA1*03:01-DQB1*02:04',
         'HLA-DQA1*03:01-DQB1*02:05', 'HLA-DQA1*03:01-DQB1*02:06',
         'HLA-DQA1*03:01-DQB1*03:01', 'HLA-DQA1*03:01-DQB1*03:02', 'HLA-DQA1*03:01-DQB1*03:03', 'HLA-DQA1*03:01-DQB1*03:04',
         'HLA-DQA1*03:01-DQB1*03:05', 'HLA-DQA1*03:01-DQB1*03:06',
         'HLA-DQA1*03:01-DQB1*03:07', 'HLA-DQA1*03:01-DQB1*03:08', 'HLA-DQA1*03:01-DQB1*03:09', 'HLA-DQA1*03:01-DQB1*03:10',
         'HLA-DQA1*03:01-DQB1*03:11', 'HLA-DQA1*03:01-DQB1*03:12',
         'HLA-DQA1*03:01-DQB1*03:13', 'HLA-DQA1*03:01-DQB1*03:14', 'HLA-DQA1*03:01-DQB1*03:15', 'HLA-DQA1*03:01-DQB1*03:16',
         'HLA-DQA1*03:01-DQB1*03:17', 'HLA-DQA1*03:01-DQB1*03:18',
         'HLA-DQA1*03:01-DQB1*03:19', 'HLA-DQA1*03:01-DQB1*03:20', 'HLA-DQA1*03:01-DQB1*03:21', 'HLA-DQA1*03:01-DQB1*03:22',
         'HLA-DQA1*03:01-DQB1*03:23', 'HLA-DQA1*03:01-DQB1*03:24',
         'HLA-DQA1*03:01-DQB1*03:25', 'HLA-DQA1*03:01-DQB1*03:26', 'HLA-DQA1*03:01-DQB1*03:27', 'HLA-DQA1*03:01-DQB1*03:28',
         'HLA-DQA1*03:01-DQB1*03:29', 'HLA-DQA1*03:01-DQB1*03:30',
         'HLA-DQA1*03:01-DQB1*03:31', 'HLA-DQA1*03:01-DQB1*03:32', 'HLA-DQA1*03:01-DQB1*03:33', 'HLA-DQA1*03:01-DQB1*03:34',
         'HLA-DQA1*03:01-DQB1*03:35', 'HLA-DQA1*03:01-DQB1*03:36',
         'HLA-DQA1*03:01-DQB1*03:37', 'HLA-DQA1*03:01-DQB1*03:38', 'HLA-DQA1*03:01-DQB1*04:01', 'HLA-DQA1*03:01-DQB1*04:02',
         'HLA-DQA1*03:01-DQB1*04:03', 'HLA-DQA1*03:01-DQB1*04:04',
         'HLA-DQA1*03:01-DQB1*04:05', 'HLA-DQA1*03:01-DQB1*04:06', 'HLA-DQA1*03:01-DQB1*04:07', 'HLA-DQA1*03:01-DQB1*04:08',
         'HLA-DQA1*03:01-DQB1*05:01', 'HLA-DQA1*03:01-DQB1*05:02',
         'HLA-DQA1*03:01-DQB1*05:03', 'HLA-DQA1*03:01-DQB1*05:05', 'HLA-DQA1*03:01-DQB1*05:06', 'HLA-DQA1*03:01-DQB1*05:07',
         'HLA-DQA1*03:01-DQB1*05:08', 'HLA-DQA1*03:01-DQB1*05:09',
         'HLA-DQA1*03:01-DQB1*05:10', 'HLA-DQA1*03:01-DQB1*05:11', 'HLA-DQA1*03:01-DQB1*05:12', 'HLA-DQA1*03:01-DQB1*05:13',
         'HLA-DQA1*03:01-DQB1*05:14', 'HLA-DQA1*03:01-DQB1*06:01',
         'HLA-DQA1*03:01-DQB1*06:02', 'HLA-DQA1*03:01-DQB1*06:03', 'HLA-DQA1*03:01-DQB1*06:04', 'HLA-DQA1*03:01-DQB1*06:07',
         'HLA-DQA1*03:01-DQB1*06:08', 'HLA-DQA1*03:01-DQB1*06:09',
         'HLA-DQA1*03:01-DQB1*06:10', 'HLA-DQA1*03:01-DQB1*06:11', 'HLA-DQA1*03:01-DQB1*06:12', 'HLA-DQA1*03:01-DQB1*06:14',
         'HLA-DQA1*03:01-DQB1*06:15', 'HLA-DQA1*03:01-DQB1*06:16',
         'HLA-DQA1*03:01-DQB1*06:17', 'HLA-DQA1*03:01-DQB1*06:18', 'HLA-DQA1*03:01-DQB1*06:19', 'HLA-DQA1*03:01-DQB1*06:21',
         'HLA-DQA1*03:01-DQB1*06:22', 'HLA-DQA1*03:01-DQB1*06:23',
         'HLA-DQA1*03:01-DQB1*06:24', 'HLA-DQA1*03:01-DQB1*06:25', 'HLA-DQA1*03:01-DQB1*06:27', 'HLA-DQA1*03:01-DQB1*06:28',
         'HLA-DQA1*03:01-DQB1*06:29', 'HLA-DQA1*03:01-DQB1*06:30',
         'HLA-DQA1*03:01-DQB1*06:31', 'HLA-DQA1*03:01-DQB1*06:32', 'HLA-DQA1*03:01-DQB1*06:33', 'HLA-DQA1*03:01-DQB1*06:34',
         'HLA-DQA1*03:01-DQB1*06:35', 'HLA-DQA1*03:01-DQB1*06:36',
         'HLA-DQA1*03:01-DQB1*06:37', 'HLA-DQA1*03:01-DQB1*06:38', 'HLA-DQA1*03:01-DQB1*06:39', 'HLA-DQA1*03:01-DQB1*06:40',
         'HLA-DQA1*03:01-DQB1*06:41', 'HLA-DQA1*03:01-DQB1*06:42',
         'HLA-DQA1*03:01-DQB1*06:43', 'HLA-DQA1*03:01-DQB1*06:44', 'HLA-DQA1*03:02-DQB1*02:01', 'HLA-DQA1*03:02-DQB1*02:02',
         'HLA-DQA1*03:02-DQB1*02:03', 'HLA-DQA1*03:02-DQB1*02:04',
         'HLA-DQA1*03:02-DQB1*02:05', 'HLA-DQA1*03:02-DQB1*02:06', 'HLA-DQA1*03:02-DQB1*03:01', 'HLA-DQA1*03:02-DQB1*03:02',
         'HLA-DQA1*03:02-DQB1*03:03', 'HLA-DQA1*03:02-DQB1*03:04',
         'HLA-DQA1*03:02-DQB1*03:05', 'HLA-DQA1*03:02-DQB1*03:06', 'HLA-DQA1*03:02-DQB1*03:07', 'HLA-DQA1*03:02-DQB1*03:08',
         'HLA-DQA1*03:02-DQB1*03:09', 'HLA-DQA1*03:02-DQB1*03:10',
         'HLA-DQA1*03:02-DQB1*03:11', 'HLA-DQA1*03:02-DQB1*03:12', 'HLA-DQA1*03:02-DQB1*03:13', 'HLA-DQA1*03:02-DQB1*03:14',
         'HLA-DQA1*03:02-DQB1*03:15', 'HLA-DQA1*03:02-DQB1*03:16',
         'HLA-DQA1*03:02-DQB1*03:17', 'HLA-DQA1*03:02-DQB1*03:18', 'HLA-DQA1*03:02-DQB1*03:19', 'HLA-DQA1*03:02-DQB1*03:20',
         'HLA-DQA1*03:02-DQB1*03:21', 'HLA-DQA1*03:02-DQB1*03:22',
         'HLA-DQA1*03:02-DQB1*03:23', 'HLA-DQA1*03:02-DQB1*03:24', 'HLA-DQA1*03:02-DQB1*03:25', 'HLA-DQA1*03:02-DQB1*03:26',
         'HLA-DQA1*03:02-DQB1*03:27', 'HLA-DQA1*03:02-DQB1*03:28',
         'HLA-DQA1*03:02-DQB1*03:29', 'HLA-DQA1*03:02-DQB1*03:30', 'HLA-DQA1*03:02-DQB1*03:31', 'HLA-DQA1*03:02-DQB1*03:32',
         'HLA-DQA1*03:02-DQB1*03:33', 'HLA-DQA1*03:02-DQB1*03:34',
         'HLA-DQA1*03:02-DQB1*03:35', 'HLA-DQA1*03:02-DQB1*03:36', 'HLA-DQA1*03:02-DQB1*03:37', 'HLA-DQA1*03:02-DQB1*03:38',
         'HLA-DQA1*03:02-DQB1*04:01', 'HLA-DQA1*03:02-DQB1*04:02',
         'HLA-DQA1*03:02-DQB1*04:03', 'HLA-DQA1*03:02-DQB1*04:04', 'HLA-DQA1*03:02-DQB1*04:05', 'HLA-DQA1*03:02-DQB1*04:06',
         'HLA-DQA1*03:02-DQB1*04:07', 'HLA-DQA1*03:02-DQB1*04:08',
         'HLA-DQA1*03:02-DQB1*05:01', 'HLA-DQA1*03:02-DQB1*05:02', 'HLA-DQA1*03:02-DQB1*05:03', 'HLA-DQA1*03:02-DQB1*05:05',
         'HLA-DQA1*03:02-DQB1*05:06', 'HLA-DQA1*03:02-DQB1*05:07',
         'HLA-DQA1*03:02-DQB1*05:08', 'HLA-DQA1*03:02-DQB1*05:09', 'HLA-DQA1*03:02-DQB1*05:10', 'HLA-DQA1*03:02-DQB1*05:11',
         'HLA-DQA1*03:02-DQB1*05:12', 'HLA-DQA1*03:02-DQB1*05:13',
         'HLA-DQA1*03:02-DQB1*05:14', 'HLA-DQA1*03:02-DQB1*06:01', 'HLA-DQA1*03:02-DQB1*06:02', 'HLA-DQA1*03:02-DQB1*06:03',
         'HLA-DQA1*03:02-DQB1*06:04', 'HLA-DQA1*03:02-DQB1*06:07',
         'HLA-DQA1*03:02-DQB1*06:08', 'HLA-DQA1*03:02-DQB1*06:09', 'HLA-DQA1*03:02-DQB1*06:10', 'HLA-DQA1*03:02-DQB1*06:11',
         'HLA-DQA1*03:02-DQB1*06:12', 'HLA-DQA1*03:02-DQB1*06:14',
         'HLA-DQA1*03:02-DQB1*06:15', 'HLA-DQA1*03:02-DQB1*06:16', 'HLA-DQA1*03:02-DQB1*06:17', 'HLA-DQA1*03:02-DQB1*06:18',
         'HLA-DQA1*03:02-DQB1*06:19', 'HLA-DQA1*03:02-DQB1*06:21',
         'HLA-DQA1*03:02-DQB1*06:22', 'HLA-DQA1*03:02-DQB1*06:23', 'HLA-DQA1*03:02-DQB1*06:24', 'HLA-DQA1*03:02-DQB1*06:25',
         'HLA-DQA1*03:02-DQB1*06:27', 'HLA-DQA1*03:02-DQB1*06:28',
         'HLA-DQA1*03:02-DQB1*06:29', 'HLA-DQA1*03:02-DQB1*06:30', 'HLA-DQA1*03:02-DQB1*06:31', 'HLA-DQA1*03:02-DQB1*06:32',
         'HLA-DQA1*03:02-DQB1*06:33', 'HLA-DQA1*03:02-DQB1*06:34',
         'HLA-DQA1*03:02-DQB1*06:35', 'HLA-DQA1*03:02-DQB1*06:36', 'HLA-DQA1*03:02-DQB1*06:37', 'HLA-DQA1*03:02-DQB1*06:38',
         'HLA-DQA1*03:02-DQB1*06:39', 'HLA-DQA1*03:02-DQB1*06:40',
         'HLA-DQA1*03:02-DQB1*06:41', 'HLA-DQA1*03:02-DQB1*06:42', 'HLA-DQA1*03:02-DQB1*06:43', 'HLA-DQA1*03:02-DQB1*06:44',
         'HLA-DQA1*03:03-DQB1*02:01', 'HLA-DQA1*03:03-DQB1*02:02',
         'HLA-DQA1*03:03-DQB1*02:03', 'HLA-DQA1*03:03-DQB1*02:04', 'HLA-DQA1*03:03-DQB1*02:05', 'HLA-DQA1*03:03-DQB1*02:06',
         'HLA-DQA1*03:03-DQB1*03:01', 'HLA-DQA1*03:03-DQB1*03:02',
         'HLA-DQA1*03:03-DQB1*03:03', 'HLA-DQA1*03:03-DQB1*03:04', 'HLA-DQA1*03:03-DQB1*03:05', 'HLA-DQA1*03:03-DQB1*03:06',
         'HLA-DQA1*03:03-DQB1*03:07', 'HLA-DQA1*03:03-DQB1*03:08',
         'HLA-DQA1*03:03-DQB1*03:09', 'HLA-DQA1*03:03-DQB1*03:10', 'HLA-DQA1*03:03-DQB1*03:11', 'HLA-DQA1*03:03-DQB1*03:12',
         'HLA-DQA1*03:03-DQB1*03:13', 'HLA-DQA1*03:03-DQB1*03:14',
         'HLA-DQA1*03:03-DQB1*03:15', 'HLA-DQA1*03:03-DQB1*03:16', 'HLA-DQA1*03:03-DQB1*03:17', 'HLA-DQA1*03:03-DQB1*03:18',
         'HLA-DQA1*03:03-DQB1*03:19', 'HLA-DQA1*03:03-DQB1*03:20',
         'HLA-DQA1*03:03-DQB1*03:21', 'HLA-DQA1*03:03-DQB1*03:22', 'HLA-DQA1*03:03-DQB1*03:23', 'HLA-DQA1*03:03-DQB1*03:24',
         'HLA-DQA1*03:03-DQB1*03:25', 'HLA-DQA1*03:03-DQB1*03:26',
         'HLA-DQA1*03:03-DQB1*03:27', 'HLA-DQA1*03:03-DQB1*03:28', 'HLA-DQA1*03:03-DQB1*03:29', 'HLA-DQA1*03:03-DQB1*03:30',
         'HLA-DQA1*03:03-DQB1*03:31', 'HLA-DQA1*03:03-DQB1*03:32',
         'HLA-DQA1*03:03-DQB1*03:33', 'HLA-DQA1*03:03-DQB1*03:34', 'HLA-DQA1*03:03-DQB1*03:35', 'HLA-DQA1*03:03-DQB1*03:36',
         'HLA-DQA1*03:03-DQB1*03:37', 'HLA-DQA1*03:03-DQB1*03:38',
         'HLA-DQA1*03:03-DQB1*04:01', 'HLA-DQA1*03:03-DQB1*04:02', 'HLA-DQA1*03:03-DQB1*04:03', 'HLA-DQA1*03:03-DQB1*04:04',
         'HLA-DQA1*03:03-DQB1*04:05', 'HLA-DQA1*03:03-DQB1*04:06',
         'HLA-DQA1*03:03-DQB1*04:07', 'HLA-DQA1*03:03-DQB1*04:08', 'HLA-DQA1*03:03-DQB1*05:01', 'HLA-DQA1*03:03-DQB1*05:02',
         'HLA-DQA1*03:03-DQB1*05:03', 'HLA-DQA1*03:03-DQB1*05:05',
         'HLA-DQA1*03:03-DQB1*05:06', 'HLA-DQA1*03:03-DQB1*05:07', 'HLA-DQA1*03:03-DQB1*05:08', 'HLA-DQA1*03:03-DQB1*05:09',
         'HLA-DQA1*03:03-DQB1*05:10', 'HLA-DQA1*03:03-DQB1*05:11',
         'HLA-DQA1*03:03-DQB1*05:12', 'HLA-DQA1*03:03-DQB1*05:13', 'HLA-DQA1*03:03-DQB1*05:14', 'HLA-DQA1*03:03-DQB1*06:01',
         'HLA-DQA1*03:03-DQB1*06:02', 'HLA-DQA1*03:03-DQB1*06:03',
         'HLA-DQA1*03:03-DQB1*06:04', 'HLA-DQA1*03:03-DQB1*06:07', 'HLA-DQA1*03:03-DQB1*06:08', 'HLA-DQA1*03:03-DQB1*06:09',
         'HLA-DQA1*03:03-DQB1*06:10', 'HLA-DQA1*03:03-DQB1*06:11',
         'HLA-DQA1*03:03-DQB1*06:12', 'HLA-DQA1*03:03-DQB1*06:14', 'HLA-DQA1*03:03-DQB1*06:15', 'HLA-DQA1*03:03-DQB1*06:16',
         'HLA-DQA1*03:03-DQB1*06:17', 'HLA-DQA1*03:03-DQB1*06:18',
         'HLA-DQA1*03:03-DQB1*06:19', 'HLA-DQA1*03:03-DQB1*06:21', 'HLA-DQA1*03:03-DQB1*06:22', 'HLA-DQA1*03:03-DQB1*06:23',
         'HLA-DQA1*03:03-DQB1*06:24', 'HLA-DQA1*03:03-DQB1*06:25',
         'HLA-DQA1*03:03-DQB1*06:27', 'HLA-DQA1*03:03-DQB1*06:28', 'HLA-DQA1*03:03-DQB1*06:29', 'HLA-DQA1*03:03-DQB1*06:30',
         'HLA-DQA1*03:03-DQB1*06:31', 'HLA-DQA1*03:03-DQB1*06:32',
         'HLA-DQA1*03:03-DQB1*06:33', 'HLA-DQA1*03:03-DQB1*06:34', 'HLA-DQA1*03:03-DQB1*06:35', 'HLA-DQA1*03:03-DQB1*06:36',
         'HLA-DQA1*03:03-DQB1*06:37', 'HLA-DQA1*03:03-DQB1*06:38',
         'HLA-DQA1*03:03-DQB1*06:39', 'HLA-DQA1*03:03-DQB1*06:40', 'HLA-DQA1*03:03-DQB1*06:41', 'HLA-DQA1*03:03-DQB1*06:42',
         'HLA-DQA1*03:03-DQB1*06:43', 'HLA-DQA1*03:03-DQB1*06:44',
         'HLA-DQA1*04:01-DQB1*02:01', 'HLA-DQA1*04:01-DQB1*02:02', 'HLA-DQA1*04:01-DQB1*02:03', 'HLA-DQA1*04:01-DQB1*02:04',
         'HLA-DQA1*04:01-DQB1*02:05', 'HLA-DQA1*04:01-DQB1*02:06',
         'HLA-DQA1*04:01-DQB1*03:01', 'HLA-DQA1*04:01-DQB1*03:02', 'HLA-DQA1*04:01-DQB1*03:03', 'HLA-DQA1*04:01-DQB1*03:04',
         'HLA-DQA1*04:01-DQB1*03:05', 'HLA-DQA1*04:01-DQB1*03:06',
         'HLA-DQA1*04:01-DQB1*03:07', 'HLA-DQA1*04:01-DQB1*03:08', 'HLA-DQA1*04:01-DQB1*03:09', 'HLA-DQA1*04:01-DQB1*03:10',
         'HLA-DQA1*04:01-DQB1*03:11', 'HLA-DQA1*04:01-DQB1*03:12',
         'HLA-DQA1*04:01-DQB1*03:13', 'HLA-DQA1*04:01-DQB1*03:14', 'HLA-DQA1*04:01-DQB1*03:15', 'HLA-DQA1*04:01-DQB1*03:16',
         'HLA-DQA1*04:01-DQB1*03:17', 'HLA-DQA1*04:01-DQB1*03:18',
         'HLA-DQA1*04:01-DQB1*03:19', 'HLA-DQA1*04:01-DQB1*03:20', 'HLA-DQA1*04:01-DQB1*03:21', 'HLA-DQA1*04:01-DQB1*03:22',
         'HLA-DQA1*04:01-DQB1*03:23', 'HLA-DQA1*04:01-DQB1*03:24',
         'HLA-DQA1*04:01-DQB1*03:25', 'HLA-DQA1*04:01-DQB1*03:26', 'HLA-DQA1*04:01-DQB1*03:27', 'HLA-DQA1*04:01-DQB1*03:28',
         'HLA-DQA1*04:01-DQB1*03:29', 'HLA-DQA1*04:01-DQB1*03:30',
         'HLA-DQA1*04:01-DQB1*03:31', 'HLA-DQA1*04:01-DQB1*03:32', 'HLA-DQA1*04:01-DQB1*03:33', 'HLA-DQA1*04:01-DQB1*03:34',
         'HLA-DQA1*04:01-DQB1*03:35', 'HLA-DQA1*04:01-DQB1*03:36',
         'HLA-DQA1*04:01-DQB1*03:37', 'HLA-DQA1*04:01-DQB1*03:38', 'HLA-DQA1*04:01-DQB1*04:01', 'HLA-DQA1*04:01-DQB1*04:02',
         'HLA-DQA1*04:01-DQB1*04:03', 'HLA-DQA1*04:01-DQB1*04:04',
         'HLA-DQA1*04:01-DQB1*04:05', 'HLA-DQA1*04:01-DQB1*04:06', 'HLA-DQA1*04:01-DQB1*04:07', 'HLA-DQA1*04:01-DQB1*04:08',
         'HLA-DQA1*04:01-DQB1*05:01', 'HLA-DQA1*04:01-DQB1*05:02',
         'HLA-DQA1*04:01-DQB1*05:03', 'HLA-DQA1*04:01-DQB1*05:05', 'HLA-DQA1*04:01-DQB1*05:06', 'HLA-DQA1*04:01-DQB1*05:07',
         'HLA-DQA1*04:01-DQB1*05:08', 'HLA-DQA1*04:01-DQB1*05:09',
         'HLA-DQA1*04:01-DQB1*05:10', 'HLA-DQA1*04:01-DQB1*05:11', 'HLA-DQA1*04:01-DQB1*05:12', 'HLA-DQA1*04:01-DQB1*05:13',
         'HLA-DQA1*04:01-DQB1*05:14', 'HLA-DQA1*04:01-DQB1*06:01',
         'HLA-DQA1*04:01-DQB1*06:02', 'HLA-DQA1*04:01-DQB1*06:03', 'HLA-DQA1*04:01-DQB1*06:04', 'HLA-DQA1*04:01-DQB1*06:07',
         'HLA-DQA1*04:01-DQB1*06:08', 'HLA-DQA1*04:01-DQB1*06:09',
         'HLA-DQA1*04:01-DQB1*06:10', 'HLA-DQA1*04:01-DQB1*06:11', 'HLA-DQA1*04:01-DQB1*06:12', 'HLA-DQA1*04:01-DQB1*06:14',
         'HLA-DQA1*04:01-DQB1*06:15', 'HLA-DQA1*04:01-DQB1*06:16',
         'HLA-DQA1*04:01-DQB1*06:17', 'HLA-DQA1*04:01-DQB1*06:18', 'HLA-DQA1*04:01-DQB1*06:19', 'HLA-DQA1*04:01-DQB1*06:21',
         'HLA-DQA1*04:01-DQB1*06:22', 'HLA-DQA1*04:01-DQB1*06:23',
         'HLA-DQA1*04:01-DQB1*06:24', 'HLA-DQA1*04:01-DQB1*06:25', 'HLA-DQA1*04:01-DQB1*06:27', 'HLA-DQA1*04:01-DQB1*06:28',
         'HLA-DQA1*04:01-DQB1*06:29', 'HLA-DQA1*04:01-DQB1*06:30',
         'HLA-DQA1*04:01-DQB1*06:31', 'HLA-DQA1*04:01-DQB1*06:32', 'HLA-DQA1*04:01-DQB1*06:33', 'HLA-DQA1*04:01-DQB1*06:34',
         'HLA-DQA1*04:01-DQB1*06:35', 'HLA-DQA1*04:01-DQB1*06:36',
         'HLA-DQA1*04:01-DQB1*06:37', 'HLA-DQA1*04:01-DQB1*06:38', 'HLA-DQA1*04:01-DQB1*06:39', 'HLA-DQA1*04:01-DQB1*06:40',
         'HLA-DQA1*04:01-DQB1*06:41', 'HLA-DQA1*04:01-DQB1*06:42',
         'HLA-DQA1*04:01-DQB1*06:43', 'HLA-DQA1*04:01-DQB1*06:44', 'HLA-DQA1*04:02-DQB1*02:01', 'HLA-DQA1*04:02-DQB1*02:02',
         'HLA-DQA1*04:02-DQB1*02:03', 'HLA-DQA1*04:02-DQB1*02:04',
         'HLA-DQA1*04:02-DQB1*02:05', 'HLA-DQA1*04:02-DQB1*02:06', 'HLA-DQA1*04:02-DQB1*03:01', 'HLA-DQA1*04:02-DQB1*03:02',
         'HLA-DQA1*04:02-DQB1*03:03', 'HLA-DQA1*04:02-DQB1*03:04',
         'HLA-DQA1*04:02-DQB1*03:05', 'HLA-DQA1*04:02-DQB1*03:06', 'HLA-DQA1*04:02-DQB1*03:07', 'HLA-DQA1*04:02-DQB1*03:08',
         'HLA-DQA1*04:02-DQB1*03:09', 'HLA-DQA1*04:02-DQB1*03:10',
         'HLA-DQA1*04:02-DQB1*03:11', 'HLA-DQA1*04:02-DQB1*03:12', 'HLA-DQA1*04:02-DQB1*03:13', 'HLA-DQA1*04:02-DQB1*03:14',
         'HLA-DQA1*04:02-DQB1*03:15', 'HLA-DQA1*04:02-DQB1*03:16',
         'HLA-DQA1*04:02-DQB1*03:17', 'HLA-DQA1*04:02-DQB1*03:18', 'HLA-DQA1*04:02-DQB1*03:19', 'HLA-DQA1*04:02-DQB1*03:20',
         'HLA-DQA1*04:02-DQB1*03:21', 'HLA-DQA1*04:02-DQB1*03:22',
         'HLA-DQA1*04:02-DQB1*03:23', 'HLA-DQA1*04:02-DQB1*03:24', 'HLA-DQA1*04:02-DQB1*03:25', 'HLA-DQA1*04:02-DQB1*03:26',
         'HLA-DQA1*04:02-DQB1*03:27', 'HLA-DQA1*04:02-DQB1*03:28',
         'HLA-DQA1*04:02-DQB1*03:29', 'HLA-DQA1*04:02-DQB1*03:30', 'HLA-DQA1*04:02-DQB1*03:31', 'HLA-DQA1*04:02-DQB1*03:32',
         'HLA-DQA1*04:02-DQB1*03:33', 'HLA-DQA1*04:02-DQB1*03:34',
         'HLA-DQA1*04:02-DQB1*03:35', 'HLA-DQA1*04:02-DQB1*03:36', 'HLA-DQA1*04:02-DQB1*03:37', 'HLA-DQA1*04:02-DQB1*03:38',
         'HLA-DQA1*04:02-DQB1*04:01', 'HLA-DQA1*04:02-DQB1*04:02',
         'HLA-DQA1*04:02-DQB1*04:03', 'HLA-DQA1*04:02-DQB1*04:04', 'HLA-DQA1*04:02-DQB1*04:05', 'HLA-DQA1*04:02-DQB1*04:06',
         'HLA-DQA1*04:02-DQB1*04:07', 'HLA-DQA1*04:02-DQB1*04:08',
         'HLA-DQA1*04:02-DQB1*05:01', 'HLA-DQA1*04:02-DQB1*05:02', 'HLA-DQA1*04:02-DQB1*05:03', 'HLA-DQA1*04:02-DQB1*05:05',
         'HLA-DQA1*04:02-DQB1*05:06', 'HLA-DQA1*04:02-DQB1*05:07',
         'HLA-DQA1*04:02-DQB1*05:08', 'HLA-DQA1*04:02-DQB1*05:09', 'HLA-DQA1*04:02-DQB1*05:10', 'HLA-DQA1*04:02-DQB1*05:11',
         'HLA-DQA1*04:02-DQB1*05:12', 'HLA-DQA1*04:02-DQB1*05:13',
         'HLA-DQA1*04:02-DQB1*05:14', 'HLA-DQA1*04:02-DQB1*06:01', 'HLA-DQA1*04:02-DQB1*06:02', 'HLA-DQA1*04:02-DQB1*06:03',
         'HLA-DQA1*04:02-DQB1*06:04', 'HLA-DQA1*04:02-DQB1*06:07',
         'HLA-DQA1*04:02-DQB1*06:08', 'HLA-DQA1*04:02-DQB1*06:09', 'HLA-DQA1*04:02-DQB1*06:10', 'HLA-DQA1*04:02-DQB1*06:11',
         'HLA-DQA1*04:02-DQB1*06:12', 'HLA-DQA1*04:02-DQB1*06:14',
         'HLA-DQA1*04:02-DQB1*06:15', 'HLA-DQA1*04:02-DQB1*06:16', 'HLA-DQA1*04:02-DQB1*06:17', 'HLA-DQA1*04:02-DQB1*06:18',
         'HLA-DQA1*04:02-DQB1*06:19', 'HLA-DQA1*04:02-DQB1*06:21',
         'HLA-DQA1*04:02-DQB1*06:22', 'HLA-DQA1*04:02-DQB1*06:23', 'HLA-DQA1*04:02-DQB1*06:24', 'HLA-DQA1*04:02-DQB1*06:25',
         'HLA-DQA1*04:02-DQB1*06:27', 'HLA-DQA1*04:02-DQB1*06:28',
         'HLA-DQA1*04:02-DQB1*06:29', 'HLA-DQA1*04:02-DQB1*06:30', 'HLA-DQA1*04:02-DQB1*06:31', 'HLA-DQA1*04:02-DQB1*06:32',
         'HLA-DQA1*04:02-DQB1*06:33', 'HLA-DQA1*04:02-DQB1*06:34',
         'HLA-DQA1*04:02-DQB1*06:35', 'HLA-DQA1*04:02-DQB1*06:36', 'HLA-DQA1*04:02-DQB1*06:37', 'HLA-DQA1*04:02-DQB1*06:38',
         'HLA-DQA1*04:02-DQB1*06:39', 'HLA-DQA1*04:02-DQB1*06:40',
         'HLA-DQA1*04:02-DQB1*06:41', 'HLA-DQA1*04:02-DQB1*06:42', 'HLA-DQA1*04:02-DQB1*06:43', 'HLA-DQA1*04:02-DQB1*06:44',
         'HLA-DQA1*04:04-DQB1*02:01', 'HLA-DQA1*04:04-DQB1*02:02',
         'HLA-DQA1*04:04-DQB1*02:03', 'HLA-DQA1*04:04-DQB1*02:04', 'HLA-DQA1*04:04-DQB1*02:05', 'HLA-DQA1*04:04-DQB1*02:06',
         'HLA-DQA1*04:04-DQB1*03:01', 'HLA-DQA1*04:04-DQB1*03:02',
         'HLA-DQA1*04:04-DQB1*03:03', 'HLA-DQA1*04:04-DQB1*03:04', 'HLA-DQA1*04:04-DQB1*03:05', 'HLA-DQA1*04:04-DQB1*03:06',
         'HLA-DQA1*04:04-DQB1*03:07', 'HLA-DQA1*04:04-DQB1*03:08',
         'HLA-DQA1*04:04-DQB1*03:09', 'HLA-DQA1*04:04-DQB1*03:10', 'HLA-DQA1*04:04-DQB1*03:11', 'HLA-DQA1*04:04-DQB1*03:12',
         'HLA-DQA1*04:04-DQB1*03:13', 'HLA-DQA1*04:04-DQB1*03:14',
         'HLA-DQA1*04:04-DQB1*03:15', 'HLA-DQA1*04:04-DQB1*03:16', 'HLA-DQA1*04:04-DQB1*03:17', 'HLA-DQA1*04:04-DQB1*03:18',
         'HLA-DQA1*04:04-DQB1*03:19', 'HLA-DQA1*04:04-DQB1*03:20',
         'HLA-DQA1*04:04-DQB1*03:21', 'HLA-DQA1*04:04-DQB1*03:22', 'HLA-DQA1*04:04-DQB1*03:23', 'HLA-DQA1*04:04-DQB1*03:24',
         'HLA-DQA1*04:04-DQB1*03:25', 'HLA-DQA1*04:04-DQB1*03:26',
         'HLA-DQA1*04:04-DQB1*03:27', 'HLA-DQA1*04:04-DQB1*03:28', 'HLA-DQA1*04:04-DQB1*03:29', 'HLA-DQA1*04:04-DQB1*03:30',
         'HLA-DQA1*04:04-DQB1*03:31', 'HLA-DQA1*04:04-DQB1*03:32',
         'HLA-DQA1*04:04-DQB1*03:33', 'HLA-DQA1*04:04-DQB1*03:34', 'HLA-DQA1*04:04-DQB1*03:35', 'HLA-DQA1*04:04-DQB1*03:36',
         'HLA-DQA1*04:04-DQB1*03:37', 'HLA-DQA1*04:04-DQB1*03:38',
         'HLA-DQA1*04:04-DQB1*04:01', 'HLA-DQA1*04:04-DQB1*04:02', 'HLA-DQA1*04:04-DQB1*04:03', 'HLA-DQA1*04:04-DQB1*04:04',
         'HLA-DQA1*04:04-DQB1*04:05', 'HLA-DQA1*04:04-DQB1*04:06',
         'HLA-DQA1*04:04-DQB1*04:07', 'HLA-DQA1*04:04-DQB1*04:08', 'HLA-DQA1*04:04-DQB1*05:01', 'HLA-DQA1*04:04-DQB1*05:02',
         'HLA-DQA1*04:04-DQB1*05:03', 'HLA-DQA1*04:04-DQB1*05:05',
         'HLA-DQA1*04:04-DQB1*05:06', 'HLA-DQA1*04:04-DQB1*05:07', 'HLA-DQA1*04:04-DQB1*05:08', 'HLA-DQA1*04:04-DQB1*05:09',
         'HLA-DQA1*04:04-DQB1*05:10', 'HLA-DQA1*04:04-DQB1*05:11',
         'HLA-DQA1*04:04-DQB1*05:12', 'HLA-DQA1*04:04-DQB1*05:13', 'HLA-DQA1*04:04-DQB1*05:14', 'HLA-DQA1*04:04-DQB1*06:01',
         'HLA-DQA1*04:04-DQB1*06:02', 'HLA-DQA1*04:04-DQB1*06:03',
         'HLA-DQA1*04:04-DQB1*06:04', 'HLA-DQA1*04:04-DQB1*06:07', 'HLA-DQA1*04:04-DQB1*06:08', 'HLA-DQA1*04:04-DQB1*06:09',
         'HLA-DQA1*04:04-DQB1*06:10', 'HLA-DQA1*04:04-DQB1*06:11',
         'HLA-DQA1*04:04-DQB1*06:12', 'HLA-DQA1*04:04-DQB1*06:14', 'HLA-DQA1*04:04-DQB1*06:15', 'HLA-DQA1*04:04-DQB1*06:16',
         'HLA-DQA1*04:04-DQB1*06:17', 'HLA-DQA1*04:04-DQB1*06:18',
         'HLA-DQA1*04:04-DQB1*06:19', 'HLA-DQA1*04:04-DQB1*06:21', 'HLA-DQA1*04:04-DQB1*06:22', 'HLA-DQA1*04:04-DQB1*06:23',
         'HLA-DQA1*04:04-DQB1*06:24', 'HLA-DQA1*04:04-DQB1*06:25',
         'HLA-DQA1*04:04-DQB1*06:27', 'HLA-DQA1*04:04-DQB1*06:28', 'HLA-DQA1*04:04-DQB1*06:29', 'HLA-DQA1*04:04-DQB1*06:30',
         'HLA-DQA1*04:04-DQB1*06:31', 'HLA-DQA1*04:04-DQB1*06:32',
         'HLA-DQA1*04:04-DQB1*06:33', 'HLA-DQA1*04:04-DQB1*06:34', 'HLA-DQA1*04:04-DQB1*06:35', 'HLA-DQA1*04:04-DQB1*06:36',
         'HLA-DQA1*04:04-DQB1*06:37', 'HLA-DQA1*04:04-DQB1*06:38',
         'HLA-DQA1*04:04-DQB1*06:39', 'HLA-DQA1*04:04-DQB1*06:40', 'HLA-DQA1*04:04-DQB1*06:41', 'HLA-DQA1*04:04-DQB1*06:42',
         'HLA-DQA1*04:04-DQB1*06:43', 'HLA-DQA1*04:04-DQB1*06:44',
         'HLA-DQA1*05:01-DQB1*02:01', 'HLA-DQA1*05:01-DQB1*02:02', 'HLA-DQA1*05:01-DQB1*02:03', 'HLA-DQA1*05:01-DQB1*02:04',
         'HLA-DQA1*05:01-DQB1*02:05', 'HLA-DQA1*05:01-DQB1*02:06',
         'HLA-DQA1*05:01-DQB1*03:01', 'HLA-DQA1*05:01-DQB1*03:02', 'HLA-DQA1*05:01-DQB1*03:03', 'HLA-DQA1*05:01-DQB1*03:04',
         'HLA-DQA1*05:01-DQB1*03:05', 'HLA-DQA1*05:01-DQB1*03:06',
         'HLA-DQA1*05:01-DQB1*03:07', 'HLA-DQA1*05:01-DQB1*03:08', 'HLA-DQA1*05:01-DQB1*03:09', 'HLA-DQA1*05:01-DQB1*03:10',
         'HLA-DQA1*05:01-DQB1*03:11', 'HLA-DQA1*05:01-DQB1*03:12',
         'HLA-DQA1*05:01-DQB1*03:13', 'HLA-DQA1*05:01-DQB1*03:14', 'HLA-DQA1*05:01-DQB1*03:15', 'HLA-DQA1*05:01-DQB1*03:16',
         'HLA-DQA1*05:01-DQB1*03:17', 'HLA-DQA1*05:01-DQB1*03:18',
         'HLA-DQA1*05:01-DQB1*03:19', 'HLA-DQA1*05:01-DQB1*03:20', 'HLA-DQA1*05:01-DQB1*03:21', 'HLA-DQA1*05:01-DQB1*03:22',
         'HLA-DQA1*05:01-DQB1*03:23', 'HLA-DQA1*05:01-DQB1*03:24',
         'HLA-DQA1*05:01-DQB1*03:25', 'HLA-DQA1*05:01-DQB1*03:26', 'HLA-DQA1*05:01-DQB1*03:27', 'HLA-DQA1*05:01-DQB1*03:28',
         'HLA-DQA1*05:01-DQB1*03:29', 'HLA-DQA1*05:01-DQB1*03:30',
         'HLA-DQA1*05:01-DQB1*03:31', 'HLA-DQA1*05:01-DQB1*03:32', 'HLA-DQA1*05:01-DQB1*03:33', 'HLA-DQA1*05:01-DQB1*03:34',
         'HLA-DQA1*05:01-DQB1*03:35', 'HLA-DQA1*05:01-DQB1*03:36',
         'HLA-DQA1*05:01-DQB1*03:37', 'HLA-DQA1*05:01-DQB1*03:38', 'HLA-DQA1*05:01-DQB1*04:01', 'HLA-DQA1*05:01-DQB1*04:02',
         'HLA-DQA1*05:01-DQB1*04:03', 'HLA-DQA1*05:01-DQB1*04:04',
         'HLA-DQA1*05:01-DQB1*04:05', 'HLA-DQA1*05:01-DQB1*04:06', 'HLA-DQA1*05:01-DQB1*04:07', 'HLA-DQA1*05:01-DQB1*04:08',
         'HLA-DQA1*05:01-DQB1*05:01', 'HLA-DQA1*05:01-DQB1*05:02',
         'HLA-DQA1*05:01-DQB1*05:03', 'HLA-DQA1*05:01-DQB1*05:05', 'HLA-DQA1*05:01-DQB1*05:06', 'HLA-DQA1*05:01-DQB1*05:07',
         'HLA-DQA1*05:01-DQB1*05:08', 'HLA-DQA1*05:01-DQB1*05:09',
         'HLA-DQA1*05:01-DQB1*05:10', 'HLA-DQA1*05:01-DQB1*05:11', 'HLA-DQA1*05:01-DQB1*05:12', 'HLA-DQA1*05:01-DQB1*05:13',
         'HLA-DQA1*05:01-DQB1*05:14', 'HLA-DQA1*05:01-DQB1*06:01',
         'HLA-DQA1*05:01-DQB1*06:02', 'HLA-DQA1*05:01-DQB1*06:03', 'HLA-DQA1*05:01-DQB1*06:04', 'HLA-DQA1*05:01-DQB1*06:07',
         'HLA-DQA1*05:01-DQB1*06:08', 'HLA-DQA1*05:01-DQB1*06:09',
         'HLA-DQA1*05:01-DQB1*06:10', 'HLA-DQA1*05:01-DQB1*06:11', 'HLA-DQA1*05:01-DQB1*06:12', 'HLA-DQA1*05:01-DQB1*06:14',
         'HLA-DQA1*05:01-DQB1*06:15', 'HLA-DQA1*05:01-DQB1*06:16',
         'HLA-DQA1*05:01-DQB1*06:17', 'HLA-DQA1*05:01-DQB1*06:18', 'HLA-DQA1*05:01-DQB1*06:19', 'HLA-DQA1*05:01-DQB1*06:21',
         'HLA-DQA1*05:01-DQB1*06:22', 'HLA-DQA1*05:01-DQB1*06:23',
         'HLA-DQA1*05:01-DQB1*06:24', 'HLA-DQA1*05:01-DQB1*06:25', 'HLA-DQA1*05:01-DQB1*06:27', 'HLA-DQA1*05:01-DQB1*06:28',
         'HLA-DQA1*05:01-DQB1*06:29', 'HLA-DQA1*05:01-DQB1*06:30',
         'HLA-DQA1*05:01-DQB1*06:31', 'HLA-DQA1*05:01-DQB1*06:32', 'HLA-DQA1*05:01-DQB1*06:33', 'HLA-DQA1*05:01-DQB1*06:34',
         'HLA-DQA1*05:01-DQB1*06:35', 'HLA-DQA1*05:01-DQB1*06:36',
         'HLA-DQA1*05:01-DQB1*06:37', 'HLA-DQA1*05:01-DQB1*06:38', 'HLA-DQA1*05:01-DQB1*06:39', 'HLA-DQA1*05:01-DQB1*06:40',
         'HLA-DQA1*05:01-DQB1*06:41', 'HLA-DQA1*05:01-DQB1*06:42',
         'HLA-DQA1*05:01-DQB1*06:43', 'HLA-DQA1*05:01-DQB1*06:44', 'HLA-DQA1*05:03-DQB1*02:01', 'HLA-DQA1*05:03-DQB1*02:02',
         'HLA-DQA1*05:03-DQB1*02:03', 'HLA-DQA1*05:03-DQB1*02:04',
         'HLA-DQA1*05:03-DQB1*02:05', 'HLA-DQA1*05:03-DQB1*02:06', 'HLA-DQA1*05:03-DQB1*03:01', 'HLA-DQA1*05:03-DQB1*03:02',
         'HLA-DQA1*05:03-DQB1*03:03', 'HLA-DQA1*05:03-DQB1*03:04',
         'HLA-DQA1*05:03-DQB1*03:05', 'HLA-DQA1*05:03-DQB1*03:06', 'HLA-DQA1*05:03-DQB1*03:07', 'HLA-DQA1*05:03-DQB1*03:08',
         'HLA-DQA1*05:03-DQB1*03:09', 'HLA-DQA1*05:03-DQB1*03:10',
         'HLA-DQA1*05:03-DQB1*03:11', 'HLA-DQA1*05:03-DQB1*03:12', 'HLA-DQA1*05:03-DQB1*03:13', 'HLA-DQA1*05:03-DQB1*03:14',
         'HLA-DQA1*05:03-DQB1*03:15', 'HLA-DQA1*05:03-DQB1*03:16',
         'HLA-DQA1*05:03-DQB1*03:17', 'HLA-DQA1*05:03-DQB1*03:18', 'HLA-DQA1*05:03-DQB1*03:19', 'HLA-DQA1*05:03-DQB1*03:20',
         'HLA-DQA1*05:03-DQB1*03:21', 'HLA-DQA1*05:03-DQB1*03:22',
         'HLA-DQA1*05:03-DQB1*03:23', 'HLA-DQA1*05:03-DQB1*03:24', 'HLA-DQA1*05:03-DQB1*03:25', 'HLA-DQA1*05:03-DQB1*03:26',
         'HLA-DQA1*05:03-DQB1*03:27', 'HLA-DQA1*05:03-DQB1*03:28',
         'HLA-DQA1*05:03-DQB1*03:29', 'HLA-DQA1*05:03-DQB1*03:30', 'HLA-DQA1*05:03-DQB1*03:31', 'HLA-DQA1*05:03-DQB1*03:32',
         'HLA-DQA1*05:03-DQB1*03:33', 'HLA-DQA1*05:03-DQB1*03:34',
         'HLA-DQA1*05:03-DQB1*03:35', 'HLA-DQA1*05:03-DQB1*03:36', 'HLA-DQA1*05:03-DQB1*03:37', 'HLA-DQA1*05:03-DQB1*03:38',
         'HLA-DQA1*05:03-DQB1*04:01', 'HLA-DQA1*05:03-DQB1*04:02',
         'HLA-DQA1*05:03-DQB1*04:03', 'HLA-DQA1*05:03-DQB1*04:04', 'HLA-DQA1*05:03-DQB1*04:05', 'HLA-DQA1*05:03-DQB1*04:06',
         'HLA-DQA1*05:03-DQB1*04:07', 'HLA-DQA1*05:03-DQB1*04:08',
         'HLA-DQA1*05:03-DQB1*05:01', 'HLA-DQA1*05:03-DQB1*05:02', 'HLA-DQA1*05:03-DQB1*05:03', 'HLA-DQA1*05:03-DQB1*05:05',
         'HLA-DQA1*05:03-DQB1*05:06', 'HLA-DQA1*05:03-DQB1*05:07',
         'HLA-DQA1*05:03-DQB1*05:08', 'HLA-DQA1*05:03-DQB1*05:09', 'HLA-DQA1*05:03-DQB1*05:10', 'HLA-DQA1*05:03-DQB1*05:11',
         'HLA-DQA1*05:03-DQB1*05:12', 'HLA-DQA1*05:03-DQB1*05:13',
         'HLA-DQA1*05:03-DQB1*05:14', 'HLA-DQA1*05:03-DQB1*06:01', 'HLA-DQA1*05:03-DQB1*06:02', 'HLA-DQA1*05:03-DQB1*06:03',
         'HLA-DQA1*05:03-DQB1*06:04', 'HLA-DQA1*05:03-DQB1*06:07',
         'HLA-DQA1*05:03-DQB1*06:08', 'HLA-DQA1*05:03-DQB1*06:09', 'HLA-DQA1*05:03-DQB1*06:10', 'HLA-DQA1*05:03-DQB1*06:11',
         'HLA-DQA1*05:03-DQB1*06:12', 'HLA-DQA1*05:03-DQB1*06:14',
         'HLA-DQA1*05:03-DQB1*06:15', 'HLA-DQA1*05:03-DQB1*06:16', 'HLA-DQA1*05:03-DQB1*06:17', 'HLA-DQA1*05:03-DQB1*06:18',
         'HLA-DQA1*05:03-DQB1*06:19', 'HLA-DQA1*05:03-DQB1*06:21',
         'HLA-DQA1*05:03-DQB1*06:22', 'HLA-DQA1*05:03-DQB1*06:23', 'HLA-DQA1*05:03-DQB1*06:24', 'HLA-DQA1*05:03-DQB1*06:25',
         'HLA-DQA1*05:03-DQB1*06:27', 'HLA-DQA1*05:03-DQB1*06:28',
         'HLA-DQA1*05:03-DQB1*06:29', 'HLA-DQA1*05:03-DQB1*06:30', 'HLA-DQA1*05:03-DQB1*06:31', 'HLA-DQA1*05:03-DQB1*06:32',
         'HLA-DQA1*05:03-DQB1*06:33', 'HLA-DQA1*05:03-DQB1*06:34',
         'HLA-DQA1*05:03-DQB1*06:35', 'HLA-DQA1*05:03-DQB1*06:36', 'HLA-DQA1*05:03-DQB1*06:37', 'HLA-DQA1*05:03-DQB1*06:38',
         'HLA-DQA1*05:03-DQB1*06:39', 'HLA-DQA1*05:03-DQB1*06:40',
         'HLA-DQA1*05:03-DQB1*06:41', 'HLA-DQA1*05:03-DQB1*06:42', 'HLA-DQA1*05:03-DQB1*06:43', 'HLA-DQA1*05:03-DQB1*06:44',
         'HLA-DQA1*05:04-DQB1*02:01', 'HLA-DQA1*05:04-DQB1*02:02',
         'HLA-DQA1*05:04-DQB1*02:03', 'HLA-DQA1*05:04-DQB1*02:04', 'HLA-DQA1*05:04-DQB1*02:05', 'HLA-DQA1*05:04-DQB1*02:06',
         'HLA-DQA1*05:04-DQB1*03:01', 'HLA-DQA1*05:04-DQB1*03:02',
         'HLA-DQA1*05:04-DQB1*03:03', 'HLA-DQA1*05:04-DQB1*03:04', 'HLA-DQA1*05:04-DQB1*03:05', 'HLA-DQA1*05:04-DQB1*03:06',
         'HLA-DQA1*05:04-DQB1*03:07', 'HLA-DQA1*05:04-DQB1*03:08',
         'HLA-DQA1*05:04-DQB1*03:09', 'HLA-DQA1*05:04-DQB1*03:10', 'HLA-DQA1*05:04-DQB1*03:11', 'HLA-DQA1*05:04-DQB1*03:12',
         'HLA-DQA1*05:04-DQB1*03:13', 'HLA-DQA1*05:04-DQB1*03:14',
         'HLA-DQA1*05:04-DQB1*03:15', 'HLA-DQA1*05:04-DQB1*03:16', 'HLA-DQA1*05:04-DQB1*03:17', 'HLA-DQA1*05:04-DQB1*03:18',
         'HLA-DQA1*05:04-DQB1*03:19', 'HLA-DQA1*05:04-DQB1*03:20',
         'HLA-DQA1*05:04-DQB1*03:21', 'HLA-DQA1*05:04-DQB1*03:22', 'HLA-DQA1*05:04-DQB1*03:23', 'HLA-DQA1*05:04-DQB1*03:24',
         'HLA-DQA1*05:04-DQB1*03:25', 'HLA-DQA1*05:04-DQB1*03:26',
         'HLA-DQA1*05:04-DQB1*03:27', 'HLA-DQA1*05:04-DQB1*03:28', 'HLA-DQA1*05:04-DQB1*03:29', 'HLA-DQA1*05:04-DQB1*03:30',
         'HLA-DQA1*05:04-DQB1*03:31', 'HLA-DQA1*05:04-DQB1*03:32',
         'HLA-DQA1*05:04-DQB1*03:33', 'HLA-DQA1*05:04-DQB1*03:34', 'HLA-DQA1*05:04-DQB1*03:35', 'HLA-DQA1*05:04-DQB1*03:36',
         'HLA-DQA1*05:04-DQB1*03:37', 'HLA-DQA1*05:04-DQB1*03:38',
         'HLA-DQA1*05:04-DQB1*04:01', 'HLA-DQA1*05:04-DQB1*04:02', 'HLA-DQA1*05:04-DQB1*04:03', 'HLA-DQA1*05:04-DQB1*04:04',
         'HLA-DQA1*05:04-DQB1*04:05', 'HLA-DQA1*05:04-DQB1*04:06',
         'HLA-DQA1*05:04-DQB1*04:07', 'HLA-DQA1*05:04-DQB1*04:08', 'HLA-DQA1*05:04-DQB1*05:01', 'HLA-DQA1*05:04-DQB1*05:02',
         'HLA-DQA1*05:04-DQB1*05:03', 'HLA-DQA1*05:04-DQB1*05:05',
         'HLA-DQA1*05:04-DQB1*05:06', 'HLA-DQA1*05:04-DQB1*05:07', 'HLA-DQA1*05:04-DQB1*05:08', 'HLA-DQA1*05:04-DQB1*05:09',
         'HLA-DQA1*05:04-DQB1*05:10', 'HLA-DQA1*05:04-DQB1*05:11',
         'HLA-DQA1*05:04-DQB1*05:12', 'HLA-DQA1*05:04-DQB1*05:13', 'HLA-DQA1*05:04-DQB1*05:14', 'HLA-DQA1*05:04-DQB1*06:01',
         'HLA-DQA1*05:04-DQB1*06:02', 'HLA-DQA1*05:04-DQB1*06:03',
         'HLA-DQA1*05:04-DQB1*06:04', 'HLA-DQA1*05:04-DQB1*06:07', 'HLA-DQA1*05:04-DQB1*06:08', 'HLA-DQA1*05:04-DQB1*06:09',
         'HLA-DQA1*05:04-DQB1*06:10', 'HLA-DQA1*05:04-DQB1*06:11',
         'HLA-DQA1*05:04-DQB1*06:12', 'HLA-DQA1*05:04-DQB1*06:14', 'HLA-DQA1*05:04-DQB1*06:15', 'HLA-DQA1*05:04-DQB1*06:16',
         'HLA-DQA1*05:04-DQB1*06:17', 'HLA-DQA1*05:04-DQB1*06:18',
         'HLA-DQA1*05:04-DQB1*06:19', 'HLA-DQA1*05:04-DQB1*06:21', 'HLA-DQA1*05:04-DQB1*06:22', 'HLA-DQA1*05:04-DQB1*06:23',
         'HLA-DQA1*05:04-DQB1*06:24', 'HLA-DQA1*05:04-DQB1*06:25',
         'HLA-DQA1*05:04-DQB1*06:27', 'HLA-DQA1*05:04-DQB1*06:28', 'HLA-DQA1*05:04-DQB1*06:29', 'HLA-DQA1*05:04-DQB1*06:30',
         'HLA-DQA1*05:04-DQB1*06:31', 'HLA-DQA1*05:04-DQB1*06:32',
         'HLA-DQA1*05:04-DQB1*06:33', 'HLA-DQA1*05:04-DQB1*06:34', 'HLA-DQA1*05:04-DQB1*06:35', 'HLA-DQA1*05:04-DQB1*06:36',
         'HLA-DQA1*05:04-DQB1*06:37', 'HLA-DQA1*05:04-DQB1*06:38',
         'HLA-DQA1*05:04-DQB1*06:39', 'HLA-DQA1*05:04-DQB1*06:40', 'HLA-DQA1*05:04-DQB1*06:41', 'HLA-DQA1*05:04-DQB1*06:42',
         'HLA-DQA1*05:04-DQB1*06:43', 'HLA-DQA1*05:04-DQB1*06:44',
         'HLA-DQA1*05:05-DQB1*02:01', 'HLA-DQA1*05:05-DQB1*02:02', 'HLA-DQA1*05:05-DQB1*02:03', 'HLA-DQA1*05:05-DQB1*02:04',
         'HLA-DQA1*05:05-DQB1*02:05', 'HLA-DQA1*05:05-DQB1*02:06',
         'HLA-DQA1*05:05-DQB1*03:01', 'HLA-DQA1*05:05-DQB1*03:02', 'HLA-DQA1*05:05-DQB1*03:03', 'HLA-DQA1*05:05-DQB1*03:04',
         'HLA-DQA1*05:05-DQB1*03:05', 'HLA-DQA1*05:05-DQB1*03:06',
         'HLA-DQA1*05:05-DQB1*03:07', 'HLA-DQA1*05:05-DQB1*03:08', 'HLA-DQA1*05:05-DQB1*03:09', 'HLA-DQA1*05:05-DQB1*03:10',
         'HLA-DQA1*05:05-DQB1*03:11', 'HLA-DQA1*05:05-DQB1*03:12',
         'HLA-DQA1*05:05-DQB1*03:13', 'HLA-DQA1*05:05-DQB1*03:14', 'HLA-DQA1*05:05-DQB1*03:15', 'HLA-DQA1*05:05-DQB1*03:16',
         'HLA-DQA1*05:05-DQB1*03:17', 'HLA-DQA1*05:05-DQB1*03:18',
         'HLA-DQA1*05:05-DQB1*03:19', 'HLA-DQA1*05:05-DQB1*03:20', 'HLA-DQA1*05:05-DQB1*03:21', 'HLA-DQA1*05:05-DQB1*03:22',
         'HLA-DQA1*05:05-DQB1*03:23', 'HLA-DQA1*05:05-DQB1*03:24',
         'HLA-DQA1*05:05-DQB1*03:25', 'HLA-DQA1*05:05-DQB1*03:26', 'HLA-DQA1*05:05-DQB1*03:27', 'HLA-DQA1*05:05-DQB1*03:28',
         'HLA-DQA1*05:05-DQB1*03:29', 'HLA-DQA1*05:05-DQB1*03:30',
         'HLA-DQA1*05:05-DQB1*03:31', 'HLA-DQA1*05:05-DQB1*03:32', 'HLA-DQA1*05:05-DQB1*03:33', 'HLA-DQA1*05:05-DQB1*03:34',
         'HLA-DQA1*05:05-DQB1*03:35', 'HLA-DQA1*05:05-DQB1*03:36',
         'HLA-DQA1*05:05-DQB1*03:37', 'HLA-DQA1*05:05-DQB1*03:38', 'HLA-DQA1*05:05-DQB1*04:01', 'HLA-DQA1*05:05-DQB1*04:02',
         'HLA-DQA1*05:05-DQB1*04:03', 'HLA-DQA1*05:05-DQB1*04:04',
         'HLA-DQA1*05:05-DQB1*04:05', 'HLA-DQA1*05:05-DQB1*04:06', 'HLA-DQA1*05:05-DQB1*04:07', 'HLA-DQA1*05:05-DQB1*04:08',
         'HLA-DQA1*05:05-DQB1*05:01', 'HLA-DQA1*05:05-DQB1*05:02',
         'HLA-DQA1*05:05-DQB1*05:03', 'HLA-DQA1*05:05-DQB1*05:05', 'HLA-DQA1*05:05-DQB1*05:06', 'HLA-DQA1*05:05-DQB1*05:07',
         'HLA-DQA1*05:05-DQB1*05:08', 'HLA-DQA1*05:05-DQB1*05:09',
         'HLA-DQA1*05:05-DQB1*05:10', 'HLA-DQA1*05:05-DQB1*05:11', 'HLA-DQA1*05:05-DQB1*05:12', 'HLA-DQA1*05:05-DQB1*05:13',
         'HLA-DQA1*05:05-DQB1*05:14', 'HLA-DQA1*05:05-DQB1*06:01',
         'HLA-DQA1*05:05-DQB1*06:02', 'HLA-DQA1*05:05-DQB1*06:03', 'HLA-DQA1*05:05-DQB1*06:04', 'HLA-DQA1*05:05-DQB1*06:07',
         'HLA-DQA1*05:05-DQB1*06:08', 'HLA-DQA1*05:05-DQB1*06:09',
         'HLA-DQA1*05:05-DQB1*06:10', 'HLA-DQA1*05:05-DQB1*06:11', 'HLA-DQA1*05:05-DQB1*06:12', 'HLA-DQA1*05:05-DQB1*06:14',
         'HLA-DQA1*05:05-DQB1*06:15', 'HLA-DQA1*05:05-DQB1*06:16',
         'HLA-DQA1*05:05-DQB1*06:17', 'HLA-DQA1*05:05-DQB1*06:18', 'HLA-DQA1*05:05-DQB1*06:19', 'HLA-DQA1*05:05-DQB1*06:21',
         'HLA-DQA1*05:05-DQB1*06:22', 'HLA-DQA1*05:05-DQB1*06:23',
         'HLA-DQA1*05:05-DQB1*06:24', 'HLA-DQA1*05:05-DQB1*06:25', 'HLA-DQA1*05:05-DQB1*06:27', 'HLA-DQA1*05:05-DQB1*06:28',
         'HLA-DQA1*05:05-DQB1*06:29', 'HLA-DQA1*05:05-DQB1*06:30',
         'HLA-DQA1*05:05-DQB1*06:31', 'HLA-DQA1*05:05-DQB1*06:32', 'HLA-DQA1*05:05-DQB1*06:33', 'HLA-DQA1*05:05-DQB1*06:34',
         'HLA-DQA1*05:05-DQB1*06:35', 'HLA-DQA1*05:05-DQB1*06:36',
         'HLA-DQA1*05:05-DQB1*06:37', 'HLA-DQA1*05:05-DQB1*06:38', 'HLA-DQA1*05:05-DQB1*06:39', 'HLA-DQA1*05:05-DQB1*06:40',
         'HLA-DQA1*05:05-DQB1*06:41', 'HLA-DQA1*05:05-DQB1*06:42',
         'HLA-DQA1*05:05-DQB1*06:43', 'HLA-DQA1*05:05-DQB1*06:44', 'HLA-DQA1*05:06-DQB1*02:01', 'HLA-DQA1*05:06-DQB1*02:02',
         'HLA-DQA1*05:06-DQB1*02:03', 'HLA-DQA1*05:06-DQB1*02:04',
         'HLA-DQA1*05:06-DQB1*02:05', 'HLA-DQA1*05:06-DQB1*02:06', 'HLA-DQA1*05:06-DQB1*03:01', 'HLA-DQA1*05:06-DQB1*03:02',
         'HLA-DQA1*05:06-DQB1*03:03', 'HLA-DQA1*05:06-DQB1*03:04',
         'HLA-DQA1*05:06-DQB1*03:05', 'HLA-DQA1*05:06-DQB1*03:06', 'HLA-DQA1*05:06-DQB1*03:07', 'HLA-DQA1*05:06-DQB1*03:08',
         'HLA-DQA1*05:06-DQB1*03:09', 'HLA-DQA1*05:06-DQB1*03:10',
         'HLA-DQA1*05:06-DQB1*03:11', 'HLA-DQA1*05:06-DQB1*03:12', 'HLA-DQA1*05:06-DQB1*03:13', 'HLA-DQA1*05:06-DQB1*03:14',
         'HLA-DQA1*05:06-DQB1*03:15', 'HLA-DQA1*05:06-DQB1*03:16',
         'HLA-DQA1*05:06-DQB1*03:17', 'HLA-DQA1*05:06-DQB1*03:18', 'HLA-DQA1*05:06-DQB1*03:19', 'HLA-DQA1*05:06-DQB1*03:20',
         'HLA-DQA1*05:06-DQB1*03:21', 'HLA-DQA1*05:06-DQB1*03:22',
         'HLA-DQA1*05:06-DQB1*03:23', 'HLA-DQA1*05:06-DQB1*03:24', 'HLA-DQA1*05:06-DQB1*03:25', 'HLA-DQA1*05:06-DQB1*03:26',
         'HLA-DQA1*05:06-DQB1*03:27', 'HLA-DQA1*05:06-DQB1*03:28',
         'HLA-DQA1*05:06-DQB1*03:29', 'HLA-DQA1*05:06-DQB1*03:30', 'HLA-DQA1*05:06-DQB1*03:31', 'HLA-DQA1*05:06-DQB1*03:32',
         'HLA-DQA1*05:06-DQB1*03:33', 'HLA-DQA1*05:06-DQB1*03:34',
         'HLA-DQA1*05:06-DQB1*03:35', 'HLA-DQA1*05:06-DQB1*03:36', 'HLA-DQA1*05:06-DQB1*03:37', 'HLA-DQA1*05:06-DQB1*03:38',
         'HLA-DQA1*05:06-DQB1*04:01', 'HLA-DQA1*05:06-DQB1*04:02',
         'HLA-DQA1*05:06-DQB1*04:03', 'HLA-DQA1*05:06-DQB1*04:04', 'HLA-DQA1*05:06-DQB1*04:05', 'HLA-DQA1*05:06-DQB1*04:06',
         'HLA-DQA1*05:06-DQB1*04:07', 'HLA-DQA1*05:06-DQB1*04:08',
         'HLA-DQA1*05:06-DQB1*05:01', 'HLA-DQA1*05:06-DQB1*05:02', 'HLA-DQA1*05:06-DQB1*05:03', 'HLA-DQA1*05:06-DQB1*05:05',
         'HLA-DQA1*05:06-DQB1*05:06', 'HLA-DQA1*05:06-DQB1*05:07',
         'HLA-DQA1*05:06-DQB1*05:08', 'HLA-DQA1*05:06-DQB1*05:09', 'HLA-DQA1*05:06-DQB1*05:10', 'HLA-DQA1*05:06-DQB1*05:11',
         'HLA-DQA1*05:06-DQB1*05:12', 'HLA-DQA1*05:06-DQB1*05:13',
         'HLA-DQA1*05:06-DQB1*05:14', 'HLA-DQA1*05:06-DQB1*06:01', 'HLA-DQA1*05:06-DQB1*06:02', 'HLA-DQA1*05:06-DQB1*06:03',
         'HLA-DQA1*05:06-DQB1*06:04', 'HLA-DQA1*05:06-DQB1*06:07',
         'HLA-DQA1*05:06-DQB1*06:08', 'HLA-DQA1*05:06-DQB1*06:09', 'HLA-DQA1*05:06-DQB1*06:10', 'HLA-DQA1*05:06-DQB1*06:11',
         'HLA-DQA1*05:06-DQB1*06:12', 'HLA-DQA1*05:06-DQB1*06:14',
         'HLA-DQA1*05:06-DQB1*06:15', 'HLA-DQA1*05:06-DQB1*06:16', 'HLA-DQA1*05:06-DQB1*06:17', 'HLA-DQA1*05:06-DQB1*06:18',
         'HLA-DQA1*05:06-DQB1*06:19', 'HLA-DQA1*05:06-DQB1*06:21',
         'HLA-DQA1*05:06-DQB1*06:22', 'HLA-DQA1*05:06-DQB1*06:23', 'HLA-DQA1*05:06-DQB1*06:24', 'HLA-DQA1*05:06-DQB1*06:25',
         'HLA-DQA1*05:06-DQB1*06:27', 'HLA-DQA1*05:06-DQB1*06:28',
         'HLA-DQA1*05:06-DQB1*06:29', 'HLA-DQA1*05:06-DQB1*06:30', 'HLA-DQA1*05:06-DQB1*06:31', 'HLA-DQA1*05:06-DQB1*06:32',
         'HLA-DQA1*05:06-DQB1*06:33', 'HLA-DQA1*05:06-DQB1*06:34',
         'HLA-DQA1*05:06-DQB1*06:35', 'HLA-DQA1*05:06-DQB1*06:36', 'HLA-DQA1*05:06-DQB1*06:37', 'HLA-DQA1*05:06-DQB1*06:38',
         'HLA-DQA1*05:06-DQB1*06:39', 'HLA-DQA1*05:06-DQB1*06:40',
         'HLA-DQA1*05:06-DQB1*06:41', 'HLA-DQA1*05:06-DQB1*06:42', 'HLA-DQA1*05:06-DQB1*06:43', 'HLA-DQA1*05:06-DQB1*06:44',
         'HLA-DQA1*05:07-DQB1*02:01', 'HLA-DQA1*05:07-DQB1*02:02',
         'HLA-DQA1*05:07-DQB1*02:03', 'HLA-DQA1*05:07-DQB1*02:04', 'HLA-DQA1*05:07-DQB1*02:05', 'HLA-DQA1*05:07-DQB1*02:06',
         'HLA-DQA1*05:07-DQB1*03:01', 'HLA-DQA1*05:07-DQB1*03:02',
         'HLA-DQA1*05:07-DQB1*03:03', 'HLA-DQA1*05:07-DQB1*03:04', 'HLA-DQA1*05:07-DQB1*03:05', 'HLA-DQA1*05:07-DQB1*03:06',
         'HLA-DQA1*05:07-DQB1*03:07', 'HLA-DQA1*05:07-DQB1*03:08',
         'HLA-DQA1*05:07-DQB1*03:09', 'HLA-DQA1*05:07-DQB1*03:10', 'HLA-DQA1*05:07-DQB1*03:11', 'HLA-DQA1*05:07-DQB1*03:12',
         'HLA-DQA1*05:07-DQB1*03:13', 'HLA-DQA1*05:07-DQB1*03:14',
         'HLA-DQA1*05:07-DQB1*03:15', 'HLA-DQA1*05:07-DQB1*03:16', 'HLA-DQA1*05:07-DQB1*03:17', 'HLA-DQA1*05:07-DQB1*03:18',
         'HLA-DQA1*05:07-DQB1*03:19', 'HLA-DQA1*05:07-DQB1*03:20',
         'HLA-DQA1*05:07-DQB1*03:21', 'HLA-DQA1*05:07-DQB1*03:22', 'HLA-DQA1*05:07-DQB1*03:23', 'HLA-DQA1*05:07-DQB1*03:24',
         'HLA-DQA1*05:07-DQB1*03:25', 'HLA-DQA1*05:07-DQB1*03:26',
         'HLA-DQA1*05:07-DQB1*03:27', 'HLA-DQA1*05:07-DQB1*03:28', 'HLA-DQA1*05:07-DQB1*03:29', 'HLA-DQA1*05:07-DQB1*03:30',
         'HLA-DQA1*05:07-DQB1*03:31', 'HLA-DQA1*05:07-DQB1*03:32',
         'HLA-DQA1*05:07-DQB1*03:33', 'HLA-DQA1*05:07-DQB1*03:34', 'HLA-DQA1*05:07-DQB1*03:35', 'HLA-DQA1*05:07-DQB1*03:36',
         'HLA-DQA1*05:07-DQB1*03:37', 'HLA-DQA1*05:07-DQB1*03:38',
         'HLA-DQA1*05:07-DQB1*04:01', 'HLA-DQA1*05:07-DQB1*04:02', 'HLA-DQA1*05:07-DQB1*04:03', 'HLA-DQA1*05:07-DQB1*04:04',
         'HLA-DQA1*05:07-DQB1*04:05', 'HLA-DQA1*05:07-DQB1*04:06',
         'HLA-DQA1*05:07-DQB1*04:07', 'HLA-DQA1*05:07-DQB1*04:08', 'HLA-DQA1*05:07-DQB1*05:01', 'HLA-DQA1*05:07-DQB1*05:02',
         'HLA-DQA1*05:07-DQB1*05:03', 'HLA-DQA1*05:07-DQB1*05:05',
         'HLA-DQA1*05:07-DQB1*05:06', 'HLA-DQA1*05:07-DQB1*05:07', 'HLA-DQA1*05:07-DQB1*05:08', 'HLA-DQA1*05:07-DQB1*05:09',
         'HLA-DQA1*05:07-DQB1*05:10', 'HLA-DQA1*05:07-DQB1*05:11',
         'HLA-DQA1*05:07-DQB1*05:12', 'HLA-DQA1*05:07-DQB1*05:13', 'HLA-DQA1*05:07-DQB1*05:14', 'HLA-DQA1*05:07-DQB1*06:01',
         'HLA-DQA1*05:07-DQB1*06:02', 'HLA-DQA1*05:07-DQB1*06:03',
         'HLA-DQA1*05:07-DQB1*06:04', 'HLA-DQA1*05:07-DQB1*06:07', 'HLA-DQA1*05:07-DQB1*06:08', 'HLA-DQA1*05:07-DQB1*06:09',
         'HLA-DQA1*05:07-DQB1*06:10', 'HLA-DQA1*05:07-DQB1*06:11',
         'HLA-DQA1*05:07-DQB1*06:12', 'HLA-DQA1*05:07-DQB1*06:14', 'HLA-DQA1*05:07-DQB1*06:15', 'HLA-DQA1*05:07-DQB1*06:16',
         'HLA-DQA1*05:07-DQB1*06:17', 'HLA-DQA1*05:07-DQB1*06:18',
         'HLA-DQA1*05:07-DQB1*06:19', 'HLA-DQA1*05:07-DQB1*06:21', 'HLA-DQA1*05:07-DQB1*06:22', 'HLA-DQA1*05:07-DQB1*06:23',
         'HLA-DQA1*05:07-DQB1*06:24', 'HLA-DQA1*05:07-DQB1*06:25',
         'HLA-DQA1*05:07-DQB1*06:27', 'HLA-DQA1*05:07-DQB1*06:28', 'HLA-DQA1*05:07-DQB1*06:29', 'HLA-DQA1*05:07-DQB1*06:30',
         'HLA-DQA1*05:07-DQB1*06:31', 'HLA-DQA1*05:07-DQB1*06:32',
         'HLA-DQA1*05:07-DQB1*06:33', 'HLA-DQA1*05:07-DQB1*06:34', 'HLA-DQA1*05:07-DQB1*06:35', 'HLA-DQA1*05:07-DQB1*06:36',
         'HLA-DQA1*05:07-DQB1*06:37', 'HLA-DQA1*05:07-DQB1*06:38',
         'HLA-DQA1*05:07-DQB1*06:39', 'HLA-DQA1*05:07-DQB1*06:40', 'HLA-DQA1*05:07-DQB1*06:41', 'HLA-DQA1*05:07-DQB1*06:42',
         'HLA-DQA1*05:07-DQB1*06:43', 'HLA-DQA1*05:07-DQB1*06:44',
         'HLA-DQA1*05:08-DQB1*02:01', 'HLA-DQA1*05:08-DQB1*02:02', 'HLA-DQA1*05:08-DQB1*02:03', 'HLA-DQA1*05:08-DQB1*02:04',
         'HLA-DQA1*05:08-DQB1*02:05', 'HLA-DQA1*05:08-DQB1*02:06',
         'HLA-DQA1*05:08-DQB1*03:01', 'HLA-DQA1*05:08-DQB1*03:02', 'HLA-DQA1*05:08-DQB1*03:03', 'HLA-DQA1*05:08-DQB1*03:04',
         'HLA-DQA1*05:08-DQB1*03:05', 'HLA-DQA1*05:08-DQB1*03:06',
         'HLA-DQA1*05:08-DQB1*03:07', 'HLA-DQA1*05:08-DQB1*03:08', 'HLA-DQA1*05:08-DQB1*03:09', 'HLA-DQA1*05:08-DQB1*03:10',
         'HLA-DQA1*05:08-DQB1*03:11', 'HLA-DQA1*05:08-DQB1*03:12',
         'HLA-DQA1*05:08-DQB1*03:13', 'HLA-DQA1*05:08-DQB1*03:14', 'HLA-DQA1*05:08-DQB1*03:15', 'HLA-DQA1*05:08-DQB1*03:16',
         'HLA-DQA1*05:08-DQB1*03:17', 'HLA-DQA1*05:08-DQB1*03:18',
         'HLA-DQA1*05:08-DQB1*03:19', 'HLA-DQA1*05:08-DQB1*03:20', 'HLA-DQA1*05:08-DQB1*03:21', 'HLA-DQA1*05:08-DQB1*03:22',
         'HLA-DQA1*05:08-DQB1*03:23', 'HLA-DQA1*05:08-DQB1*03:24',
         'HLA-DQA1*05:08-DQB1*03:25', 'HLA-DQA1*05:08-DQB1*03:26', 'HLA-DQA1*05:08-DQB1*03:27', 'HLA-DQA1*05:08-DQB1*03:28',
         'HLA-DQA1*05:08-DQB1*03:29', 'HLA-DQA1*05:08-DQB1*03:30',
         'HLA-DQA1*05:08-DQB1*03:31', 'HLA-DQA1*05:08-DQB1*03:32', 'HLA-DQA1*05:08-DQB1*03:33', 'HLA-DQA1*05:08-DQB1*03:34',
         'HLA-DQA1*05:08-DQB1*03:35', 'HLA-DQA1*05:08-DQB1*03:36',
         'HLA-DQA1*05:08-DQB1*03:37', 'HLA-DQA1*05:08-DQB1*03:38', 'HLA-DQA1*05:08-DQB1*04:01', 'HLA-DQA1*05:08-DQB1*04:02',
         'HLA-DQA1*05:08-DQB1*04:03', 'HLA-DQA1*05:08-DQB1*04:04',
         'HLA-DQA1*05:08-DQB1*04:05', 'HLA-DQA1*05:08-DQB1*04:06', 'HLA-DQA1*05:08-DQB1*04:07', 'HLA-DQA1*05:08-DQB1*04:08',
         'HLA-DQA1*05:08-DQB1*05:01', 'HLA-DQA1*05:08-DQB1*05:02',
         'HLA-DQA1*05:08-DQB1*05:03', 'HLA-DQA1*05:08-DQB1*05:05', 'HLA-DQA1*05:08-DQB1*05:06', 'HLA-DQA1*05:08-DQB1*05:07',
         'HLA-DQA1*05:08-DQB1*05:08', 'HLA-DQA1*05:08-DQB1*05:09',
         'HLA-DQA1*05:08-DQB1*05:10', 'HLA-DQA1*05:08-DQB1*05:11', 'HLA-DQA1*05:08-DQB1*05:12', 'HLA-DQA1*05:08-DQB1*05:13',
         'HLA-DQA1*05:08-DQB1*05:14', 'HLA-DQA1*05:08-DQB1*06:01',
         'HLA-DQA1*05:08-DQB1*06:02', 'HLA-DQA1*05:08-DQB1*06:03', 'HLA-DQA1*05:08-DQB1*06:04', 'HLA-DQA1*05:08-DQB1*06:07',
         'HLA-DQA1*05:08-DQB1*06:08', 'HLA-DQA1*05:08-DQB1*06:09',
         'HLA-DQA1*05:08-DQB1*06:10', 'HLA-DQA1*05:08-DQB1*06:11', 'HLA-DQA1*05:08-DQB1*06:12', 'HLA-DQA1*05:08-DQB1*06:14',
         'HLA-DQA1*05:08-DQB1*06:15', 'HLA-DQA1*05:08-DQB1*06:16',
         'HLA-DQA1*05:08-DQB1*06:17', 'HLA-DQA1*05:08-DQB1*06:18', 'HLA-DQA1*05:08-DQB1*06:19', 'HLA-DQA1*05:08-DQB1*06:21',
         'HLA-DQA1*05:08-DQB1*06:22', 'HLA-DQA1*05:08-DQB1*06:23',
         'HLA-DQA1*05:08-DQB1*06:24', 'HLA-DQA1*05:08-DQB1*06:25', 'HLA-DQA1*05:08-DQB1*06:27', 'HLA-DQA1*05:08-DQB1*06:28',
         'HLA-DQA1*05:08-DQB1*06:29', 'HLA-DQA1*05:08-DQB1*06:30',
         'HLA-DQA1*05:08-DQB1*06:31', 'HLA-DQA1*05:08-DQB1*06:32', 'HLA-DQA1*05:08-DQB1*06:33', 'HLA-DQA1*05:08-DQB1*06:34',
         'HLA-DQA1*05:08-DQB1*06:35', 'HLA-DQA1*05:08-DQB1*06:36',
         'HLA-DQA1*05:08-DQB1*06:37', 'HLA-DQA1*05:08-DQB1*06:38', 'HLA-DQA1*05:08-DQB1*06:39', 'HLA-DQA1*05:08-DQB1*06:40',
         'HLA-DQA1*05:08-DQB1*06:41', 'HLA-DQA1*05:08-DQB1*06:42',
         'HLA-DQA1*05:08-DQB1*06:43', 'HLA-DQA1*05:08-DQB1*06:44', 'HLA-DQA1*05:09-DQB1*02:01', 'HLA-DQA1*05:09-DQB1*02:02',
         'HLA-DQA1*05:09-DQB1*02:03', 'HLA-DQA1*05:09-DQB1*02:04',
         'HLA-DQA1*05:09-DQB1*02:05', 'HLA-DQA1*05:09-DQB1*02:06', 'HLA-DQA1*05:09-DQB1*03:01', 'HLA-DQA1*05:09-DQB1*03:02',
         'HLA-DQA1*05:09-DQB1*03:03', 'HLA-DQA1*05:09-DQB1*03:04',
         'HLA-DQA1*05:09-DQB1*03:05', 'HLA-DQA1*05:09-DQB1*03:06', 'HLA-DQA1*05:09-DQB1*03:07', 'HLA-DQA1*05:09-DQB1*03:08',
         'HLA-DQA1*05:09-DQB1*03:09', 'HLA-DQA1*05:09-DQB1*03:10',
         'HLA-DQA1*05:09-DQB1*03:11', 'HLA-DQA1*05:09-DQB1*03:12', 'HLA-DQA1*05:09-DQB1*03:13', 'HLA-DQA1*05:09-DQB1*03:14',
         'HLA-DQA1*05:09-DQB1*03:15', 'HLA-DQA1*05:09-DQB1*03:16',
         'HLA-DQA1*05:09-DQB1*03:17', 'HLA-DQA1*05:09-DQB1*03:18', 'HLA-DQA1*05:09-DQB1*03:19', 'HLA-DQA1*05:09-DQB1*03:20',
         'HLA-DQA1*05:09-DQB1*03:21', 'HLA-DQA1*05:09-DQB1*03:22',
         'HLA-DQA1*05:09-DQB1*03:23', 'HLA-DQA1*05:09-DQB1*03:24', 'HLA-DQA1*05:09-DQB1*03:25', 'HLA-DQA1*05:09-DQB1*03:26',
         'HLA-DQA1*05:09-DQB1*03:27', 'HLA-DQA1*05:09-DQB1*03:28',
         'HLA-DQA1*05:09-DQB1*03:29', 'HLA-DQA1*05:09-DQB1*03:30', 'HLA-DQA1*05:09-DQB1*03:31', 'HLA-DQA1*05:09-DQB1*03:32',
         'HLA-DQA1*05:09-DQB1*03:33', 'HLA-DQA1*05:09-DQB1*03:34',
         'HLA-DQA1*05:09-DQB1*03:35', 'HLA-DQA1*05:09-DQB1*03:36', 'HLA-DQA1*05:09-DQB1*03:37', 'HLA-DQA1*05:09-DQB1*03:38',
         'HLA-DQA1*05:09-DQB1*04:01', 'HLA-DQA1*05:09-DQB1*04:02',
         'HLA-DQA1*05:09-DQB1*04:03', 'HLA-DQA1*05:09-DQB1*04:04', 'HLA-DQA1*05:09-DQB1*04:05', 'HLA-DQA1*05:09-DQB1*04:06',
         'HLA-DQA1*05:09-DQB1*04:07', 'HLA-DQA1*05:09-DQB1*04:08',
         'HLA-DQA1*05:09-DQB1*05:01', 'HLA-DQA1*05:09-DQB1*05:02', 'HLA-DQA1*05:09-DQB1*05:03', 'HLA-DQA1*05:09-DQB1*05:05',
         'HLA-DQA1*05:09-DQB1*05:06', 'HLA-DQA1*05:09-DQB1*05:07',
         'HLA-DQA1*05:09-DQB1*05:08', 'HLA-DQA1*05:09-DQB1*05:09', 'HLA-DQA1*05:09-DQB1*05:10', 'HLA-DQA1*05:09-DQB1*05:11',
         'HLA-DQA1*05:09-DQB1*05:12', 'HLA-DQA1*05:09-DQB1*05:13',
         'HLA-DQA1*05:09-DQB1*05:14', 'HLA-DQA1*05:09-DQB1*06:01', 'HLA-DQA1*05:09-DQB1*06:02', 'HLA-DQA1*05:09-DQB1*06:03',
         'HLA-DQA1*05:09-DQB1*06:04', 'HLA-DQA1*05:09-DQB1*06:07',
         'HLA-DQA1*05:09-DQB1*06:08', 'HLA-DQA1*05:09-DQB1*06:09', 'HLA-DQA1*05:09-DQB1*06:10', 'HLA-DQA1*05:09-DQB1*06:11',
         'HLA-DQA1*05:09-DQB1*06:12', 'HLA-DQA1*05:09-DQB1*06:14',
         'HLA-DQA1*05:09-DQB1*06:15', 'HLA-DQA1*05:09-DQB1*06:16', 'HLA-DQA1*05:09-DQB1*06:17', 'HLA-DQA1*05:09-DQB1*06:18',
         'HLA-DQA1*05:09-DQB1*06:19', 'HLA-DQA1*05:09-DQB1*06:21',
         'HLA-DQA1*05:09-DQB1*06:22', 'HLA-DQA1*05:09-DQB1*06:23', 'HLA-DQA1*05:09-DQB1*06:24', 'HLA-DQA1*05:09-DQB1*06:25',
         'HLA-DQA1*05:09-DQB1*06:27', 'HLA-DQA1*05:09-DQB1*06:28',
         'HLA-DQA1*05:09-DQB1*06:29', 'HLA-DQA1*05:09-DQB1*06:30', 'HLA-DQA1*05:09-DQB1*06:31', 'HLA-DQA1*05:09-DQB1*06:32',
         'HLA-DQA1*05:09-DQB1*06:33', 'HLA-DQA1*05:09-DQB1*06:34',
         'HLA-DQA1*05:09-DQB1*06:35', 'HLA-DQA1*05:09-DQB1*06:36', 'HLA-DQA1*05:09-DQB1*06:37', 'HLA-DQA1*05:09-DQB1*06:38',
         'HLA-DQA1*05:09-DQB1*06:39', 'HLA-DQA1*05:09-DQB1*06:40',
         'HLA-DQA1*05:09-DQB1*06:41', 'HLA-DQA1*05:09-DQB1*06:42', 'HLA-DQA1*05:09-DQB1*06:43', 'HLA-DQA1*05:09-DQB1*06:44',
         'HLA-DQA1*05:10-DQB1*02:01', 'HLA-DQA1*05:10-DQB1*02:02',
         'HLA-DQA1*05:10-DQB1*02:03', 'HLA-DQA1*05:10-DQB1*02:04', 'HLA-DQA1*05:10-DQB1*02:05', 'HLA-DQA1*05:10-DQB1*02:06',
         'HLA-DQA1*05:10-DQB1*03:01', 'HLA-DQA1*05:10-DQB1*03:02',
         'HLA-DQA1*05:10-DQB1*03:03', 'HLA-DQA1*05:10-DQB1*03:04', 'HLA-DQA1*05:10-DQB1*03:05', 'HLA-DQA1*05:10-DQB1*03:06',
         'HLA-DQA1*05:10-DQB1*03:07', 'HLA-DQA1*05:10-DQB1*03:08',
         'HLA-DQA1*05:10-DQB1*03:09', 'HLA-DQA1*05:10-DQB1*03:10', 'HLA-DQA1*05:10-DQB1*03:11', 'HLA-DQA1*05:10-DQB1*03:12',
         'HLA-DQA1*05:10-DQB1*03:13', 'HLA-DQA1*05:10-DQB1*03:14',
         'HLA-DQA1*05:10-DQB1*03:15', 'HLA-DQA1*05:10-DQB1*03:16', 'HLA-DQA1*05:10-DQB1*03:17', 'HLA-DQA1*05:10-DQB1*03:18',
         'HLA-DQA1*05:10-DQB1*03:19', 'HLA-DQA1*05:10-DQB1*03:20',
         'HLA-DQA1*05:10-DQB1*03:21', 'HLA-DQA1*05:10-DQB1*03:22', 'HLA-DQA1*05:10-DQB1*03:23', 'HLA-DQA1*05:10-DQB1*03:24',
         'HLA-DQA1*05:10-DQB1*03:25', 'HLA-DQA1*05:10-DQB1*03:26',
         'HLA-DQA1*05:10-DQB1*03:27', 'HLA-DQA1*05:10-DQB1*03:28', 'HLA-DQA1*05:10-DQB1*03:29', 'HLA-DQA1*05:10-DQB1*03:30',
         'HLA-DQA1*05:10-DQB1*03:31', 'HLA-DQA1*05:10-DQB1*03:32',
         'HLA-DQA1*05:10-DQB1*03:33', 'HLA-DQA1*05:10-DQB1*03:34', 'HLA-DQA1*05:10-DQB1*03:35', 'HLA-DQA1*05:10-DQB1*03:36',
         'HLA-DQA1*05:10-DQB1*03:37', 'HLA-DQA1*05:10-DQB1*03:38',
         'HLA-DQA1*05:10-DQB1*04:01', 'HLA-DQA1*05:10-DQB1*04:02', 'HLA-DQA1*05:10-DQB1*04:03', 'HLA-DQA1*05:10-DQB1*04:04',
         'HLA-DQA1*05:10-DQB1*04:05', 'HLA-DQA1*05:10-DQB1*04:06',
         'HLA-DQA1*05:10-DQB1*04:07', 'HLA-DQA1*05:10-DQB1*04:08', 'HLA-DQA1*05:10-DQB1*05:01', 'HLA-DQA1*05:10-DQB1*05:02',
         'HLA-DQA1*05:10-DQB1*05:03', 'HLA-DQA1*05:10-DQB1*05:05',
         'HLA-DQA1*05:10-DQB1*05:06', 'HLA-DQA1*05:10-DQB1*05:07', 'HLA-DQA1*05:10-DQB1*05:08', 'HLA-DQA1*05:10-DQB1*05:09',
         'HLA-DQA1*05:10-DQB1*05:10', 'HLA-DQA1*05:10-DQB1*05:11',
         'HLA-DQA1*05:10-DQB1*05:12', 'HLA-DQA1*05:10-DQB1*05:13', 'HLA-DQA1*05:10-DQB1*05:14', 'HLA-DQA1*05:10-DQB1*06:01',
         'HLA-DQA1*05:10-DQB1*06:02', 'HLA-DQA1*05:10-DQB1*06:03',
         'HLA-DQA1*05:10-DQB1*06:04', 'HLA-DQA1*05:10-DQB1*06:07', 'HLA-DQA1*05:10-DQB1*06:08', 'HLA-DQA1*05:10-DQB1*06:09',
         'HLA-DQA1*05:10-DQB1*06:10', 'HLA-DQA1*05:10-DQB1*06:11',
         'HLA-DQA1*05:10-DQB1*06:12', 'HLA-DQA1*05:10-DQB1*06:14', 'HLA-DQA1*05:10-DQB1*06:15', 'HLA-DQA1*05:10-DQB1*06:16',
         'HLA-DQA1*05:10-DQB1*06:17', 'HLA-DQA1*05:10-DQB1*06:18',
         'HLA-DQA1*05:10-DQB1*06:19', 'HLA-DQA1*05:10-DQB1*06:21', 'HLA-DQA1*05:10-DQB1*06:22', 'HLA-DQA1*05:10-DQB1*06:23',
         'HLA-DQA1*05:10-DQB1*06:24', 'HLA-DQA1*05:10-DQB1*06:25',
         'HLA-DQA1*05:10-DQB1*06:27', 'HLA-DQA1*05:10-DQB1*06:28', 'HLA-DQA1*05:10-DQB1*06:29', 'HLA-DQA1*05:10-DQB1*06:30',
         'HLA-DQA1*05:10-DQB1*06:31', 'HLA-DQA1*05:10-DQB1*06:32',
         'HLA-DQA1*05:10-DQB1*06:33', 'HLA-DQA1*05:10-DQB1*06:34', 'HLA-DQA1*05:10-DQB1*06:35', 'HLA-DQA1*05:10-DQB1*06:36',
         'HLA-DQA1*05:10-DQB1*06:37', 'HLA-DQA1*05:10-DQB1*06:38',
         'HLA-DQA1*05:10-DQB1*06:39', 'HLA-DQA1*05:10-DQB1*06:40', 'HLA-DQA1*05:10-DQB1*06:41', 'HLA-DQA1*05:10-DQB1*06:42',
         'HLA-DQA1*05:10-DQB1*06:43', 'HLA-DQA1*05:10-DQB1*06:44',
         'HLA-DQA1*05:11-DQB1*02:01', 'HLA-DQA1*05:11-DQB1*02:02', 'HLA-DQA1*05:11-DQB1*02:03', 'HLA-DQA1*05:11-DQB1*02:04',
         'HLA-DQA1*05:11-DQB1*02:05', 'HLA-DQA1*05:11-DQB1*02:06',
         'HLA-DQA1*05:11-DQB1*03:01', 'HLA-DQA1*05:11-DQB1*03:02', 'HLA-DQA1*05:11-DQB1*03:03', 'HLA-DQA1*05:11-DQB1*03:04',
         'HLA-DQA1*05:11-DQB1*03:05', 'HLA-DQA1*05:11-DQB1*03:06',
         'HLA-DQA1*05:11-DQB1*03:07', 'HLA-DQA1*05:11-DQB1*03:08', 'HLA-DQA1*05:11-DQB1*03:09', 'HLA-DQA1*05:11-DQB1*03:10',
         'HLA-DQA1*05:11-DQB1*03:11', 'HLA-DQA1*05:11-DQB1*03:12',
         'HLA-DQA1*05:11-DQB1*03:13', 'HLA-DQA1*05:11-DQB1*03:14', 'HLA-DQA1*05:11-DQB1*03:15', 'HLA-DQA1*05:11-DQB1*03:16',
         'HLA-DQA1*05:11-DQB1*03:17', 'HLA-DQA1*05:11-DQB1*03:18',
         'HLA-DQA1*05:11-DQB1*03:19', 'HLA-DQA1*05:11-DQB1*03:20', 'HLA-DQA1*05:11-DQB1*03:21', 'HLA-DQA1*05:11-DQB1*03:22',
         'HLA-DQA1*05:11-DQB1*03:23', 'HLA-DQA1*05:11-DQB1*03:24',
         'HLA-DQA1*05:11-DQB1*03:25', 'HLA-DQA1*05:11-DQB1*03:26', 'HLA-DQA1*05:11-DQB1*03:27', 'HLA-DQA1*05:11-DQB1*03:28',
         'HLA-DQA1*05:11-DQB1*03:29', 'HLA-DQA1*05:11-DQB1*03:30',
         'HLA-DQA1*05:11-DQB1*03:31', 'HLA-DQA1*05:11-DQB1*03:32', 'HLA-DQA1*05:11-DQB1*03:33', 'HLA-DQA1*05:11-DQB1*03:34',
         'HLA-DQA1*05:11-DQB1*03:35', 'HLA-DQA1*05:11-DQB1*03:36',
         'HLA-DQA1*05:11-DQB1*03:37', 'HLA-DQA1*05:11-DQB1*03:38', 'HLA-DQA1*05:11-DQB1*04:01', 'HLA-DQA1*05:11-DQB1*04:02',
         'HLA-DQA1*05:11-DQB1*04:03', 'HLA-DQA1*05:11-DQB1*04:04',
         'HLA-DQA1*05:11-DQB1*04:05', 'HLA-DQA1*05:11-DQB1*04:06', 'HLA-DQA1*05:11-DQB1*04:07', 'HLA-DQA1*05:11-DQB1*04:08',
         'HLA-DQA1*05:11-DQB1*05:01', 'HLA-DQA1*05:11-DQB1*05:02',
         'HLA-DQA1*05:11-DQB1*05:03', 'HLA-DQA1*05:11-DQB1*05:05', 'HLA-DQA1*05:11-DQB1*05:06', 'HLA-DQA1*05:11-DQB1*05:07',
         'HLA-DQA1*05:11-DQB1*05:08', 'HLA-DQA1*05:11-DQB1*05:09',
         'HLA-DQA1*05:11-DQB1*05:10', 'HLA-DQA1*05:11-DQB1*05:11', 'HLA-DQA1*05:11-DQB1*05:12', 'HLA-DQA1*05:11-DQB1*05:13',
         'HLA-DQA1*05:11-DQB1*05:14', 'HLA-DQA1*05:11-DQB1*06:01',
         'HLA-DQA1*05:11-DQB1*06:02', 'HLA-DQA1*05:11-DQB1*06:03', 'HLA-DQA1*05:11-DQB1*06:04', 'HLA-DQA1*05:11-DQB1*06:07',
         'HLA-DQA1*05:11-DQB1*06:08', 'HLA-DQA1*05:11-DQB1*06:09',
         'HLA-DQA1*05:11-DQB1*06:10', 'HLA-DQA1*05:11-DQB1*06:11', 'HLA-DQA1*05:11-DQB1*06:12', 'HLA-DQA1*05:11-DQB1*06:14',
         'HLA-DQA1*05:11-DQB1*06:15', 'HLA-DQA1*05:11-DQB1*06:16',
         'HLA-DQA1*05:11-DQB1*06:17', 'HLA-DQA1*05:11-DQB1*06:18', 'HLA-DQA1*05:11-DQB1*06:19', 'HLA-DQA1*05:11-DQB1*06:21',
         'HLA-DQA1*05:11-DQB1*06:22', 'HLA-DQA1*05:11-DQB1*06:23',
         'HLA-DQA1*05:11-DQB1*06:24', 'HLA-DQA1*05:11-DQB1*06:25', 'HLA-DQA1*05:11-DQB1*06:27', 'HLA-DQA1*05:11-DQB1*06:28',
         'HLA-DQA1*05:11-DQB1*06:29', 'HLA-DQA1*05:11-DQB1*06:30',
         'HLA-DQA1*05:11-DQB1*06:31', 'HLA-DQA1*05:11-DQB1*06:32', 'HLA-DQA1*05:11-DQB1*06:33', 'HLA-DQA1*05:11-DQB1*06:34',
         'HLA-DQA1*05:11-DQB1*06:35', 'HLA-DQA1*05:11-DQB1*06:36',
         'HLA-DQA1*05:11-DQB1*06:37', 'HLA-DQA1*05:11-DQB1*06:38', 'HLA-DQA1*05:11-DQB1*06:39', 'HLA-DQA1*05:11-DQB1*06:40',
         'HLA-DQA1*05:11-DQB1*06:41', 'HLA-DQA1*05:11-DQB1*06:42',
         'HLA-DQA1*05:11-DQB1*06:43', 'HLA-DQA1*05:11-DQB1*06:44', 'HLA-DQA1*06:01-DQB1*02:01', 'HLA-DQA1*06:01-DQB1*02:02',
         'HLA-DQA1*06:01-DQB1*02:03', 'HLA-DQA1*06:01-DQB1*02:04',
         'HLA-DQA1*06:01-DQB1*02:05', 'HLA-DQA1*06:01-DQB1*02:06', 'HLA-DQA1*06:01-DQB1*03:01', 'HLA-DQA1*06:01-DQB1*03:02',
         'HLA-DQA1*06:01-DQB1*03:03', 'HLA-DQA1*06:01-DQB1*03:04',
         'HLA-DQA1*06:01-DQB1*03:05', 'HLA-DQA1*06:01-DQB1*03:06', 'HLA-DQA1*06:01-DQB1*03:07', 'HLA-DQA1*06:01-DQB1*03:08',
         'HLA-DQA1*06:01-DQB1*03:09', 'HLA-DQA1*06:01-DQB1*03:10',
         'HLA-DQA1*06:01-DQB1*03:11', 'HLA-DQA1*06:01-DQB1*03:12', 'HLA-DQA1*06:01-DQB1*03:13', 'HLA-DQA1*06:01-DQB1*03:14',
         'HLA-DQA1*06:01-DQB1*03:15', 'HLA-DQA1*06:01-DQB1*03:16',
         'HLA-DQA1*06:01-DQB1*03:17', 'HLA-DQA1*06:01-DQB1*03:18', 'HLA-DQA1*06:01-DQB1*03:19', 'HLA-DQA1*06:01-DQB1*03:20',
         'HLA-DQA1*06:01-DQB1*03:21', 'HLA-DQA1*06:01-DQB1*03:22',
         'HLA-DQA1*06:01-DQB1*03:23', 'HLA-DQA1*06:01-DQB1*03:24', 'HLA-DQA1*06:01-DQB1*03:25', 'HLA-DQA1*06:01-DQB1*03:26',
         'HLA-DQA1*06:01-DQB1*03:27', 'HLA-DQA1*06:01-DQB1*03:28',
         'HLA-DQA1*06:01-DQB1*03:29', 'HLA-DQA1*06:01-DQB1*03:30', 'HLA-DQA1*06:01-DQB1*03:31', 'HLA-DQA1*06:01-DQB1*03:32',
         'HLA-DQA1*06:01-DQB1*03:33', 'HLA-DQA1*06:01-DQB1*03:34',
         'HLA-DQA1*06:01-DQB1*03:35', 'HLA-DQA1*06:01-DQB1*03:36', 'HLA-DQA1*06:01-DQB1*03:37', 'HLA-DQA1*06:01-DQB1*03:38',
         'HLA-DQA1*06:01-DQB1*04:01', 'HLA-DQA1*06:01-DQB1*04:02',
         'HLA-DQA1*06:01-DQB1*04:03', 'HLA-DQA1*06:01-DQB1*04:04', 'HLA-DQA1*06:01-DQB1*04:05', 'HLA-DQA1*06:01-DQB1*04:06',
         'HLA-DQA1*06:01-DQB1*04:07', 'HLA-DQA1*06:01-DQB1*04:08',
         'HLA-DQA1*06:01-DQB1*05:01', 'HLA-DQA1*06:01-DQB1*05:02', 'HLA-DQA1*06:01-DQB1*05:03', 'HLA-DQA1*06:01-DQB1*05:05',
         'HLA-DQA1*06:01-DQB1*05:06', 'HLA-DQA1*06:01-DQB1*05:07',
         'HLA-DQA1*06:01-DQB1*05:08', 'HLA-DQA1*06:01-DQB1*05:09', 'HLA-DQA1*06:01-DQB1*05:10', 'HLA-DQA1*06:01-DQB1*05:11',
         'HLA-DQA1*06:01-DQB1*05:12', 'HLA-DQA1*06:01-DQB1*05:13',
         'HLA-DQA1*06:01-DQB1*05:14', 'HLA-DQA1*06:01-DQB1*06:01', 'HLA-DQA1*06:01-DQB1*06:02', 'HLA-DQA1*06:01-DQB1*06:03',
         'HLA-DQA1*06:01-DQB1*06:04', 'HLA-DQA1*06:01-DQB1*06:07',
         'HLA-DQA1*06:01-DQB1*06:08', 'HLA-DQA1*06:01-DQB1*06:09', 'HLA-DQA1*06:01-DQB1*06:10', 'HLA-DQA1*06:01-DQB1*06:11',
         'HLA-DQA1*06:01-DQB1*06:12', 'HLA-DQA1*06:01-DQB1*06:14',
         'HLA-DQA1*06:01-DQB1*06:15', 'HLA-DQA1*06:01-DQB1*06:16', 'HLA-DQA1*06:01-DQB1*06:17', 'HLA-DQA1*06:01-DQB1*06:18',
         'HLA-DQA1*06:01-DQB1*06:19', 'HLA-DQA1*06:01-DQB1*06:21',
         'HLA-DQA1*06:01-DQB1*06:22', 'HLA-DQA1*06:01-DQB1*06:23', 'HLA-DQA1*06:01-DQB1*06:24', 'HLA-DQA1*06:01-DQB1*06:25',
         'HLA-DQA1*06:01-DQB1*06:27', 'HLA-DQA1*06:01-DQB1*06:28',
         'HLA-DQA1*06:01-DQB1*06:29', 'HLA-DQA1*06:01-DQB1*06:30', 'HLA-DQA1*06:01-DQB1*06:31', 'HLA-DQA1*06:01-DQB1*06:32',
         'HLA-DQA1*06:01-DQB1*06:33', 'HLA-DQA1*06:01-DQB1*06:34',
         'HLA-DQA1*06:01-DQB1*06:35', 'HLA-DQA1*06:01-DQB1*06:36', 'HLA-DQA1*06:01-DQB1*06:37', 'HLA-DQA1*06:01-DQB1*06:38',
         'HLA-DQA1*06:01-DQB1*06:39', 'HLA-DQA1*06:01-DQB1*06:40',
         'HLA-DQA1*06:01-DQB1*06:41', 'HLA-DQA1*06:01-DQB1*06:42', 'HLA-DQA1*06:01-DQB1*06:43', 'HLA-DQA1*06:01-DQB1*06:44',
         'HLA-DQA1*06:02-DQB1*02:01', 'HLA-DQA1*06:02-DQB1*02:02',
         'HLA-DQA1*06:02-DQB1*02:03', 'HLA-DQA1*06:02-DQB1*02:04', 'HLA-DQA1*06:02-DQB1*02:05', 'HLA-DQA1*06:02-DQB1*02:06',
         'HLA-DQA1*06:02-DQB1*03:01', 'HLA-DQA1*06:02-DQB1*03:02',
         'HLA-DQA1*06:02-DQB1*03:03', 'HLA-DQA1*06:02-DQB1*03:04', 'HLA-DQA1*06:02-DQB1*03:05', 'HLA-DQA1*06:02-DQB1*03:06',
         'HLA-DQA1*06:02-DQB1*03:07', 'HLA-DQA1*06:02-DQB1*03:08',
         'HLA-DQA1*06:02-DQB1*03:09', 'HLA-DQA1*06:02-DQB1*03:10', 'HLA-DQA1*06:02-DQB1*03:11', 'HLA-DQA1*06:02-DQB1*03:12',
         'HLA-DQA1*06:02-DQB1*03:13', 'HLA-DQA1*06:02-DQB1*03:14',
         'HLA-DQA1*06:02-DQB1*03:15', 'HLA-DQA1*06:02-DQB1*03:16', 'HLA-DQA1*06:02-DQB1*03:17', 'HLA-DQA1*06:02-DQB1*03:18',
         'HLA-DQA1*06:02-DQB1*03:19', 'HLA-DQA1*06:02-DQB1*03:20',
         'HLA-DQA1*06:02-DQB1*03:21', 'HLA-DQA1*06:02-DQB1*03:22', 'HLA-DQA1*06:02-DQB1*03:23', 'HLA-DQA1*06:02-DQB1*03:24',
         'HLA-DQA1*06:02-DQB1*03:25', 'HLA-DQA1*06:02-DQB1*03:26',
         'HLA-DQA1*06:02-DQB1*03:27', 'HLA-DQA1*06:02-DQB1*03:28', 'HLA-DQA1*06:02-DQB1*03:29', 'HLA-DQA1*06:02-DQB1*03:30',
         'HLA-DQA1*06:02-DQB1*03:31', 'HLA-DQA1*06:02-DQB1*03:32',
         'HLA-DQA1*06:02-DQB1*03:33', 'HLA-DQA1*06:02-DQB1*03:34', 'HLA-DQA1*06:02-DQB1*03:35', 'HLA-DQA1*06:02-DQB1*03:36',
         'HLA-DQA1*06:02-DQB1*03:37', 'HLA-DQA1*06:02-DQB1*03:38',
         'HLA-DQA1*06:02-DQB1*04:01', 'HLA-DQA1*06:02-DQB1*04:02', 'HLA-DQA1*06:02-DQB1*04:03', 'HLA-DQA1*06:02-DQB1*04:04',
         'HLA-DQA1*06:02-DQB1*04:05', 'HLA-DQA1*06:02-DQB1*04:06',
         'HLA-DQA1*06:02-DQB1*04:07', 'HLA-DQA1*06:02-DQB1*04:08', 'HLA-DQA1*06:02-DQB1*05:01', 'HLA-DQA1*06:02-DQB1*05:02',
         'HLA-DQA1*06:02-DQB1*05:03', 'HLA-DQA1*06:02-DQB1*05:05',
         'HLA-DQA1*06:02-DQB1*05:06', 'HLA-DQA1*06:02-DQB1*05:07', 'HLA-DQA1*06:02-DQB1*05:08', 'HLA-DQA1*06:02-DQB1*05:09',
         'HLA-DQA1*06:02-DQB1*05:10', 'HLA-DQA1*06:02-DQB1*05:11',
         'HLA-DQA1*06:02-DQB1*05:12', 'HLA-DQA1*06:02-DQB1*05:13', 'HLA-DQA1*06:02-DQB1*05:14', 'HLA-DQA1*06:02-DQB1*06:01',
         'HLA-DQA1*06:02-DQB1*06:02', 'HLA-DQA1*06:02-DQB1*06:03',
         'HLA-DQA1*06:02-DQB1*06:04', 'HLA-DQA1*06:02-DQB1*06:07', 'HLA-DQA1*06:02-DQB1*06:08', 'HLA-DQA1*06:02-DQB1*06:09',
         'HLA-DQA1*06:02-DQB1*06:10', 'HLA-DQA1*06:02-DQB1*06:11',
         'HLA-DQA1*06:02-DQB1*06:12', 'HLA-DQA1*06:02-DQB1*06:14', 'HLA-DQA1*06:02-DQB1*06:15', 'HLA-DQA1*06:02-DQB1*06:16',
         'HLA-DQA1*06:02-DQB1*06:17', 'HLA-DQA1*06:02-DQB1*06:18',
         'HLA-DQA1*06:02-DQB1*06:19', 'HLA-DQA1*06:02-DQB1*06:21', 'HLA-DQA1*06:02-DQB1*06:22', 'HLA-DQA1*06:02-DQB1*06:23',
         'HLA-DQA1*06:02-DQB1*06:24', 'HLA-DQA1*06:02-DQB1*06:25',
         'HLA-DQA1*06:02-DQB1*06:27', 'HLA-DQA1*06:02-DQB1*06:28', 'HLA-DQA1*06:02-DQB1*06:29', 'HLA-DQA1*06:02-DQB1*06:30',
         'HLA-DQA1*06:02-DQB1*06:31', 'HLA-DQA1*06:02-DQB1*06:32',
         'HLA-DQA1*06:02-DQB1*06:33', 'HLA-DQA1*06:02-DQB1*06:34', 'HLA-DQA1*06:02-DQB1*06:35', 'HLA-DQA1*06:02-DQB1*06:36',
         'HLA-DQA1*06:02-DQB1*06:37', 'HLA-DQA1*06:02-DQB1*06:38',
         'HLA-DQA1*06:02-DQB1*06:39', 'HLA-DQA1*06:02-DQB1*06:40', 'HLA-DQA1*06:02-DQB1*06:41', 'HLA-DQA1*06:02-DQB1*06:42',
         'HLA-DQA1*06:02-DQB1*06:43', 'HLA-DQA1*06:02-DQB1*06:44',
         'H-2-Iab', 'H-2-Iad'])
    __version = "3.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype.upper(), allele.subtype)
        elif isinstance(allele, CombinedAllele):
            return "HLA-%s%s%s-%s%s%s" % (allele.alpha_locus, allele.alpha_supertype, allele.alpha_subtype,
                                          allele.beta_locus, allele.beta_supertype, allele.beta_subtype)
        else:
            return "%s_%s%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        def parse_allele_from_external_result(alleles_str_out):
            """
            Parses allele string from external result to allele string representation of input

            :param str allele_str_out: The allele string representation from the external result output
            :return: str allele_str_in: The allele string representation from the external result input
            :rtype: str
            """
            alleles_str_in = []
            for allele_str_out in alleles_str_out:
                if allele_str_out.startswith('HLA-'):
                    allele_str_in = allele_str_out.replace('*','').replace(':','')
                elif allele_str_out.startswith('D'):
                    allele_str_in = allele_str_out.replace('*','_').replace(':','')
                else:
                    allele_str_in = allele_str_out
                alleles_str_in.append(allele_str_in)

            return(alleles_str_in)

        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
        # Convert output representation of allele to input representation of allele, because they differ
        alleles = parse_allele_from_external_result(alleles)
        
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_3_0]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_3_0 + i * Offset.NETMHCIIPAN_3_0])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_3_0 + i * Offset.NETMHCIIPAN_3_0])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools
        and writes them to _file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetMHCIIpan_3_1(NetMHCIIpan_3_0):
    """
    Implementation of NetMHCIIpan 3.1 adapter.

    .. note::

        Andreatta, M., Karosiene, E., Rasmussen, M., Stryhn, A., Buus, S., & Nielsen, M. (2015). Accurate pan-specific
        prediction of peptide-MHC class II binding affinity with improved binding core identification.
        Immunogenetics, 1-10.
    """

    __supported_length = frozenset([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    __name = "netmhcIIpan"
    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"
    __alleles = frozenset(
        ['HLA-DRB1*01:01', 'HLA-DRB1*01:02', 'HLA-DRB1*01:03', 'HLA-DRB1*01:04', 'HLA-DRB1*01:05', 'HLA-DRB1*01:06',
         'HLA-DRB1*01:07', 'HLA-DRB1*01:08', 'HLA-DRB1*01:09', 'HLA-DRB1*01:10', 'HLA-DRB1*01:11', 'HLA-DRB1*01:12',
         'HLA-DRB1*01:13', 'HLA-DRB1*01:14', 'HLA-DRB1*01:15', 'HLA-DRB1*01:16', 'HLA-DRB1*01:17', 'HLA-DRB1*01:18',
         'HLA-DRB1*01:19', 'HLA-DRB1*01:20', 'HLA-DRB1*01:21', 'HLA-DRB1*01:22', 'HLA-DRB1*01:23', 'HLA-DRB1*01:24',
         'HLA-DRB1*01:25', 'HLA-DRB1*01:26', 'HLA-DRB1*01:27', 'HLA-DRB1*01:28', 'HLA-DRB1*01:29', 'HLA-DRB1*01:30',
         'HLA-DRB1*01:31', 'HLA-DRB1*01:32', 'HLA-DRB1*03:01', 'HLA-DRB1*03:02', 'HLA-DRB1*03:03', 'HLA-DRB1*03:04',
         'HLA-DRB1*03:05', 'HLA-DRB1*03:06', 'HLA-DRB1*03:07', 'HLA-DRB1*03:08', 'HLA-DRB1*03:10', 'HLA-DRB1*03:11',
         'HLA-DRB1*03:13', 'HLA-DRB1*03:14', 'HLA-DRB1*03:15', 'HLA-DRB1*03:17', 'HLA-DRB1*03:18', 'HLA-DRB1*03:19',
         'HLA-DRB1*03:20', 'HLA-DRB1*03:21', 'HLA-DRB1*03:22', 'HLA-DRB1*03:23', 'HLA-DRB1*03:24', 'HLA-DRB1*03:25',
         'HLA-DRB1*03:26', 'HLA-DRB1*03:27', 'HLA-DRB1*03:28', 'HLA-DRB1*03:29', 'HLA-DRB1*03:30', 'HLA-DRB1*03:31',
         'HLA-DRB1*03:32', 'HLA-DRB1*03:33', 'HLA-DRB1*03:34', 'HLA-DRB1*03:35', 'HLA-DRB1*03:36', 'HLA-DRB1*03:37',
         'HLA-DRB1*03:38', 'HLA-DRB1*03:39', 'HLA-DRB1*03:40', 'HLA-DRB1*03:41', 'HLA-DRB1*03:42', 'HLA-DRB1*03:43',
         'HLA-DRB1*03:44', 'HLA-DRB1*03:45', 'HLA-DRB1*03:46', 'HLA-DRB1*03:47', 'HLA-DRB1*03:48', 'HLA-DRB1*03:49',
         'HLA-DRB1*03:50', 'HLA-DRB1*03:51', 'HLA-DRB1*03:52', 'HLA-DRB1*03:53', 'HLA-DRB1*03:54', 'HLA-DRB1*03:55',
         'HLA-DRB1*04:01', 'HLA-DRB1*04:02', 'HLA-DRB1*04:03', 'HLA-DRB1*04:04', 'HLA-DRB1*04:05', 'HLA-DRB1*04:06',
         'HLA-DRB1*04:07', 'HLA-DRB1*04:08', 'HLA-DRB1*04:09', 'HLA-DRB1*04:10', 'HLA-DRB1*04:11', 'HLA-DRB1*04:12',
         'HLA-DRB1*04:13', 'HLA-DRB1*04:14', 'HLA-DRB1*04:15', 'HLA-DRB1*04:16', 'HLA-DRB1*04:17', 'HLA-DRB1*04:18',
         'HLA-DRB1*04:19', 'HLA-DRB1*04:21', 'HLA-DRB1*04:22', 'HLA-DRB1*04:23', 'HLA-DRB1*04:24', 'HLA-DRB1*04:26',
         'HLA-DRB1*04:27', 'HLA-DRB1*04:28', 'HLA-DRB1*04:29', 'HLA-DRB1*04:30', 'HLA-DRB1*04:31', 'HLA-DRB1*04:33',
         'HLA-DRB1*04:34', 'HLA-DRB1*04:35', 'HLA-DRB1*04:36', 'HLA-DRB1*04:37', 'HLA-DRB1*04:38', 'HLA-DRB1*04:39',
         'HLA-DRB1*04:40', 'HLA-DRB1*04:41', 'HLA-DRB1*04:42', 'HLA-DRB1*04:43', 'HLA-DRB1*04:44', 'HLA-DRB1*04:45',
         'HLA-DRB1*04:46', 'HLA-DRB1*04:47', 'HLA-DRB1*04:48', 'HLA-DRB1*04:49', 'HLA-DRB1*04:50', 'HLA-DRB1*04:51',
         'HLA-DRB1*04:52', 'HLA-DRB1*04:53', 'HLA-DRB1*04:54', 'HLA-DRB1*04:55', 'HLA-DRB1*04:56', 'HLA-DRB1*04:57',
         'HLA-DRB1*04:58', 'HLA-DRB1*04:59', 'HLA-DRB1*04:60', 'HLA-DRB1*04:61', 'HLA-DRB1*04:62', 'HLA-DRB1*04:63',
         'HLA-DRB1*04:64', 'HLA-DRB1*04:65', 'HLA-DRB1*04:66', 'HLA-DRB1*04:67', 'HLA-DRB1*04:68', 'HLA-DRB1*04:69',
         'HLA-DRB1*04:70', 'HLA-DRB1*04:71', 'HLA-DRB1*04:72', 'HLA-DRB1*04:73', 'HLA-DRB1*04:74', 'HLA-DRB1*04:75',
         'HLA-DRB1*04:76', 'HLA-DRB1*04:77', 'HLA-DRB1*04:78', 'HLA-DRB1*04:79', 'HLA-DRB1*04:80', 'HLA-DRB1*04:82',
         'HLA-DRB1*04:83', 'HLA-DRB1*04:84', 'HLA-DRB1*04:85', 'HLA-DRB1*04:86', 'HLA-DRB1*04:87', 'HLA-DRB1*04:88',
         'HLA-DRB1*04:89', 'HLA-DRB1*04:91', 'HLA-DRB1*07:01', 'HLA-DRB1*07:03', 'HLA-DRB1*07:04', 'HLA-DRB1*07:05',
         'HLA-DRB1*07:06', 'HLA-DRB1*07:07', 'HLA-DRB1*07:08', 'HLA-DRB1*07:09', 'HLA-DRB1*07:11', 'HLA-DRB1*07:12',
         'HLA-DRB1*07:13', 'HLA-DRB1*07:14', 'HLA-DRB1*07:15', 'HLA-DRB1*07:16', 'HLA-DRB1*07:17', 'HLA-DRB1*07:19',
         'HLA-DRB1*08:01', 'HLA-DRB1*08:02', 'HLA-DRB1*08:03', 'HLA-DRB1*08:04', 'HLA-DRB1*08:05', 'HLA-DRB1*08:06',
         'HLA-DRB1*08:07', 'HLA-DRB1*08:08', 'HLA-DRB1*08:09', 'HLA-DRB1*08:10', 'HLA-DRB1*08:11', 'HLA-DRB1*08:12',
         'HLA-DRB1*08:13', 'HLA-DRB1*08:14', 'HLA-DRB1*08:15', 'HLA-DRB1*08:16', 'HLA-DRB1*08:18', 'HLA-DRB1*08:19',
         'HLA-DRB1*08:20', 'HLA-DRB1*08:21', 'HLA-DRB1*08:22', 'HLA-DRB1*08:23', 'HLA-DRB1*08:24', 'HLA-DRB1*08:25',
         'HLA-DRB1*08:26', 'HLA-DRB1*08:27', 'HLA-DRB1*08:28', 'HLA-DRB1*08:29', 'HLA-DRB1*08:30', 'HLA-DRB1*08:31',
         'HLA-DRB1*08:32', 'HLA-DRB1*08:33', 'HLA-DRB1*08:34', 'HLA-DRB1*08:35', 'HLA-DRB1*08:36', 'HLA-DRB1*08:37',
         'HLA-DRB1*08:38', 'HLA-DRB1*08:39', 'HLA-DRB1*08:40', 'HLA-DRB1*09:01', 'HLA-DRB1*09:02', 'HLA-DRB1*09:03',
         'HLA-DRB1*09:04', 'HLA-DRB1*09:05', 'HLA-DRB1*09:06', 'HLA-DRB1*09:07', 'HLA-DRB1*09:08', 'HLA-DRB1*09:09',
         'HLA-DRB1*10:01', 'HLA-DRB1*10:02', 'HLA-DRB1*10:03', 'HLA-DRB1*11:01', 'HLA-DRB1*11:02', 'HLA-DRB1*11:03',
         'HLA-DRB1*11:04', 'HLA-DRB1*11:05', 'HLA-DRB1*11:06', 'HLA-DRB1*11:07', 'HLA-DRB1*11:08', 'HLA-DRB1*11:09',
         'HLA-DRB1*11:10', 'HLA-DRB1*11:11', 'HLA-DRB1*11:12', 'HLA-DRB1*11:13', 'HLA-DRB1*11:14', 'HLA-DRB1*11:15',
         'HLA-DRB1*11:16', 'HLA-DRB1*11:17', 'HLA-DRB1*11:18', 'HLA-DRB1*11:19', 'HLA-DRB1*11:20', 'HLA-DRB1*11:21',
         'HLA-DRB1*11:24', 'HLA-DRB1*11:25', 'HLA-DRB1*11:27', 'HLA-DRB1*11:28', 'HLA-DRB1*11:29', 'HLA-DRB1*11:30',
         'HLA-DRB1*11:31', 'HLA-DRB1*11:32', 'HLA-DRB1*11:33', 'HLA-DRB1*11:34', 'HLA-DRB1*11:35', 'HLA-DRB1*11:36',
         'HLA-DRB1*11:37', 'HLA-DRB1*11:38', 'HLA-DRB1*11:39', 'HLA-DRB1*11:41', 'HLA-DRB1*11:42', 'HLA-DRB1*11:43',
         'HLA-DRB1*11:44', 'HLA-DRB1*11:45', 'HLA-DRB1*11:46', 'HLA-DRB1*11:47', 'HLA-DRB1*11:48', 'HLA-DRB1*11:49',
         'HLA-DRB1*11:50', 'HLA-DRB1*11:51', 'HLA-DRB1*11:52', 'HLA-DRB1*11:53', 'HLA-DRB1*11:54', 'HLA-DRB1*11:55',
         'HLA-DRB1*11:56', 'HLA-DRB1*11:57', 'HLA-DRB1*11:58', 'HLA-DRB1*11:59', 'HLA-DRB1*11:60', 'HLA-DRB1*11:61',
         'HLA-DRB1*11:62', 'HLA-DRB1*11:63', 'HLA-DRB1*11:64', 'HLA-DRB1*11:65', 'HLA-DRB1*11:66', 'HLA-DRB1*11:67',
         'HLA-DRB1*11:68', 'HLA-DRB1*11:69', 'HLA-DRB1*11:70', 'HLA-DRB1*11:72', 'HLA-DRB1*11:73', 'HLA-DRB1*11:74',
         'HLA-DRB1*11:75', 'HLA-DRB1*11:76', 'HLA-DRB1*11:77', 'HLA-DRB1*11:78', 'HLA-DRB1*11:79', 'HLA-DRB1*11:80',
         'HLA-DRB1*11:81', 'HLA-DRB1*11:82', 'HLA-DRB1*11:83', 'HLA-DRB1*11:84', 'HLA-DRB1*11:85', 'HLA-DRB1*11:86',
         'HLA-DRB1*11:87', 'HLA-DRB1*11:88', 'HLA-DRB1*11:89', 'HLA-DRB1*11:90', 'HLA-DRB1*11:91', 'HLA-DRB1*11:92',
         'HLA-DRB1*11:93', 'HLA-DRB1*11:94', 'HLA-DRB1*11:95', 'HLA-DRB1*11:96', 'HLA-DRB1*12:01', 'HLA-DRB1*12:02',
         'HLA-DRB1*12:03', 'HLA-DRB1*12:04', 'HLA-DRB1*12:05', 'HLA-DRB1*12:06', 'HLA-DRB1*12:07', 'HLA-DRB1*12:08',
         'HLA-DRB1*12:09', 'HLA-DRB1*12:10', 'HLA-DRB1*12:11', 'HLA-DRB1*12:12', 'HLA-DRB1*12:13', 'HLA-DRB1*12:14',
         'HLA-DRB1*12:15', 'HLA-DRB1*12:16', 'HLA-DRB1*12:17', 'HLA-DRB1*12:18', 'HLA-DRB1*12:19', 'HLA-DRB1*12:20',
         'HLA-DRB1*12:21', 'HLA-DRB1*12:22', 'HLA-DRB1*12:23', 'HLA-DRB1*13:01', 'HLA-DRB1*13:02', 'HLA-DRB1*13:03',
         'HLA-DRB1*13:04', 'HLA-DRB1*13:05', 'HLA-DRB1*13:06', 'HLA-DRB1*13:07', 'HLA-DRB1*13:08', 'HLA-DRB1*13:09',
         'HLA-DRB1*13:10', 'HLA-DRB1*13:100', 'HLA-DRB1*13:101', 'HLA-DRB1*13:11', 'HLA-DRB1*13:12', 'HLA-DRB1*13:13',
         'HLA-DRB1*13:14', 'HLA-DRB1*13:15', 'HLA-DRB1*13:16', 'HLA-DRB1*13:17', 'HLA-DRB1*13:18', 'HLA-DRB1*13:19',
         'HLA-DRB1*13:20', 'HLA-DRB1*13:21', 'HLA-DRB1*13:22', 'HLA-DRB1*13:23', 'HLA-DRB1*13:24', 'HLA-DRB1*13:26',
         'HLA-DRB1*13:27', 'HLA-DRB1*13:29', 'HLA-DRB1*13:30', 'HLA-DRB1*13:31', 'HLA-DRB1*13:32', 'HLA-DRB1*13:33',
         'HLA-DRB1*13:34', 'HLA-DRB1*13:35', 'HLA-DRB1*13:36', 'HLA-DRB1*13:37', 'HLA-DRB1*13:38', 'HLA-DRB1*13:39',
         'HLA-DRB1*13:41', 'HLA-DRB1*13:42', 'HLA-DRB1*13:43', 'HLA-DRB1*13:44', 'HLA-DRB1*13:46', 'HLA-DRB1*13:47',
         'HLA-DRB1*13:48', 'HLA-DRB1*13:49', 'HLA-DRB1*13:50', 'HLA-DRB1*13:51', 'HLA-DRB1*13:52', 'HLA-DRB1*13:53',
         'HLA-DRB1*13:54', 'HLA-DRB1*13:55', 'HLA-DRB1*13:56', 'HLA-DRB1*13:57', 'HLA-DRB1*13:58', 'HLA-DRB1*13:59',
         'HLA-DRB1*13:60', 'HLA-DRB1*13:61', 'HLA-DRB1*13:62', 'HLA-DRB1*13:63', 'HLA-DRB1*13:64', 'HLA-DRB1*13:65',
         'HLA-DRB1*13:66', 'HLA-DRB1*13:67', 'HLA-DRB1*13:68', 'HLA-DRB1*13:69', 'HLA-DRB1*13:70', 'HLA-DRB1*13:71',
         'HLA-DRB1*13:72', 'HLA-DRB1*13:73', 'HLA-DRB1*13:74', 'HLA-DRB1*13:75', 'HLA-DRB1*13:76', 'HLA-DRB1*13:77',
         'HLA-DRB1*13:78', 'HLA-DRB1*13:79', 'HLA-DRB1*13:80', 'HLA-DRB1*13:81', 'HLA-DRB1*13:82', 'HLA-DRB1*13:83',
         'HLA-DRB1*13:84', 'HLA-DRB1*13:85', 'HLA-DRB1*13:86', 'HLA-DRB1*13:87', 'HLA-DRB1*13:88', 'HLA-DRB1*13:89',
         'HLA-DRB1*13:90', 'HLA-DRB1*13:91', 'HLA-DRB1*13:92', 'HLA-DRB1*13:93', 'HLA-DRB1*13:94', 'HLA-DRB1*13:95',
         'HLA-DRB1*13:96', 'HLA-DRB1*13:97', 'HLA-DRB1*13:98', 'HLA-DRB1*13:99', 'HLA-DRB1*14:01', 'HLA-DRB1*14:02',
         'HLA-DRB1*14:03', 'HLA-DRB1*14:04', 'HLA-DRB1*14:05', 'HLA-DRB1*14:06', 'HLA-DRB1*14:07', 'HLA-DRB1*14:08',
         'HLA-DRB1*14:09', 'HLA-DRB1*14:10', 'HLA-DRB1*14:11', 'HLA-DRB1*14:12', 'HLA-DRB1*14:13', 'HLA-DRB1*14:14',
         'HLA-DRB1*14:15', 'HLA-DRB1*14:16', 'HLA-DRB1*14:17', 'HLA-DRB1*14:18', 'HLA-DRB1*14:19', 'HLA-DRB1*14:20',
         'HLA-DRB1*14:21', 'HLA-DRB1*14:22', 'HLA-DRB1*14:23', 'HLA-DRB1*14:24', 'HLA-DRB1*14:25', 'HLA-DRB1*14:26',
         'HLA-DRB1*14:27', 'HLA-DRB1*14:28', 'HLA-DRB1*14:29', 'HLA-DRB1*14:30', 'HLA-DRB1*14:31', 'HLA-DRB1*14:32',
         'HLA-DRB1*14:33', 'HLA-DRB1*14:34', 'HLA-DRB1*14:35', 'HLA-DRB1*14:36', 'HLA-DRB1*14:37', 'HLA-DRB1*14:38',
         'HLA-DRB1*14:39', 'HLA-DRB1*14:40', 'HLA-DRB1*14:41', 'HLA-DRB1*14:42', 'HLA-DRB1*14:43', 'HLA-DRB1*14:44',
         'HLA-DRB1*14:45', 'HLA-DRB1*14:46', 'HLA-DRB1*14:47', 'HLA-DRB1*14:48', 'HLA-DRB1*14:49', 'HLA-DRB1*14:50',
         'HLA-DRB1*14:51', 'HLA-DRB1*14:52', 'HLA-DRB1*14:53', 'HLA-DRB1*14:54', 'HLA-DRB1*14:55', 'HLA-DRB1*14:56',
         'HLA-DRB1*14:57', 'HLA-DRB1*14:58', 'HLA-DRB1*14:59', 'HLA-DRB1*14:60', 'HLA-DRB1*14:61', 'HLA-DRB1*14:62',
         'HLA-DRB1*14:63', 'HLA-DRB1*14:64', 'HLA-DRB1*14:65', 'HLA-DRB1*14:67', 'HLA-DRB1*14:68', 'HLA-DRB1*14:69',
         'HLA-DRB1*14:70', 'HLA-DRB1*14:71', 'HLA-DRB1*14:72', 'HLA-DRB1*14:73', 'HLA-DRB1*14:74', 'HLA-DRB1*14:75',
         'HLA-DRB1*14:76', 'HLA-DRB1*14:77', 'HLA-DRB1*14:78', 'HLA-DRB1*14:79', 'HLA-DRB1*14:80', 'HLA-DRB1*14:81',
         'HLA-DRB1*14:82', 'HLA-DRB1*14:83', 'HLA-DRB1*14:84', 'HLA-DRB1*14:85', 'HLA-DRB1*14:86', 'HLA-DRB1*14:87',
         'HLA-DRB1*14:88', 'HLA-DRB1*14:89', 'HLA-DRB1*14:90', 'HLA-DRB1*14:91', 'HLA-DRB1*14:93', 'HLA-DRB1*14:94',
         'HLA-DRB1*14:95', 'HLA-DRB1*14:96', 'HLA-DRB1*14:97', 'HLA-DRB1*14:98', 'HLA-DRB1*14:99', 'HLA-DRB1*15:01',
         'HLA-DRB1*15:02', 'HLA-DRB1*15:03', 'HLA-DRB1*15:04', 'HLA-DRB1*15:05', 'HLA-DRB1*15:06', 'HLA-DRB1*15:07',
         'HLA-DRB1*15:08', 'HLA-DRB1*15:09', 'HLA-DRB1*15:10', 'HLA-DRB1*15:11', 'HLA-DRB1*15:12', 'HLA-DRB1*15:13',
         'HLA-DRB1*15:14', 'HLA-DRB1*15:15', 'HLA-DRB1*15:16', 'HLA-DRB1*15:18', 'HLA-DRB1*15:19', 'HLA-DRB1*15:20',
         'HLA-DRB1*15:21', 'HLA-DRB1*15:22', 'HLA-DRB1*15:23', 'HLA-DRB1*15:24', 'HLA-DRB1*15:25', 'HLA-DRB1*15:26',
         'HLA-DRB1*15:27', 'HLA-DRB1*15:28', 'HLA-DRB1*15:29', 'HLA-DRB1*15:30', 'HLA-DRB1*15:31', 'HLA-DRB1*15:32',
         'HLA-DRB1*15:33', 'HLA-DRB1*15:34', 'HLA-DRB1*15:35', 'HLA-DRB1*15:36', 'HLA-DRB1*15:37', 'HLA-DRB1*15:38',
         'HLA-DRB1*15:39', 'HLA-DRB1*15:40', 'HLA-DRB1*15:41', 'HLA-DRB1*15:42', 'HLA-DRB1*15:43', 'HLA-DRB1*15:44',
         'HLA-DRB1*15:45', 'HLA-DRB1*15:46', 'HLA-DRB1*15:47', 'HLA-DRB1*15:48', 'HLA-DRB1*15:49', 'HLA-DRB1*16:01',
         'HLA-DRB1*16:02', 'HLA-DRB1*16:03', 'HLA-DRB1*16:04', 'HLA-DRB1*16:05', 'HLA-DRB1*16:07', 'HLA-DRB1*16:08',
         'HLA-DRB1*16:09', 'HLA-DRB1*16:10', 'HLA-DRB1*16:11', 'HLA-DRB1*16:12', 'HLA-DRB1*16:14', 'HLA-DRB1*16:15',
         'HLA-DRB1*16:16', 'HLA-DRB3*01:01', 'HLA-DRB3*01:04', 'HLA-DRB3*01:05', 'HLA-DRB3*01:08', 'HLA-DRB3*01:09',
         'HLA-DRB3*01:11', 'HLA-DRB3*01:12', 'HLA-DRB3*01:13', 'HLA-DRB3*01:14', 'HLA-DRB3*02:01', 'HLA-DRB3*02:02',
         'HLA-DRB3*02:04', 'HLA-DRB3*02:05', 'HLA-DRB3*02:09', 'HLA-DRB3*02:10', 'HLA-DRB3*02:11', 'HLA-DRB3*02:12',
         'HLA-DRB3*02:13', 'HLA-DRB3*02:14', 'HLA-DRB3*02:15', 'HLA-DRB3*02:16', 'HLA-DRB3*02:17', 'HLA-DRB3*02:18',
         'HLA-DRB3*02:19', 'HLA-DRB3*02:20', 'HLA-DRB3*02:21', 'HLA-DRB3*02:22', 'HLA-DRB3*02:23', 'HLA-DRB3*02:24',
         'HLA-DRB3*02:25', 'HLA-DRB3*03:01', 'HLA-DRB3*03:03', 'HLA-DRB4*01:01', 'HLA-DRB4*01:03', 'HLA-DRB4*01:04',
         'HLA-DRB4*01:06', 'HLA-DRB4*01:07', 'HLA-DRB4*01:08', 'HLA-DRB5*01:01', 'HLA-DRB5*01:02', 'HLA-DRB5*01:03',
         'HLA-DRB5*01:04', 'HLA-DRB5*01:05', 'HLA-DRB5*01:06', 'HLA-DRB5*01:08N', 'HLA-DRB5*01:11', 'HLA-DRB5*01:12',
         'HLA-DRB5*01:13', 'HLA-DRB5*01:14', 'HLA-DRB5*02:02', 'HLA-DRB5*02:03', 'HLA-DRB5*02:04', 'HLA-DRB5*02:05',
         'HLA-DPA1*01:03-DPB1*01:01', 'HLA-DPA1*01:03-DPB1*02:01', 'HLA-DPA1*01:03-DPB1*02:02', 'HLA-DPA1*01:03-DPB1*03:01',
         'HLA-DPA1*01:03-DPB1*04:01', 'HLA-DPA1*01:03-DPB1*04:02', 'HLA-DPA1*01:03-DPB1*05:01', 'HLA-DPA1*01:03-DPB1*06:01',
         'HLA-DPA1*01:03-DPB1*08:01', 'HLA-DPA1*01:03-DPB1*09:01', 'HLA-DPA1*01:03-DPB1*10:001', 'HLA-DPA1*01:03-DPB1*10:01',
         'HLA-DPA1*01:03-DPB1*10:101', 'HLA-DPA1*01:03-DPB1*10:201',
         'HLA-DPA1*01:03-DPB1*10:301', 'HLA-DPA1*01:03-DPB1*10:401',
         'HLA-DPA1*01:03-DPB1*10:501', 'HLA-DPA1*01:03-DPB1*10:601', 'HLA-DPA1*01:03-DPB1*10:701', 'HLA-DPA1*01:03-DPB1*10:801',
         'HLA-DPA1*01:03-DPB1*10:901', 'HLA-DPA1*01:03-DPB1*11:001',
         'HLA-DPA1*01:03-DPB1*11:01', 'HLA-DPA1*01:03-DPB1*11:101', 'HLA-DPA1*01:03-DPB1*11:201', 'HLA-DPA1*01:03-DPB1*11:301',
         'HLA-DPA1*01:03-DPB1*11:401', 'HLA-DPA1*01:03-DPB1*11:501',
         'HLA-DPA1*01:03-DPB1*11:601', 'HLA-DPA1*01:03-DPB1*11:701', 'HLA-DPA1*01:03-DPB1*11:801', 'HLA-DPA1*01:03-DPB1*11:901',
         'HLA-DPA1*01:03-DPB1*12:101', 'HLA-DPA1*01:03-DPB1*12:201',
         'HLA-DPA1*01:03-DPB1*12:301', 'HLA-DPA1*01:03-DPB1*12:401', 'HLA-DPA1*01:03-DPB1*12:501', 'HLA-DPA1*01:03-DPB1*12:601',
         'HLA-DPA1*01:03-DPB1*12:701', 'HLA-DPA1*01:03-DPB1*12:801',
         'HLA-DPA1*01:03-DPB1*12:901', 'HLA-DPA1*01:03-DPB1*13:001', 'HLA-DPA1*01:03-DPB1*13:01', 'HLA-DPA1*01:03-DPB1*13:101',
         'HLA-DPA1*01:03-DPB1*13:201', 'HLA-DPA1*01:03-DPB1*13:301',
         'HLA-DPA1*01:03-DPB1*13:401', 'HLA-DPA1*01:03-DPB1*14:01', 'HLA-DPA1*01:03-DPB1*15:01', 'HLA-DPA1*01:03-DPB1*16:01',
         'HLA-DPA1*01:03-DPB1*17:01', 'HLA-DPA1*01:03-DPB1*18:01',
         'HLA-DPA1*01:03-DPB1*19:01', 'HLA-DPA1*01:03-DPB1*20:01', 'HLA-DPA1*01:03-DPB1*21:01', 'HLA-DPA1*01:03-DPB1*22:01',
         'HLA-DPA1*01:03-DPB1*23:01', 'HLA-DPA1*01:03-DPB1*24:01',
         'HLA-DPA1*01:03-DPB1*25:01', 'HLA-DPA1*01:03-DPB1*26:01', 'HLA-DPA1*01:03-DPB1*27:01', 'HLA-DPA1*01:03-DPB1*28:01',
         'HLA-DPA1*01:03-DPB1*29:01', 'HLA-DPA1*01:03-DPB1*30:01',
         'HLA-DPA1*01:03-DPB1*31:01', 'HLA-DPA1*01:03-DPB1*32:01', 'HLA-DPA1*01:03-DPB1*33:01', 'HLA-DPA1*01:03-DPB1*34:01',
         'HLA-DPA1*01:03-DPB1*35:01', 'HLA-DPA1*01:03-DPB1*36:01',
         'HLA-DPA1*01:03-DPB1*37:01', 'HLA-DPA1*01:03-DPB1*38:01', 'HLA-DPA1*01:03-DPB1*39:01', 'HLA-DPA1*01:03-DPB1*40:01',
         'HLA-DPA1*01:03-DPB1*41:01', 'HLA-DPA1*01:03-DPB1*44:01',
         'HLA-DPA1*01:03-DPB1*45:01', 'HLA-DPA1*01:03-DPB1*46:01', 'HLA-DPA1*01:03-DPB1*47:01', 'HLA-DPA1*01:03-DPB1*48:01',
         'HLA-DPA1*01:03-DPB1*49:01', 'HLA-DPA1*01:03-DPB1*50:01',
         'HLA-DPA1*01:03-DPB1*51:01', 'HLA-DPA1*01:03-DPB1*52:01', 'HLA-DPA1*01:03-DPB1*53:01', 'HLA-DPA1*01:03-DPB1*54:01',
         'HLA-DPA1*01:03-DPB1*55:01', 'HLA-DPA1*01:03-DPB1*56:01',
         'HLA-DPA1*01:03-DPB1*58:01', 'HLA-DPA1*01:03-DPB1*59:01', 'HLA-DPA1*01:03-DPB1*60:01', 'HLA-DPA1*01:03-DPB1*62:01',
         'HLA-DPA1*01:03-DPB1*63:01', 'HLA-DPA1*01:03-DPB1*65:01',
         'HLA-DPA1*01:03-DPB1*66:01', 'HLA-DPA1*01:03-DPB1*67:01', 'HLA-DPA1*01:03-DPB1*68:01', 'HLA-DPA1*01:03-DPB1*69:01',
         'HLA-DPA1*01:03-DPB1*70:01', 'HLA-DPA1*01:03-DPB1*71:01',
         'HLA-DPA1*01:03-DPB1*72:01', 'HLA-DPA1*01:03-DPB1*73:01', 'HLA-DPA1*01:03-DPB1*74:01', 'HLA-DPA1*01:03-DPB1*75:01',
         'HLA-DPA1*01:03-DPB1*76:01', 'HLA-DPA1*01:03-DPB1*77:01',
         'HLA-DPA1*01:03-DPB1*78:01', 'HLA-DPA1*01:03-DPB1*79:01', 'HLA-DPA1*01:03-DPB1*80:01', 'HLA-DPA1*01:03-DPB1*81:01',
         'HLA-DPA1*01:03-DPB1*82:01', 'HLA-DPA1*01:03-DPB1*83:01',
         'HLA-DPA1*01:03-DPB1*84:01', 'HLA-DPA1*01:03-DPB1*85:01', 'HLA-DPA1*01:03-DPB1*86:01', 'HLA-DPA1*01:03-DPB1*87:01',
         'HLA-DPA1*01:03-DPB1*88:01', 'HLA-DPA1*01:03-DPB1*89:01',
         'HLA-DPA1*01:03-DPB1*90:01', 'HLA-DPA1*01:03-DPB1*91:01', 'HLA-DPA1*01:03-DPB1*92:01', 'HLA-DPA1*01:03-DPB1*93:01',
         'HLA-DPA1*01:03-DPB1*94:01', 'HLA-DPA1*01:03-DPB1*95:01',
         'HLA-DPA1*01:03-DPB1*96:01', 'HLA-DPA1*01:03-DPB1*97:01', 'HLA-DPA1*01:03-DPB1*98:01', 'HLA-DPA1*01:03-DPB1*99:01',
         'HLA-DPA1*01:04-DPB1*01:01', 'HLA-DPA1*01:04-DPB1*02:01',
         'HLA-DPA1*01:04-DPB1*02:02', 'HLA-DPA1*01:04-DPB1*03:01', 'HLA-DPA1*01:04-DPB1*04:01', 'HLA-DPA1*01:04-DPB1*04:02',
         'HLA-DPA1*01:04-DPB1*05:01', 'HLA-DPA1*01:04-DPB1*06:01',
         'HLA-DPA1*01:04-DPB1*08:01', 'HLA-DPA1*01:04-DPB1*09:01', 'HLA-DPA1*01:04-DPB1*10:001', 'HLA-DPA1*01:04-DPB1*10:01',
         'HLA-DPA1*01:04-DPB1*10:101', 'HLA-DPA1*01:04-DPB1*10:201',
         'HLA-DPA1*01:04-DPB1*10:301', 'HLA-DPA1*01:04-DPB1*10:401', 'HLA-DPA1*01:04-DPB1*10:501', 'HLA-DPA1*01:04-DPB1*10:601',
         'HLA-DPA1*01:04-DPB1*10:701', 'HLA-DPA1*01:04-DPB1*10:801',
         'HLA-DPA1*01:04-DPB1*10:901', 'HLA-DPA1*01:04-DPB1*11:001', 'HLA-DPA1*01:04-DPB1*11:01', 'HLA-DPA1*01:04-DPB1*11:101',
         'HLA-DPA1*01:04-DPB1*11:201', 'HLA-DPA1*01:04-DPB1*11:301',
         'HLA-DPA1*01:04-DPB1*11:401', 'HLA-DPA1*01:04-DPB1*11:501', 'HLA-DPA1*01:04-DPB1*11:601', 'HLA-DPA1*01:04-DPB1*11:701',
         'HLA-DPA1*01:04-DPB1*11:801', 'HLA-DPA1*01:04-DPB1*11:901',
         'HLA-DPA1*01:04-DPB1*12:101', 'HLA-DPA1*01:04-DPB1*12:201', 'HLA-DPA1*01:04-DPB1*12:301', 'HLA-DPA1*01:04-DPB1*12:401',
         'HLA-DPA1*01:04-DPB1*12:501', 'HLA-DPA1*01:04-DPB1*12:601',
         'HLA-DPA1*01:04-DPB1*12:701', 'HLA-DPA1*01:04-DPB1*12:801', 'HLA-DPA1*01:04-DPB1*12:901', 'HLA-DPA1*01:04-DPB1*13:001',
         'HLA-DPA1*01:04-DPB1*13:01', 'HLA-DPA1*01:04-DPB1*13:101',
         'HLA-DPA1*01:04-DPB1*13:201', 'HLA-DPA1*01:04-DPB1*13:301', 'HLA-DPA1*01:04-DPB1*13:401', 'HLA-DPA1*01:04-DPB1*14:01',
         'HLA-DPA1*01:04-DPB1*15:01', 'HLA-DPA1*01:04-DPB1*16:01',
         'HLA-DPA1*01:04-DPB1*17:01', 'HLA-DPA1*01:04-DPB1*18:01', 'HLA-DPA1*01:04-DPB1*19:01', 'HLA-DPA1*01:04-DPB1*20:01',
         'HLA-DPA1*01:04-DPB1*21:01', 'HLA-DPA1*01:04-DPB1*22:01',
         'HLA-DPA1*01:04-DPB1*23:01', 'HLA-DPA1*01:04-DPB1*24:01', 'HLA-DPA1*01:04-DPB1*25:01', 'HLA-DPA1*01:04-DPB1*26:01',
         'HLA-DPA1*01:04-DPB1*27:01', 'HLA-DPA1*01:04-DPB1*28:01',
         'HLA-DPA1*01:04-DPB1*29:01', 'HLA-DPA1*01:04-DPB1*30:01', 'HLA-DPA1*01:04-DPB1*31:01', 'HLA-DPA1*01:04-DPB1*32:01',
         'HLA-DPA1*01:04-DPB1*33:01', 'HLA-DPA1*01:04-DPB1*34:01',
         'HLA-DPA1*01:04-DPB1*35:01', 'HLA-DPA1*01:04-DPB1*36:01', 'HLA-DPA1*01:04-DPB1*37:01', 'HLA-DPA1*01:04-DPB1*38:01',
         'HLA-DPA1*01:04-DPB1*39:01', 'HLA-DPA1*01:04-DPB1*40:01',
         'HLA-DPA1*01:04-DPB1*41:01', 'HLA-DPA1*01:04-DPB1*44:01', 'HLA-DPA1*01:04-DPB1*45:01', 'HLA-DPA1*01:04-DPB1*46:01',
         'HLA-DPA1*01:04-DPB1*47:01', 'HLA-DPA1*01:04-DPB1*48:01',
         'HLA-DPA1*01:04-DPB1*49:01', 'HLA-DPA1*01:04-DPB1*50:01', 'HLA-DPA1*01:04-DPB1*51:01', 'HLA-DPA1*01:04-DPB1*52:01',
         'HLA-DPA1*01:04-DPB1*53:01', 'HLA-DPA1*01:04-DPB1*54:01',
         'HLA-DPA1*01:04-DPB1*55:01', 'HLA-DPA1*01:04-DPB1*56:01', 'HLA-DPA1*01:04-DPB1*58:01', 'HLA-DPA1*01:04-DPB1*59:01',
         'HLA-DPA1*01:04-DPB1*60:01', 'HLA-DPA1*01:04-DPB1*62:01',
         'HLA-DPA1*01:04-DPB1*63:01', 'HLA-DPA1*01:04-DPB1*65:01', 'HLA-DPA1*01:04-DPB1*66:01', 'HLA-DPA1*01:04-DPB1*67:01',
         'HLA-DPA1*01:04-DPB1*68:01', 'HLA-DPA1*01:04-DPB1*69:01',
         'HLA-DPA1*01:04-DPB1*70:01', 'HLA-DPA1*01:04-DPB1*71:01', 'HLA-DPA1*01:04-DPB1*72:01', 'HLA-DPA1*01:04-DPB1*73:01',
         'HLA-DPA1*01:04-DPB1*74:01', 'HLA-DPA1*01:04-DPB1*75:01',
         'HLA-DPA1*01:04-DPB1*76:01', 'HLA-DPA1*01:04-DPB1*77:01', 'HLA-DPA1*01:04-DPB1*78:01', 'HLA-DPA1*01:04-DPB1*79:01',
         'HLA-DPA1*01:04-DPB1*80:01', 'HLA-DPA1*01:04-DPB1*81:01',
         'HLA-DPA1*01:04-DPB1*82:01', 'HLA-DPA1*01:04-DPB1*83:01', 'HLA-DPA1*01:04-DPB1*84:01', 'HLA-DPA1*01:04-DPB1*85:01',
         'HLA-DPA1*01:04-DPB1*86:01', 'HLA-DPA1*01:04-DPB1*87:01',
         'HLA-DPA1*01:04-DPB1*88:01', 'HLA-DPA1*01:04-DPB1*89:01', 'HLA-DPA1*01:04-DPB1*90:01', 'HLA-DPA1*01:04-DPB1*91:01',
         'HLA-DPA1*01:04-DPB1*92:01', 'HLA-DPA1*01:04-DPB1*93:01',
         'HLA-DPA1*01:04-DPB1*94:01', 'HLA-DPA1*01:04-DPB1*95:01', 'HLA-DPA1*01:04-DPB1*96:01', 'HLA-DPA1*01:04-DPB1*97:01',
         'HLA-DPA1*01:04-DPB1*98:01', 'HLA-DPA1*01:04-DPB1*99:01',
         'HLA-DPA1*01:05-DPB1*01:01', 'HLA-DPA1*01:05-DPB1*02:01', 'HLA-DPA1*01:05-DPB1*02:02', 'HLA-DPA1*01:05-DPB1*03:01',
         'HLA-DPA1*01:05-DPB1*04:01', 'HLA-DPA1*01:05-DPB1*04:02',
         'HLA-DPA1*01:05-DPB1*05:01', 'HLA-DPA1*01:05-DPB1*06:01', 'HLA-DPA1*01:05-DPB1*08:01', 'HLA-DPA1*01:05-DPB1*09:01',
         'HLA-DPA1*01:05-DPB1*10:001', 'HLA-DPA1*01:05-DPB1*10:01',
         'HLA-DPA1*01:05-DPB1*10:101', 'HLA-DPA1*01:05-DPB1*10:201', 'HLA-DPA1*01:05-DPB1*10:301', 'HLA-DPA1*01:05-DPB1*10:401',
         'HLA-DPA1*01:05-DPB1*10:501', 'HLA-DPA1*01:05-DPB1*10:601',
         'HLA-DPA1*01:05-DPB1*10:701', 'HLA-DPA1*01:05-DPB1*10:801', 'HLA-DPA1*01:05-DPB1*10:901', 'HLA-DPA1*01:05-DPB1*11:001',
         'HLA-DPA1*01:05-DPB1*11:01', 'HLA-DPA1*01:05-DPB1*11:101',
         'HLA-DPA1*01:05-DPB1*11:201', 'HLA-DPA1*01:05-DPB1*11:301', 'HLA-DPA1*01:05-DPB1*11:401', 'HLA-DPA1*01:05-DPB1*11:501',
         'HLA-DPA1*01:05-DPB1*11:601', 'HLA-DPA1*01:05-DPB1*11:701',
         'HLA-DPA1*01:05-DPB1*11:801', 'HLA-DPA1*01:05-DPB1*11:901', 'HLA-DPA1*01:05-DPB1*12:101', 'HLA-DPA1*01:05-DPB1*12:201',
         'HLA-DPA1*01:05-DPB1*12:301', 'HLA-DPA1*01:05-DPB1*12:401',
         'HLA-DPA1*01:05-DPB1*12:501', 'HLA-DPA1*01:05-DPB1*12:601', 'HLA-DPA1*01:05-DPB1*12:701', 'HLA-DPA1*01:05-DPB1*12:801',
         'HLA-DPA1*01:05-DPB1*12:901', 'HLA-DPA1*01:05-DPB1*13:001',
         'HLA-DPA1*01:05-DPB1*13:01', 'HLA-DPA1*01:05-DPB1*13:101', 'HLA-DPA1*01:05-DPB1*13:201', 'HLA-DPA1*01:05-DPB1*13:301',
         'HLA-DPA1*01:05-DPB1*13:401', 'HLA-DPA1*01:05-DPB1*14:01',
         'HLA-DPA1*01:05-DPB1*15:01', 'HLA-DPA1*01:05-DPB1*16:01', 'HLA-DPA1*01:05-DPB1*17:01', 'HLA-DPA1*01:05-DPB1*18:01',
         'HLA-DPA1*01:05-DPB1*19:01', 'HLA-DPA1*01:05-DPB1*20:01',
         'HLA-DPA1*01:05-DPB1*21:01', 'HLA-DPA1*01:05-DPB1*22:01', 'HLA-DPA1*01:05-DPB1*23:01', 'HLA-DPA1*01:05-DPB1*24:01',
         'HLA-DPA1*01:05-DPB1*25:01', 'HLA-DPA1*01:05-DPB1*26:01',
         'HLA-DPA1*01:05-DPB1*27:01', 'HLA-DPA1*01:05-DPB1*28:01', 'HLA-DPA1*01:05-DPB1*29:01', 'HLA-DPA1*01:05-DPB1*30:01',
         'HLA-DPA1*01:05-DPB1*31:01', 'HLA-DPA1*01:05-DPB1*32:01',
         'HLA-DPA1*01:05-DPB1*33:01', 'HLA-DPA1*01:05-DPB1*34:01', 'HLA-DPA1*01:05-DPB1*35:01', 'HLA-DPA1*01:05-DPB1*36:01',
         'HLA-DPA1*01:05-DPB1*37:01', 'HLA-DPA1*01:05-DPB1*38:01',
         'HLA-DPA1*01:05-DPB1*39:01', 'HLA-DPA1*01:05-DPB1*40:01', 'HLA-DPA1*01:05-DPB1*41:01', 'HLA-DPA1*01:05-DPB1*44:01',
         'HLA-DPA1*01:05-DPB1*45:01', 'HLA-DPA1*01:05-DPB1*46:01',
         'HLA-DPA1*01:05-DPB1*47:01', 'HLA-DPA1*01:05-DPB1*48:01', 'HLA-DPA1*01:05-DPB1*49:01', 'HLA-DPA1*01:05-DPB1*50:01',
         'HLA-DPA1*01:05-DPB1*51:01', 'HLA-DPA1*01:05-DPB1*52:01',
         'HLA-DPA1*01:05-DPB1*53:01', 'HLA-DPA1*01:05-DPB1*54:01', 'HLA-DPA1*01:05-DPB1*55:01', 'HLA-DPA1*01:05-DPB1*56:01',
         'HLA-DPA1*01:05-DPB1*58:01', 'HLA-DPA1*01:05-DPB1*59:01',
         'HLA-DPA1*01:05-DPB1*60:01', 'HLA-DPA1*01:05-DPB1*62:01', 'HLA-DPA1*01:05-DPB1*63:01', 'HLA-DPA1*01:05-DPB1*65:01',
         'HLA-DPA1*01:05-DPB1*66:01', 'HLA-DPA1*01:05-DPB1*67:01',
         'HLA-DPA1*01:05-DPB1*68:01', 'HLA-DPA1*01:05-DPB1*69:01', 'HLA-DPA1*01:05-DPB1*70:01', 'HLA-DPA1*01:05-DPB1*71:01',
         'HLA-DPA1*01:05-DPB1*72:01', 'HLA-DPA1*01:05-DPB1*73:01',
         'HLA-DPA1*01:05-DPB1*74:01', 'HLA-DPA1*01:05-DPB1*75:01', 'HLA-DPA1*01:05-DPB1*76:01', 'HLA-DPA1*01:05-DPB1*77:01',
         'HLA-DPA1*01:05-DPB1*78:01', 'HLA-DPA1*01:05-DPB1*79:01',
         'HLA-DPA1*01:05-DPB1*80:01', 'HLA-DPA1*01:05-DPB1*81:01', 'HLA-DPA1*01:05-DPB1*82:01', 'HLA-DPA1*01:05-DPB1*83:01',
         'HLA-DPA1*01:05-DPB1*84:01', 'HLA-DPA1*01:05-DPB1*85:01',
         'HLA-DPA1*01:05-DPB1*86:01', 'HLA-DPA1*01:05-DPB1*87:01', 'HLA-DPA1*01:05-DPB1*88:01', 'HLA-DPA1*01:05-DPB1*89:01',
         'HLA-DPA1*01:05-DPB1*90:01', 'HLA-DPA1*01:05-DPB1*91:01',
         'HLA-DPA1*01:05-DPB1*92:01', 'HLA-DPA1*01:05-DPB1*93:01', 'HLA-DPA1*01:05-DPB1*94:01', 'HLA-DPA1*01:05-DPB1*95:01',
         'HLA-DPA1*01:05-DPB1*96:01', 'HLA-DPA1*01:05-DPB1*97:01',
         'HLA-DPA1*01:05-DPB1*98:01', 'HLA-DPA1*01:05-DPB1*99:01', 'HLA-DPA1*01:06-DPB1*01:01', 'HLA-DPA1*01:06-DPB1*02:01',
         'HLA-DPA1*01:06-DPB1*02:02', 'HLA-DPA1*01:06-DPB1*03:01',
         'HLA-DPA1*01:06-DPB1*04:01', 'HLA-DPA1*01:06-DPB1*04:02', 'HLA-DPA1*01:06-DPB1*05:01', 'HLA-DPA1*01:06-DPB1*06:01',
         'HLA-DPA1*01:06-DPB1*08:01', 'HLA-DPA1*01:06-DPB1*09:01',
         'HLA-DPA1*01:06-DPB1*10:001', 'HLA-DPA1*01:06-DPB1*10:01', 'HLA-DPA1*01:06-DPB1*10:101', 'HLA-DPA1*01:06-DPB1*10:201',
         'HLA-DPA1*01:06-DPB1*10:301', 'HLA-DPA1*01:06-DPB1*10:401',
         'HLA-DPA1*01:06-DPB1*10:501', 'HLA-DPA1*01:06-DPB1*10:601', 'HLA-DPA1*01:06-DPB1*10:701', 'HLA-DPA1*01:06-DPB1*10:801',
         'HLA-DPA1*01:06-DPB1*10:901', 'HLA-DPA1*01:06-DPB1*11:001',
         'HLA-DPA1*01:06-DPB1*11:01', 'HLA-DPA1*01:06-DPB1*11:101', 'HLA-DPA1*01:06-DPB1*11:201', 'HLA-DPA1*01:06-DPB1*11:301',
         'HLA-DPA1*01:06-DPB1*11:401', 'HLA-DPA1*01:06-DPB1*11:501',
         'HLA-DPA1*01:06-DPB1*11:601', 'HLA-DPA1*01:06-DPB1*11:701', 'HLA-DPA1*01:06-DPB1*11:801', 'HLA-DPA1*01:06-DPB1*11:901',
         'HLA-DPA1*01:06-DPB1*12:101', 'HLA-DPA1*01:06-DPB1*12:201',
         'HLA-DPA1*01:06-DPB1*12:301', 'HLA-DPA1*01:06-DPB1*12:401', 'HLA-DPA1*01:06-DPB1*12:501', 'HLA-DPA1*01:06-DPB1*12:601',
         'HLA-DPA1*01:06-DPB1*12:701', 'HLA-DPA1*01:06-DPB1*12:801',
         'HLA-DPA1*01:06-DPB1*12:901', 'HLA-DPA1*01:06-DPB1*13:001', 'HLA-DPA1*01:06-DPB1*13:01', 'HLA-DPA1*01:06-DPB1*13:101',
         'HLA-DPA1*01:06-DPB1*13:201', 'HLA-DPA1*01:06-DPB1*13:301',
         'HLA-DPA1*01:06-DPB1*13:401', 'HLA-DPA1*01:06-DPB1*14:01', 'HLA-DPA1*01:06-DPB1*15:01', 'HLA-DPA1*01:06-DPB1*16:01',
         'HLA-DPA1*01:06-DPB1*17:01', 'HLA-DPA1*01:06-DPB1*18:01',
         'HLA-DPA1*01:06-DPB1*19:01', 'HLA-DPA1*01:06-DPB1*20:01', 'HLA-DPA1*01:06-DPB1*21:01', 'HLA-DPA1*01:06-DPB1*22:01',
         'HLA-DPA1*01:06-DPB1*23:01', 'HLA-DPA1*01:06-DPB1*24:01',
         'HLA-DPA1*01:06-DPB1*25:01', 'HLA-DPA1*01:06-DPB1*26:01', 'HLA-DPA1*01:06-DPB1*27:01', 'HLA-DPA1*01:06-DPB1*28:01',
         'HLA-DPA1*01:06-DPB1*29:01', 'HLA-DPA1*01:06-DPB1*30:01',
         'HLA-DPA1*01:06-DPB1*31:01', 'HLA-DPA1*01:06-DPB1*32:01', 'HLA-DPA1*01:06-DPB1*33:01', 'HLA-DPA1*01:06-DPB1*34:01',
         'HLA-DPA1*01:06-DPB1*35:01', 'HLA-DPA1*01:06-DPB1*36:01',
         'HLA-DPA1*01:06-DPB1*37:01', 'HLA-DPA1*01:06-DPB1*38:01', 'HLA-DPA1*01:06-DPB1*39:01', 'HLA-DPA1*01:06-DPB1*40:01',
         'HLA-DPA1*01:06-DPB1*41:01', 'HLA-DPA1*01:06-DPB1*44:01',
         'HLA-DPA1*01:06-DPB1*45:01', 'HLA-DPA1*01:06-DPB1*46:01', 'HLA-DPA1*01:06-DPB1*47:01', 'HLA-DPA1*01:06-DPB1*48:01',
         'HLA-DPA1*01:06-DPB1*49:01', 'HLA-DPA1*01:06-DPB1*50:01',
         'HLA-DPA1*01:06-DPB1*51:01', 'HLA-DPA1*01:06-DPB1*52:01', 'HLA-DPA1*01:06-DPB1*53:01', 'HLA-DPA1*01:06-DPB1*54:01',
         'HLA-DPA1*01:06-DPB1*55:01', 'HLA-DPA1*01:06-DPB1*56:01',
         'HLA-DPA1*01:06-DPB1*58:01', 'HLA-DPA1*01:06-DPB1*59:01', 'HLA-DPA1*01:06-DPB1*60:01', 'HLA-DPA1*01:06-DPB1*62:01',
         'HLA-DPA1*01:06-DPB1*63:01', 'HLA-DPA1*01:06-DPB1*65:01',
         'HLA-DPA1*01:06-DPB1*66:01', 'HLA-DPA1*01:06-DPB1*67:01', 'HLA-DPA1*01:06-DPB1*68:01', 'HLA-DPA1*01:06-DPB1*69:01',
         'HLA-DPA1*01:06-DPB1*70:01', 'HLA-DPA1*01:06-DPB1*71:01',
         'HLA-DPA1*01:06-DPB1*72:01', 'HLA-DPA1*01:06-DPB1*73:01', 'HLA-DPA1*01:06-DPB1*74:01', 'HLA-DPA1*01:06-DPB1*75:01',
         'HLA-DPA1*01:06-DPB1*76:01', 'HLA-DPA1*01:06-DPB1*77:01',
         'HLA-DPA1*01:06-DPB1*78:01', 'HLA-DPA1*01:06-DPB1*79:01', 'HLA-DPA1*01:06-DPB1*80:01', 'HLA-DPA1*01:06-DPB1*81:01',
         'HLA-DPA1*01:06-DPB1*82:01', 'HLA-DPA1*01:06-DPB1*83:01',
         'HLA-DPA1*01:06-DPB1*84:01', 'HLA-DPA1*01:06-DPB1*85:01', 'HLA-DPA1*01:06-DPB1*86:01', 'HLA-DPA1*01:06-DPB1*87:01',
         'HLA-DPA1*01:06-DPB1*88:01', 'HLA-DPA1*01:06-DPB1*89:01',
         'HLA-DPA1*01:06-DPB1*90:01', 'HLA-DPA1*01:06-DPB1*91:01', 'HLA-DPA1*01:06-DPB1*92:01', 'HLA-DPA1*01:06-DPB1*93:01',
         'HLA-DPA1*01:06-DPB1*94:01', 'HLA-DPA1*01:06-DPB1*95:01',
         'HLA-DPA1*01:06-DPB1*96:01', 'HLA-DPA1*01:06-DPB1*97:01', 'HLA-DPA1*01:06-DPB1*98:01', 'HLA-DPA1*01:06-DPB1*99:01',
         'HLA-DPA1*01:07-DPB1*01:01', 'HLA-DPA1*01:07-DPB1*02:01',
         'HLA-DPA1*01:07-DPB1*02:02', 'HLA-DPA1*01:07-DPB1*03:01', 'HLA-DPA1*01:07-DPB1*04:01', 'HLA-DPA1*01:07-DPB1*04:02',
         'HLA-DPA1*01:07-DPB1*05:01', 'HLA-DPA1*01:07-DPB1*06:01',
         'HLA-DPA1*01:07-DPB1*08:01', 'HLA-DPA1*01:07-DPB1*09:01', 'HLA-DPA1*01:07-DPB1*10:001', 'HLA-DPA1*01:07-DPB1*10:01',
         'HLA-DPA1*01:07-DPB1*10:101', 'HLA-DPA1*01:07-DPB1*10:201',
         'HLA-DPA1*01:07-DPB1*10:301', 'HLA-DPA1*01:07-DPB1*10:401', 'HLA-DPA1*01:07-DPB1*10:501', 'HLA-DPA1*01:07-DPB1*10:601',
         'HLA-DPA1*01:07-DPB1*10:701', 'HLA-DPA1*01:07-DPB1*10:801',
         'HLA-DPA1*01:07-DPB1*10:901', 'HLA-DPA1*01:07-DPB1*11:001', 'HLA-DPA1*01:07-DPB1*11:01', 'HLA-DPA1*01:07-DPB1*11:101',
         'HLA-DPA1*01:07-DPB1*11:201', 'HLA-DPA1*01:07-DPB1*11:301',
         'HLA-DPA1*01:07-DPB1*11:401', 'HLA-DPA1*01:07-DPB1*11:501', 'HLA-DPA1*01:07-DPB1*11:601', 'HLA-DPA1*01:07-DPB1*11:701',
         'HLA-DPA1*01:07-DPB1*11:801', 'HLA-DPA1*01:07-DPB1*11:901',
         'HLA-DPA1*01:07-DPB1*12:101', 'HLA-DPA1*01:07-DPB1*12:201', 'HLA-DPA1*01:07-DPB1*12:301', 'HLA-DPA1*01:07-DPB1*12:401',
         'HLA-DPA1*01:07-DPB1*12:501', 'HLA-DPA1*01:07-DPB1*12:601',
         'HLA-DPA1*01:07-DPB1*12:701', 'HLA-DPA1*01:07-DPB1*12:801', 'HLA-DPA1*01:07-DPB1*12:901', 'HLA-DPA1*01:07-DPB1*13:001',
         'HLA-DPA1*01:07-DPB1*13:01', 'HLA-DPA1*01:07-DPB1*13:101',
         'HLA-DPA1*01:07-DPB1*13:201', 'HLA-DPA1*01:07-DPB1*13:301', 'HLA-DPA1*01:07-DPB1*13:401', 'HLA-DPA1*01:07-DPB1*14:01',
         'HLA-DPA1*01:07-DPB1*15:01', 'HLA-DPA1*01:07-DPB1*16:01',
         'HLA-DPA1*01:07-DPB1*17:01', 'HLA-DPA1*01:07-DPB1*18:01', 'HLA-DPA1*01:07-DPB1*19:01', 'HLA-DPA1*01:07-DPB1*20:01',
         'HLA-DPA1*01:07-DPB1*21:01', 'HLA-DPA1*01:07-DPB1*22:01',
         'HLA-DPA1*01:07-DPB1*23:01', 'HLA-DPA1*01:07-DPB1*24:01', 'HLA-DPA1*01:07-DPB1*25:01', 'HLA-DPA1*01:07-DPB1*26:01',
         'HLA-DPA1*01:07-DPB1*27:01', 'HLA-DPA1*01:07-DPB1*28:01',
         'HLA-DPA1*01:07-DPB1*29:01', 'HLA-DPA1*01:07-DPB1*30:01', 'HLA-DPA1*01:07-DPB1*31:01', 'HLA-DPA1*01:07-DPB1*32:01',
         'HLA-DPA1*01:07-DPB1*33:01', 'HLA-DPA1*01:07-DPB1*34:01',
         'HLA-DPA1*01:07-DPB1*35:01', 'HLA-DPA1*01:07-DPB1*36:01', 'HLA-DPA1*01:07-DPB1*37:01', 'HLA-DPA1*01:07-DPB1*38:01',
         'HLA-DPA1*01:07-DPB1*39:01', 'HLA-DPA1*01:07-DPB1*40:01',
         'HLA-DPA1*01:07-DPB1*41:01', 'HLA-DPA1*01:07-DPB1*44:01', 'HLA-DPA1*01:07-DPB1*45:01', 'HLA-DPA1*01:07-DPB1*46:01',
         'HLA-DPA1*01:07-DPB1*47:01', 'HLA-DPA1*01:07-DPB1*48:01',
         'HLA-DPA1*01:07-DPB1*49:01', 'HLA-DPA1*01:07-DPB1*50:01', 'HLA-DPA1*01:07-DPB1*51:01', 'HLA-DPA1*01:07-DPB1*52:01',
         'HLA-DPA1*01:07-DPB1*53:01', 'HLA-DPA1*01:07-DPB1*54:01',
         'HLA-DPA1*01:07-DPB1*55:01', 'HLA-DPA1*01:07-DPB1*56:01', 'HLA-DPA1*01:07-DPB1*58:01', 'HLA-DPA1*01:07-DPB1*59:01',
         'HLA-DPA1*01:07-DPB1*60:01', 'HLA-DPA1*01:07-DPB1*62:01',
         'HLA-DPA1*01:07-DPB1*63:01', 'HLA-DPA1*01:07-DPB1*65:01', 'HLA-DPA1*01:07-DPB1*66:01', 'HLA-DPA1*01:07-DPB1*67:01',
         'HLA-DPA1*01:07-DPB1*68:01', 'HLA-DPA1*01:07-DPB1*69:01',
         'HLA-DPA1*01:07-DPB1*70:01', 'HLA-DPA1*01:07-DPB1*71:01', 'HLA-DPA1*01:07-DPB1*72:01', 'HLA-DPA1*01:07-DPB1*73:01',
         'HLA-DPA1*01:07-DPB1*74:01', 'HLA-DPA1*01:07-DPB1*75:01',
         'HLA-DPA1*01:07-DPB1*76:01', 'HLA-DPA1*01:07-DPB1*77:01', 'HLA-DPA1*01:07-DPB1*78:01', 'HLA-DPA1*01:07-DPB1*79:01',
         'HLA-DPA1*01:07-DPB1*80:01', 'HLA-DPA1*01:07-DPB1*81:01',
         'HLA-DPA1*01:07-DPB1*82:01', 'HLA-DPA1*01:07-DPB1*83:01', 'HLA-DPA1*01:07-DPB1*84:01', 'HLA-DPA1*01:07-DPB1*85:01',
         'HLA-DPA1*01:07-DPB1*86:01', 'HLA-DPA1*01:07-DPB1*87:01',
         'HLA-DPA1*01:07-DPB1*88:01', 'HLA-DPA1*01:07-DPB1*89:01', 'HLA-DPA1*01:07-DPB1*90:01', 'HLA-DPA1*01:07-DPB1*91:01',
         'HLA-DPA1*01:07-DPB1*92:01', 'HLA-DPA1*01:07-DPB1*93:01',
         'HLA-DPA1*01:07-DPB1*94:01', 'HLA-DPA1*01:07-DPB1*95:01', 'HLA-DPA1*01:07-DPB1*96:01', 'HLA-DPA1*01:07-DPB1*97:01',
         'HLA-DPA1*01:07-DPB1*98:01', 'HLA-DPA1*01:07-DPB1*99:01',
         'HLA-DPA1*01:08-DPB1*01:01', 'HLA-DPA1*01:08-DPB1*02:01', 'HLA-DPA1*01:08-DPB1*02:02', 'HLA-DPA1*01:08-DPB1*03:01',
         'HLA-DPA1*01:08-DPB1*04:01', 'HLA-DPA1*01:08-DPB1*04:02',
         'HLA-DPA1*01:08-DPB1*05:01', 'HLA-DPA1*01:08-DPB1*06:01', 'HLA-DPA1*01:08-DPB1*08:01', 'HLA-DPA1*01:08-DPB1*09:01',
         'HLA-DPA1*01:08-DPB1*10:001', 'HLA-DPA1*01:08-DPB1*10:01',
         'HLA-DPA1*01:08-DPB1*10:101', 'HLA-DPA1*01:08-DPB1*10:201', 'HLA-DPA1*01:08-DPB1*10:301', 'HLA-DPA1*01:08-DPB1*10:401',
         'HLA-DPA1*01:08-DPB1*10:501', 'HLA-DPA1*01:08-DPB1*10:601',
         'HLA-DPA1*01:08-DPB1*10:701', 'HLA-DPA1*01:08-DPB1*10:801', 'HLA-DPA1*01:08-DPB1*10:901', 'HLA-DPA1*01:08-DPB1*11:001',
         'HLA-DPA1*01:08-DPB1*11:01', 'HLA-DPA1*01:08-DPB1*11:101',
         'HLA-DPA1*01:08-DPB1*11:201', 'HLA-DPA1*01:08-DPB1*11:301', 'HLA-DPA1*01:08-DPB1*11:401', 'HLA-DPA1*01:08-DPB1*11:501',
         'HLA-DPA1*01:08-DPB1*11:601', 'HLA-DPA1*01:08-DPB1*11:701',
         'HLA-DPA1*01:08-DPB1*11:801', 'HLA-DPA1*01:08-DPB1*11:901', 'HLA-DPA1*01:08-DPB1*12:101', 'HLA-DPA1*01:08-DPB1*12:201',
         'HLA-DPA1*01:08-DPB1*12:301', 'HLA-DPA1*01:08-DPB1*12:401',
         'HLA-DPA1*01:08-DPB1*12:501', 'HLA-DPA1*01:08-DPB1*12:601', 'HLA-DPA1*01:08-DPB1*12:701', 'HLA-DPA1*01:08-DPB1*12:801',
         'HLA-DPA1*01:08-DPB1*12:901', 'HLA-DPA1*01:08-DPB1*13:001',
         'HLA-DPA1*01:08-DPB1*13:01', 'HLA-DPA1*01:08-DPB1*13:101', 'HLA-DPA1*01:08-DPB1*13:201', 'HLA-DPA1*01:08-DPB1*13:301',
         'HLA-DPA1*01:08-DPB1*13:401', 'HLA-DPA1*01:08-DPB1*14:01',
         'HLA-DPA1*01:08-DPB1*15:01', 'HLA-DPA1*01:08-DPB1*16:01', 'HLA-DPA1*01:08-DPB1*17:01', 'HLA-DPA1*01:08-DPB1*18:01',
         'HLA-DPA1*01:08-DPB1*19:01', 'HLA-DPA1*01:08-DPB1*20:01',
         'HLA-DPA1*01:08-DPB1*21:01', 'HLA-DPA1*01:08-DPB1*22:01', 'HLA-DPA1*01:08-DPB1*23:01', 'HLA-DPA1*01:08-DPB1*24:01',
         'HLA-DPA1*01:08-DPB1*25:01', 'HLA-DPA1*01:08-DPB1*26:01',
         'HLA-DPA1*01:08-DPB1*27:01', 'HLA-DPA1*01:08-DPB1*28:01', 'HLA-DPA1*01:08-DPB1*29:01', 'HLA-DPA1*01:08-DPB1*30:01',
         'HLA-DPA1*01:08-DPB1*31:01', 'HLA-DPA1*01:08-DPB1*32:01',
         'HLA-DPA1*01:08-DPB1*33:01', 'HLA-DPA1*01:08-DPB1*34:01', 'HLA-DPA1*01:08-DPB1*35:01', 'HLA-DPA1*01:08-DPB1*36:01',
         'HLA-DPA1*01:08-DPB1*37:01', 'HLA-DPA1*01:08-DPB1*38:01',
         'HLA-DPA1*01:08-DPB1*39:01', 'HLA-DPA1*01:08-DPB1*40:01', 'HLA-DPA1*01:08-DPB1*41:01', 'HLA-DPA1*01:08-DPB1*44:01',
         'HLA-DPA1*01:08-DPB1*45:01', 'HLA-DPA1*01:08-DPB1*46:01',
         'HLA-DPA1*01:08-DPB1*47:01', 'HLA-DPA1*01:08-DPB1*48:01', 'HLA-DPA1*01:08-DPB1*49:01', 'HLA-DPA1*01:08-DPB1*50:01',
         'HLA-DPA1*01:08-DPB1*51:01', 'HLA-DPA1*01:08-DPB1*52:01',
         'HLA-DPA1*01:08-DPB1*53:01', 'HLA-DPA1*01:08-DPB1*54:01', 'HLA-DPA1*01:08-DPB1*55:01', 'HLA-DPA1*01:08-DPB1*56:01',
         'HLA-DPA1*01:08-DPB1*58:01', 'HLA-DPA1*01:08-DPB1*59:01',
         'HLA-DPA1*01:08-DPB1*60:01', 'HLA-DPA1*01:08-DPB1*62:01', 'HLA-DPA1*01:08-DPB1*63:01', 'HLA-DPA1*01:08-DPB1*65:01',
         'HLA-DPA1*01:08-DPB1*66:01', 'HLA-DPA1*01:08-DPB1*67:01',
         'HLA-DPA1*01:08-DPB1*68:01', 'HLA-DPA1*01:08-DPB1*69:01', 'HLA-DPA1*01:08-DPB1*70:01', 'HLA-DPA1*01:08-DPB1*71:01',
         'HLA-DPA1*01:08-DPB1*72:01', 'HLA-DPA1*01:08-DPB1*73:01',
         'HLA-DPA1*01:08-DPB1*74:01', 'HLA-DPA1*01:08-DPB1*75:01', 'HLA-DPA1*01:08-DPB1*76:01', 'HLA-DPA1*01:08-DPB1*77:01',
         'HLA-DPA1*01:08-DPB1*78:01', 'HLA-DPA1*01:08-DPB1*79:01',
         'HLA-DPA1*01:08-DPB1*80:01', 'HLA-DPA1*01:08-DPB1*81:01', 'HLA-DPA1*01:08-DPB1*82:01', 'HLA-DPA1*01:08-DPB1*83:01',
         'HLA-DPA1*01:08-DPB1*84:01', 'HLA-DPA1*01:08-DPB1*85:01',
         'HLA-DPA1*01:08-DPB1*86:01', 'HLA-DPA1*01:08-DPB1*87:01', 'HLA-DPA1*01:08-DPB1*88:01', 'HLA-DPA1*01:08-DPB1*89:01',
         'HLA-DPA1*01:08-DPB1*90:01', 'HLA-DPA1*01:08-DPB1*91:01',
         'HLA-DPA1*01:08-DPB1*92:01', 'HLA-DPA1*01:08-DPB1*93:01', 'HLA-DPA1*01:08-DPB1*94:01', 'HLA-DPA1*01:08-DPB1*95:01',
         'HLA-DPA1*01:08-DPB1*96:01', 'HLA-DPA1*01:08-DPB1*97:01',
         'HLA-DPA1*01:08-DPB1*98:01', 'HLA-DPA1*01:08-DPB1*99:01', 'HLA-DPA1*01:09-DPB1*01:01', 'HLA-DPA1*01:09-DPB1*02:01',
         'HLA-DPA1*01:09-DPB1*02:02', 'HLA-DPA1*01:09-DPB1*03:01',
         'HLA-DPA1*01:09-DPB1*04:01', 'HLA-DPA1*01:09-DPB1*04:02', 'HLA-DPA1*01:09-DPB1*05:01', 'HLA-DPA1*01:09-DPB1*06:01',
         'HLA-DPA1*01:09-DPB1*08:01', 'HLA-DPA1*01:09-DPB1*09:01',
         'HLA-DPA1*01:09-DPB1*10:001', 'HLA-DPA1*01:09-DPB1*10:01', 'HLA-DPA1*01:09-DPB1*10:101', 'HLA-DPA1*01:09-DPB1*10:201',
         'HLA-DPA1*01:09-DPB1*10:301', 'HLA-DPA1*01:09-DPB1*10:401',
         'HLA-DPA1*01:09-DPB1*10:501', 'HLA-DPA1*01:09-DPB1*10:601', 'HLA-DPA1*01:09-DPB1*10:701', 'HLA-DPA1*01:09-DPB1*10:801',
         'HLA-DPA1*01:09-DPB1*10:901', 'HLA-DPA1*01:09-DPB1*11:001',
         'HLA-DPA1*01:09-DPB1*11:01', 'HLA-DPA1*01:09-DPB1*11:101', 'HLA-DPA1*01:09-DPB1*11:201', 'HLA-DPA1*01:09-DPB1*11:301',
         'HLA-DPA1*01:09-DPB1*11:401', 'HLA-DPA1*01:09-DPB1*11:501',
         'HLA-DPA1*01:09-DPB1*11:601', 'HLA-DPA1*01:09-DPB1*11:701', 'HLA-DPA1*01:09-DPB1*11:801', 'HLA-DPA1*01:09-DPB1*11:901',
         'HLA-DPA1*01:09-DPB1*12:101', 'HLA-DPA1*01:09-DPB1*12:201',
         'HLA-DPA1*01:09-DPB1*12:301', 'HLA-DPA1*01:09-DPB1*12:401', 'HLA-DPA1*01:09-DPB1*12:501', 'HLA-DPA1*01:09-DPB1*12:601',
         'HLA-DPA1*01:09-DPB1*12:701', 'HLA-DPA1*01:09-DPB1*12:801',
         'HLA-DPA1*01:09-DPB1*12:901', 'HLA-DPA1*01:09-DPB1*13:001', 'HLA-DPA1*01:09-DPB1*13:01', 'HLA-DPA1*01:09-DPB1*13:101',
         'HLA-DPA1*01:09-DPB1*13:201', 'HLA-DPA1*01:09-DPB1*13:301',
         'HLA-DPA1*01:09-DPB1*13:401', 'HLA-DPA1*01:09-DPB1*14:01', 'HLA-DPA1*01:09-DPB1*15:01', 'HLA-DPA1*01:09-DPB1*16:01',
         'HLA-DPA1*01:09-DPB1*17:01', 'HLA-DPA1*01:09-DPB1*18:01',
         'HLA-DPA1*01:09-DPB1*19:01', 'HLA-DPA1*01:09-DPB1*20:01', 'HLA-DPA1*01:09-DPB1*21:01', 'HLA-DPA1*01:09-DPB1*22:01',
         'HLA-DPA1*01:09-DPB1*23:01', 'HLA-DPA1*01:09-DPB1*24:01',
         'HLA-DPA1*01:09-DPB1*25:01', 'HLA-DPA1*01:09-DPB1*26:01', 'HLA-DPA1*01:09-DPB1*27:01', 'HLA-DPA1*01:09-DPB1*28:01',
         'HLA-DPA1*01:09-DPB1*29:01', 'HLA-DPA1*01:09-DPB1*30:01',
         'HLA-DPA1*01:09-DPB1*31:01', 'HLA-DPA1*01:09-DPB1*32:01', 'HLA-DPA1*01:09-DPB1*33:01', 'HLA-DPA1*01:09-DPB1*34:01',
         'HLA-DPA1*01:09-DPB1*35:01', 'HLA-DPA1*01:09-DPB1*36:01',
         'HLA-DPA1*01:09-DPB1*37:01', 'HLA-DPA1*01:09-DPB1*38:01', 'HLA-DPA1*01:09-DPB1*39:01', 'HLA-DPA1*01:09-DPB1*40:01',
         'HLA-DPA1*01:09-DPB1*41:01', 'HLA-DPA1*01:09-DPB1*44:01',
         'HLA-DPA1*01:09-DPB1*45:01', 'HLA-DPA1*01:09-DPB1*46:01', 'HLA-DPA1*01:09-DPB1*47:01', 'HLA-DPA1*01:09-DPB1*48:01',
         'HLA-DPA1*01:09-DPB1*49:01', 'HLA-DPA1*01:09-DPB1*50:01',
         'HLA-DPA1*01:09-DPB1*51:01', 'HLA-DPA1*01:09-DPB1*52:01', 'HLA-DPA1*01:09-DPB1*53:01', 'HLA-DPA1*01:09-DPB1*54:01',
         'HLA-DPA1*01:09-DPB1*55:01', 'HLA-DPA1*01:09-DPB1*56:01',
         'HLA-DPA1*01:09-DPB1*58:01', 'HLA-DPA1*01:09-DPB1*59:01', 'HLA-DPA1*01:09-DPB1*60:01', 'HLA-DPA1*01:09-DPB1*62:01',
         'HLA-DPA1*01:09-DPB1*63:01', 'HLA-DPA1*01:09-DPB1*65:01',
         'HLA-DPA1*01:09-DPB1*66:01', 'HLA-DPA1*01:09-DPB1*67:01', 'HLA-DPA1*01:09-DPB1*68:01', 'HLA-DPA1*01:09-DPB1*69:01',
         'HLA-DPA1*01:09-DPB1*70:01', 'HLA-DPA1*01:09-DPB1*71:01',
         'HLA-DPA1*01:09-DPB1*72:01', 'HLA-DPA1*01:09-DPB1*73:01', 'HLA-DPA1*01:09-DPB1*74:01', 'HLA-DPA1*01:09-DPB1*75:01',
         'HLA-DPA1*01:09-DPB1*76:01', 'HLA-DPA1*01:09-DPB1*77:01',
         'HLA-DPA1*01:09-DPB1*78:01', 'HLA-DPA1*01:09-DPB1*79:01', 'HLA-DPA1*01:09-DPB1*80:01', 'HLA-DPA1*01:09-DPB1*81:01',
         'HLA-DPA1*01:09-DPB1*82:01', 'HLA-DPA1*01:09-DPB1*83:01',
         'HLA-DPA1*01:09-DPB1*84:01', 'HLA-DPA1*01:09-DPB1*85:01', 'HLA-DPA1*01:09-DPB1*86:01', 'HLA-DPA1*01:09-DPB1*87:01',
         'HLA-DPA1*01:09-DPB1*88:01', 'HLA-DPA1*01:09-DPB1*89:01',
         'HLA-DPA1*01:09-DPB1*90:01', 'HLA-DPA1*01:09-DPB1*91:01', 'HLA-DPA1*01:09-DPB1*92:01', 'HLA-DPA1*01:09-DPB1*93:01',
         'HLA-DPA1*01:09-DPB1*94:01', 'HLA-DPA1*01:09-DPB1*95:01',
         'HLA-DPA1*01:09-DPB1*96:01', 'HLA-DPA1*01:09-DPB1*97:01', 'HLA-DPA1*01:09-DPB1*98:01', 'HLA-DPA1*01:09-DPB1*99:01',
         'HLA-DPA1*01:10-DPB1*01:01', 'HLA-DPA1*01:10-DPB1*02:01',
         'HLA-DPA1*01:10-DPB1*02:02', 'HLA-DPA1*01:10-DPB1*03:01', 'HLA-DPA1*01:10-DPB1*04:01', 'HLA-DPA1*01:10-DPB1*04:02',
         'HLA-DPA1*01:10-DPB1*05:01', 'HLA-DPA1*01:10-DPB1*06:01',
         'HLA-DPA1*01:10-DPB1*08:01', 'HLA-DPA1*01:10-DPB1*09:01', 'HLA-DPA1*01:10-DPB1*10:001', 'HLA-DPA1*01:10-DPB1*10:01',
         'HLA-DPA1*01:10-DPB1*10:101', 'HLA-DPA1*01:10-DPB1*10:201',
         'HLA-DPA1*01:10-DPB1*10:301', 'HLA-DPA1*01:10-DPB1*10:401', 'HLA-DPA1*01:10-DPB1*10:501', 'HLA-DPA1*01:10-DPB1*10:601',
         'HLA-DPA1*01:10-DPB1*10:701', 'HLA-DPA1*01:10-DPB1*10:801',
         'HLA-DPA1*01:10-DPB1*10:901', 'HLA-DPA1*01:10-DPB1*11:001', 'HLA-DPA1*01:10-DPB1*11:01', 'HLA-DPA1*01:10-DPB1*11:101',
         'HLA-DPA1*01:10-DPB1*11:201', 'HLA-DPA1*01:10-DPB1*11:301',
         'HLA-DPA1*01:10-DPB1*11:401', 'HLA-DPA1*01:10-DPB1*11:501', 'HLA-DPA1*01:10-DPB1*11:601', 'HLA-DPA1*01:10-DPB1*11:701',
         'HLA-DPA1*01:10-DPB1*11:801', 'HLA-DPA1*01:10-DPB1*11:901',
         'HLA-DPA1*01:10-DPB1*12:101', 'HLA-DPA1*01:10-DPB1*12:201', 'HLA-DPA1*01:10-DPB1*12:301', 'HLA-DPA1*01:10-DPB1*12:401',
         'HLA-DPA1*01:10-DPB1*12:501', 'HLA-DPA1*01:10-DPB1*12:601',
         'HLA-DPA1*01:10-DPB1*12:701', 'HLA-DPA1*01:10-DPB1*12:801', 'HLA-DPA1*01:10-DPB1*12:901', 'HLA-DPA1*01:10-DPB1*13:001',
         'HLA-DPA1*01:10-DPB1*13:01', 'HLA-DPA1*01:10-DPB1*13:101',
         'HLA-DPA1*01:10-DPB1*13:201', 'HLA-DPA1*01:10-DPB1*13:301', 'HLA-DPA1*01:10-DPB1*13:401', 'HLA-DPA1*01:10-DPB1*14:01',
         'HLA-DPA1*01:10-DPB1*15:01', 'HLA-DPA1*01:10-DPB1*16:01',
         'HLA-DPA1*01:10-DPB1*17:01', 'HLA-DPA1*01:10-DPB1*18:01', 'HLA-DPA1*01:10-DPB1*19:01', 'HLA-DPA1*01:10-DPB1*20:01',
         'HLA-DPA1*01:10-DPB1*21:01', 'HLA-DPA1*01:10-DPB1*22:01',
         'HLA-DPA1*01:10-DPB1*23:01', 'HLA-DPA1*01:10-DPB1*24:01', 'HLA-DPA1*01:10-DPB1*25:01', 'HLA-DPA1*01:10-DPB1*26:01',
         'HLA-DPA1*01:10-DPB1*27:01', 'HLA-DPA1*01:10-DPB1*28:01',
         'HLA-DPA1*01:10-DPB1*29:01', 'HLA-DPA1*01:10-DPB1*30:01', 'HLA-DPA1*01:10-DPB1*31:01', 'HLA-DPA1*01:10-DPB1*32:01',
         'HLA-DPA1*01:10-DPB1*33:01', 'HLA-DPA1*01:10-DPB1*34:01',
         'HLA-DPA1*01:10-DPB1*35:01', 'HLA-DPA1*01:10-DPB1*36:01', 'HLA-DPA1*01:10-DPB1*37:01', 'HLA-DPA1*01:10-DPB1*38:01',
         'HLA-DPA1*01:10-DPB1*39:01', 'HLA-DPA1*01:10-DPB1*40:01',
         'HLA-DPA1*01:10-DPB1*41:01', 'HLA-DPA1*01:10-DPB1*44:01', 'HLA-DPA1*01:10-DPB1*45:01', 'HLA-DPA1*01:10-DPB1*46:01',
         'HLA-DPA1*01:10-DPB1*47:01', 'HLA-DPA1*01:10-DPB1*48:01',
         'HLA-DPA1*01:10-DPB1*49:01', 'HLA-DPA1*01:10-DPB1*50:01', 'HLA-DPA1*01:10-DPB1*51:01', 'HLA-DPA1*01:10-DPB1*52:01',
         'HLA-DPA1*01:10-DPB1*53:01', 'HLA-DPA1*01:10-DPB1*54:01',
         'HLA-DPA1*01:10-DPB1*55:01', 'HLA-DPA1*01:10-DPB1*56:01', 'HLA-DPA1*01:10-DPB1*58:01', 'HLA-DPA1*01:10-DPB1*59:01',
         'HLA-DPA1*01:10-DPB1*60:01', 'HLA-DPA1*01:10-DPB1*62:01',
         'HLA-DPA1*01:10-DPB1*63:01', 'HLA-DPA1*01:10-DPB1*65:01', 'HLA-DPA1*01:10-DPB1*66:01', 'HLA-DPA1*01:10-DPB1*67:01',
         'HLA-DPA1*01:10-DPB1*68:01', 'HLA-DPA1*01:10-DPB1*69:01',
         'HLA-DPA1*01:10-DPB1*70:01', 'HLA-DPA1*01:10-DPB1*71:01', 'HLA-DPA1*01:10-DPB1*72:01', 'HLA-DPA1*01:10-DPB1*73:01',
         'HLA-DPA1*01:10-DPB1*74:01', 'HLA-DPA1*01:10-DPB1*75:01',
         'HLA-DPA1*01:10-DPB1*76:01', 'HLA-DPA1*01:10-DPB1*77:01', 'HLA-DPA1*01:10-DPB1*78:01', 'HLA-DPA1*01:10-DPB1*79:01',
         'HLA-DPA1*01:10-DPB1*80:01', 'HLA-DPA1*01:10-DPB1*81:01',
         'HLA-DPA1*01:10-DPB1*82:01', 'HLA-DPA1*01:10-DPB1*83:01', 'HLA-DPA1*01:10-DPB1*84:01', 'HLA-DPA1*01:10-DPB1*85:01',
         'HLA-DPA1*01:10-DPB1*86:01', 'HLA-DPA1*01:10-DPB1*87:01',
         'HLA-DPA1*01:10-DPB1*88:01', 'HLA-DPA1*01:10-DPB1*89:01', 'HLA-DPA1*01:10-DPB1*90:01', 'HLA-DPA1*01:10-DPB1*91:01',
         'HLA-DPA1*01:10-DPB1*92:01', 'HLA-DPA1*01:10-DPB1*93:01',
         'HLA-DPA1*01:10-DPB1*94:01', 'HLA-DPA1*01:10-DPB1*95:01', 'HLA-DPA1*01:10-DPB1*96:01', 'HLA-DPA1*01:10-DPB1*97:01',
         'HLA-DPA1*01:10-DPB1*98:01', 'HLA-DPA1*01:10-DPB1*99:01',
         'HLA-DPA1*02:01-DPB1*01:01', 'HLA-DPA1*02:01-DPB1*02:01', 'HLA-DPA1*02:01-DPB1*02:02', 'HLA-DPA1*02:01-DPB1*03:01',
         'HLA-DPA1*02:01-DPB1*04:01', 'HLA-DPA1*02:01-DPB1*04:02',
         'HLA-DPA1*02:01-DPB1*05:01', 'HLA-DPA1*02:01-DPB1*06:01', 'HLA-DPA1*02:01-DPB1*08:01', 'HLA-DPA1*02:01-DPB1*09:01',
         'HLA-DPA1*02:01-DPB1*10:001', 'HLA-DPA1*02:01-DPB1*10:01',
         'HLA-DPA1*02:01-DPB1*10:101', 'HLA-DPA1*02:01-DPB1*10:201', 'HLA-DPA1*02:01-DPB1*10:301', 'HLA-DPA1*02:01-DPB1*10:401',
         'HLA-DPA1*02:01-DPB1*10:501', 'HLA-DPA1*02:01-DPB1*10:601',
         'HLA-DPA1*02:01-DPB1*10:701', 'HLA-DPA1*02:01-DPB1*10:801', 'HLA-DPA1*02:01-DPB1*10:901', 'HLA-DPA1*02:01-DPB1*11:001',
         'HLA-DPA1*02:01-DPB1*11:01', 'HLA-DPA1*02:01-DPB1*11:101',
         'HLA-DPA1*02:01-DPB1*11:201', 'HLA-DPA1*02:01-DPB1*11:301', 'HLA-DPA1*02:01-DPB1*11:401', 'HLA-DPA1*02:01-DPB1*11:501',
         'HLA-DPA1*02:01-DPB1*11:601', 'HLA-DPA1*02:01-DPB1*11:701',
         'HLA-DPA1*02:01-DPB1*11:801', 'HLA-DPA1*02:01-DPB1*11:901', 'HLA-DPA1*02:01-DPB1*12:101', 'HLA-DPA1*02:01-DPB1*12:201',
         'HLA-DPA1*02:01-DPB1*12:301', 'HLA-DPA1*02:01-DPB1*12:401',
         'HLA-DPA1*02:01-DPB1*12:501', 'HLA-DPA1*02:01-DPB1*12:601', 'HLA-DPA1*02:01-DPB1*12:701', 'HLA-DPA1*02:01-DPB1*12:801',
         'HLA-DPA1*02:01-DPB1*12:901', 'HLA-DPA1*02:01-DPB1*13:001',
         'HLA-DPA1*02:01-DPB1*13:01', 'HLA-DPA1*02:01-DPB1*13:101', 'HLA-DPA1*02:01-DPB1*13:201', 'HLA-DPA1*02:01-DPB1*13:301',
         'HLA-DPA1*02:01-DPB1*13:401', 'HLA-DPA1*02:01-DPB1*14:01',
         'HLA-DPA1*02:01-DPB1*15:01', 'HLA-DPA1*02:01-DPB1*16:01', 'HLA-DPA1*02:01-DPB1*17:01', 'HLA-DPA1*02:01-DPB1*18:01',
         'HLA-DPA1*02:01-DPB1*19:01', 'HLA-DPA1*02:01-DPB1*20:01',
         'HLA-DPA1*02:01-DPB1*21:01', 'HLA-DPA1*02:01-DPB1*22:01', 'HLA-DPA1*02:01-DPB1*23:01', 'HLA-DPA1*02:01-DPB1*24:01',
         'HLA-DPA1*02:01-DPB1*25:01', 'HLA-DPA1*02:01-DPB1*26:01',
         'HLA-DPA1*02:01-DPB1*27:01', 'HLA-DPA1*02:01-DPB1*28:01', 'HLA-DPA1*02:01-DPB1*29:01', 'HLA-DPA1*02:01-DPB1*30:01',
         'HLA-DPA1*02:01-DPB1*31:01', 'HLA-DPA1*02:01-DPB1*32:01',
         'HLA-DPA1*02:01-DPB1*33:01', 'HLA-DPA1*02:01-DPB1*34:01', 'HLA-DPA1*02:01-DPB1*35:01', 'HLA-DPA1*02:01-DPB1*36:01',
         'HLA-DPA1*02:01-DPB1*37:01', 'HLA-DPA1*02:01-DPB1*38:01',
         'HLA-DPA1*02:01-DPB1*39:01', 'HLA-DPA1*02:01-DPB1*40:01', 'HLA-DPA1*02:01-DPB1*41:01', 'HLA-DPA1*02:01-DPB1*44:01',
         'HLA-DPA1*02:01-DPB1*45:01', 'HLA-DPA1*02:01-DPB1*46:01',
         'HLA-DPA1*02:01-DPB1*47:01', 'HLA-DPA1*02:01-DPB1*48:01', 'HLA-DPA1*02:01-DPB1*49:01', 'HLA-DPA1*02:01-DPB1*50:01',
         'HLA-DPA1*02:01-DPB1*51:01', 'HLA-DPA1*02:01-DPB1*52:01',
         'HLA-DPA1*02:01-DPB1*53:01', 'HLA-DPA1*02:01-DPB1*54:01', 'HLA-DPA1*02:01-DPB1*55:01', 'HLA-DPA1*02:01-DPB1*56:01',
         'HLA-DPA1*02:01-DPB1*58:01', 'HLA-DPA1*02:01-DPB1*59:01',
         'HLA-DPA1*02:01-DPB1*60:01', 'HLA-DPA1*02:01-DPB1*62:01', 'HLA-DPA1*02:01-DPB1*63:01', 'HLA-DPA1*02:01-DPB1*65:01',
         'HLA-DPA1*02:01-DPB1*66:01', 'HLA-DPA1*02:01-DPB1*67:01',
         'HLA-DPA1*02:01-DPB1*68:01', 'HLA-DPA1*02:01-DPB1*69:01', 'HLA-DPA1*02:01-DPB1*70:01', 'HLA-DPA1*02:01-DPB1*71:01',
         'HLA-DPA1*02:01-DPB1*72:01', 'HLA-DPA1*02:01-DPB1*73:01',
         'HLA-DPA1*02:01-DPB1*74:01', 'HLA-DPA1*02:01-DPB1*75:01', 'HLA-DPA1*02:01-DPB1*76:01', 'HLA-DPA1*02:01-DPB1*77:01',
         'HLA-DPA1*02:01-DPB1*78:01', 'HLA-DPA1*02:01-DPB1*79:01',
         'HLA-DPA1*02:01-DPB1*80:01', 'HLA-DPA1*02:01-DPB1*81:01', 'HLA-DPA1*02:01-DPB1*82:01', 'HLA-DPA1*02:01-DPB1*83:01',
         'HLA-DPA1*02:01-DPB1*84:01', 'HLA-DPA1*02:01-DPB1*85:01',
         'HLA-DPA1*02:01-DPB1*86:01', 'HLA-DPA1*02:01-DPB1*87:01', 'HLA-DPA1*02:01-DPB1*88:01', 'HLA-DPA1*02:01-DPB1*89:01',
         'HLA-DPA1*02:01-DPB1*90:01', 'HLA-DPA1*02:01-DPB1*91:01',
         'HLA-DPA1*02:01-DPB1*92:01', 'HLA-DPA1*02:01-DPB1*93:01', 'HLA-DPA1*02:01-DPB1*94:01', 'HLA-DPA1*02:01-DPB1*95:01',
         'HLA-DPA1*02:01-DPB1*96:01', 'HLA-DPA1*02:01-DPB1*97:01',
         'HLA-DPA1*02:01-DPB1*98:01', 'HLA-DPA1*02:01-DPB1*99:01', 'HLA-DPA1*02:02-DPB1*01:01', 'HLA-DPA1*02:02-DPB1*02:01',
         'HLA-DPA1*02:02-DPB1*02:02', 'HLA-DPA1*02:02-DPB1*03:01',
         'HLA-DPA1*02:02-DPB1*04:01', 'HLA-DPA1*02:02-DPB1*04:02', 'HLA-DPA1*02:02-DPB1*05:01', 'HLA-DPA1*02:02-DPB1*06:01',
         'HLA-DPA1*02:02-DPB1*08:01', 'HLA-DPA1*02:02-DPB1*09:01',
         'HLA-DPA1*02:02-DPB1*10:001', 'HLA-DPA1*02:02-DPB1*10:01', 'HLA-DPA1*02:02-DPB1*10:101', 'HLA-DPA1*02:02-DPB1*10:201',
         'HLA-DPA1*02:02-DPB1*10:301', 'HLA-DPA1*02:02-DPB1*10:401',
         'HLA-DPA1*02:02-DPB1*10:501', 'HLA-DPA1*02:02-DPB1*10:601', 'HLA-DPA1*02:02-DPB1*10:701', 'HLA-DPA1*02:02-DPB1*10:801',
         'HLA-DPA1*02:02-DPB1*10:901', 'HLA-DPA1*02:02-DPB1*11:001',
         'HLA-DPA1*02:02-DPB1*11:01', 'HLA-DPA1*02:02-DPB1*11:101', 'HLA-DPA1*02:02-DPB1*11:201', 'HLA-DPA1*02:02-DPB1*11:301',
         'HLA-DPA1*02:02-DPB1*11:401', 'HLA-DPA1*02:02-DPB1*11:501',
         'HLA-DPA1*02:02-DPB1*11:601', 'HLA-DPA1*02:02-DPB1*11:701', 'HLA-DPA1*02:02-DPB1*11:801', 'HLA-DPA1*02:02-DPB1*11:901',
         'HLA-DPA1*02:02-DPB1*12:101', 'HLA-DPA1*02:02-DPB1*12:201',
         'HLA-DPA1*02:02-DPB1*12:301', 'HLA-DPA1*02:02-DPB1*12:401', 'HLA-DPA1*02:02-DPB1*12:501', 'HLA-DPA1*02:02-DPB1*12:601',
         'HLA-DPA1*02:02-DPB1*12:701', 'HLA-DPA1*02:02-DPB1*12:801',
         'HLA-DPA1*02:02-DPB1*12:901', 'HLA-DPA1*02:02-DPB1*13:001', 'HLA-DPA1*02:02-DPB1*13:01', 'HLA-DPA1*02:02-DPB1*13:101',
         'HLA-DPA1*02:02-DPB1*13:201', 'HLA-DPA1*02:02-DPB1*13:301',
         'HLA-DPA1*02:02-DPB1*13:401', 'HLA-DPA1*02:02-DPB1*14:01', 'HLA-DPA1*02:02-DPB1*15:01', 'HLA-DPA1*02:02-DPB1*16:01',
         'HLA-DPA1*02:02-DPB1*17:01', 'HLA-DPA1*02:02-DPB1*18:01',
         'HLA-DPA1*02:02-DPB1*19:01', 'HLA-DPA1*02:02-DPB1*20:01', 'HLA-DPA1*02:02-DPB1*21:01', 'HLA-DPA1*02:02-DPB1*22:01',
         'HLA-DPA1*02:02-DPB1*23:01', 'HLA-DPA1*02:02-DPB1*24:01',
         'HLA-DPA1*02:02-DPB1*25:01', 'HLA-DPA1*02:02-DPB1*26:01', 'HLA-DPA1*02:02-DPB1*27:01', 'HLA-DPA1*02:02-DPB1*28:01',
         'HLA-DPA1*02:02-DPB1*29:01', 'HLA-DPA1*02:02-DPB1*30:01',
         'HLA-DPA1*02:02-DPB1*31:01', 'HLA-DPA1*02:02-DPB1*32:01', 'HLA-DPA1*02:02-DPB1*33:01', 'HLA-DPA1*02:02-DPB1*34:01',
         'HLA-DPA1*02:02-DPB1*35:01', 'HLA-DPA1*02:02-DPB1*36:01',
         'HLA-DPA1*02:02-DPB1*37:01', 'HLA-DPA1*02:02-DPB1*38:01', 'HLA-DPA1*02:02-DPB1*39:01', 'HLA-DPA1*02:02-DPB1*40:01',
         'HLA-DPA1*02:02-DPB1*41:01', 'HLA-DPA1*02:02-DPB1*44:01',
         'HLA-DPA1*02:02-DPB1*45:01', 'HLA-DPA1*02:02-DPB1*46:01', 'HLA-DPA1*02:02-DPB1*47:01', 'HLA-DPA1*02:02-DPB1*48:01',
         'HLA-DPA1*02:02-DPB1*49:01', 'HLA-DPA1*02:02-DPB1*50:01',
         'HLA-DPA1*02:02-DPB1*51:01', 'HLA-DPA1*02:02-DPB1*52:01', 'HLA-DPA1*02:02-DPB1*53:01', 'HLA-DPA1*02:02-DPB1*54:01',
         'HLA-DPA1*02:02-DPB1*55:01', 'HLA-DPA1*02:02-DPB1*56:01',
         'HLA-DPA1*02:02-DPB1*58:01', 'HLA-DPA1*02:02-DPB1*59:01', 'HLA-DPA1*02:02-DPB1*60:01', 'HLA-DPA1*02:02-DPB1*62:01',
         'HLA-DPA1*02:02-DPB1*63:01', 'HLA-DPA1*02:02-DPB1*65:01',
         'HLA-DPA1*02:02-DPB1*66:01', 'HLA-DPA1*02:02-DPB1*67:01', 'HLA-DPA1*02:02-DPB1*68:01', 'HLA-DPA1*02:02-DPB1*69:01',
         'HLA-DPA1*02:02-DPB1*70:01', 'HLA-DPA1*02:02-DPB1*71:01',
         'HLA-DPA1*02:02-DPB1*72:01', 'HLA-DPA1*02:02-DPB1*73:01', 'HLA-DPA1*02:02-DPB1*74:01', 'HLA-DPA1*02:02-DPB1*75:01',
         'HLA-DPA1*02:02-DPB1*76:01', 'HLA-DPA1*02:02-DPB1*77:01',
         'HLA-DPA1*02:02-DPB1*78:01', 'HLA-DPA1*02:02-DPB1*79:01', 'HLA-DPA1*02:02-DPB1*80:01', 'HLA-DPA1*02:02-DPB1*81:01',
         'HLA-DPA1*02:02-DPB1*82:01', 'HLA-DPA1*02:02-DPB1*83:01',
         'HLA-DPA1*02:02-DPB1*84:01', 'HLA-DPA1*02:02-DPB1*85:01', 'HLA-DPA1*02:02-DPB1*86:01', 'HLA-DPA1*02:02-DPB1*87:01',
         'HLA-DPA1*02:02-DPB1*88:01', 'HLA-DPA1*02:02-DPB1*89:01',
         'HLA-DPA1*02:02-DPB1*90:01', 'HLA-DPA1*02:02-DPB1*91:01', 'HLA-DPA1*02:02-DPB1*92:01', 'HLA-DPA1*02:02-DPB1*93:01',
         'HLA-DPA1*02:02-DPB1*94:01', 'HLA-DPA1*02:02-DPB1*95:01',
         'HLA-DPA1*02:02-DPB1*96:01', 'HLA-DPA1*02:02-DPB1*97:01', 'HLA-DPA1*02:02-DPB1*98:01', 'HLA-DPA1*02:02-DPB1*99:01',
         'HLA-DPA1*02:03-DPB1*01:01', 'HLA-DPA1*02:03-DPB1*02:01',
         'HLA-DPA1*02:03-DPB1*02:02', 'HLA-DPA1*02:03-DPB1*03:01', 'HLA-DPA1*02:03-DPB1*04:01', 'HLA-DPA1*02:03-DPB1*04:02',
         'HLA-DPA1*02:03-DPB1*05:01', 'HLA-DPA1*02:03-DPB1*06:01',
         'HLA-DPA1*02:03-DPB1*08:01', 'HLA-DPA1*02:03-DPB1*09:01', 'HLA-DPA1*02:03-DPB1*10:001', 'HLA-DPA1*02:03-DPB1*10:01',
         'HLA-DPA1*02:03-DPB1*10:101', 'HLA-DPA1*02:03-DPB1*10:201',
         'HLA-DPA1*02:03-DPB1*10:301', 'HLA-DPA1*02:03-DPB1*10:401', 'HLA-DPA1*02:03-DPB1*10:501', 'HLA-DPA1*02:03-DPB1*10:601',
         'HLA-DPA1*02:03-DPB1*10:701', 'HLA-DPA1*02:03-DPB1*10:801',
         'HLA-DPA1*02:03-DPB1*10:901', 'HLA-DPA1*02:03-DPB1*11:001', 'HLA-DPA1*02:03-DPB1*11:01', 'HLA-DPA1*02:03-DPB1*11:101',
         'HLA-DPA1*02:03-DPB1*11:201', 'HLA-DPA1*02:03-DPB1*11:301',
         'HLA-DPA1*02:03-DPB1*11:401', 'HLA-DPA1*02:03-DPB1*11:501', 'HLA-DPA1*02:03-DPB1*11:601', 'HLA-DPA1*02:03-DPB1*11:701',
         'HLA-DPA1*02:03-DPB1*11:801', 'HLA-DPA1*02:03-DPB1*11:901',
         'HLA-DPA1*02:03-DPB1*12:101', 'HLA-DPA1*02:03-DPB1*12:201', 'HLA-DPA1*02:03-DPB1*12:301', 'HLA-DPA1*02:03-DPB1*12:401',
         'HLA-DPA1*02:03-DPB1*12:501', 'HLA-DPA1*02:03-DPB1*12:601',
         'HLA-DPA1*02:03-DPB1*12:701', 'HLA-DPA1*02:03-DPB1*12:801', 'HLA-DPA1*02:03-DPB1*12:901', 'HLA-DPA1*02:03-DPB1*13:001',
         'HLA-DPA1*02:03-DPB1*13:01', 'HLA-DPA1*02:03-DPB1*13:101',
         'HLA-DPA1*02:03-DPB1*13:201', 'HLA-DPA1*02:03-DPB1*13:301', 'HLA-DPA1*02:03-DPB1*13:401', 'HLA-DPA1*02:03-DPB1*14:01',
         'HLA-DPA1*02:03-DPB1*15:01', 'HLA-DPA1*02:03-DPB1*16:01',
         'HLA-DPA1*02:03-DPB1*17:01', 'HLA-DPA1*02:03-DPB1*18:01', 'HLA-DPA1*02:03-DPB1*19:01', 'HLA-DPA1*02:03-DPB1*20:01',
         'HLA-DPA1*02:03-DPB1*21:01', 'HLA-DPA1*02:03-DPB1*22:01',
         'HLA-DPA1*02:03-DPB1*23:01', 'HLA-DPA1*02:03-DPB1*24:01', 'HLA-DPA1*02:03-DPB1*25:01', 'HLA-DPA1*02:03-DPB1*26:01',
         'HLA-DPA1*02:03-DPB1*27:01', 'HLA-DPA1*02:03-DPB1*28:01',
         'HLA-DPA1*02:03-DPB1*29:01', 'HLA-DPA1*02:03-DPB1*30:01', 'HLA-DPA1*02:03-DPB1*31:01', 'HLA-DPA1*02:03-DPB1*32:01',
         'HLA-DPA1*02:03-DPB1*33:01', 'HLA-DPA1*02:03-DPB1*34:01',
         'HLA-DPA1*02:03-DPB1*35:01', 'HLA-DPA1*02:03-DPB1*36:01', 'HLA-DPA1*02:03-DPB1*37:01', 'HLA-DPA1*02:03-DPB1*38:01',
         'HLA-DPA1*02:03-DPB1*39:01', 'HLA-DPA1*02:03-DPB1*40:01',
         'HLA-DPA1*02:03-DPB1*41:01', 'HLA-DPA1*02:03-DPB1*44:01', 'HLA-DPA1*02:03-DPB1*45:01', 'HLA-DPA1*02:03-DPB1*46:01',
         'HLA-DPA1*02:03-DPB1*47:01', 'HLA-DPA1*02:03-DPB1*48:01',
         'HLA-DPA1*02:03-DPB1*49:01', 'HLA-DPA1*02:03-DPB1*50:01', 'HLA-DPA1*02:03-DPB1*51:01', 'HLA-DPA1*02:03-DPB1*52:01',
         'HLA-DPA1*02:03-DPB1*53:01', 'HLA-DPA1*02:03-DPB1*54:01',
         'HLA-DPA1*02:03-DPB1*55:01', 'HLA-DPA1*02:03-DPB1*56:01', 'HLA-DPA1*02:03-DPB1*58:01', 'HLA-DPA1*02:03-DPB1*59:01',
         'HLA-DPA1*02:03-DPB1*60:01', 'HLA-DPA1*02:03-DPB1*62:01',
         'HLA-DPA1*02:03-DPB1*63:01', 'HLA-DPA1*02:03-DPB1*65:01', 'HLA-DPA1*02:03-DPB1*66:01', 'HLA-DPA1*02:03-DPB1*67:01',
         'HLA-DPA1*02:03-DPB1*68:01', 'HLA-DPA1*02:03-DPB1*69:01',
         'HLA-DPA1*02:03-DPB1*70:01', 'HLA-DPA1*02:03-DPB1*71:01', 'HLA-DPA1*02:03-DPB1*72:01', 'HLA-DPA1*02:03-DPB1*73:01',
         'HLA-DPA1*02:03-DPB1*74:01', 'HLA-DPA1*02:03-DPB1*75:01',
         'HLA-DPA1*02:03-DPB1*76:01', 'HLA-DPA1*02:03-DPB1*77:01', 'HLA-DPA1*02:03-DPB1*78:01', 'HLA-DPA1*02:03-DPB1*79:01',
         'HLA-DPA1*02:03-DPB1*80:01', 'HLA-DPA1*02:03-DPB1*81:01',
         'HLA-DPA1*02:03-DPB1*82:01', 'HLA-DPA1*02:03-DPB1*83:01', 'HLA-DPA1*02:03-DPB1*84:01', 'HLA-DPA1*02:03-DPB1*85:01',
         'HLA-DPA1*02:03-DPB1*86:01', 'HLA-DPA1*02:03-DPB1*87:01',
         'HLA-DPA1*02:03-DPB1*88:01', 'HLA-DPA1*02:03-DPB1*89:01', 'HLA-DPA1*02:03-DPB1*90:01', 'HLA-DPA1*02:03-DPB1*91:01',
         'HLA-DPA1*02:03-DPB1*92:01', 'HLA-DPA1*02:03-DPB1*93:01',
         'HLA-DPA1*02:03-DPB1*94:01', 'HLA-DPA1*02:03-DPB1*95:01', 'HLA-DPA1*02:03-DPB1*96:01', 'HLA-DPA1*02:03-DPB1*97:01',
         'HLA-DPA1*02:03-DPB1*98:01', 'HLA-DPA1*02:03-DPB1*99:01',
         'HLA-DPA1*02:04-DPB1*01:01', 'HLA-DPA1*02:04-DPB1*02:01', 'HLA-DPA1*02:04-DPB1*02:02', 'HLA-DPA1*02:04-DPB1*03:01',
         'HLA-DPA1*02:04-DPB1*04:01', 'HLA-DPA1*02:04-DPB1*04:02',
         'HLA-DPA1*02:04-DPB1*05:01', 'HLA-DPA1*02:04-DPB1*06:01', 'HLA-DPA1*02:04-DPB1*08:01', 'HLA-DPA1*02:04-DPB1*09:01',
         'HLA-DPA1*02:04-DPB1*10:001', 'HLA-DPA1*02:04-DPB1*10:01',
         'HLA-DPA1*02:04-DPB1*10:101', 'HLA-DPA1*02:04-DPB1*10:201', 'HLA-DPA1*02:04-DPB1*10:301', 'HLA-DPA1*02:04-DPB1*10:401',
         'HLA-DPA1*02:04-DPB1*10:501', 'HLA-DPA1*02:04-DPB1*10:601',
         'HLA-DPA1*02:04-DPB1*10:701', 'HLA-DPA1*02:04-DPB1*10:801', 'HLA-DPA1*02:04-DPB1*10:901', 'HLA-DPA1*02:04-DPB1*11:001',
         'HLA-DPA1*02:04-DPB1*11:01', 'HLA-DPA1*02:04-DPB1*11:101',
         'HLA-DPA1*02:04-DPB1*11:201', 'HLA-DPA1*02:04-DPB1*11:301', 'HLA-DPA1*02:04-DPB1*11:401', 'HLA-DPA1*02:04-DPB1*11:501',
         'HLA-DPA1*02:04-DPB1*11:601', 'HLA-DPA1*02:04-DPB1*11:701',
         'HLA-DPA1*02:04-DPB1*11:801', 'HLA-DPA1*02:04-DPB1*11:901', 'HLA-DPA1*02:04-DPB1*12:101', 'HLA-DPA1*02:04-DPB1*12:201',
         'HLA-DPA1*02:04-DPB1*12:301', 'HLA-DPA1*02:04-DPB1*12:401',
         'HLA-DPA1*02:04-DPB1*12:501', 'HLA-DPA1*02:04-DPB1*12:601', 'HLA-DPA1*02:04-DPB1*12:701', 'HLA-DPA1*02:04-DPB1*12:801',
         'HLA-DPA1*02:04-DPB1*12:901', 'HLA-DPA1*02:04-DPB1*13:001',
         'HLA-DPA1*02:04-DPB1*13:01', 'HLA-DPA1*02:04-DPB1*13:101', 'HLA-DPA1*02:04-DPB1*13:201', 'HLA-DPA1*02:04-DPB1*13:301',
         'HLA-DPA1*02:04-DPB1*13:401', 'HLA-DPA1*02:04-DPB1*14:01',
         'HLA-DPA1*02:04-DPB1*15:01', 'HLA-DPA1*02:04-DPB1*16:01', 'HLA-DPA1*02:04-DPB1*17:01', 'HLA-DPA1*02:04-DPB1*18:01',
         'HLA-DPA1*02:04-DPB1*19:01', 'HLA-DPA1*02:04-DPB1*20:01',
         'HLA-DPA1*02:04-DPB1*21:01', 'HLA-DPA1*02:04-DPB1*22:01', 'HLA-DPA1*02:04-DPB1*23:01', 'HLA-DPA1*02:04-DPB1*24:01',
         'HLA-DPA1*02:04-DPB1*25:01', 'HLA-DPA1*02:04-DPB1*26:01',
         'HLA-DPA1*02:04-DPB1*27:01', 'HLA-DPA1*02:04-DPB1*28:01', 'HLA-DPA1*02:04-DPB1*29:01', 'HLA-DPA1*02:04-DPB1*30:01',
         'HLA-DPA1*02:04-DPB1*31:01', 'HLA-DPA1*02:04-DPB1*32:01',
         'HLA-DPA1*02:04-DPB1*33:01', 'HLA-DPA1*02:04-DPB1*34:01', 'HLA-DPA1*02:04-DPB1*35:01', 'HLA-DPA1*02:04-DPB1*36:01',
         'HLA-DPA1*02:04-DPB1*37:01', 'HLA-DPA1*02:04-DPB1*38:01',
         'HLA-DPA1*02:04-DPB1*39:01', 'HLA-DPA1*02:04-DPB1*40:01', 'HLA-DPA1*02:04-DPB1*41:01', 'HLA-DPA1*02:04-DPB1*44:01',
         'HLA-DPA1*02:04-DPB1*45:01', 'HLA-DPA1*02:04-DPB1*46:01',
         'HLA-DPA1*02:04-DPB1*47:01', 'HLA-DPA1*02:04-DPB1*48:01', 'HLA-DPA1*02:04-DPB1*49:01', 'HLA-DPA1*02:04-DPB1*50:01',
         'HLA-DPA1*02:04-DPB1*51:01', 'HLA-DPA1*02:04-DPB1*52:01',
         'HLA-DPA1*02:04-DPB1*53:01', 'HLA-DPA1*02:04-DPB1*54:01', 'HLA-DPA1*02:04-DPB1*55:01', 'HLA-DPA1*02:04-DPB1*56:01',
         'HLA-DPA1*02:04-DPB1*58:01', 'HLA-DPA1*02:04-DPB1*59:01',
         'HLA-DPA1*02:04-DPB1*60:01', 'HLA-DPA1*02:04-DPB1*62:01', 'HLA-DPA1*02:04-DPB1*63:01', 'HLA-DPA1*02:04-DPB1*65:01',
         'HLA-DPA1*02:04-DPB1*66:01', 'HLA-DPA1*02:04-DPB1*67:01',
         'HLA-DPA1*02:04-DPB1*68:01', 'HLA-DPA1*02:04-DPB1*69:01', 'HLA-DPA1*02:04-DPB1*70:01', 'HLA-DPA1*02:04-DPB1*71:01',
         'HLA-DPA1*02:04-DPB1*72:01', 'HLA-DPA1*02:04-DPB1*73:01',
         'HLA-DPA1*02:04-DPB1*74:01', 'HLA-DPA1*02:04-DPB1*75:01', 'HLA-DPA1*02:04-DPB1*76:01', 'HLA-DPA1*02:04-DPB1*77:01',
         'HLA-DPA1*02:04-DPB1*78:01', 'HLA-DPA1*02:04-DPB1*79:01',
         'HLA-DPA1*02:04-DPB1*80:01', 'HLA-DPA1*02:04-DPB1*81:01', 'HLA-DPA1*02:04-DPB1*82:01', 'HLA-DPA1*02:04-DPB1*83:01',
         'HLA-DPA1*02:04-DPB1*84:01', 'HLA-DPA1*02:04-DPB1*85:01',
         'HLA-DPA1*02:04-DPB1*86:01', 'HLA-DPA1*02:04-DPB1*87:01', 'HLA-DPA1*02:04-DPB1*88:01', 'HLA-DPA1*02:04-DPB1*89:01',
         'HLA-DPA1*02:04-DPB1*90:01', 'HLA-DPA1*02:04-DPB1*91:01',
         'HLA-DPA1*02:04-DPB1*92:01', 'HLA-DPA1*02:04-DPB1*93:01', 'HLA-DPA1*02:04-DPB1*94:01', 'HLA-DPA1*02:04-DPB1*95:01',
         'HLA-DPA1*02:04-DPB1*96:01', 'HLA-DPA1*02:04-DPB1*97:01',
         'HLA-DPA1*02:04-DPB1*98:01', 'HLA-DPA1*02:04-DPB1*99:01', 'HLA-DPA1*03:01-DPB1*01:01', 'HLA-DPA1*03:01-DPB1*02:01',
         'HLA-DPA1*03:01-DPB1*02:02', 'HLA-DPA1*03:01-DPB1*03:01',
         'HLA-DPA1*03:01-DPB1*04:01', 'HLA-DPA1*03:01-DPB1*04:02', 'HLA-DPA1*03:01-DPB1*05:01', 'HLA-DPA1*03:01-DPB1*06:01',
         'HLA-DPA1*03:01-DPB1*08:01', 'HLA-DPA1*03:01-DPB1*09:01',
         'HLA-DPA1*03:01-DPB1*10:001', 'HLA-DPA1*03:01-DPB1*10:01', 'HLA-DPA1*03:01-DPB1*10:101', 'HLA-DPA1*03:01-DPB1*10:201',
         'HLA-DPA1*03:01-DPB1*10:301', 'HLA-DPA1*03:01-DPB1*10:401',
         'HLA-DPA1*03:01-DPB1*10:501', 'HLA-DPA1*03:01-DPB1*10:601', 'HLA-DPA1*03:01-DPB1*10:701', 'HLA-DPA1*03:01-DPB1*10:801',
         'HLA-DPA1*03:01-DPB1*10:901', 'HLA-DPA1*03:01-DPB1*11:001',
         'HLA-DPA1*03:01-DPB1*11:01', 'HLA-DPA1*03:01-DPB1*11:101', 'HLA-DPA1*03:01-DPB1*11:201', 'HLA-DPA1*03:01-DPB1*11:301',
         'HLA-DPA1*03:01-DPB1*11:401', 'HLA-DPA1*03:01-DPB1*11:501',
         'HLA-DPA1*03:01-DPB1*11:601', 'HLA-DPA1*03:01-DPB1*11:701', 'HLA-DPA1*03:01-DPB1*11:801', 'HLA-DPA1*03:01-DPB1*11:901',
         'HLA-DPA1*03:01-DPB1*12:101', 'HLA-DPA1*03:01-DPB1*12:201',
         'HLA-DPA1*03:01-DPB1*12:301', 'HLA-DPA1*03:01-DPB1*12:401', 'HLA-DPA1*03:01-DPB1*12:501', 'HLA-DPA1*03:01-DPB1*12:601',
         'HLA-DPA1*03:01-DPB1*12:701', 'HLA-DPA1*03:01-DPB1*12:801',
         'HLA-DPA1*03:01-DPB1*12:901', 'HLA-DPA1*03:01-DPB1*13:001', 'HLA-DPA1*03:01-DPB1*13:01', 'HLA-DPA1*03:01-DPB1*13:101',
         'HLA-DPA1*03:01-DPB1*13:201', 'HLA-DPA1*03:01-DPB1*13:301',
         'HLA-DPA1*03:01-DPB1*13:401', 'HLA-DPA1*03:01-DPB1*14:01', 'HLA-DPA1*03:01-DPB1*15:01', 'HLA-DPA1*03:01-DPB1*16:01',
         'HLA-DPA1*03:01-DPB1*17:01', 'HLA-DPA1*03:01-DPB1*18:01',
         'HLA-DPA1*03:01-DPB1*19:01', 'HLA-DPA1*03:01-DPB1*20:01', 'HLA-DPA1*03:01-DPB1*21:01', 'HLA-DPA1*03:01-DPB1*22:01',
         'HLA-DPA1*03:01-DPB1*23:01', 'HLA-DPA1*03:01-DPB1*24:01',
         'HLA-DPA1*03:01-DPB1*25:01', 'HLA-DPA1*03:01-DPB1*26:01', 'HLA-DPA1*03:01-DPB1*27:01', 'HLA-DPA1*03:01-DPB1*28:01',
         'HLA-DPA1*03:01-DPB1*29:01', 'HLA-DPA1*03:01-DPB1*30:01',
         'HLA-DPA1*03:01-DPB1*31:01', 'HLA-DPA1*03:01-DPB1*32:01', 'HLA-DPA1*03:01-DPB1*33:01', 'HLA-DPA1*03:01-DPB1*34:01',
         'HLA-DPA1*03:01-DPB1*35:01', 'HLA-DPA1*03:01-DPB1*36:01',
         'HLA-DPA1*03:01-DPB1*37:01', 'HLA-DPA1*03:01-DPB1*38:01', 'HLA-DPA1*03:01-DPB1*39:01', 'HLA-DPA1*03:01-DPB1*40:01',
         'HLA-DPA1*03:01-DPB1*41:01', 'HLA-DPA1*03:01-DPB1*44:01',
         'HLA-DPA1*03:01-DPB1*45:01', 'HLA-DPA1*03:01-DPB1*46:01', 'HLA-DPA1*03:01-DPB1*47:01', 'HLA-DPA1*03:01-DPB1*48:01',
         'HLA-DPA1*03:01-DPB1*49:01', 'HLA-DPA1*03:01-DPB1*50:01',
         'HLA-DPA1*03:01-DPB1*51:01', 'HLA-DPA1*03:01-DPB1*52:01', 'HLA-DPA1*03:01-DPB1*53:01', 'HLA-DPA1*03:01-DPB1*54:01',
         'HLA-DPA1*03:01-DPB1*55:01', 'HLA-DPA1*03:01-DPB1*56:01',
         'HLA-DPA1*03:01-DPB1*58:01', 'HLA-DPA1*03:01-DPB1*59:01', 'HLA-DPA1*03:01-DPB1*60:01', 'HLA-DPA1*03:01-DPB1*62:01',
         'HLA-DPA1*03:01-DPB1*63:01', 'HLA-DPA1*03:01-DPB1*65:01',
         'HLA-DPA1*03:01-DPB1*66:01', 'HLA-DPA1*03:01-DPB1*67:01', 'HLA-DPA1*03:01-DPB1*68:01', 'HLA-DPA1*03:01-DPB1*69:01',
         'HLA-DPA1*03:01-DPB1*70:01', 'HLA-DPA1*03:01-DPB1*71:01',
         'HLA-DPA1*03:01-DPB1*72:01', 'HLA-DPA1*03:01-DPB1*73:01', 'HLA-DPA1*03:01-DPB1*74:01', 'HLA-DPA1*03:01-DPB1*75:01',
         'HLA-DPA1*03:01-DPB1*76:01', 'HLA-DPA1*03:01-DPB1*77:01',
         'HLA-DPA1*03:01-DPB1*78:01', 'HLA-DPA1*03:01-DPB1*79:01', 'HLA-DPA1*03:01-DPB1*80:01', 'HLA-DPA1*03:01-DPB1*81:01',
         'HLA-DPA1*03:01-DPB1*82:01', 'HLA-DPA1*03:01-DPB1*83:01',
         'HLA-DPA1*03:01-DPB1*84:01', 'HLA-DPA1*03:01-DPB1*85:01', 'HLA-DPA1*03:01-DPB1*86:01', 'HLA-DPA1*03:01-DPB1*87:01',
         'HLA-DPA1*03:01-DPB1*88:01', 'HLA-DPA1*03:01-DPB1*89:01',
         'HLA-DPA1*03:01-DPB1*90:01', 'HLA-DPA1*03:01-DPB1*91:01', 'HLA-DPA1*03:01-DPB1*92:01', 'HLA-DPA1*03:01-DPB1*93:01',
         'HLA-DPA1*03:01-DPB1*94:01', 'HLA-DPA1*03:01-DPB1*95:01',
         'HLA-DPA1*03:01-DPB1*96:01', 'HLA-DPA1*03:01-DPB1*97:01', 'HLA-DPA1*03:01-DPB1*98:01', 'HLA-DPA1*03:01-DPB1*99:01',
         'HLA-DPA1*03:02-DPB1*01:01', 'HLA-DPA1*03:02-DPB1*02:01',
         'HLA-DPA1*03:02-DPB1*02:02', 'HLA-DPA1*03:02-DPB1*03:01', 'HLA-DPA1*03:02-DPB1*04:01', 'HLA-DPA1*03:02-DPB1*04:02',
         'HLA-DPA1*03:02-DPB1*05:01', 'HLA-DPA1*03:02-DPB1*06:01',
         'HLA-DPA1*03:02-DPB1*08:01', 'HLA-DPA1*03:02-DPB1*09:01', 'HLA-DPA1*03:02-DPB1*10:001', 'HLA-DPA1*03:02-DPB1*10:01',
         'HLA-DPA1*03:02-DPB1*10:101', 'HLA-DPA1*03:02-DPB1*10:201',
         'HLA-DPA1*03:02-DPB1*10:301', 'HLA-DPA1*03:02-DPB1*10:401', 'HLA-DPA1*03:02-DPB1*10:501', 'HLA-DPA1*03:02-DPB1*10:601',
         'HLA-DPA1*03:02-DPB1*10:701', 'HLA-DPA1*03:02-DPB1*10:801',
         'HLA-DPA1*03:02-DPB1*10:901', 'HLA-DPA1*03:02-DPB1*11:001', 'HLA-DPA1*03:02-DPB1*11:01', 'HLA-DPA1*03:02-DPB1*11:101',
         'HLA-DPA1*03:02-DPB1*11:201', 'HLA-DPA1*03:02-DPB1*11:301',
         'HLA-DPA1*03:02-DPB1*11:401', 'HLA-DPA1*03:02-DPB1*11:501', 'HLA-DPA1*03:02-DPB1*11:601', 'HLA-DPA1*03:02-DPB1*11:701',
         'HLA-DPA1*03:02-DPB1*11:801', 'HLA-DPA1*03:02-DPB1*11:901',
         'HLA-DPA1*03:02-DPB1*12:101', 'HLA-DPA1*03:02-DPB1*12:201', 'HLA-DPA1*03:02-DPB1*12:301', 'HLA-DPA1*03:02-DPB1*12:401',
         'HLA-DPA1*03:02-DPB1*12:501', 'HLA-DPA1*03:02-DPB1*12:601',
         'HLA-DPA1*03:02-DPB1*12:701', 'HLA-DPA1*03:02-DPB1*12:801', 'HLA-DPA1*03:02-DPB1*12:901', 'HLA-DPA1*03:02-DPB1*13:001',
         'HLA-DPA1*03:02-DPB1*13:01', 'HLA-DPA1*03:02-DPB1*13:101',
         'HLA-DPA1*03:02-DPB1*13:201', 'HLA-DPA1*03:02-DPB1*13:301', 'HLA-DPA1*03:02-DPB1*13:401', 'HLA-DPA1*03:02-DPB1*14:01',
         'HLA-DPA1*03:02-DPB1*15:01', 'HLA-DPA1*03:02-DPB1*16:01',
         'HLA-DPA1*03:02-DPB1*17:01', 'HLA-DPA1*03:02-DPB1*18:01', 'HLA-DPA1*03:02-DPB1*19:01', 'HLA-DPA1*03:02-DPB1*20:01',
         'HLA-DPA1*03:02-DPB1*21:01', 'HLA-DPA1*03:02-DPB1*22:01',
         'HLA-DPA1*03:02-DPB1*23:01', 'HLA-DPA1*03:02-DPB1*24:01', 'HLA-DPA1*03:02-DPB1*25:01', 'HLA-DPA1*03:02-DPB1*26:01',
         'HLA-DPA1*03:02-DPB1*27:01', 'HLA-DPA1*03:02-DPB1*28:01',
         'HLA-DPA1*03:02-DPB1*29:01', 'HLA-DPA1*03:02-DPB1*30:01', 'HLA-DPA1*03:02-DPB1*31:01', 'HLA-DPA1*03:02-DPB1*32:01',
         'HLA-DPA1*03:02-DPB1*33:01', 'HLA-DPA1*03:02-DPB1*34:01',
         'HLA-DPA1*03:02-DPB1*35:01', 'HLA-DPA1*03:02-DPB1*36:01', 'HLA-DPA1*03:02-DPB1*37:01', 'HLA-DPA1*03:02-DPB1*38:01',
         'HLA-DPA1*03:02-DPB1*39:01', 'HLA-DPA1*03:02-DPB1*40:01',
         'HLA-DPA1*03:02-DPB1*41:01', 'HLA-DPA1*03:02-DPB1*44:01', 'HLA-DPA1*03:02-DPB1*45:01', 'HLA-DPA1*03:02-DPB1*46:01',
         'HLA-DPA1*03:02-DPB1*47:01', 'HLA-DPA1*03:02-DPB1*48:01',
         'HLA-DPA1*03:02-DPB1*49:01', 'HLA-DPA1*03:02-DPB1*50:01', 'HLA-DPA1*03:02-DPB1*51:01', 'HLA-DPA1*03:02-DPB1*52:01',
         'HLA-DPA1*03:02-DPB1*53:01', 'HLA-DPA1*03:02-DPB1*54:01',
         'HLA-DPA1*03:02-DPB1*55:01', 'HLA-DPA1*03:02-DPB1*56:01', 'HLA-DPA1*03:02-DPB1*58:01', 'HLA-DPA1*03:02-DPB1*59:01',
         'HLA-DPA1*03:02-DPB1*60:01', 'HLA-DPA1*03:02-DPB1*62:01',
         'HLA-DPA1*03:02-DPB1*63:01', 'HLA-DPA1*03:02-DPB1*65:01', 'HLA-DPA1*03:02-DPB1*66:01', 'HLA-DPA1*03:02-DPB1*67:01',
         'HLA-DPA1*03:02-DPB1*68:01', 'HLA-DPA1*03:02-DPB1*69:01',
         'HLA-DPA1*03:02-DPB1*70:01', 'HLA-DPA1*03:02-DPB1*71:01', 'HLA-DPA1*03:02-DPB1*72:01', 'HLA-DPA1*03:02-DPB1*73:01',
         'HLA-DPA1*03:02-DPB1*74:01', 'HLA-DPA1*03:02-DPB1*75:01',
         'HLA-DPA1*03:02-DPB1*76:01', 'HLA-DPA1*03:02-DPB1*77:01', 'HLA-DPA1*03:02-DPB1*78:01', 'HLA-DPA1*03:02-DPB1*79:01',
         'HLA-DPA1*03:02-DPB1*80:01', 'HLA-DPA1*03:02-DPB1*81:01',
         'HLA-DPA1*03:02-DPB1*82:01', 'HLA-DPA1*03:02-DPB1*83:01', 'HLA-DPA1*03:02-DPB1*84:01', 'HLA-DPA1*03:02-DPB1*85:01',
         'HLA-DPA1*03:02-DPB1*86:01', 'HLA-DPA1*03:02-DPB1*87:01',
         'HLA-DPA1*03:02-DPB1*88:01', 'HLA-DPA1*03:02-DPB1*89:01', 'HLA-DPA1*03:02-DPB1*90:01', 'HLA-DPA1*03:02-DPB1*91:01',
         'HLA-DPA1*03:02-DPB1*92:01', 'HLA-DPA1*03:02-DPB1*93:01',
         'HLA-DPA1*03:02-DPB1*94:01', 'HLA-DPA1*03:02-DPB1*95:01', 'HLA-DPA1*03:02-DPB1*96:01', 'HLA-DPA1*03:02-DPB1*97:01',
         'HLA-DPA1*03:02-DPB1*98:01', 'HLA-DPA1*03:02-DPB1*99:01',
         'HLA-DPA1*03:03-DPB1*01:01', 'HLA-DPA1*03:03-DPB1*02:01', 'HLA-DPA1*03:03-DPB1*02:02', 'HLA-DPA1*03:03-DPB1*03:01',
         'HLA-DPA1*03:03-DPB1*04:01', 'HLA-DPA1*03:03-DPB1*04:02',
         'HLA-DPA1*03:03-DPB1*05:01', 'HLA-DPA1*03:03-DPB1*06:01', 'HLA-DPA1*03:03-DPB1*08:01', 'HLA-DPA1*03:03-DPB1*09:01',
         'HLA-DPA1*03:03-DPB1*10:001', 'HLA-DPA1*03:03-DPB1*10:01',
         'HLA-DPA1*03:03-DPB1*10:101', 'HLA-DPA1*03:03-DPB1*10:201', 'HLA-DPA1*03:03-DPB1*10:301', 'HLA-DPA1*03:03-DPB1*10:401',
         'HLA-DPA1*03:03-DPB1*10:501', 'HLA-DPA1*03:03-DPB1*10:601',
         'HLA-DPA1*03:03-DPB1*10:701', 'HLA-DPA1*03:03-DPB1*10:801', 'HLA-DPA1*03:03-DPB1*10:901', 'HLA-DPA1*03:03-DPB1*11:001',
         'HLA-DPA1*03:03-DPB1*11:01', 'HLA-DPA1*03:03-DPB1*11:101',
         'HLA-DPA1*03:03-DPB1*11:201', 'HLA-DPA1*03:03-DPB1*11:301', 'HLA-DPA1*03:03-DPB1*11:401', 'HLA-DPA1*03:03-DPB1*11:501',
         'HLA-DPA1*03:03-DPB1*11:601', 'HLA-DPA1*03:03-DPB1*11:701',
         'HLA-DPA1*03:03-DPB1*11:801', 'HLA-DPA1*03:03-DPB1*11:901', 'HLA-DPA1*03:03-DPB1*12:101', 'HLA-DPA1*03:03-DPB1*12:201',
         'HLA-DPA1*03:03-DPB1*12:301', 'HLA-DPA1*03:03-DPB1*12:401',
         'HLA-DPA1*03:03-DPB1*12:501', 'HLA-DPA1*03:03-DPB1*12:601', 'HLA-DPA1*03:03-DPB1*12:701', 'HLA-DPA1*03:03-DPB1*12:801',
         'HLA-DPA1*03:03-DPB1*12:901', 'HLA-DPA1*03:03-DPB1*13:001',
         'HLA-DPA1*03:03-DPB1*13:01', 'HLA-DPA1*03:03-DPB1*13:101', 'HLA-DPA1*03:03-DPB1*13:201', 'HLA-DPA1*03:03-DPB1*13:301',
         'HLA-DPA1*03:03-DPB1*13:401', 'HLA-DPA1*03:03-DPB1*14:01',
         'HLA-DPA1*03:03-DPB1*15:01', 'HLA-DPA1*03:03-DPB1*16:01', 'HLA-DPA1*03:03-DPB1*17:01', 'HLA-DPA1*03:03-DPB1*18:01',
         'HLA-DPA1*03:03-DPB1*19:01', 'HLA-DPA1*03:03-DPB1*20:01',
         'HLA-DPA1*03:03-DPB1*21:01', 'HLA-DPA1*03:03-DPB1*22:01', 'HLA-DPA1*03:03-DPB1*23:01', 'HLA-DPA1*03:03-DPB1*24:01',
         'HLA-DPA1*03:03-DPB1*25:01', 'HLA-DPA1*03:03-DPB1*26:01',
         'HLA-DPA1*03:03-DPB1*27:01', 'HLA-DPA1*03:03-DPB1*28:01', 'HLA-DPA1*03:03-DPB1*29:01', 'HLA-DPA1*03:03-DPB1*30:01',
         'HLA-DPA1*03:03-DPB1*31:01', 'HLA-DPA1*03:03-DPB1*32:01',
         'HLA-DPA1*03:03-DPB1*33:01', 'HLA-DPA1*03:03-DPB1*34:01', 'HLA-DPA1*03:03-DPB1*35:01', 'HLA-DPA1*03:03-DPB1*36:01',
         'HLA-DPA1*03:03-DPB1*37:01', 'HLA-DPA1*03:03-DPB1*38:01',
         'HLA-DPA1*03:03-DPB1*39:01', 'HLA-DPA1*03:03-DPB1*40:01', 'HLA-DPA1*03:03-DPB1*41:01', 'HLA-DPA1*03:03-DPB1*44:01',
         'HLA-DPA1*03:03-DPB1*45:01', 'HLA-DPA1*03:03-DPB1*46:01',
         'HLA-DPA1*03:03-DPB1*47:01', 'HLA-DPA1*03:03-DPB1*48:01', 'HLA-DPA1*03:03-DPB1*49:01', 'HLA-DPA1*03:03-DPB1*50:01',
         'HLA-DPA1*03:03-DPB1*51:01', 'HLA-DPA1*03:03-DPB1*52:01',
         'HLA-DPA1*03:03-DPB1*53:01', 'HLA-DPA1*03:03-DPB1*54:01', 'HLA-DPA1*03:03-DPB1*55:01', 'HLA-DPA1*03:03-DPB1*56:01',
         'HLA-DPA1*03:03-DPB1*58:01', 'HLA-DPA1*03:03-DPB1*59:01',
         'HLA-DPA1*03:03-DPB1*60:01', 'HLA-DPA1*03:03-DPB1*62:01', 'HLA-DPA1*03:03-DPB1*63:01', 'HLA-DPA1*03:03-DPB1*65:01',
         'HLA-DPA1*03:03-DPB1*66:01', 'HLA-DPA1*03:03-DPB1*67:01',
         'HLA-DPA1*03:03-DPB1*68:01', 'HLA-DPA1*03:03-DPB1*69:01', 'HLA-DPA1*03:03-DPB1*70:01', 'HLA-DPA1*03:03-DPB1*71:01',
         'HLA-DPA1*03:03-DPB1*72:01', 'HLA-DPA1*03:03-DPB1*73:01',
         'HLA-DPA1*03:03-DPB1*74:01', 'HLA-DPA1*03:03-DPB1*75:01', 'HLA-DPA1*03:03-DPB1*76:01', 'HLA-DPA1*03:03-DPB1*77:01',
         'HLA-DPA1*03:03-DPB1*78:01', 'HLA-DPA1*03:03-DPB1*79:01',
         'HLA-DPA1*03:03-DPB1*80:01', 'HLA-DPA1*03:03-DPB1*81:01', 'HLA-DPA1*03:03-DPB1*82:01', 'HLA-DPA1*03:03-DPB1*83:01',
         'HLA-DPA1*03:03-DPB1*84:01', 'HLA-DPA1*03:03-DPB1*85:01',
         'HLA-DPA1*03:03-DPB1*86:01', 'HLA-DPA1*03:03-DPB1*87:01', 'HLA-DPA1*03:03-DPB1*88:01', 'HLA-DPA1*03:03-DPB1*89:01',
         'HLA-DPA1*03:03-DPB1*90:01', 'HLA-DPA1*03:03-DPB1*91:01',
         'HLA-DPA1*03:03-DPB1*92:01', 'HLA-DPA1*03:03-DPB1*93:01', 'HLA-DPA1*03:03-DPB1*94:01', 'HLA-DPA1*03:03-DPB1*95:01',
         'HLA-DPA1*03:03-DPB1*96:01', 'HLA-DPA1*03:03-DPB1*97:01',
         'HLA-DPA1*03:03-DPB1*98:01', 'HLA-DPA1*03:03-DPB1*99:01', 'HLA-DPA1*04:01-DPB1*01:01', 'HLA-DPA1*04:01-DPB1*02:01',
         'HLA-DPA1*04:01-DPB1*02:02', 'HLA-DPA1*04:01-DPB1*03:01',
         'HLA-DPA1*04:01-DPB1*04:01', 'HLA-DPA1*04:01-DPB1*04:02', 'HLA-DPA1*04:01-DPB1*05:01', 'HLA-DPA1*04:01-DPB1*06:01',
         'HLA-DPA1*04:01-DPB1*08:01', 'HLA-DPA1*04:01-DPB1*09:01',
         'HLA-DPA1*04:01-DPB1*10:001', 'HLA-DPA1*04:01-DPB1*10:01', 'HLA-DPA1*04:01-DPB1*10:101', 'HLA-DPA1*04:01-DPB1*10:201',
         'HLA-DPA1*04:01-DPB1*10:301', 'HLA-DPA1*04:01-DPB1*10:401',
         'HLA-DPA1*04:01-DPB1*10:501', 'HLA-DPA1*04:01-DPB1*10:601', 'HLA-DPA1*04:01-DPB1*10:701', 'HLA-DPA1*04:01-DPB1*10:801',
         'HLA-DPA1*04:01-DPB1*10:901', 'HLA-DPA1*04:01-DPB1*11:001',
         'HLA-DPA1*04:01-DPB1*11:01', 'HLA-DPA1*04:01-DPB1*11:101', 'HLA-DPA1*04:01-DPB1*11:201', 'HLA-DPA1*04:01-DPB1*11:301',
         'HLA-DPA1*04:01-DPB1*11:401', 'HLA-DPA1*04:01-DPB1*11:501',
         'HLA-DPA1*04:01-DPB1*11:601', 'HLA-DPA1*04:01-DPB1*11:701', 'HLA-DPA1*04:01-DPB1*11:801', 'HLA-DPA1*04:01-DPB1*11:901',
         'HLA-DPA1*04:01-DPB1*12:101', 'HLA-DPA1*04:01-DPB1*12:201',
         'HLA-DPA1*04:01-DPB1*12:301', 'HLA-DPA1*04:01-DPB1*12:401', 'HLA-DPA1*04:01-DPB1*12:501', 'HLA-DPA1*04:01-DPB1*12:601',
         'HLA-DPA1*04:01-DPB1*12:701', 'HLA-DPA1*04:01-DPB1*12:801',
         'HLA-DPA1*04:01-DPB1*12:901', 'HLA-DPA1*04:01-DPB1*13:001', 'HLA-DPA1*04:01-DPB1*13:01', 'HLA-DPA1*04:01-DPB1*13:101',
         'HLA-DPA1*04:01-DPB1*13:201', 'HLA-DPA1*04:01-DPB1*13:301',
         'HLA-DPA1*04:01-DPB1*13:401', 'HLA-DPA1*04:01-DPB1*14:01', 'HLA-DPA1*04:01-DPB1*15:01', 'HLA-DPA1*04:01-DPB1*16:01',
         'HLA-DPA1*04:01-DPB1*17:01', 'HLA-DPA1*04:01-DPB1*18:01',
         'HLA-DPA1*04:01-DPB1*19:01', 'HLA-DPA1*04:01-DPB1*20:01', 'HLA-DPA1*04:01-DPB1*21:01', 'HLA-DPA1*04:01-DPB1*22:01',
         'HLA-DPA1*04:01-DPB1*23:01', 'HLA-DPA1*04:01-DPB1*24:01',
         'HLA-DPA1*04:01-DPB1*25:01', 'HLA-DPA1*04:01-DPB1*26:01', 'HLA-DPA1*04:01-DPB1*27:01', 'HLA-DPA1*04:01-DPB1*28:01',
         'HLA-DPA1*04:01-DPB1*29:01', 'HLA-DPA1*04:01-DPB1*30:01',
         'HLA-DPA1*04:01-DPB1*31:01', 'HLA-DPA1*04:01-DPB1*32:01', 'HLA-DPA1*04:01-DPB1*33:01', 'HLA-DPA1*04:01-DPB1*34:01',
         'HLA-DPA1*04:01-DPB1*35:01', 'HLA-DPA1*04:01-DPB1*36:01',
         'HLA-DPA1*04:01-DPB1*37:01', 'HLA-DPA1*04:01-DPB1*38:01', 'HLA-DPA1*04:01-DPB1*39:01', 'HLA-DPA1*04:01-DPB1*40:01',
         'HLA-DPA1*04:01-DPB1*41:01', 'HLA-DPA1*04:01-DPB1*44:01',
         'HLA-DPA1*04:01-DPB1*45:01', 'HLA-DPA1*04:01-DPB1*46:01', 'HLA-DPA1*04:01-DPB1*47:01', 'HLA-DPA1*04:01-DPB1*48:01',
         'HLA-DPA1*04:01-DPB1*49:01', 'HLA-DPA1*04:01-DPB1*50:01',
         'HLA-DPA1*04:01-DPB1*51:01', 'HLA-DPA1*04:01-DPB1*52:01', 'HLA-DPA1*04:01-DPB1*53:01', 'HLA-DPA1*04:01-DPB1*54:01',
         'HLA-DPA1*04:01-DPB1*55:01', 'HLA-DPA1*04:01-DPB1*56:01',
         'HLA-DPA1*04:01-DPB1*58:01', 'HLA-DPA1*04:01-DPB1*59:01', 'HLA-DPA1*04:01-DPB1*60:01', 'HLA-DPA1*04:01-DPB1*62:01',
         'HLA-DPA1*04:01-DPB1*63:01', 'HLA-DPA1*04:01-DPB1*65:01',
         'HLA-DPA1*04:01-DPB1*66:01', 'HLA-DPA1*04:01-DPB1*67:01', 'HLA-DPA1*04:01-DPB1*68:01', 'HLA-DPA1*04:01-DPB1*69:01',
         'HLA-DPA1*04:01-DPB1*70:01', 'HLA-DPA1*04:01-DPB1*71:01',
         'HLA-DPA1*04:01-DPB1*72:01', 'HLA-DPA1*04:01-DPB1*73:01', 'HLA-DPA1*04:01-DPB1*74:01', 'HLA-DPA1*04:01-DPB1*75:01',
         'HLA-DPA1*04:01-DPB1*76:01', 'HLA-DPA1*04:01-DPB1*77:01',
         'HLA-DPA1*04:01-DPB1*78:01', 'HLA-DPA1*04:01-DPB1*79:01', 'HLA-DPA1*04:01-DPB1*80:01', 'HLA-DPA1*04:01-DPB1*81:01',
         'HLA-DPA1*04:01-DPB1*82:01', 'HLA-DPA1*04:01-DPB1*83:01',
         'HLA-DPA1*04:01-DPB1*84:01', 'HLA-DPA1*04:01-DPB1*85:01', 'HLA-DPA1*04:01-DPB1*86:01', 'HLA-DPA1*04:01-DPB1*87:01',
         'HLA-DPA1*04:01-DPB1*88:01', 'HLA-DPA1*04:01-DPB1*89:01',
         'HLA-DPA1*04:01-DPB1*90:01', 'HLA-DPA1*04:01-DPB1*91:01', 'HLA-DPA1*04:01-DPB1*92:01', 'HLA-DPA1*04:01-DPB1*93:01',
         'HLA-DPA1*04:01-DPB1*94:01', 'HLA-DPA1*04:01-DPB1*95:01',
         'HLA-DPA1*04:01-DPB1*96:01', 'HLA-DPA1*04:01-DPB1*97:01', 'HLA-DPA1*04:01-DPB1*98:01', 'HLA-DPA1*04:01-DPB1*99:01',
         'HLA-DQA1*01:01-DQB1*02:01', 'HLA-DQA1*01:01-DQB1*02:02',
         'HLA-DQA1*01:01-DQB1*02:03', 'HLA-DQA1*01:01-DQB1*02:04', 'HLA-DQA1*01:01-DQB1*02:05', 'HLA-DQA1*01:01-DQB1*02:06',
         'HLA-DQA1*01:01-DQB1*03:01', 'HLA-DQA1*01:01-DQB1*03:02',
         'HLA-DQA1*01:01-DQB1*03:03', 'HLA-DQA1*01:01-DQB1*03:04', 'HLA-DQA1*01:01-DQB1*03:05', 'HLA-DQA1*01:01-DQB1*03:06',
         'HLA-DQA1*01:01-DQB1*03:07', 'HLA-DQA1*01:01-DQB1*03:08',
         'HLA-DQA1*01:01-DQB1*03:09', 'HLA-DQA1*01:01-DQB1*03:10', 'HLA-DQA1*01:01-DQB1*03:11', 'HLA-DQA1*01:01-DQB1*03:12',
         'HLA-DQA1*01:01-DQB1*03:13', 'HLA-DQA1*01:01-DQB1*03:14',
         'HLA-DQA1*01:01-DQB1*03:15', 'HLA-DQA1*01:01-DQB1*03:16', 'HLA-DQA1*01:01-DQB1*03:17', 'HLA-DQA1*01:01-DQB1*03:18',
         'HLA-DQA1*01:01-DQB1*03:19', 'HLA-DQA1*01:01-DQB1*03:20',
         'HLA-DQA1*01:01-DQB1*03:21', 'HLA-DQA1*01:01-DQB1*03:22', 'HLA-DQA1*01:01-DQB1*03:23', 'HLA-DQA1*01:01-DQB1*03:24',
         'HLA-DQA1*01:01-DQB1*03:25', 'HLA-DQA1*01:01-DQB1*03:26',
         'HLA-DQA1*01:01-DQB1*03:27', 'HLA-DQA1*01:01-DQB1*03:28', 'HLA-DQA1*01:01-DQB1*03:29', 'HLA-DQA1*01:01-DQB1*03:30',
         'HLA-DQA1*01:01-DQB1*03:31', 'HLA-DQA1*01:01-DQB1*03:32',
         'HLA-DQA1*01:01-DQB1*03:33', 'HLA-DQA1*01:01-DQB1*03:34', 'HLA-DQA1*01:01-DQB1*03:35', 'HLA-DQA1*01:01-DQB1*03:36',
         'HLA-DQA1*01:01-DQB1*03:37', 'HLA-DQA1*01:01-DQB1*03:38',
         'HLA-DQA1*01:01-DQB1*04:01', 'HLA-DQA1*01:01-DQB1*04:02', 'HLA-DQA1*01:01-DQB1*04:03', 'HLA-DQA1*01:01-DQB1*04:04',
         'HLA-DQA1*01:01-DQB1*04:05', 'HLA-DQA1*01:01-DQB1*04:06',
         'HLA-DQA1*01:01-DQB1*04:07', 'HLA-DQA1*01:01-DQB1*04:08', 'HLA-DQA1*01:01-DQB1*05:01', 'HLA-DQA1*01:01-DQB1*05:02',
         'HLA-DQA1*01:01-DQB1*05:03', 'HLA-DQA1*01:01-DQB1*05:05',
         'HLA-DQA1*01:01-DQB1*05:06', 'HLA-DQA1*01:01-DQB1*05:07', 'HLA-DQA1*01:01-DQB1*05:08', 'HLA-DQA1*01:01-DQB1*05:09',
         'HLA-DQA1*01:01-DQB1*05:10', 'HLA-DQA1*01:01-DQB1*05:11',
         'HLA-DQA1*01:01-DQB1*05:12', 'HLA-DQA1*01:01-DQB1*05:13', 'HLA-DQA1*01:01-DQB1*05:14', 'HLA-DQA1*01:01-DQB1*06:01',
         'HLA-DQA1*01:01-DQB1*06:02', 'HLA-DQA1*01:01-DQB1*06:03',
         'HLA-DQA1*01:01-DQB1*06:04', 'HLA-DQA1*01:01-DQB1*06:07', 'HLA-DQA1*01:01-DQB1*06:08', 'HLA-DQA1*01:01-DQB1*06:09',
         'HLA-DQA1*01:01-DQB1*06:10', 'HLA-DQA1*01:01-DQB1*06:11',
         'HLA-DQA1*01:01-DQB1*06:12', 'HLA-DQA1*01:01-DQB1*06:14', 'HLA-DQA1*01:01-DQB1*06:15', 'HLA-DQA1*01:01-DQB1*06:16',
         'HLA-DQA1*01:01-DQB1*06:17', 'HLA-DQA1*01:01-DQB1*06:18',
         'HLA-DQA1*01:01-DQB1*06:19', 'HLA-DQA1*01:01-DQB1*06:21', 'HLA-DQA1*01:01-DQB1*06:22', 'HLA-DQA1*01:01-DQB1*06:23',
         'HLA-DQA1*01:01-DQB1*06:24', 'HLA-DQA1*01:01-DQB1*06:25',
         'HLA-DQA1*01:01-DQB1*06:27', 'HLA-DQA1*01:01-DQB1*06:28', 'HLA-DQA1*01:01-DQB1*06:29', 'HLA-DQA1*01:01-DQB1*06:30',
         'HLA-DQA1*01:01-DQB1*06:31', 'HLA-DQA1*01:01-DQB1*06:32',
         'HLA-DQA1*01:01-DQB1*06:33', 'HLA-DQA1*01:01-DQB1*06:34', 'HLA-DQA1*01:01-DQB1*06:35', 'HLA-DQA1*01:01-DQB1*06:36',
         'HLA-DQA1*01:01-DQB1*06:37', 'HLA-DQA1*01:01-DQB1*06:38',
         'HLA-DQA1*01:01-DQB1*06:39', 'HLA-DQA1*01:01-DQB1*06:40', 'HLA-DQA1*01:01-DQB1*06:41', 'HLA-DQA1*01:01-DQB1*06:42',
         'HLA-DQA1*01:01-DQB1*06:43', 'HLA-DQA1*01:01-DQB1*06:44',
         'HLA-DQA1*01:02-DQB1*02:01', 'HLA-DQA1*01:02-DQB1*02:02', 'HLA-DQA1*01:02-DQB1*02:03', 'HLA-DQA1*01:02-DQB1*02:04',
         'HLA-DQA1*01:02-DQB1*02:05', 'HLA-DQA1*01:02-DQB1*02:06',
         'HLA-DQA1*01:02-DQB1*03:01', 'HLA-DQA1*01:02-DQB1*03:02', 'HLA-DQA1*01:02-DQB1*03:03', 'HLA-DQA1*01:02-DQB1*03:04',
         'HLA-DQA1*01:02-DQB1*03:05', 'HLA-DQA1*01:02-DQB1*03:06',
         'HLA-DQA1*01:02-DQB1*03:07', 'HLA-DQA1*01:02-DQB1*03:08', 'HLA-DQA1*01:02-DQB1*03:09', 'HLA-DQA1*01:02-DQB1*03:10',
         'HLA-DQA1*01:02-DQB1*03:11', 'HLA-DQA1*01:02-DQB1*03:12',
         'HLA-DQA1*01:02-DQB1*03:13', 'HLA-DQA1*01:02-DQB1*03:14', 'HLA-DQA1*01:02-DQB1*03:15', 'HLA-DQA1*01:02-DQB1*03:16',
         'HLA-DQA1*01:02-DQB1*03:17', 'HLA-DQA1*01:02-DQB1*03:18',
         'HLA-DQA1*01:02-DQB1*03:19', 'HLA-DQA1*01:02-DQB1*03:20', 'HLA-DQA1*01:02-DQB1*03:21', 'HLA-DQA1*01:02-DQB1*03:22',
         'HLA-DQA1*01:02-DQB1*03:23', 'HLA-DQA1*01:02-DQB1*03:24',
         'HLA-DQA1*01:02-DQB1*03:25', 'HLA-DQA1*01:02-DQB1*03:26', 'HLA-DQA1*01:02-DQB1*03:27', 'HLA-DQA1*01:02-DQB1*03:28',
         'HLA-DQA1*01:02-DQB1*03:29', 'HLA-DQA1*01:02-DQB1*03:30',
         'HLA-DQA1*01:02-DQB1*03:31', 'HLA-DQA1*01:02-DQB1*03:32', 'HLA-DQA1*01:02-DQB1*03:33', 'HLA-DQA1*01:02-DQB1*03:34',
         'HLA-DQA1*01:02-DQB1*03:35', 'HLA-DQA1*01:02-DQB1*03:36',
         'HLA-DQA1*01:02-DQB1*03:37', 'HLA-DQA1*01:02-DQB1*03:38', 'HLA-DQA1*01:02-DQB1*04:01', 'HLA-DQA1*01:02-DQB1*04:02',
         'HLA-DQA1*01:02-DQB1*04:03', 'HLA-DQA1*01:02-DQB1*04:04',
         'HLA-DQA1*01:02-DQB1*04:05', 'HLA-DQA1*01:02-DQB1*04:06', 'HLA-DQA1*01:02-DQB1*04:07', 'HLA-DQA1*01:02-DQB1*04:08',
         'HLA-DQA1*01:02-DQB1*05:01', 'HLA-DQA1*01:02-DQB1*05:02',
         'HLA-DQA1*01:02-DQB1*05:03', 'HLA-DQA1*01:02-DQB1*05:05', 'HLA-DQA1*01:02-DQB1*05:06', 'HLA-DQA1*01:02-DQB1*05:07',
         'HLA-DQA1*01:02-DQB1*05:08', 'HLA-DQA1*01:02-DQB1*05:09',
         'HLA-DQA1*01:02-DQB1*05:10', 'HLA-DQA1*01:02-DQB1*05:11', 'HLA-DQA1*01:02-DQB1*05:12', 'HLA-DQA1*01:02-DQB1*05:13',
         'HLA-DQA1*01:02-DQB1*05:14', 'HLA-DQA1*01:02-DQB1*06:01',
         'HLA-DQA1*01:02-DQB1*06:02', 'HLA-DQA1*01:02-DQB1*06:03', 'HLA-DQA1*01:02-DQB1*06:04', 'HLA-DQA1*01:02-DQB1*06:07',
         'HLA-DQA1*01:02-DQB1*06:08', 'HLA-DQA1*01:02-DQB1*06:09',
         'HLA-DQA1*01:02-DQB1*06:10', 'HLA-DQA1*01:02-DQB1*06:11', 'HLA-DQA1*01:02-DQB1*06:12', 'HLA-DQA1*01:02-DQB1*06:14',
         'HLA-DQA1*01:02-DQB1*06:15', 'HLA-DQA1*01:02-DQB1*06:16',
         'HLA-DQA1*01:02-DQB1*06:17', 'HLA-DQA1*01:02-DQB1*06:18', 'HLA-DQA1*01:02-DQB1*06:19', 'HLA-DQA1*01:02-DQB1*06:21',
         'HLA-DQA1*01:02-DQB1*06:22', 'HLA-DQA1*01:02-DQB1*06:23',
         'HLA-DQA1*01:02-DQB1*06:24', 'HLA-DQA1*01:02-DQB1*06:25', 'HLA-DQA1*01:02-DQB1*06:27', 'HLA-DQA1*01:02-DQB1*06:28',
         'HLA-DQA1*01:02-DQB1*06:29', 'HLA-DQA1*01:02-DQB1*06:30',
         'HLA-DQA1*01:02-DQB1*06:31', 'HLA-DQA1*01:02-DQB1*06:32', 'HLA-DQA1*01:02-DQB1*06:33', 'HLA-DQA1*01:02-DQB1*06:34',
         'HLA-DQA1*01:02-DQB1*06:35', 'HLA-DQA1*01:02-DQB1*06:36',
         'HLA-DQA1*01:02-DQB1*06:37', 'HLA-DQA1*01:02-DQB1*06:38', 'HLA-DQA1*01:02-DQB1*06:39', 'HLA-DQA1*01:02-DQB1*06:40',
         'HLA-DQA1*01:02-DQB1*06:41', 'HLA-DQA1*01:02-DQB1*06:42',
         'HLA-DQA1*01:02-DQB1*06:43', 'HLA-DQA1*01:02-DQB1*06:44', 'HLA-DQA1*01:03-DQB1*02:01', 'HLA-DQA1*01:03-DQB1*02:02',
         'HLA-DQA1*01:03-DQB1*02:03', 'HLA-DQA1*01:03-DQB1*02:04',
         'HLA-DQA1*01:03-DQB1*02:05', 'HLA-DQA1*01:03-DQB1*02:06', 'HLA-DQA1*01:03-DQB1*03:01', 'HLA-DQA1*01:03-DQB1*03:02',
         'HLA-DQA1*01:03-DQB1*03:03', 'HLA-DQA1*01:03-DQB1*03:04',
         'HLA-DQA1*01:03-DQB1*03:05', 'HLA-DQA1*01:03-DQB1*03:06', 'HLA-DQA1*01:03-DQB1*03:07', 'HLA-DQA1*01:03-DQB1*03:08',
         'HLA-DQA1*01:03-DQB1*03:09', 'HLA-DQA1*01:03-DQB1*03:10',
         'HLA-DQA1*01:03-DQB1*03:11', 'HLA-DQA1*01:03-DQB1*03:12', 'HLA-DQA1*01:03-DQB1*03:13', 'HLA-DQA1*01:03-DQB1*03:14',
         'HLA-DQA1*01:03-DQB1*03:15', 'HLA-DQA1*01:03-DQB1*03:16',
         'HLA-DQA1*01:03-DQB1*03:17', 'HLA-DQA1*01:03-DQB1*03:18', 'HLA-DQA1*01:03-DQB1*03:19', 'HLA-DQA1*01:03-DQB1*03:20',
         'HLA-DQA1*01:03-DQB1*03:21', 'HLA-DQA1*01:03-DQB1*03:22',
         'HLA-DQA1*01:03-DQB1*03:23', 'HLA-DQA1*01:03-DQB1*03:24', 'HLA-DQA1*01:03-DQB1*03:25', 'HLA-DQA1*01:03-DQB1*03:26',
         'HLA-DQA1*01:03-DQB1*03:27', 'HLA-DQA1*01:03-DQB1*03:28',
         'HLA-DQA1*01:03-DQB1*03:29', 'HLA-DQA1*01:03-DQB1*03:30', 'HLA-DQA1*01:03-DQB1*03:31', 'HLA-DQA1*01:03-DQB1*03:32',
         'HLA-DQA1*01:03-DQB1*03:33', 'HLA-DQA1*01:03-DQB1*03:34',
         'HLA-DQA1*01:03-DQB1*03:35', 'HLA-DQA1*01:03-DQB1*03:36', 'HLA-DQA1*01:03-DQB1*03:37', 'HLA-DQA1*01:03-DQB1*03:38',
         'HLA-DQA1*01:03-DQB1*04:01', 'HLA-DQA1*01:03-DQB1*04:02',
         'HLA-DQA1*01:03-DQB1*04:03', 'HLA-DQA1*01:03-DQB1*04:04', 'HLA-DQA1*01:03-DQB1*04:05', 'HLA-DQA1*01:03-DQB1*04:06',
         'HLA-DQA1*01:03-DQB1*04:07', 'HLA-DQA1*01:03-DQB1*04:08',
         'HLA-DQA1*01:03-DQB1*05:01', 'HLA-DQA1*01:03-DQB1*05:02', 'HLA-DQA1*01:03-DQB1*05:03', 'HLA-DQA1*01:03-DQB1*05:05',
         'HLA-DQA1*01:03-DQB1*05:06', 'HLA-DQA1*01:03-DQB1*05:07',
         'HLA-DQA1*01:03-DQB1*05:08', 'HLA-DQA1*01:03-DQB1*05:09', 'HLA-DQA1*01:03-DQB1*05:10', 'HLA-DQA1*01:03-DQB1*05:11',
         'HLA-DQA1*01:03-DQB1*05:12', 'HLA-DQA1*01:03-DQB1*05:13',
         'HLA-DQA1*01:03-DQB1*05:14', 'HLA-DQA1*01:03-DQB1*06:01', 'HLA-DQA1*01:03-DQB1*06:02', 'HLA-DQA1*01:03-DQB1*06:03',
         'HLA-DQA1*01:03-DQB1*06:04', 'HLA-DQA1*01:03-DQB1*06:07',
         'HLA-DQA1*01:03-DQB1*06:08', 'HLA-DQA1*01:03-DQB1*06:09', 'HLA-DQA1*01:03-DQB1*06:10', 'HLA-DQA1*01:03-DQB1*06:11',
         'HLA-DQA1*01:03-DQB1*06:12', 'HLA-DQA1*01:03-DQB1*06:14',
         'HLA-DQA1*01:03-DQB1*06:15', 'HLA-DQA1*01:03-DQB1*06:16', 'HLA-DQA1*01:03-DQB1*06:17', 'HLA-DQA1*01:03-DQB1*06:18',
         'HLA-DQA1*01:03-DQB1*06:19', 'HLA-DQA1*01:03-DQB1*06:21',
         'HLA-DQA1*01:03-DQB1*06:22', 'HLA-DQA1*01:03-DQB1*06:23', 'HLA-DQA1*01:03-DQB1*06:24', 'HLA-DQA1*01:03-DQB1*06:25',
         'HLA-DQA1*01:03-DQB1*06:27', 'HLA-DQA1*01:03-DQB1*06:28',
         'HLA-DQA1*01:03-DQB1*06:29', 'HLA-DQA1*01:03-DQB1*06:30', 'HLA-DQA1*01:03-DQB1*06:31', 'HLA-DQA1*01:03-DQB1*06:32',
         'HLA-DQA1*01:03-DQB1*06:33', 'HLA-DQA1*01:03-DQB1*06:34',
         'HLA-DQA1*01:03-DQB1*06:35', 'HLA-DQA1*01:03-DQB1*06:36', 'HLA-DQA1*01:03-DQB1*06:37', 'HLA-DQA1*01:03-DQB1*06:38',
         'HLA-DQA1*01:03-DQB1*06:39', 'HLA-DQA1*01:03-DQB1*06:40',
         'HLA-DQA1*01:03-DQB1*06:41', 'HLA-DQA1*01:03-DQB1*06:42', 'HLA-DQA1*01:03-DQB1*06:43', 'HLA-DQA1*01:03-DQB1*06:44',
         'HLA-DQA1*01:04-DQB1*02:01', 'HLA-DQA1*01:04-DQB1*02:02',
         'HLA-DQA1*01:04-DQB1*02:03', 'HLA-DQA1*01:04-DQB1*02:04', 'HLA-DQA1*01:04-DQB1*02:05', 'HLA-DQA1*01:04-DQB1*02:06',
         'HLA-DQA1*01:04-DQB1*03:01', 'HLA-DQA1*01:04-DQB1*03:02',
         'HLA-DQA1*01:04-DQB1*03:03', 'HLA-DQA1*01:04-DQB1*03:04', 'HLA-DQA1*01:04-DQB1*03:05', 'HLA-DQA1*01:04-DQB1*03:06',
         'HLA-DQA1*01:04-DQB1*03:07', 'HLA-DQA1*01:04-DQB1*03:08',
         'HLA-DQA1*01:04-DQB1*03:09', 'HLA-DQA1*01:04-DQB1*03:10', 'HLA-DQA1*01:04-DQB1*03:11', 'HLA-DQA1*01:04-DQB1*03:12',
         'HLA-DQA1*01:04-DQB1*03:13', 'HLA-DQA1*01:04-DQB1*03:14',
         'HLA-DQA1*01:04-DQB1*03:15', 'HLA-DQA1*01:04-DQB1*03:16', 'HLA-DQA1*01:04-DQB1*03:17', 'HLA-DQA1*01:04-DQB1*03:18',
         'HLA-DQA1*01:04-DQB1*03:19', 'HLA-DQA1*01:04-DQB1*03:20',
         'HLA-DQA1*01:04-DQB1*03:21', 'HLA-DQA1*01:04-DQB1*03:22', 'HLA-DQA1*01:04-DQB1*03:23', 'HLA-DQA1*01:04-DQB1*03:24',
         'HLA-DQA1*01:04-DQB1*03:25', 'HLA-DQA1*01:04-DQB1*03:26',
         'HLA-DQA1*01:04-DQB1*03:27', 'HLA-DQA1*01:04-DQB1*03:28', 'HLA-DQA1*01:04-DQB1*03:29', 'HLA-DQA1*01:04-DQB1*03:30',
         'HLA-DQA1*01:04-DQB1*03:31', 'HLA-DQA1*01:04-DQB1*03:32',
         'HLA-DQA1*01:04-DQB1*03:33', 'HLA-DQA1*01:04-DQB1*03:34', 'HLA-DQA1*01:04-DQB1*03:35', 'HLA-DQA1*01:04-DQB1*03:36',
         'HLA-DQA1*01:04-DQB1*03:37', 'HLA-DQA1*01:04-DQB1*03:38',
         'HLA-DQA1*01:04-DQB1*04:01', 'HLA-DQA1*01:04-DQB1*04:02', 'HLA-DQA1*01:04-DQB1*04:03', 'HLA-DQA1*01:04-DQB1*04:04',
         'HLA-DQA1*01:04-DQB1*04:05', 'HLA-DQA1*01:04-DQB1*04:06',
         'HLA-DQA1*01:04-DQB1*04:07', 'HLA-DQA1*01:04-DQB1*04:08', 'HLA-DQA1*01:04-DQB1*05:01', 'HLA-DQA1*01:04-DQB1*05:02',
         'HLA-DQA1*01:04-DQB1*05:03', 'HLA-DQA1*01:04-DQB1*05:05',
         'HLA-DQA1*01:04-DQB1*05:06', 'HLA-DQA1*01:04-DQB1*05:07', 'HLA-DQA1*01:04-DQB1*05:08', 'HLA-DQA1*01:04-DQB1*05:09',
         'HLA-DQA1*01:04-DQB1*05:10', 'HLA-DQA1*01:04-DQB1*05:11',
         'HLA-DQA1*01:04-DQB1*05:12', 'HLA-DQA1*01:04-DQB1*05:13', 'HLA-DQA1*01:04-DQB1*05:14', 'HLA-DQA1*01:04-DQB1*06:01',
         'HLA-DQA1*01:04-DQB1*06:02', 'HLA-DQA1*01:04-DQB1*06:03',
         'HLA-DQA1*01:04-DQB1*06:04', 'HLA-DQA1*01:04-DQB1*06:07', 'HLA-DQA1*01:04-DQB1*06:08', 'HLA-DQA1*01:04-DQB1*06:09',
         'HLA-DQA1*01:04-DQB1*06:10', 'HLA-DQA1*01:04-DQB1*06:11',
         'HLA-DQA1*01:04-DQB1*06:12', 'HLA-DQA1*01:04-DQB1*06:14', 'HLA-DQA1*01:04-DQB1*06:15', 'HLA-DQA1*01:04-DQB1*06:16',
         'HLA-DQA1*01:04-DQB1*06:17', 'HLA-DQA1*01:04-DQB1*06:18',
         'HLA-DQA1*01:04-DQB1*06:19', 'HLA-DQA1*01:04-DQB1*06:21', 'HLA-DQA1*01:04-DQB1*06:22', 'HLA-DQA1*01:04-DQB1*06:23',
         'HLA-DQA1*01:04-DQB1*06:24', 'HLA-DQA1*01:04-DQB1*06:25',
         'HLA-DQA1*01:04-DQB1*06:27', 'HLA-DQA1*01:04-DQB1*06:28', 'HLA-DQA1*01:04-DQB1*06:29', 'HLA-DQA1*01:04-DQB1*06:30',
         'HLA-DQA1*01:04-DQB1*06:31', 'HLA-DQA1*01:04-DQB1*06:32',
         'HLA-DQA1*01:04-DQB1*06:33', 'HLA-DQA1*01:04-DQB1*06:34', 'HLA-DQA1*01:04-DQB1*06:35', 'HLA-DQA1*01:04-DQB1*06:36',
         'HLA-DQA1*01:04-DQB1*06:37', 'HLA-DQA1*01:04-DQB1*06:38',
         'HLA-DQA1*01:04-DQB1*06:39', 'HLA-DQA1*01:04-DQB1*06:40', 'HLA-DQA1*01:04-DQB1*06:41', 'HLA-DQA1*01:04-DQB1*06:42',
         'HLA-DQA1*01:04-DQB1*06:43', 'HLA-DQA1*01:04-DQB1*06:44',
         'HLA-DQA1*01:05-DQB1*02:01', 'HLA-DQA1*01:05-DQB1*02:02', 'HLA-DQA1*01:05-DQB1*02:03', 'HLA-DQA1*01:05-DQB1*02:04',
         'HLA-DQA1*01:05-DQB1*02:05', 'HLA-DQA1*01:05-DQB1*02:06',
         'HLA-DQA1*01:05-DQB1*03:01', 'HLA-DQA1*01:05-DQB1*03:02', 'HLA-DQA1*01:05-DQB1*03:03', 'HLA-DQA1*01:05-DQB1*03:04',
         'HLA-DQA1*01:05-DQB1*03:05', 'HLA-DQA1*01:05-DQB1*03:06',
         'HLA-DQA1*01:05-DQB1*03:07', 'HLA-DQA1*01:05-DQB1*03:08', 'HLA-DQA1*01:05-DQB1*03:09', 'HLA-DQA1*01:05-DQB1*03:10',
         'HLA-DQA1*01:05-DQB1*03:11', 'HLA-DQA1*01:05-DQB1*03:12',
         'HLA-DQA1*01:05-DQB1*03:13', 'HLA-DQA1*01:05-DQB1*03:14', 'HLA-DQA1*01:05-DQB1*03:15', 'HLA-DQA1*01:05-DQB1*03:16',
         'HLA-DQA1*01:05-DQB1*03:17', 'HLA-DQA1*01:05-DQB1*03:18',
         'HLA-DQA1*01:05-DQB1*03:19', 'HLA-DQA1*01:05-DQB1*03:20', 'HLA-DQA1*01:05-DQB1*03:21', 'HLA-DQA1*01:05-DQB1*03:22',
         'HLA-DQA1*01:05-DQB1*03:23', 'HLA-DQA1*01:05-DQB1*03:24',
         'HLA-DQA1*01:05-DQB1*03:25', 'HLA-DQA1*01:05-DQB1*03:26', 'HLA-DQA1*01:05-DQB1*03:27', 'HLA-DQA1*01:05-DQB1*03:28',
         'HLA-DQA1*01:05-DQB1*03:29', 'HLA-DQA1*01:05-DQB1*03:30',
         'HLA-DQA1*01:05-DQB1*03:31', 'HLA-DQA1*01:05-DQB1*03:32', 'HLA-DQA1*01:05-DQB1*03:33', 'HLA-DQA1*01:05-DQB1*03:34',
         'HLA-DQA1*01:05-DQB1*03:35', 'HLA-DQA1*01:05-DQB1*03:36',
         'HLA-DQA1*01:05-DQB1*03:37', 'HLA-DQA1*01:05-DQB1*03:38', 'HLA-DQA1*01:05-DQB1*04:01', 'HLA-DQA1*01:05-DQB1*04:02',
         'HLA-DQA1*01:05-DQB1*04:03', 'HLA-DQA1*01:05-DQB1*04:04',
         'HLA-DQA1*01:05-DQB1*04:05', 'HLA-DQA1*01:05-DQB1*04:06', 'HLA-DQA1*01:05-DQB1*04:07', 'HLA-DQA1*01:05-DQB1*04:08',
         'HLA-DQA1*01:05-DQB1*05:01', 'HLA-DQA1*01:05-DQB1*05:02',
         'HLA-DQA1*01:05-DQB1*05:03', 'HLA-DQA1*01:05-DQB1*05:05', 'HLA-DQA1*01:05-DQB1*05:06', 'HLA-DQA1*01:05-DQB1*05:07',
         'HLA-DQA1*01:05-DQB1*05:08', 'HLA-DQA1*01:05-DQB1*05:09',
         'HLA-DQA1*01:05-DQB1*05:10', 'HLA-DQA1*01:05-DQB1*05:11', 'HLA-DQA1*01:05-DQB1*05:12', 'HLA-DQA1*01:05-DQB1*05:13',
         'HLA-DQA1*01:05-DQB1*05:14', 'HLA-DQA1*01:05-DQB1*06:01',
         'HLA-DQA1*01:05-DQB1*06:02', 'HLA-DQA1*01:05-DQB1*06:03', 'HLA-DQA1*01:05-DQB1*06:04', 'HLA-DQA1*01:05-DQB1*06:07',
         'HLA-DQA1*01:05-DQB1*06:08', 'HLA-DQA1*01:05-DQB1*06:09',
         'HLA-DQA1*01:05-DQB1*06:10', 'HLA-DQA1*01:05-DQB1*06:11', 'HLA-DQA1*01:05-DQB1*06:12', 'HLA-DQA1*01:05-DQB1*06:14',
         'HLA-DQA1*01:05-DQB1*06:15', 'HLA-DQA1*01:05-DQB1*06:16',
         'HLA-DQA1*01:05-DQB1*06:17', 'HLA-DQA1*01:05-DQB1*06:18', 'HLA-DQA1*01:05-DQB1*06:19', 'HLA-DQA1*01:05-DQB1*06:21',
         'HLA-DQA1*01:05-DQB1*06:22', 'HLA-DQA1*01:05-DQB1*06:23',
         'HLA-DQA1*01:05-DQB1*06:24', 'HLA-DQA1*01:05-DQB1*06:25', 'HLA-DQA1*01:05-DQB1*06:27', 'HLA-DQA1*01:05-DQB1*06:28',
         'HLA-DQA1*01:05-DQB1*06:29', 'HLA-DQA1*01:05-DQB1*06:30',
         'HLA-DQA1*01:05-DQB1*06:31', 'HLA-DQA1*01:05-DQB1*06:32', 'HLA-DQA1*01:05-DQB1*06:33', 'HLA-DQA1*01:05-DQB1*06:34',
         'HLA-DQA1*01:05-DQB1*06:35', 'HLA-DQA1*01:05-DQB1*06:36',
         'HLA-DQA1*01:05-DQB1*06:37', 'HLA-DQA1*01:05-DQB1*06:38', 'HLA-DQA1*01:05-DQB1*06:39', 'HLA-DQA1*01:05-DQB1*06:40',
         'HLA-DQA1*01:05-DQB1*06:41', 'HLA-DQA1*01:05-DQB1*06:42',
         'HLA-DQA1*01:05-DQB1*06:43', 'HLA-DQA1*01:05-DQB1*06:44', 'HLA-DQA1*01:06-DQB1*02:01', 'HLA-DQA1*01:06-DQB1*02:02',
         'HLA-DQA1*01:06-DQB1*02:03', 'HLA-DQA1*01:06-DQB1*02:04',
         'HLA-DQA1*01:06-DQB1*02:05', 'HLA-DQA1*01:06-DQB1*02:06', 'HLA-DQA1*01:06-DQB1*03:01', 'HLA-DQA1*01:06-DQB1*03:02',
         'HLA-DQA1*01:06-DQB1*03:03', 'HLA-DQA1*01:06-DQB1*03:04',
         'HLA-DQA1*01:06-DQB1*03:05', 'HLA-DQA1*01:06-DQB1*03:06', 'HLA-DQA1*01:06-DQB1*03:07', 'HLA-DQA1*01:06-DQB1*03:08',
         'HLA-DQA1*01:06-DQB1*03:09', 'HLA-DQA1*01:06-DQB1*03:10',
         'HLA-DQA1*01:06-DQB1*03:11', 'HLA-DQA1*01:06-DQB1*03:12', 'HLA-DQA1*01:06-DQB1*03:13', 'HLA-DQA1*01:06-DQB1*03:14',
         'HLA-DQA1*01:06-DQB1*03:15', 'HLA-DQA1*01:06-DQB1*03:16',
         'HLA-DQA1*01:06-DQB1*03:17', 'HLA-DQA1*01:06-DQB1*03:18', 'HLA-DQA1*01:06-DQB1*03:19', 'HLA-DQA1*01:06-DQB1*03:20',
         'HLA-DQA1*01:06-DQB1*03:21', 'HLA-DQA1*01:06-DQB1*03:22',
         'HLA-DQA1*01:06-DQB1*03:23', 'HLA-DQA1*01:06-DQB1*03:24', 'HLA-DQA1*01:06-DQB1*03:25', 'HLA-DQA1*01:06-DQB1*03:26',
         'HLA-DQA1*01:06-DQB1*03:27', 'HLA-DQA1*01:06-DQB1*03:28',
         'HLA-DQA1*01:06-DQB1*03:29', 'HLA-DQA1*01:06-DQB1*03:30', 'HLA-DQA1*01:06-DQB1*03:31', 'HLA-DQA1*01:06-DQB1*03:32',
         'HLA-DQA1*01:06-DQB1*03:33', 'HLA-DQA1*01:06-DQB1*03:34',
         'HLA-DQA1*01:06-DQB1*03:35', 'HLA-DQA1*01:06-DQB1*03:36', 'HLA-DQA1*01:06-DQB1*03:37', 'HLA-DQA1*01:06-DQB1*03:38',
         'HLA-DQA1*01:06-DQB1*04:01', 'HLA-DQA1*01:06-DQB1*04:02',
         'HLA-DQA1*01:06-DQB1*04:03', 'HLA-DQA1*01:06-DQB1*04:04', 'HLA-DQA1*01:06-DQB1*04:05', 'HLA-DQA1*01:06-DQB1*04:06',
         'HLA-DQA1*01:06-DQB1*04:07', 'HLA-DQA1*01:06-DQB1*04:08',
         'HLA-DQA1*01:06-DQB1*05:01', 'HLA-DQA1*01:06-DQB1*05:02', 'HLA-DQA1*01:06-DQB1*05:03', 'HLA-DQA1*01:06-DQB1*05:05',
         'HLA-DQA1*01:06-DQB1*05:06', 'HLA-DQA1*01:06-DQB1*05:07',
         'HLA-DQA1*01:06-DQB1*05:08', 'HLA-DQA1*01:06-DQB1*05:09', 'HLA-DQA1*01:06-DQB1*05:10', 'HLA-DQA1*01:06-DQB1*05:11',
         'HLA-DQA1*01:06-DQB1*05:12', 'HLA-DQA1*01:06-DQB1*05:13',
         'HLA-DQA1*01:06-DQB1*05:14', 'HLA-DQA1*01:06-DQB1*06:01', 'HLA-DQA1*01:06-DQB1*06:02', 'HLA-DQA1*01:06-DQB1*06:03',
         'HLA-DQA1*01:06-DQB1*06:04', 'HLA-DQA1*01:06-DQB1*06:07',
         'HLA-DQA1*01:06-DQB1*06:08', 'HLA-DQA1*01:06-DQB1*06:09', 'HLA-DQA1*01:06-DQB1*06:10', 'HLA-DQA1*01:06-DQB1*06:11',
         'HLA-DQA1*01:06-DQB1*06:12', 'HLA-DQA1*01:06-DQB1*06:14',
         'HLA-DQA1*01:06-DQB1*06:15', 'HLA-DQA1*01:06-DQB1*06:16', 'HLA-DQA1*01:06-DQB1*06:17', 'HLA-DQA1*01:06-DQB1*06:18',
         'HLA-DQA1*01:06-DQB1*06:19', 'HLA-DQA1*01:06-DQB1*06:21',
         'HLA-DQA1*01:06-DQB1*06:22', 'HLA-DQA1*01:06-DQB1*06:23', 'HLA-DQA1*01:06-DQB1*06:24', 'HLA-DQA1*01:06-DQB1*06:25',
         'HLA-DQA1*01:06-DQB1*06:27', 'HLA-DQA1*01:06-DQB1*06:28',
         'HLA-DQA1*01:06-DQB1*06:29', 'HLA-DQA1*01:06-DQB1*06:30', 'HLA-DQA1*01:06-DQB1*06:31', 'HLA-DQA1*01:06-DQB1*06:32',
         'HLA-DQA1*01:06-DQB1*06:33', 'HLA-DQA1*01:06-DQB1*06:34',
         'HLA-DQA1*01:06-DQB1*06:35', 'HLA-DQA1*01:06-DQB1*06:36', 'HLA-DQA1*01:06-DQB1*06:37', 'HLA-DQA1*01:06-DQB1*06:38',
         'HLA-DQA1*01:06-DQB1*06:39', 'HLA-DQA1*01:06-DQB1*06:40',
         'HLA-DQA1*01:06-DQB1*06:41', 'HLA-DQA1*01:06-DQB1*06:42', 'HLA-DQA1*01:06-DQB1*06:43', 'HLA-DQA1*01:06-DQB1*06:44',
         'HLA-DQA1*01:07-DQB1*02:01', 'HLA-DQA1*01:07-DQB1*02:02',
         'HLA-DQA1*01:07-DQB1*02:03', 'HLA-DQA1*01:07-DQB1*02:04', 'HLA-DQA1*01:07-DQB1*02:05', 'HLA-DQA1*01:07-DQB1*02:06',
         'HLA-DQA1*01:07-DQB1*03:01', 'HLA-DQA1*01:07-DQB1*03:02',
         'HLA-DQA1*01:07-DQB1*03:03', 'HLA-DQA1*01:07-DQB1*03:04', 'HLA-DQA1*01:07-DQB1*03:05', 'HLA-DQA1*01:07-DQB1*03:06',
         'HLA-DQA1*01:07-DQB1*03:07', 'HLA-DQA1*01:07-DQB1*03:08',
         'HLA-DQA1*01:07-DQB1*03:09', 'HLA-DQA1*01:07-DQB1*03:10', 'HLA-DQA1*01:07-DQB1*03:11', 'HLA-DQA1*01:07-DQB1*03:12',
         'HLA-DQA1*01:07-DQB1*03:13', 'HLA-DQA1*01:07-DQB1*03:14',
         'HLA-DQA1*01:07-DQB1*03:15', 'HLA-DQA1*01:07-DQB1*03:16', 'HLA-DQA1*01:07-DQB1*03:17', 'HLA-DQA1*01:07-DQB1*03:18',
         'HLA-DQA1*01:07-DQB1*03:19', 'HLA-DQA1*01:07-DQB1*03:20',
         'HLA-DQA1*01:07-DQB1*03:21', 'HLA-DQA1*01:07-DQB1*03:22', 'HLA-DQA1*01:07-DQB1*03:23', 'HLA-DQA1*01:07-DQB1*03:24',
         'HLA-DQA1*01:07-DQB1*03:25', 'HLA-DQA1*01:07-DQB1*03:26',
         'HLA-DQA1*01:07-DQB1*03:27', 'HLA-DQA1*01:07-DQB1*03:28', 'HLA-DQA1*01:07-DQB1*03:29', 'HLA-DQA1*01:07-DQB1*03:30',
         'HLA-DQA1*01:07-DQB1*03:31', 'HLA-DQA1*01:07-DQB1*03:32',
         'HLA-DQA1*01:07-DQB1*03:33', 'HLA-DQA1*01:07-DQB1*03:34', 'HLA-DQA1*01:07-DQB1*03:35', 'HLA-DQA1*01:07-DQB1*03:36',
         'HLA-DQA1*01:07-DQB1*03:37', 'HLA-DQA1*01:07-DQB1*03:38',
         'HLA-DQA1*01:07-DQB1*04:01', 'HLA-DQA1*01:07-DQB1*04:02', 'HLA-DQA1*01:07-DQB1*04:03', 'HLA-DQA1*01:07-DQB1*04:04',
         'HLA-DQA1*01:07-DQB1*04:05', 'HLA-DQA1*01:07-DQB1*04:06',
         'HLA-DQA1*01:07-DQB1*04:07', 'HLA-DQA1*01:07-DQB1*04:08', 'HLA-DQA1*01:07-DQB1*05:01', 'HLA-DQA1*01:07-DQB1*05:02',
         'HLA-DQA1*01:07-DQB1*05:03', 'HLA-DQA1*01:07-DQB1*05:05',
         'HLA-DQA1*01:07-DQB1*05:06', 'HLA-DQA1*01:07-DQB1*05:07', 'HLA-DQA1*01:07-DQB1*05:08', 'HLA-DQA1*01:07-DQB1*05:09',
         'HLA-DQA1*01:07-DQB1*05:10', 'HLA-DQA1*01:07-DQB1*05:11',
         'HLA-DQA1*01:07-DQB1*05:12', 'HLA-DQA1*01:07-DQB1*05:13', 'HLA-DQA1*01:07-DQB1*05:14', 'HLA-DQA1*01:07-DQB1*06:01',
         'HLA-DQA1*01:07-DQB1*06:02', 'HLA-DQA1*01:07-DQB1*06:03',
         'HLA-DQA1*01:07-DQB1*06:04', 'HLA-DQA1*01:07-DQB1*06:07', 'HLA-DQA1*01:07-DQB1*06:08', 'HLA-DQA1*01:07-DQB1*06:09',
         'HLA-DQA1*01:07-DQB1*06:10', 'HLA-DQA1*01:07-DQB1*06:11',
         'HLA-DQA1*01:07-DQB1*06:12', 'HLA-DQA1*01:07-DQB1*06:14', 'HLA-DQA1*01:07-DQB1*06:15', 'HLA-DQA1*01:07-DQB1*06:16',
         'HLA-DQA1*01:07-DQB1*06:17', 'HLA-DQA1*01:07-DQB1*06:18',
         'HLA-DQA1*01:07-DQB1*06:19', 'HLA-DQA1*01:07-DQB1*06:21', 'HLA-DQA1*01:07-DQB1*06:22', 'HLA-DQA1*01:07-DQB1*06:23',
         'HLA-DQA1*01:07-DQB1*06:24', 'HLA-DQA1*01:07-DQB1*06:25',
         'HLA-DQA1*01:07-DQB1*06:27', 'HLA-DQA1*01:07-DQB1*06:28', 'HLA-DQA1*01:07-DQB1*06:29', 'HLA-DQA1*01:07-DQB1*06:30',
         'HLA-DQA1*01:07-DQB1*06:31', 'HLA-DQA1*01:07-DQB1*06:32',
         'HLA-DQA1*01:07-DQB1*06:33', 'HLA-DQA1*01:07-DQB1*06:34', 'HLA-DQA1*01:07-DQB1*06:35', 'HLA-DQA1*01:07-DQB1*06:36',
         'HLA-DQA1*01:07-DQB1*06:37', 'HLA-DQA1*01:07-DQB1*06:38',
         'HLA-DQA1*01:07-DQB1*06:39', 'HLA-DQA1*01:07-DQB1*06:40', 'HLA-DQA1*01:07-DQB1*06:41', 'HLA-DQA1*01:07-DQB1*06:42',
         'HLA-DQA1*01:07-DQB1*06:43', 'HLA-DQA1*01:07-DQB1*06:44',
         'HLA-DQA1*01:08-DQB1*02:01', 'HLA-DQA1*01:08-DQB1*02:02', 'HLA-DQA1*01:08-DQB1*02:03', 'HLA-DQA1*01:08-DQB1*02:04',
         'HLA-DQA1*01:08-DQB1*02:05', 'HLA-DQA1*01:08-DQB1*02:06',
         'HLA-DQA1*01:08-DQB1*03:01', 'HLA-DQA1*01:08-DQB1*03:02', 'HLA-DQA1*01:08-DQB1*03:03', 'HLA-DQA1*01:08-DQB1*03:04',
         'HLA-DQA1*01:08-DQB1*03:05', 'HLA-DQA1*01:08-DQB1*03:06',
         'HLA-DQA1*01:08-DQB1*03:07', 'HLA-DQA1*01:08-DQB1*03:08', 'HLA-DQA1*01:08-DQB1*03:09', 'HLA-DQA1*01:08-DQB1*03:10',
         'HLA-DQA1*01:08-DQB1*03:11', 'HLA-DQA1*01:08-DQB1*03:12',
         'HLA-DQA1*01:08-DQB1*03:13', 'HLA-DQA1*01:08-DQB1*03:14', 'HLA-DQA1*01:08-DQB1*03:15', 'HLA-DQA1*01:08-DQB1*03:16',
         'HLA-DQA1*01:08-DQB1*03:17', 'HLA-DQA1*01:08-DQB1*03:18',
         'HLA-DQA1*01:08-DQB1*03:19', 'HLA-DQA1*01:08-DQB1*03:20', 'HLA-DQA1*01:08-DQB1*03:21', 'HLA-DQA1*01:08-DQB1*03:22',
         'HLA-DQA1*01:08-DQB1*03:23', 'HLA-DQA1*01:08-DQB1*03:24',
         'HLA-DQA1*01:08-DQB1*03:25', 'HLA-DQA1*01:08-DQB1*03:26', 'HLA-DQA1*01:08-DQB1*03:27', 'HLA-DQA1*01:08-DQB1*03:28',
         'HLA-DQA1*01:08-DQB1*03:29', 'HLA-DQA1*01:08-DQB1*03:30',
         'HLA-DQA1*01:08-DQB1*03:31', 'HLA-DQA1*01:08-DQB1*03:32', 'HLA-DQA1*01:08-DQB1*03:33', 'HLA-DQA1*01:08-DQB1*03:34',
         'HLA-DQA1*01:08-DQB1*03:35', 'HLA-DQA1*01:08-DQB1*03:36',
         'HLA-DQA1*01:08-DQB1*03:37', 'HLA-DQA1*01:08-DQB1*03:38', 'HLA-DQA1*01:08-DQB1*04:01', 'HLA-DQA1*01:08-DQB1*04:02',
         'HLA-DQA1*01:08-DQB1*04:03', 'HLA-DQA1*01:08-DQB1*04:04',
         'HLA-DQA1*01:08-DQB1*04:05', 'HLA-DQA1*01:08-DQB1*04:06', 'HLA-DQA1*01:08-DQB1*04:07', 'HLA-DQA1*01:08-DQB1*04:08',
         'HLA-DQA1*01:08-DQB1*05:01', 'HLA-DQA1*01:08-DQB1*05:02',
         'HLA-DQA1*01:08-DQB1*05:03', 'HLA-DQA1*01:08-DQB1*05:05', 'HLA-DQA1*01:08-DQB1*05:06', 'HLA-DQA1*01:08-DQB1*05:07',
         'HLA-DQA1*01:08-DQB1*05:08', 'HLA-DQA1*01:08-DQB1*05:09',
         'HLA-DQA1*01:08-DQB1*05:10', 'HLA-DQA1*01:08-DQB1*05:11', 'HLA-DQA1*01:08-DQB1*05:12', 'HLA-DQA1*01:08-DQB1*05:13',
         'HLA-DQA1*01:08-DQB1*05:14', 'HLA-DQA1*01:08-DQB1*06:01',
         'HLA-DQA1*01:08-DQB1*06:02', 'HLA-DQA1*01:08-DQB1*06:03', 'HLA-DQA1*01:08-DQB1*06:04', 'HLA-DQA1*01:08-DQB1*06:07',
         'HLA-DQA1*01:08-DQB1*06:08', 'HLA-DQA1*01:08-DQB1*06:09',
         'HLA-DQA1*01:08-DQB1*06:10', 'HLA-DQA1*01:08-DQB1*06:11', 'HLA-DQA1*01:08-DQB1*06:12', 'HLA-DQA1*01:08-DQB1*06:14',
         'HLA-DQA1*01:08-DQB1*06:15', 'HLA-DQA1*01:08-DQB1*06:16',
         'HLA-DQA1*01:08-DQB1*06:17', 'HLA-DQA1*01:08-DQB1*06:18', 'HLA-DQA1*01:08-DQB1*06:19', 'HLA-DQA1*01:08-DQB1*06:21',
         'HLA-DQA1*01:08-DQB1*06:22', 'HLA-DQA1*01:08-DQB1*06:23',
         'HLA-DQA1*01:08-DQB1*06:24', 'HLA-DQA1*01:08-DQB1*06:25', 'HLA-DQA1*01:08-DQB1*06:27', 'HLA-DQA1*01:08-DQB1*06:28',
         'HLA-DQA1*01:08-DQB1*06:29', 'HLA-DQA1*01:08-DQB1*06:30',
         'HLA-DQA1*01:08-DQB1*06:31', 'HLA-DQA1*01:08-DQB1*06:32', 'HLA-DQA1*01:08-DQB1*06:33', 'HLA-DQA1*01:08-DQB1*06:34',
         'HLA-DQA1*01:08-DQB1*06:35', 'HLA-DQA1*01:08-DQB1*06:36',
         'HLA-DQA1*01:08-DQB1*06:37', 'HLA-DQA1*01:08-DQB1*06:38', 'HLA-DQA1*01:08-DQB1*06:39', 'HLA-DQA1*01:08-DQB1*06:40',
         'HLA-DQA1*01:08-DQB1*06:41', 'HLA-DQA1*01:08-DQB1*06:42',
         'HLA-DQA1*01:08-DQB1*06:43', 'HLA-DQA1*01:08-DQB1*06:44', 'HLA-DQA1*01:09-DQB1*02:01', 'HLA-DQA1*01:09-DQB1*02:02',
         'HLA-DQA1*01:09-DQB1*02:03', 'HLA-DQA1*01:09-DQB1*02:04',
         'HLA-DQA1*01:09-DQB1*02:05', 'HLA-DQA1*01:09-DQB1*02:06', 'HLA-DQA1*01:09-DQB1*03:01', 'HLA-DQA1*01:09-DQB1*03:02',
         'HLA-DQA1*01:09-DQB1*03:03', 'HLA-DQA1*01:09-DQB1*03:04',
         'HLA-DQA1*01:09-DQB1*03:05', 'HLA-DQA1*01:09-DQB1*03:06', 'HLA-DQA1*01:09-DQB1*03:07', 'HLA-DQA1*01:09-DQB1*03:08',
         'HLA-DQA1*01:09-DQB1*03:09', 'HLA-DQA1*01:09-DQB1*03:10',
         'HLA-DQA1*01:09-DQB1*03:11', 'HLA-DQA1*01:09-DQB1*03:12', 'HLA-DQA1*01:09-DQB1*03:13', 'HLA-DQA1*01:09-DQB1*03:14',
         'HLA-DQA1*01:09-DQB1*03:15', 'HLA-DQA1*01:09-DQB1*03:16',
         'HLA-DQA1*01:09-DQB1*03:17', 'HLA-DQA1*01:09-DQB1*03:18', 'HLA-DQA1*01:09-DQB1*03:19', 'HLA-DQA1*01:09-DQB1*03:20',
         'HLA-DQA1*01:09-DQB1*03:21', 'HLA-DQA1*01:09-DQB1*03:22',
         'HLA-DQA1*01:09-DQB1*03:23', 'HLA-DQA1*01:09-DQB1*03:24', 'HLA-DQA1*01:09-DQB1*03:25', 'HLA-DQA1*01:09-DQB1*03:26',
         'HLA-DQA1*01:09-DQB1*03:27', 'HLA-DQA1*01:09-DQB1*03:28',
         'HLA-DQA1*01:09-DQB1*03:29', 'HLA-DQA1*01:09-DQB1*03:30', 'HLA-DQA1*01:09-DQB1*03:31', 'HLA-DQA1*01:09-DQB1*03:32',
         'HLA-DQA1*01:09-DQB1*03:33', 'HLA-DQA1*01:09-DQB1*03:34',
         'HLA-DQA1*01:09-DQB1*03:35', 'HLA-DQA1*01:09-DQB1*03:36', 'HLA-DQA1*01:09-DQB1*03:37', 'HLA-DQA1*01:09-DQB1*03:38',
         'HLA-DQA1*01:09-DQB1*04:01', 'HLA-DQA1*01:09-DQB1*04:02',
         'HLA-DQA1*01:09-DQB1*04:03', 'HLA-DQA1*01:09-DQB1*04:04', 'HLA-DQA1*01:09-DQB1*04:05', 'HLA-DQA1*01:09-DQB1*04:06',
         'HLA-DQA1*01:09-DQB1*04:07', 'HLA-DQA1*01:09-DQB1*04:08',
         'HLA-DQA1*01:09-DQB1*05:01', 'HLA-DQA1*01:09-DQB1*05:02', 'HLA-DQA1*01:09-DQB1*05:03', 'HLA-DQA1*01:09-DQB1*05:05',
         'HLA-DQA1*01:09-DQB1*05:06', 'HLA-DQA1*01:09-DQB1*05:07',
         'HLA-DQA1*01:09-DQB1*05:08', 'HLA-DQA1*01:09-DQB1*05:09', 'HLA-DQA1*01:09-DQB1*05:10', 'HLA-DQA1*01:09-DQB1*05:11',
         'HLA-DQA1*01:09-DQB1*05:12', 'HLA-DQA1*01:09-DQB1*05:13',
         'HLA-DQA1*01:09-DQB1*05:14', 'HLA-DQA1*01:09-DQB1*06:01', 'HLA-DQA1*01:09-DQB1*06:02', 'HLA-DQA1*01:09-DQB1*06:03',
         'HLA-DQA1*01:09-DQB1*06:04', 'HLA-DQA1*01:09-DQB1*06:07',
         'HLA-DQA1*01:09-DQB1*06:08', 'HLA-DQA1*01:09-DQB1*06:09', 'HLA-DQA1*01:09-DQB1*06:10', 'HLA-DQA1*01:09-DQB1*06:11',
         'HLA-DQA1*01:09-DQB1*06:12', 'HLA-DQA1*01:09-DQB1*06:14',
         'HLA-DQA1*01:09-DQB1*06:15', 'HLA-DQA1*01:09-DQB1*06:16', 'HLA-DQA1*01:09-DQB1*06:17', 'HLA-DQA1*01:09-DQB1*06:18',
         'HLA-DQA1*01:09-DQB1*06:19', 'HLA-DQA1*01:09-DQB1*06:21',
         'HLA-DQA1*01:09-DQB1*06:22', 'HLA-DQA1*01:09-DQB1*06:23', 'HLA-DQA1*01:09-DQB1*06:24', 'HLA-DQA1*01:09-DQB1*06:25',
         'HLA-DQA1*01:09-DQB1*06:27', 'HLA-DQA1*01:09-DQB1*06:28',
         'HLA-DQA1*01:09-DQB1*06:29', 'HLA-DQA1*01:09-DQB1*06:30', 'HLA-DQA1*01:09-DQB1*06:31', 'HLA-DQA1*01:09-DQB1*06:32',
         'HLA-DQA1*01:09-DQB1*06:33', 'HLA-DQA1*01:09-DQB1*06:34',
         'HLA-DQA1*01:09-DQB1*06:35', 'HLA-DQA1*01:09-DQB1*06:36', 'HLA-DQA1*01:09-DQB1*06:37', 'HLA-DQA1*01:09-DQB1*06:38',
         'HLA-DQA1*01:09-DQB1*06:39', 'HLA-DQA1*01:09-DQB1*06:40',
         'HLA-DQA1*01:09-DQB1*06:41', 'HLA-DQA1*01:09-DQB1*06:42', 'HLA-DQA1*01:09-DQB1*06:43', 'HLA-DQA1*01:09-DQB1*06:44',
         'HLA-DQA1*02:01-DQB1*02:01', 'HLA-DQA1*02:01-DQB1*02:02',
         'HLA-DQA1*02:01-DQB1*02:03', 'HLA-DQA1*02:01-DQB1*02:04', 'HLA-DQA1*02:01-DQB1*02:05', 'HLA-DQA1*02:01-DQB1*02:06',
         'HLA-DQA1*02:01-DQB1*03:01', 'HLA-DQA1*02:01-DQB1*03:02',
         'HLA-DQA1*02:01-DQB1*03:03', 'HLA-DQA1*02:01-DQB1*03:04', 'HLA-DQA1*02:01-DQB1*03:05', 'HLA-DQA1*02:01-DQB1*03:06',
         'HLA-DQA1*02:01-DQB1*03:07', 'HLA-DQA1*02:01-DQB1*03:08',
         'HLA-DQA1*02:01-DQB1*03:09', 'HLA-DQA1*02:01-DQB1*03:10', 'HLA-DQA1*02:01-DQB1*03:11', 'HLA-DQA1*02:01-DQB1*03:12',
         'HLA-DQA1*02:01-DQB1*03:13', 'HLA-DQA1*02:01-DQB1*03:14',
         'HLA-DQA1*02:01-DQB1*03:15', 'HLA-DQA1*02:01-DQB1*03:16', 'HLA-DQA1*02:01-DQB1*03:17', 'HLA-DQA1*02:01-DQB1*03:18',
         'HLA-DQA1*02:01-DQB1*03:19', 'HLA-DQA1*02:01-DQB1*03:20',
         'HLA-DQA1*02:01-DQB1*03:21', 'HLA-DQA1*02:01-DQB1*03:22', 'HLA-DQA1*02:01-DQB1*03:23', 'HLA-DQA1*02:01-DQB1*03:24',
         'HLA-DQA1*02:01-DQB1*03:25', 'HLA-DQA1*02:01-DQB1*03:26',
         'HLA-DQA1*02:01-DQB1*03:27', 'HLA-DQA1*02:01-DQB1*03:28', 'HLA-DQA1*02:01-DQB1*03:29', 'HLA-DQA1*02:01-DQB1*03:30',
         'HLA-DQA1*02:01-DQB1*03:31', 'HLA-DQA1*02:01-DQB1*03:32',
         'HLA-DQA1*02:01-DQB1*03:33', 'HLA-DQA1*02:01-DQB1*03:34', 'HLA-DQA1*02:01-DQB1*03:35', 'HLA-DQA1*02:01-DQB1*03:36',
         'HLA-DQA1*02:01-DQB1*03:37', 'HLA-DQA1*02:01-DQB1*03:38',
         'HLA-DQA1*02:01-DQB1*04:01', 'HLA-DQA1*02:01-DQB1*04:02', 'HLA-DQA1*02:01-DQB1*04:03', 'HLA-DQA1*02:01-DQB1*04:04',
         'HLA-DQA1*02:01-DQB1*04:05', 'HLA-DQA1*02:01-DQB1*04:06',
         'HLA-DQA1*02:01-DQB1*04:07', 'HLA-DQA1*02:01-DQB1*04:08', 'HLA-DQA1*02:01-DQB1*05:01', 'HLA-DQA1*02:01-DQB1*05:02',
         'HLA-DQA1*02:01-DQB1*05:03', 'HLA-DQA1*02:01-DQB1*05:05',
         'HLA-DQA1*02:01-DQB1*05:06', 'HLA-DQA1*02:01-DQB1*05:07', 'HLA-DQA1*02:01-DQB1*05:08', 'HLA-DQA1*02:01-DQB1*05:09',
         'HLA-DQA1*02:01-DQB1*05:10', 'HLA-DQA1*02:01-DQB1*05:11',
         'HLA-DQA1*02:01-DQB1*05:12', 'HLA-DQA1*02:01-DQB1*05:13', 'HLA-DQA1*02:01-DQB1*05:14', 'HLA-DQA1*02:01-DQB1*06:01',
         'HLA-DQA1*02:01-DQB1*06:02', 'HLA-DQA1*02:01-DQB1*06:03',
         'HLA-DQA1*02:01-DQB1*06:04', 'HLA-DQA1*02:01-DQB1*06:07', 'HLA-DQA1*02:01-DQB1*06:08', 'HLA-DQA1*02:01-DQB1*06:09',
         'HLA-DQA1*02:01-DQB1*06:10', 'HLA-DQA1*02:01-DQB1*06:11',
         'HLA-DQA1*02:01-DQB1*06:12', 'HLA-DQA1*02:01-DQB1*06:14', 'HLA-DQA1*02:01-DQB1*06:15', 'HLA-DQA1*02:01-DQB1*06:16',
         'HLA-DQA1*02:01-DQB1*06:17', 'HLA-DQA1*02:01-DQB1*06:18',
         'HLA-DQA1*02:01-DQB1*06:19', 'HLA-DQA1*02:01-DQB1*06:21', 'HLA-DQA1*02:01-DQB1*06:22', 'HLA-DQA1*02:01-DQB1*06:23',
         'HLA-DQA1*02:01-DQB1*06:24', 'HLA-DQA1*02:01-DQB1*06:25',
         'HLA-DQA1*02:01-DQB1*06:27', 'HLA-DQA1*02:01-DQB1*06:28', 'HLA-DQA1*02:01-DQB1*06:29', 'HLA-DQA1*02:01-DQB1*06:30',
         'HLA-DQA1*02:01-DQB1*06:31', 'HLA-DQA1*02:01-DQB1*06:32',
         'HLA-DQA1*02:01-DQB1*06:33', 'HLA-DQA1*02:01-DQB1*06:34', 'HLA-DQA1*02:01-DQB1*06:35', 'HLA-DQA1*02:01-DQB1*06:36',
         'HLA-DQA1*02:01-DQB1*06:37', 'HLA-DQA1*02:01-DQB1*06:38',
         'HLA-DQA1*02:01-DQB1*06:39', 'HLA-DQA1*02:01-DQB1*06:40', 'HLA-DQA1*02:01-DQB1*06:41', 'HLA-DQA1*02:01-DQB1*06:42',
         'HLA-DQA1*02:01-DQB1*06:43', 'HLA-DQA1*02:01-DQB1*06:44',
         'HLA-DQA1*03:01-DQB1*02:01', 'HLA-DQA1*03:01-DQB1*02:02', 'HLA-DQA1*03:01-DQB1*02:03', 'HLA-DQA1*03:01-DQB1*02:04',
         'HLA-DQA1*03:01-DQB1*02:05', 'HLA-DQA1*03:01-DQB1*02:06',
         'HLA-DQA1*03:01-DQB1*03:01', 'HLA-DQA1*03:01-DQB1*03:02', 'HLA-DQA1*03:01-DQB1*03:03', 'HLA-DQA1*03:01-DQB1*03:04',
         'HLA-DQA1*03:01-DQB1*03:05', 'HLA-DQA1*03:01-DQB1*03:06',
         'HLA-DQA1*03:01-DQB1*03:07', 'HLA-DQA1*03:01-DQB1*03:08', 'HLA-DQA1*03:01-DQB1*03:09', 'HLA-DQA1*03:01-DQB1*03:10',
         'HLA-DQA1*03:01-DQB1*03:11', 'HLA-DQA1*03:01-DQB1*03:12',
         'HLA-DQA1*03:01-DQB1*03:13', 'HLA-DQA1*03:01-DQB1*03:14', 'HLA-DQA1*03:01-DQB1*03:15', 'HLA-DQA1*03:01-DQB1*03:16',
         'HLA-DQA1*03:01-DQB1*03:17', 'HLA-DQA1*03:01-DQB1*03:18',
         'HLA-DQA1*03:01-DQB1*03:19', 'HLA-DQA1*03:01-DQB1*03:20', 'HLA-DQA1*03:01-DQB1*03:21', 'HLA-DQA1*03:01-DQB1*03:22',
         'HLA-DQA1*03:01-DQB1*03:23', 'HLA-DQA1*03:01-DQB1*03:24',
         'HLA-DQA1*03:01-DQB1*03:25', 'HLA-DQA1*03:01-DQB1*03:26', 'HLA-DQA1*03:01-DQB1*03:27', 'HLA-DQA1*03:01-DQB1*03:28',
         'HLA-DQA1*03:01-DQB1*03:29', 'HLA-DQA1*03:01-DQB1*03:30',
         'HLA-DQA1*03:01-DQB1*03:31', 'HLA-DQA1*03:01-DQB1*03:32', 'HLA-DQA1*03:01-DQB1*03:33', 'HLA-DQA1*03:01-DQB1*03:34',
         'HLA-DQA1*03:01-DQB1*03:35', 'HLA-DQA1*03:01-DQB1*03:36',
         'HLA-DQA1*03:01-DQB1*03:37', 'HLA-DQA1*03:01-DQB1*03:38', 'HLA-DQA1*03:01-DQB1*04:01', 'HLA-DQA1*03:01-DQB1*04:02',
         'HLA-DQA1*03:01-DQB1*04:03', 'HLA-DQA1*03:01-DQB1*04:04',
         'HLA-DQA1*03:01-DQB1*04:05', 'HLA-DQA1*03:01-DQB1*04:06', 'HLA-DQA1*03:01-DQB1*04:07', 'HLA-DQA1*03:01-DQB1*04:08',
         'HLA-DQA1*03:01-DQB1*05:01', 'HLA-DQA1*03:01-DQB1*05:02',
         'HLA-DQA1*03:01-DQB1*05:03', 'HLA-DQA1*03:01-DQB1*05:05', 'HLA-DQA1*03:01-DQB1*05:06', 'HLA-DQA1*03:01-DQB1*05:07',
         'HLA-DQA1*03:01-DQB1*05:08', 'HLA-DQA1*03:01-DQB1*05:09',
         'HLA-DQA1*03:01-DQB1*05:10', 'HLA-DQA1*03:01-DQB1*05:11', 'HLA-DQA1*03:01-DQB1*05:12', 'HLA-DQA1*03:01-DQB1*05:13',
         'HLA-DQA1*03:01-DQB1*05:14', 'HLA-DQA1*03:01-DQB1*06:01',
         'HLA-DQA1*03:01-DQB1*06:02', 'HLA-DQA1*03:01-DQB1*06:03', 'HLA-DQA1*03:01-DQB1*06:04', 'HLA-DQA1*03:01-DQB1*06:07',
         'HLA-DQA1*03:01-DQB1*06:08', 'HLA-DQA1*03:01-DQB1*06:09',
         'HLA-DQA1*03:01-DQB1*06:10', 'HLA-DQA1*03:01-DQB1*06:11', 'HLA-DQA1*03:01-DQB1*06:12', 'HLA-DQA1*03:01-DQB1*06:14',
         'HLA-DQA1*03:01-DQB1*06:15', 'HLA-DQA1*03:01-DQB1*06:16',
         'HLA-DQA1*03:01-DQB1*06:17', 'HLA-DQA1*03:01-DQB1*06:18', 'HLA-DQA1*03:01-DQB1*06:19', 'HLA-DQA1*03:01-DQB1*06:21',
         'HLA-DQA1*03:01-DQB1*06:22', 'HLA-DQA1*03:01-DQB1*06:23',
         'HLA-DQA1*03:01-DQB1*06:24', 'HLA-DQA1*03:01-DQB1*06:25', 'HLA-DQA1*03:01-DQB1*06:27', 'HLA-DQA1*03:01-DQB1*06:28',
         'HLA-DQA1*03:01-DQB1*06:29', 'HLA-DQA1*03:01-DQB1*06:30',
         'HLA-DQA1*03:01-DQB1*06:31', 'HLA-DQA1*03:01-DQB1*06:32', 'HLA-DQA1*03:01-DQB1*06:33', 'HLA-DQA1*03:01-DQB1*06:34',
         'HLA-DQA1*03:01-DQB1*06:35', 'HLA-DQA1*03:01-DQB1*06:36',
         'HLA-DQA1*03:01-DQB1*06:37', 'HLA-DQA1*03:01-DQB1*06:38', 'HLA-DQA1*03:01-DQB1*06:39', 'HLA-DQA1*03:01-DQB1*06:40',
         'HLA-DQA1*03:01-DQB1*06:41', 'HLA-DQA1*03:01-DQB1*06:42',
         'HLA-DQA1*03:01-DQB1*06:43', 'HLA-DQA1*03:01-DQB1*06:44', 'HLA-DQA1*03:02-DQB1*02:01', 'HLA-DQA1*03:02-DQB1*02:02',
         'HLA-DQA1*03:02-DQB1*02:03', 'HLA-DQA1*03:02-DQB1*02:04',
         'HLA-DQA1*03:02-DQB1*02:05', 'HLA-DQA1*03:02-DQB1*02:06', 'HLA-DQA1*03:02-DQB1*03:01', 'HLA-DQA1*03:02-DQB1*03:02',
         'HLA-DQA1*03:02-DQB1*03:03', 'HLA-DQA1*03:02-DQB1*03:04',
         'HLA-DQA1*03:02-DQB1*03:05', 'HLA-DQA1*03:02-DQB1*03:06', 'HLA-DQA1*03:02-DQB1*03:07', 'HLA-DQA1*03:02-DQB1*03:08',
         'HLA-DQA1*03:02-DQB1*03:09', 'HLA-DQA1*03:02-DQB1*03:10',
         'HLA-DQA1*03:02-DQB1*03:11', 'HLA-DQA1*03:02-DQB1*03:12', 'HLA-DQA1*03:02-DQB1*03:13', 'HLA-DQA1*03:02-DQB1*03:14',
         'HLA-DQA1*03:02-DQB1*03:15', 'HLA-DQA1*03:02-DQB1*03:16',
         'HLA-DQA1*03:02-DQB1*03:17', 'HLA-DQA1*03:02-DQB1*03:18', 'HLA-DQA1*03:02-DQB1*03:19', 'HLA-DQA1*03:02-DQB1*03:20',
         'HLA-DQA1*03:02-DQB1*03:21', 'HLA-DQA1*03:02-DQB1*03:22',
         'HLA-DQA1*03:02-DQB1*03:23', 'HLA-DQA1*03:02-DQB1*03:24', 'HLA-DQA1*03:02-DQB1*03:25', 'HLA-DQA1*03:02-DQB1*03:26',
         'HLA-DQA1*03:02-DQB1*03:27', 'HLA-DQA1*03:02-DQB1*03:28',
         'HLA-DQA1*03:02-DQB1*03:29', 'HLA-DQA1*03:02-DQB1*03:30', 'HLA-DQA1*03:02-DQB1*03:31', 'HLA-DQA1*03:02-DQB1*03:32',
         'HLA-DQA1*03:02-DQB1*03:33', 'HLA-DQA1*03:02-DQB1*03:34',
         'HLA-DQA1*03:02-DQB1*03:35', 'HLA-DQA1*03:02-DQB1*03:36', 'HLA-DQA1*03:02-DQB1*03:37', 'HLA-DQA1*03:02-DQB1*03:38',
         'HLA-DQA1*03:02-DQB1*04:01', 'HLA-DQA1*03:02-DQB1*04:02',
         'HLA-DQA1*03:02-DQB1*04:03', 'HLA-DQA1*03:02-DQB1*04:04', 'HLA-DQA1*03:02-DQB1*04:05', 'HLA-DQA1*03:02-DQB1*04:06',
         'HLA-DQA1*03:02-DQB1*04:07', 'HLA-DQA1*03:02-DQB1*04:08',
         'HLA-DQA1*03:02-DQB1*05:01', 'HLA-DQA1*03:02-DQB1*05:02', 'HLA-DQA1*03:02-DQB1*05:03', 'HLA-DQA1*03:02-DQB1*05:05',
         'HLA-DQA1*03:02-DQB1*05:06', 'HLA-DQA1*03:02-DQB1*05:07',
         'HLA-DQA1*03:02-DQB1*05:08', 'HLA-DQA1*03:02-DQB1*05:09', 'HLA-DQA1*03:02-DQB1*05:10', 'HLA-DQA1*03:02-DQB1*05:11',
         'HLA-DQA1*03:02-DQB1*05:12', 'HLA-DQA1*03:02-DQB1*05:13',
         'HLA-DQA1*03:02-DQB1*05:14', 'HLA-DQA1*03:02-DQB1*06:01', 'HLA-DQA1*03:02-DQB1*06:02', 'HLA-DQA1*03:02-DQB1*06:03',
         'HLA-DQA1*03:02-DQB1*06:04', 'HLA-DQA1*03:02-DQB1*06:07',
         'HLA-DQA1*03:02-DQB1*06:08', 'HLA-DQA1*03:02-DQB1*06:09', 'HLA-DQA1*03:02-DQB1*06:10', 'HLA-DQA1*03:02-DQB1*06:11',
         'HLA-DQA1*03:02-DQB1*06:12', 'HLA-DQA1*03:02-DQB1*06:14',
         'HLA-DQA1*03:02-DQB1*06:15', 'HLA-DQA1*03:02-DQB1*06:16', 'HLA-DQA1*03:02-DQB1*06:17', 'HLA-DQA1*03:02-DQB1*06:18',
         'HLA-DQA1*03:02-DQB1*06:19', 'HLA-DQA1*03:02-DQB1*06:21',
         'HLA-DQA1*03:02-DQB1*06:22', 'HLA-DQA1*03:02-DQB1*06:23', 'HLA-DQA1*03:02-DQB1*06:24', 'HLA-DQA1*03:02-DQB1*06:25',
         'HLA-DQA1*03:02-DQB1*06:27', 'HLA-DQA1*03:02-DQB1*06:28',
         'HLA-DQA1*03:02-DQB1*06:29', 'HLA-DQA1*03:02-DQB1*06:30', 'HLA-DQA1*03:02-DQB1*06:31', 'HLA-DQA1*03:02-DQB1*06:32',
         'HLA-DQA1*03:02-DQB1*06:33', 'HLA-DQA1*03:02-DQB1*06:34',
         'HLA-DQA1*03:02-DQB1*06:35', 'HLA-DQA1*03:02-DQB1*06:36', 'HLA-DQA1*03:02-DQB1*06:37', 'HLA-DQA1*03:02-DQB1*06:38',
         'HLA-DQA1*03:02-DQB1*06:39', 'HLA-DQA1*03:02-DQB1*06:40',
         'HLA-DQA1*03:02-DQB1*06:41', 'HLA-DQA1*03:02-DQB1*06:42', 'HLA-DQA1*03:02-DQB1*06:43', 'HLA-DQA1*03:02-DQB1*06:44',
         'HLA-DQA1*03:03-DQB1*02:01', 'HLA-DQA1*03:03-DQB1*02:02',
         'HLA-DQA1*03:03-DQB1*02:03', 'HLA-DQA1*03:03-DQB1*02:04', 'HLA-DQA1*03:03-DQB1*02:05', 'HLA-DQA1*03:03-DQB1*02:06',
         'HLA-DQA1*03:03-DQB1*03:01', 'HLA-DQA1*03:03-DQB1*03:02',
         'HLA-DQA1*03:03-DQB1*03:03', 'HLA-DQA1*03:03-DQB1*03:04', 'HLA-DQA1*03:03-DQB1*03:05', 'HLA-DQA1*03:03-DQB1*03:06',
         'HLA-DQA1*03:03-DQB1*03:07', 'HLA-DQA1*03:03-DQB1*03:08',
         'HLA-DQA1*03:03-DQB1*03:09', 'HLA-DQA1*03:03-DQB1*03:10', 'HLA-DQA1*03:03-DQB1*03:11', 'HLA-DQA1*03:03-DQB1*03:12',
         'HLA-DQA1*03:03-DQB1*03:13', 'HLA-DQA1*03:03-DQB1*03:14',
         'HLA-DQA1*03:03-DQB1*03:15', 'HLA-DQA1*03:03-DQB1*03:16', 'HLA-DQA1*03:03-DQB1*03:17', 'HLA-DQA1*03:03-DQB1*03:18',
         'HLA-DQA1*03:03-DQB1*03:19', 'HLA-DQA1*03:03-DQB1*03:20',
         'HLA-DQA1*03:03-DQB1*03:21', 'HLA-DQA1*03:03-DQB1*03:22', 'HLA-DQA1*03:03-DQB1*03:23', 'HLA-DQA1*03:03-DQB1*03:24',
         'HLA-DQA1*03:03-DQB1*03:25', 'HLA-DQA1*03:03-DQB1*03:26',
         'HLA-DQA1*03:03-DQB1*03:27', 'HLA-DQA1*03:03-DQB1*03:28', 'HLA-DQA1*03:03-DQB1*03:29', 'HLA-DQA1*03:03-DQB1*03:30',
         'HLA-DQA1*03:03-DQB1*03:31', 'HLA-DQA1*03:03-DQB1*03:32',
         'HLA-DQA1*03:03-DQB1*03:33', 'HLA-DQA1*03:03-DQB1*03:34', 'HLA-DQA1*03:03-DQB1*03:35', 'HLA-DQA1*03:03-DQB1*03:36',
         'HLA-DQA1*03:03-DQB1*03:37', 'HLA-DQA1*03:03-DQB1*03:38',
         'HLA-DQA1*03:03-DQB1*04:01', 'HLA-DQA1*03:03-DQB1*04:02', 'HLA-DQA1*03:03-DQB1*04:03', 'HLA-DQA1*03:03-DQB1*04:04',
         'HLA-DQA1*03:03-DQB1*04:05', 'HLA-DQA1*03:03-DQB1*04:06',
         'HLA-DQA1*03:03-DQB1*04:07', 'HLA-DQA1*03:03-DQB1*04:08', 'HLA-DQA1*03:03-DQB1*05:01', 'HLA-DQA1*03:03-DQB1*05:02',
         'HLA-DQA1*03:03-DQB1*05:03', 'HLA-DQA1*03:03-DQB1*05:05',
         'HLA-DQA1*03:03-DQB1*05:06', 'HLA-DQA1*03:03-DQB1*05:07', 'HLA-DQA1*03:03-DQB1*05:08', 'HLA-DQA1*03:03-DQB1*05:09',
         'HLA-DQA1*03:03-DQB1*05:10', 'HLA-DQA1*03:03-DQB1*05:11',
         'HLA-DQA1*03:03-DQB1*05:12', 'HLA-DQA1*03:03-DQB1*05:13', 'HLA-DQA1*03:03-DQB1*05:14', 'HLA-DQA1*03:03-DQB1*06:01',
         'HLA-DQA1*03:03-DQB1*06:02', 'HLA-DQA1*03:03-DQB1*06:03',
         'HLA-DQA1*03:03-DQB1*06:04', 'HLA-DQA1*03:03-DQB1*06:07', 'HLA-DQA1*03:03-DQB1*06:08', 'HLA-DQA1*03:03-DQB1*06:09',
         'HLA-DQA1*03:03-DQB1*06:10', 'HLA-DQA1*03:03-DQB1*06:11',
         'HLA-DQA1*03:03-DQB1*06:12', 'HLA-DQA1*03:03-DQB1*06:14', 'HLA-DQA1*03:03-DQB1*06:15', 'HLA-DQA1*03:03-DQB1*06:16',
         'HLA-DQA1*03:03-DQB1*06:17', 'HLA-DQA1*03:03-DQB1*06:18',
         'HLA-DQA1*03:03-DQB1*06:19', 'HLA-DQA1*03:03-DQB1*06:21', 'HLA-DQA1*03:03-DQB1*06:22', 'HLA-DQA1*03:03-DQB1*06:23',
         'HLA-DQA1*03:03-DQB1*06:24', 'HLA-DQA1*03:03-DQB1*06:25',
         'HLA-DQA1*03:03-DQB1*06:27', 'HLA-DQA1*03:03-DQB1*06:28', 'HLA-DQA1*03:03-DQB1*06:29', 'HLA-DQA1*03:03-DQB1*06:30',
         'HLA-DQA1*03:03-DQB1*06:31', 'HLA-DQA1*03:03-DQB1*06:32',
         'HLA-DQA1*03:03-DQB1*06:33', 'HLA-DQA1*03:03-DQB1*06:34', 'HLA-DQA1*03:03-DQB1*06:35', 'HLA-DQA1*03:03-DQB1*06:36',
         'HLA-DQA1*03:03-DQB1*06:37', 'HLA-DQA1*03:03-DQB1*06:38',
         'HLA-DQA1*03:03-DQB1*06:39', 'HLA-DQA1*03:03-DQB1*06:40', 'HLA-DQA1*03:03-DQB1*06:41', 'HLA-DQA1*03:03-DQB1*06:42',
         'HLA-DQA1*03:03-DQB1*06:43', 'HLA-DQA1*03:03-DQB1*06:44',
         'HLA-DQA1*04:01-DQB1*02:01', 'HLA-DQA1*04:01-DQB1*02:02', 'HLA-DQA1*04:01-DQB1*02:03', 'HLA-DQA1*04:01-DQB1*02:04',
         'HLA-DQA1*04:01-DQB1*02:05', 'HLA-DQA1*04:01-DQB1*02:06',
         'HLA-DQA1*04:01-DQB1*03:01', 'HLA-DQA1*04:01-DQB1*03:02', 'HLA-DQA1*04:01-DQB1*03:03', 'HLA-DQA1*04:01-DQB1*03:04',
         'HLA-DQA1*04:01-DQB1*03:05', 'HLA-DQA1*04:01-DQB1*03:06',
         'HLA-DQA1*04:01-DQB1*03:07', 'HLA-DQA1*04:01-DQB1*03:08', 'HLA-DQA1*04:01-DQB1*03:09', 'HLA-DQA1*04:01-DQB1*03:10',
         'HLA-DQA1*04:01-DQB1*03:11', 'HLA-DQA1*04:01-DQB1*03:12',
         'HLA-DQA1*04:01-DQB1*03:13', 'HLA-DQA1*04:01-DQB1*03:14', 'HLA-DQA1*04:01-DQB1*03:15', 'HLA-DQA1*04:01-DQB1*03:16',
         'HLA-DQA1*04:01-DQB1*03:17', 'HLA-DQA1*04:01-DQB1*03:18',
         'HLA-DQA1*04:01-DQB1*03:19', 'HLA-DQA1*04:01-DQB1*03:20', 'HLA-DQA1*04:01-DQB1*03:21', 'HLA-DQA1*04:01-DQB1*03:22',
         'HLA-DQA1*04:01-DQB1*03:23', 'HLA-DQA1*04:01-DQB1*03:24',
         'HLA-DQA1*04:01-DQB1*03:25', 'HLA-DQA1*04:01-DQB1*03:26', 'HLA-DQA1*04:01-DQB1*03:27', 'HLA-DQA1*04:01-DQB1*03:28',
         'HLA-DQA1*04:01-DQB1*03:29', 'HLA-DQA1*04:01-DQB1*03:30',
         'HLA-DQA1*04:01-DQB1*03:31', 'HLA-DQA1*04:01-DQB1*03:32', 'HLA-DQA1*04:01-DQB1*03:33', 'HLA-DQA1*04:01-DQB1*03:34',
         'HLA-DQA1*04:01-DQB1*03:35', 'HLA-DQA1*04:01-DQB1*03:36',
         'HLA-DQA1*04:01-DQB1*03:37', 'HLA-DQA1*04:01-DQB1*03:38', 'HLA-DQA1*04:01-DQB1*04:01', 'HLA-DQA1*04:01-DQB1*04:02',
         'HLA-DQA1*04:01-DQB1*04:03', 'HLA-DQA1*04:01-DQB1*04:04',
         'HLA-DQA1*04:01-DQB1*04:05', 'HLA-DQA1*04:01-DQB1*04:06', 'HLA-DQA1*04:01-DQB1*04:07', 'HLA-DQA1*04:01-DQB1*04:08',
         'HLA-DQA1*04:01-DQB1*05:01', 'HLA-DQA1*04:01-DQB1*05:02',
         'HLA-DQA1*04:01-DQB1*05:03', 'HLA-DQA1*04:01-DQB1*05:05', 'HLA-DQA1*04:01-DQB1*05:06', 'HLA-DQA1*04:01-DQB1*05:07',
         'HLA-DQA1*04:01-DQB1*05:08', 'HLA-DQA1*04:01-DQB1*05:09',
         'HLA-DQA1*04:01-DQB1*05:10', 'HLA-DQA1*04:01-DQB1*05:11', 'HLA-DQA1*04:01-DQB1*05:12', 'HLA-DQA1*04:01-DQB1*05:13',
         'HLA-DQA1*04:01-DQB1*05:14', 'HLA-DQA1*04:01-DQB1*06:01',
         'HLA-DQA1*04:01-DQB1*06:02', 'HLA-DQA1*04:01-DQB1*06:03', 'HLA-DQA1*04:01-DQB1*06:04', 'HLA-DQA1*04:01-DQB1*06:07',
         'HLA-DQA1*04:01-DQB1*06:08', 'HLA-DQA1*04:01-DQB1*06:09',
         'HLA-DQA1*04:01-DQB1*06:10', 'HLA-DQA1*04:01-DQB1*06:11', 'HLA-DQA1*04:01-DQB1*06:12', 'HLA-DQA1*04:01-DQB1*06:14',
         'HLA-DQA1*04:01-DQB1*06:15', 'HLA-DQA1*04:01-DQB1*06:16',
         'HLA-DQA1*04:01-DQB1*06:17', 'HLA-DQA1*04:01-DQB1*06:18', 'HLA-DQA1*04:01-DQB1*06:19', 'HLA-DQA1*04:01-DQB1*06:21',
         'HLA-DQA1*04:01-DQB1*06:22', 'HLA-DQA1*04:01-DQB1*06:23',
         'HLA-DQA1*04:01-DQB1*06:24', 'HLA-DQA1*04:01-DQB1*06:25', 'HLA-DQA1*04:01-DQB1*06:27', 'HLA-DQA1*04:01-DQB1*06:28',
         'HLA-DQA1*04:01-DQB1*06:29', 'HLA-DQA1*04:01-DQB1*06:30',
         'HLA-DQA1*04:01-DQB1*06:31', 'HLA-DQA1*04:01-DQB1*06:32', 'HLA-DQA1*04:01-DQB1*06:33', 'HLA-DQA1*04:01-DQB1*06:34',
         'HLA-DQA1*04:01-DQB1*06:35', 'HLA-DQA1*04:01-DQB1*06:36',
         'HLA-DQA1*04:01-DQB1*06:37', 'HLA-DQA1*04:01-DQB1*06:38', 'HLA-DQA1*04:01-DQB1*06:39', 'HLA-DQA1*04:01-DQB1*06:40',
         'HLA-DQA1*04:01-DQB1*06:41', 'HLA-DQA1*04:01-DQB1*06:42',
         'HLA-DQA1*04:01-DQB1*06:43', 'HLA-DQA1*04:01-DQB1*06:44', 'HLA-DQA1*04:02-DQB1*02:01', 'HLA-DQA1*04:02-DQB1*02:02',
         'HLA-DQA1*04:02-DQB1*02:03', 'HLA-DQA1*04:02-DQB1*02:04',
         'HLA-DQA1*04:02-DQB1*02:05', 'HLA-DQA1*04:02-DQB1*02:06', 'HLA-DQA1*04:02-DQB1*03:01', 'HLA-DQA1*04:02-DQB1*03:02',
         'HLA-DQA1*04:02-DQB1*03:03', 'HLA-DQA1*04:02-DQB1*03:04',
         'HLA-DQA1*04:02-DQB1*03:05', 'HLA-DQA1*04:02-DQB1*03:06', 'HLA-DQA1*04:02-DQB1*03:07', 'HLA-DQA1*04:02-DQB1*03:08',
         'HLA-DQA1*04:02-DQB1*03:09', 'HLA-DQA1*04:02-DQB1*03:10',
         'HLA-DQA1*04:02-DQB1*03:11', 'HLA-DQA1*04:02-DQB1*03:12', 'HLA-DQA1*04:02-DQB1*03:13', 'HLA-DQA1*04:02-DQB1*03:14',
         'HLA-DQA1*04:02-DQB1*03:15', 'HLA-DQA1*04:02-DQB1*03:16',
         'HLA-DQA1*04:02-DQB1*03:17', 'HLA-DQA1*04:02-DQB1*03:18', 'HLA-DQA1*04:02-DQB1*03:19', 'HLA-DQA1*04:02-DQB1*03:20',
         'HLA-DQA1*04:02-DQB1*03:21', 'HLA-DQA1*04:02-DQB1*03:22',
         'HLA-DQA1*04:02-DQB1*03:23', 'HLA-DQA1*04:02-DQB1*03:24', 'HLA-DQA1*04:02-DQB1*03:25', 'HLA-DQA1*04:02-DQB1*03:26',
         'HLA-DQA1*04:02-DQB1*03:27', 'HLA-DQA1*04:02-DQB1*03:28',
         'HLA-DQA1*04:02-DQB1*03:29', 'HLA-DQA1*04:02-DQB1*03:30', 'HLA-DQA1*04:02-DQB1*03:31', 'HLA-DQA1*04:02-DQB1*03:32',
         'HLA-DQA1*04:02-DQB1*03:33', 'HLA-DQA1*04:02-DQB1*03:34',
         'HLA-DQA1*04:02-DQB1*03:35', 'HLA-DQA1*04:02-DQB1*03:36', 'HLA-DQA1*04:02-DQB1*03:37', 'HLA-DQA1*04:02-DQB1*03:38',
         'HLA-DQA1*04:02-DQB1*04:01', 'HLA-DQA1*04:02-DQB1*04:02',
         'HLA-DQA1*04:02-DQB1*04:03', 'HLA-DQA1*04:02-DQB1*04:04', 'HLA-DQA1*04:02-DQB1*04:05', 'HLA-DQA1*04:02-DQB1*04:06',
         'HLA-DQA1*04:02-DQB1*04:07', 'HLA-DQA1*04:02-DQB1*04:08',
         'HLA-DQA1*04:02-DQB1*05:01', 'HLA-DQA1*04:02-DQB1*05:02', 'HLA-DQA1*04:02-DQB1*05:03', 'HLA-DQA1*04:02-DQB1*05:05',
         'HLA-DQA1*04:02-DQB1*05:06', 'HLA-DQA1*04:02-DQB1*05:07',
         'HLA-DQA1*04:02-DQB1*05:08', 'HLA-DQA1*04:02-DQB1*05:09', 'HLA-DQA1*04:02-DQB1*05:10', 'HLA-DQA1*04:02-DQB1*05:11',
         'HLA-DQA1*04:02-DQB1*05:12', 'HLA-DQA1*04:02-DQB1*05:13',
         'HLA-DQA1*04:02-DQB1*05:14', 'HLA-DQA1*04:02-DQB1*06:01', 'HLA-DQA1*04:02-DQB1*06:02', 'HLA-DQA1*04:02-DQB1*06:03',
         'HLA-DQA1*04:02-DQB1*06:04', 'HLA-DQA1*04:02-DQB1*06:07',
         'HLA-DQA1*04:02-DQB1*06:08', 'HLA-DQA1*04:02-DQB1*06:09', 'HLA-DQA1*04:02-DQB1*06:10', 'HLA-DQA1*04:02-DQB1*06:11',
         'HLA-DQA1*04:02-DQB1*06:12', 'HLA-DQA1*04:02-DQB1*06:14',
         'HLA-DQA1*04:02-DQB1*06:15', 'HLA-DQA1*04:02-DQB1*06:16', 'HLA-DQA1*04:02-DQB1*06:17', 'HLA-DQA1*04:02-DQB1*06:18',
         'HLA-DQA1*04:02-DQB1*06:19', 'HLA-DQA1*04:02-DQB1*06:21',
         'HLA-DQA1*04:02-DQB1*06:22', 'HLA-DQA1*04:02-DQB1*06:23', 'HLA-DQA1*04:02-DQB1*06:24', 'HLA-DQA1*04:02-DQB1*06:25',
         'HLA-DQA1*04:02-DQB1*06:27', 'HLA-DQA1*04:02-DQB1*06:28',
         'HLA-DQA1*04:02-DQB1*06:29', 'HLA-DQA1*04:02-DQB1*06:30', 'HLA-DQA1*04:02-DQB1*06:31', 'HLA-DQA1*04:02-DQB1*06:32',
         'HLA-DQA1*04:02-DQB1*06:33', 'HLA-DQA1*04:02-DQB1*06:34',
         'HLA-DQA1*04:02-DQB1*06:35', 'HLA-DQA1*04:02-DQB1*06:36', 'HLA-DQA1*04:02-DQB1*06:37', 'HLA-DQA1*04:02-DQB1*06:38',
         'HLA-DQA1*04:02-DQB1*06:39', 'HLA-DQA1*04:02-DQB1*06:40',
         'HLA-DQA1*04:02-DQB1*06:41', 'HLA-DQA1*04:02-DQB1*06:42', 'HLA-DQA1*04:02-DQB1*06:43', 'HLA-DQA1*04:02-DQB1*06:44',
         'HLA-DQA1*04:04-DQB1*02:01', 'HLA-DQA1*04:04-DQB1*02:02',
         'HLA-DQA1*04:04-DQB1*02:03', 'HLA-DQA1*04:04-DQB1*02:04', 'HLA-DQA1*04:04-DQB1*02:05', 'HLA-DQA1*04:04-DQB1*02:06',
         'HLA-DQA1*04:04-DQB1*03:01', 'HLA-DQA1*04:04-DQB1*03:02',
         'HLA-DQA1*04:04-DQB1*03:03', 'HLA-DQA1*04:04-DQB1*03:04', 'HLA-DQA1*04:04-DQB1*03:05', 'HLA-DQA1*04:04-DQB1*03:06',
         'HLA-DQA1*04:04-DQB1*03:07', 'HLA-DQA1*04:04-DQB1*03:08',
         'HLA-DQA1*04:04-DQB1*03:09', 'HLA-DQA1*04:04-DQB1*03:10', 'HLA-DQA1*04:04-DQB1*03:11', 'HLA-DQA1*04:04-DQB1*03:12',
         'HLA-DQA1*04:04-DQB1*03:13', 'HLA-DQA1*04:04-DQB1*03:14',
         'HLA-DQA1*04:04-DQB1*03:15', 'HLA-DQA1*04:04-DQB1*03:16', 'HLA-DQA1*04:04-DQB1*03:17', 'HLA-DQA1*04:04-DQB1*03:18',
         'HLA-DQA1*04:04-DQB1*03:19', 'HLA-DQA1*04:04-DQB1*03:20',
         'HLA-DQA1*04:04-DQB1*03:21', 'HLA-DQA1*04:04-DQB1*03:22', 'HLA-DQA1*04:04-DQB1*03:23', 'HLA-DQA1*04:04-DQB1*03:24',
         'HLA-DQA1*04:04-DQB1*03:25', 'HLA-DQA1*04:04-DQB1*03:26',
         'HLA-DQA1*04:04-DQB1*03:27', 'HLA-DQA1*04:04-DQB1*03:28', 'HLA-DQA1*04:04-DQB1*03:29', 'HLA-DQA1*04:04-DQB1*03:30',
         'HLA-DQA1*04:04-DQB1*03:31', 'HLA-DQA1*04:04-DQB1*03:32',
         'HLA-DQA1*04:04-DQB1*03:33', 'HLA-DQA1*04:04-DQB1*03:34', 'HLA-DQA1*04:04-DQB1*03:35', 'HLA-DQA1*04:04-DQB1*03:36',
         'HLA-DQA1*04:04-DQB1*03:37', 'HLA-DQA1*04:04-DQB1*03:38',
         'HLA-DQA1*04:04-DQB1*04:01', 'HLA-DQA1*04:04-DQB1*04:02', 'HLA-DQA1*04:04-DQB1*04:03', 'HLA-DQA1*04:04-DQB1*04:04',
         'HLA-DQA1*04:04-DQB1*04:05', 'HLA-DQA1*04:04-DQB1*04:06',
         'HLA-DQA1*04:04-DQB1*04:07', 'HLA-DQA1*04:04-DQB1*04:08', 'HLA-DQA1*04:04-DQB1*05:01', 'HLA-DQA1*04:04-DQB1*05:02',
         'HLA-DQA1*04:04-DQB1*05:03', 'HLA-DQA1*04:04-DQB1*05:05',
         'HLA-DQA1*04:04-DQB1*05:06', 'HLA-DQA1*04:04-DQB1*05:07', 'HLA-DQA1*04:04-DQB1*05:08', 'HLA-DQA1*04:04-DQB1*05:09',
         'HLA-DQA1*04:04-DQB1*05:10', 'HLA-DQA1*04:04-DQB1*05:11',
         'HLA-DQA1*04:04-DQB1*05:12', 'HLA-DQA1*04:04-DQB1*05:13', 'HLA-DQA1*04:04-DQB1*05:14', 'HLA-DQA1*04:04-DQB1*06:01',
         'HLA-DQA1*04:04-DQB1*06:02', 'HLA-DQA1*04:04-DQB1*06:03',
         'HLA-DQA1*04:04-DQB1*06:04', 'HLA-DQA1*04:04-DQB1*06:07', 'HLA-DQA1*04:04-DQB1*06:08', 'HLA-DQA1*04:04-DQB1*06:09',
         'HLA-DQA1*04:04-DQB1*06:10', 'HLA-DQA1*04:04-DQB1*06:11',
         'HLA-DQA1*04:04-DQB1*06:12', 'HLA-DQA1*04:04-DQB1*06:14', 'HLA-DQA1*04:04-DQB1*06:15', 'HLA-DQA1*04:04-DQB1*06:16',
         'HLA-DQA1*04:04-DQB1*06:17', 'HLA-DQA1*04:04-DQB1*06:18',
         'HLA-DQA1*04:04-DQB1*06:19', 'HLA-DQA1*04:04-DQB1*06:21', 'HLA-DQA1*04:04-DQB1*06:22', 'HLA-DQA1*04:04-DQB1*06:23',
         'HLA-DQA1*04:04-DQB1*06:24', 'HLA-DQA1*04:04-DQB1*06:25',
         'HLA-DQA1*04:04-DQB1*06:27', 'HLA-DQA1*04:04-DQB1*06:28', 'HLA-DQA1*04:04-DQB1*06:29', 'HLA-DQA1*04:04-DQB1*06:30',
         'HLA-DQA1*04:04-DQB1*06:31', 'HLA-DQA1*04:04-DQB1*06:32',
         'HLA-DQA1*04:04-DQB1*06:33', 'HLA-DQA1*04:04-DQB1*06:34', 'HLA-DQA1*04:04-DQB1*06:35', 'HLA-DQA1*04:04-DQB1*06:36',
         'HLA-DQA1*04:04-DQB1*06:37', 'HLA-DQA1*04:04-DQB1*06:38',
         'HLA-DQA1*04:04-DQB1*06:39', 'HLA-DQA1*04:04-DQB1*06:40', 'HLA-DQA1*04:04-DQB1*06:41', 'HLA-DQA1*04:04-DQB1*06:42',
         'HLA-DQA1*04:04-DQB1*06:43', 'HLA-DQA1*04:04-DQB1*06:44',
         'HLA-DQA1*05:01-DQB1*02:01', 'HLA-DQA1*05:01-DQB1*02:02', 'HLA-DQA1*05:01-DQB1*02:03', 'HLA-DQA1*05:01-DQB1*02:04',
         'HLA-DQA1*05:01-DQB1*02:05', 'HLA-DQA1*05:01-DQB1*02:06',
         'HLA-DQA1*05:01-DQB1*03:01', 'HLA-DQA1*05:01-DQB1*03:02', 'HLA-DQA1*05:01-DQB1*03:03', 'HLA-DQA1*05:01-DQB1*03:04',
         'HLA-DQA1*05:01-DQB1*03:05', 'HLA-DQA1*05:01-DQB1*03:06',
         'HLA-DQA1*05:01-DQB1*03:07', 'HLA-DQA1*05:01-DQB1*03:08', 'HLA-DQA1*05:01-DQB1*03:09', 'HLA-DQA1*05:01-DQB1*03:10',
         'HLA-DQA1*05:01-DQB1*03:11', 'HLA-DQA1*05:01-DQB1*03:12',
         'HLA-DQA1*05:01-DQB1*03:13', 'HLA-DQA1*05:01-DQB1*03:14', 'HLA-DQA1*05:01-DQB1*03:15', 'HLA-DQA1*05:01-DQB1*03:16',
         'HLA-DQA1*05:01-DQB1*03:17', 'HLA-DQA1*05:01-DQB1*03:18',
         'HLA-DQA1*05:01-DQB1*03:19', 'HLA-DQA1*05:01-DQB1*03:20', 'HLA-DQA1*05:01-DQB1*03:21', 'HLA-DQA1*05:01-DQB1*03:22',
         'HLA-DQA1*05:01-DQB1*03:23', 'HLA-DQA1*05:01-DQB1*03:24',
         'HLA-DQA1*05:01-DQB1*03:25', 'HLA-DQA1*05:01-DQB1*03:26', 'HLA-DQA1*05:01-DQB1*03:27', 'HLA-DQA1*05:01-DQB1*03:28',
         'HLA-DQA1*05:01-DQB1*03:29', 'HLA-DQA1*05:01-DQB1*03:30',
         'HLA-DQA1*05:01-DQB1*03:31', 'HLA-DQA1*05:01-DQB1*03:32', 'HLA-DQA1*05:01-DQB1*03:33', 'HLA-DQA1*05:01-DQB1*03:34',
         'HLA-DQA1*05:01-DQB1*03:35', 'HLA-DQA1*05:01-DQB1*03:36',
         'HLA-DQA1*05:01-DQB1*03:37', 'HLA-DQA1*05:01-DQB1*03:38', 'HLA-DQA1*05:01-DQB1*04:01', 'HLA-DQA1*05:01-DQB1*04:02',
         'HLA-DQA1*05:01-DQB1*04:03', 'HLA-DQA1*05:01-DQB1*04:04',
         'HLA-DQA1*05:01-DQB1*04:05', 'HLA-DQA1*05:01-DQB1*04:06', 'HLA-DQA1*05:01-DQB1*04:07', 'HLA-DQA1*05:01-DQB1*04:08',
         'HLA-DQA1*05:01-DQB1*05:01', 'HLA-DQA1*05:01-DQB1*05:02',
         'HLA-DQA1*05:01-DQB1*05:03', 'HLA-DQA1*05:01-DQB1*05:05', 'HLA-DQA1*05:01-DQB1*05:06', 'HLA-DQA1*05:01-DQB1*05:07',
         'HLA-DQA1*05:01-DQB1*05:08', 'HLA-DQA1*05:01-DQB1*05:09',
         'HLA-DQA1*05:01-DQB1*05:10', 'HLA-DQA1*05:01-DQB1*05:11', 'HLA-DQA1*05:01-DQB1*05:12', 'HLA-DQA1*05:01-DQB1*05:13',
         'HLA-DQA1*05:01-DQB1*05:14', 'HLA-DQA1*05:01-DQB1*06:01',
         'HLA-DQA1*05:01-DQB1*06:02', 'HLA-DQA1*05:01-DQB1*06:03', 'HLA-DQA1*05:01-DQB1*06:04', 'HLA-DQA1*05:01-DQB1*06:07',
         'HLA-DQA1*05:01-DQB1*06:08', 'HLA-DQA1*05:01-DQB1*06:09',
         'HLA-DQA1*05:01-DQB1*06:10', 'HLA-DQA1*05:01-DQB1*06:11', 'HLA-DQA1*05:01-DQB1*06:12', 'HLA-DQA1*05:01-DQB1*06:14',
         'HLA-DQA1*05:01-DQB1*06:15', 'HLA-DQA1*05:01-DQB1*06:16',
         'HLA-DQA1*05:01-DQB1*06:17', 'HLA-DQA1*05:01-DQB1*06:18', 'HLA-DQA1*05:01-DQB1*06:19', 'HLA-DQA1*05:01-DQB1*06:21',
         'HLA-DQA1*05:01-DQB1*06:22', 'HLA-DQA1*05:01-DQB1*06:23',
         'HLA-DQA1*05:01-DQB1*06:24', 'HLA-DQA1*05:01-DQB1*06:25', 'HLA-DQA1*05:01-DQB1*06:27', 'HLA-DQA1*05:01-DQB1*06:28',
         'HLA-DQA1*05:01-DQB1*06:29', 'HLA-DQA1*05:01-DQB1*06:30',
         'HLA-DQA1*05:01-DQB1*06:31', 'HLA-DQA1*05:01-DQB1*06:32', 'HLA-DQA1*05:01-DQB1*06:33', 'HLA-DQA1*05:01-DQB1*06:34',
         'HLA-DQA1*05:01-DQB1*06:35', 'HLA-DQA1*05:01-DQB1*06:36',
         'HLA-DQA1*05:01-DQB1*06:37', 'HLA-DQA1*05:01-DQB1*06:38', 'HLA-DQA1*05:01-DQB1*06:39', 'HLA-DQA1*05:01-DQB1*06:40',
         'HLA-DQA1*05:01-DQB1*06:41', 'HLA-DQA1*05:01-DQB1*06:42',
         'HLA-DQA1*05:01-DQB1*06:43', 'HLA-DQA1*05:01-DQB1*06:44', 'HLA-DQA1*05:03-DQB1*02:01', 'HLA-DQA1*05:03-DQB1*02:02',
         'HLA-DQA1*05:03-DQB1*02:03', 'HLA-DQA1*05:03-DQB1*02:04',
         'HLA-DQA1*05:03-DQB1*02:05', 'HLA-DQA1*05:03-DQB1*02:06', 'HLA-DQA1*05:03-DQB1*03:01', 'HLA-DQA1*05:03-DQB1*03:02',
         'HLA-DQA1*05:03-DQB1*03:03', 'HLA-DQA1*05:03-DQB1*03:04',
         'HLA-DQA1*05:03-DQB1*03:05', 'HLA-DQA1*05:03-DQB1*03:06', 'HLA-DQA1*05:03-DQB1*03:07', 'HLA-DQA1*05:03-DQB1*03:08',
         'HLA-DQA1*05:03-DQB1*03:09', 'HLA-DQA1*05:03-DQB1*03:10',
         'HLA-DQA1*05:03-DQB1*03:11', 'HLA-DQA1*05:03-DQB1*03:12', 'HLA-DQA1*05:03-DQB1*03:13', 'HLA-DQA1*05:03-DQB1*03:14',
         'HLA-DQA1*05:03-DQB1*03:15', 'HLA-DQA1*05:03-DQB1*03:16',
         'HLA-DQA1*05:03-DQB1*03:17', 'HLA-DQA1*05:03-DQB1*03:18', 'HLA-DQA1*05:03-DQB1*03:19', 'HLA-DQA1*05:03-DQB1*03:20',
         'HLA-DQA1*05:03-DQB1*03:21', 'HLA-DQA1*05:03-DQB1*03:22',
         'HLA-DQA1*05:03-DQB1*03:23', 'HLA-DQA1*05:03-DQB1*03:24', 'HLA-DQA1*05:03-DQB1*03:25', 'HLA-DQA1*05:03-DQB1*03:26',
         'HLA-DQA1*05:03-DQB1*03:27', 'HLA-DQA1*05:03-DQB1*03:28',
         'HLA-DQA1*05:03-DQB1*03:29', 'HLA-DQA1*05:03-DQB1*03:30', 'HLA-DQA1*05:03-DQB1*03:31', 'HLA-DQA1*05:03-DQB1*03:32',
         'HLA-DQA1*05:03-DQB1*03:33', 'HLA-DQA1*05:03-DQB1*03:34',
         'HLA-DQA1*05:03-DQB1*03:35', 'HLA-DQA1*05:03-DQB1*03:36', 'HLA-DQA1*05:03-DQB1*03:37', 'HLA-DQA1*05:03-DQB1*03:38',
         'HLA-DQA1*05:03-DQB1*04:01', 'HLA-DQA1*05:03-DQB1*04:02',
         'HLA-DQA1*05:03-DQB1*04:03', 'HLA-DQA1*05:03-DQB1*04:04', 'HLA-DQA1*05:03-DQB1*04:05', 'HLA-DQA1*05:03-DQB1*04:06',
         'HLA-DQA1*05:03-DQB1*04:07', 'HLA-DQA1*05:03-DQB1*04:08',
         'HLA-DQA1*05:03-DQB1*05:01', 'HLA-DQA1*05:03-DQB1*05:02', 'HLA-DQA1*05:03-DQB1*05:03', 'HLA-DQA1*05:03-DQB1*05:05',
         'HLA-DQA1*05:03-DQB1*05:06', 'HLA-DQA1*05:03-DQB1*05:07',
         'HLA-DQA1*05:03-DQB1*05:08', 'HLA-DQA1*05:03-DQB1*05:09', 'HLA-DQA1*05:03-DQB1*05:10', 'HLA-DQA1*05:03-DQB1*05:11',
         'HLA-DQA1*05:03-DQB1*05:12', 'HLA-DQA1*05:03-DQB1*05:13',
         'HLA-DQA1*05:03-DQB1*05:14', 'HLA-DQA1*05:03-DQB1*06:01', 'HLA-DQA1*05:03-DQB1*06:02', 'HLA-DQA1*05:03-DQB1*06:03',
         'HLA-DQA1*05:03-DQB1*06:04', 'HLA-DQA1*05:03-DQB1*06:07',
         'HLA-DQA1*05:03-DQB1*06:08', 'HLA-DQA1*05:03-DQB1*06:09', 'HLA-DQA1*05:03-DQB1*06:10', 'HLA-DQA1*05:03-DQB1*06:11',
         'HLA-DQA1*05:03-DQB1*06:12', 'HLA-DQA1*05:03-DQB1*06:14',
         'HLA-DQA1*05:03-DQB1*06:15', 'HLA-DQA1*05:03-DQB1*06:16', 'HLA-DQA1*05:03-DQB1*06:17', 'HLA-DQA1*05:03-DQB1*06:18',
         'HLA-DQA1*05:03-DQB1*06:19', 'HLA-DQA1*05:03-DQB1*06:21',
         'HLA-DQA1*05:03-DQB1*06:22', 'HLA-DQA1*05:03-DQB1*06:23', 'HLA-DQA1*05:03-DQB1*06:24', 'HLA-DQA1*05:03-DQB1*06:25',
         'HLA-DQA1*05:03-DQB1*06:27', 'HLA-DQA1*05:03-DQB1*06:28',
         'HLA-DQA1*05:03-DQB1*06:29', 'HLA-DQA1*05:03-DQB1*06:30', 'HLA-DQA1*05:03-DQB1*06:31', 'HLA-DQA1*05:03-DQB1*06:32',
         'HLA-DQA1*05:03-DQB1*06:33', 'HLA-DQA1*05:03-DQB1*06:34',
         'HLA-DQA1*05:03-DQB1*06:35', 'HLA-DQA1*05:03-DQB1*06:36', 'HLA-DQA1*05:03-DQB1*06:37', 'HLA-DQA1*05:03-DQB1*06:38',
         'HLA-DQA1*05:03-DQB1*06:39', 'HLA-DQA1*05:03-DQB1*06:40',
         'HLA-DQA1*05:03-DQB1*06:41', 'HLA-DQA1*05:03-DQB1*06:42', 'HLA-DQA1*05:03-DQB1*06:43', 'HLA-DQA1*05:03-DQB1*06:44',
         'HLA-DQA1*05:04-DQB1*02:01', 'HLA-DQA1*05:04-DQB1*02:02',
         'HLA-DQA1*05:04-DQB1*02:03', 'HLA-DQA1*05:04-DQB1*02:04', 'HLA-DQA1*05:04-DQB1*02:05', 'HLA-DQA1*05:04-DQB1*02:06',
         'HLA-DQA1*05:04-DQB1*03:01', 'HLA-DQA1*05:04-DQB1*03:02',
         'HLA-DQA1*05:04-DQB1*03:03', 'HLA-DQA1*05:04-DQB1*03:04', 'HLA-DQA1*05:04-DQB1*03:05', 'HLA-DQA1*05:04-DQB1*03:06',
         'HLA-DQA1*05:04-DQB1*03:07', 'HLA-DQA1*05:04-DQB1*03:08',
         'HLA-DQA1*05:04-DQB1*03:09', 'HLA-DQA1*05:04-DQB1*03:10', 'HLA-DQA1*05:04-DQB1*03:11', 'HLA-DQA1*05:04-DQB1*03:12',
         'HLA-DQA1*05:04-DQB1*03:13', 'HLA-DQA1*05:04-DQB1*03:14',
         'HLA-DQA1*05:04-DQB1*03:15', 'HLA-DQA1*05:04-DQB1*03:16', 'HLA-DQA1*05:04-DQB1*03:17', 'HLA-DQA1*05:04-DQB1*03:18',
         'HLA-DQA1*05:04-DQB1*03:19', 'HLA-DQA1*05:04-DQB1*03:20',
         'HLA-DQA1*05:04-DQB1*03:21', 'HLA-DQA1*05:04-DQB1*03:22', 'HLA-DQA1*05:04-DQB1*03:23', 'HLA-DQA1*05:04-DQB1*03:24',
         'HLA-DQA1*05:04-DQB1*03:25', 'HLA-DQA1*05:04-DQB1*03:26',
         'HLA-DQA1*05:04-DQB1*03:27', 'HLA-DQA1*05:04-DQB1*03:28', 'HLA-DQA1*05:04-DQB1*03:29', 'HLA-DQA1*05:04-DQB1*03:30',
         'HLA-DQA1*05:04-DQB1*03:31', 'HLA-DQA1*05:04-DQB1*03:32',
         'HLA-DQA1*05:04-DQB1*03:33', 'HLA-DQA1*05:04-DQB1*03:34', 'HLA-DQA1*05:04-DQB1*03:35', 'HLA-DQA1*05:04-DQB1*03:36',
         'HLA-DQA1*05:04-DQB1*03:37', 'HLA-DQA1*05:04-DQB1*03:38',
         'HLA-DQA1*05:04-DQB1*04:01', 'HLA-DQA1*05:04-DQB1*04:02', 'HLA-DQA1*05:04-DQB1*04:03', 'HLA-DQA1*05:04-DQB1*04:04',
         'HLA-DQA1*05:04-DQB1*04:05', 'HLA-DQA1*05:04-DQB1*04:06',
         'HLA-DQA1*05:04-DQB1*04:07', 'HLA-DQA1*05:04-DQB1*04:08', 'HLA-DQA1*05:04-DQB1*05:01', 'HLA-DQA1*05:04-DQB1*05:02',
         'HLA-DQA1*05:04-DQB1*05:03', 'HLA-DQA1*05:04-DQB1*05:05',
         'HLA-DQA1*05:04-DQB1*05:06', 'HLA-DQA1*05:04-DQB1*05:07', 'HLA-DQA1*05:04-DQB1*05:08', 'HLA-DQA1*05:04-DQB1*05:09',
         'HLA-DQA1*05:04-DQB1*05:10', 'HLA-DQA1*05:04-DQB1*05:11',
         'HLA-DQA1*05:04-DQB1*05:12', 'HLA-DQA1*05:04-DQB1*05:13', 'HLA-DQA1*05:04-DQB1*05:14', 'HLA-DQA1*05:04-DQB1*06:01',
         'HLA-DQA1*05:04-DQB1*06:02', 'HLA-DQA1*05:04-DQB1*06:03',
         'HLA-DQA1*05:04-DQB1*06:04', 'HLA-DQA1*05:04-DQB1*06:07', 'HLA-DQA1*05:04-DQB1*06:08', 'HLA-DQA1*05:04-DQB1*06:09',
         'HLA-DQA1*05:04-DQB1*06:10', 'HLA-DQA1*05:04-DQB1*06:11',
         'HLA-DQA1*05:04-DQB1*06:12', 'HLA-DQA1*05:04-DQB1*06:14', 'HLA-DQA1*05:04-DQB1*06:15', 'HLA-DQA1*05:04-DQB1*06:16',
         'HLA-DQA1*05:04-DQB1*06:17', 'HLA-DQA1*05:04-DQB1*06:18',
         'HLA-DQA1*05:04-DQB1*06:19', 'HLA-DQA1*05:04-DQB1*06:21', 'HLA-DQA1*05:04-DQB1*06:22', 'HLA-DQA1*05:04-DQB1*06:23',
         'HLA-DQA1*05:04-DQB1*06:24', 'HLA-DQA1*05:04-DQB1*06:25',
         'HLA-DQA1*05:04-DQB1*06:27', 'HLA-DQA1*05:04-DQB1*06:28', 'HLA-DQA1*05:04-DQB1*06:29', 'HLA-DQA1*05:04-DQB1*06:30',
         'HLA-DQA1*05:04-DQB1*06:31', 'HLA-DQA1*05:04-DQB1*06:32',
         'HLA-DQA1*05:04-DQB1*06:33', 'HLA-DQA1*05:04-DQB1*06:34', 'HLA-DQA1*05:04-DQB1*06:35', 'HLA-DQA1*05:04-DQB1*06:36',
         'HLA-DQA1*05:04-DQB1*06:37', 'HLA-DQA1*05:04-DQB1*06:38',
         'HLA-DQA1*05:04-DQB1*06:39', 'HLA-DQA1*05:04-DQB1*06:40', 'HLA-DQA1*05:04-DQB1*06:41', 'HLA-DQA1*05:04-DQB1*06:42',
         'HLA-DQA1*05:04-DQB1*06:43', 'HLA-DQA1*05:04-DQB1*06:44',
         'HLA-DQA1*05:05-DQB1*02:01', 'HLA-DQA1*05:05-DQB1*02:02', 'HLA-DQA1*05:05-DQB1*02:03', 'HLA-DQA1*05:05-DQB1*02:04',
         'HLA-DQA1*05:05-DQB1*02:05', 'HLA-DQA1*05:05-DQB1*02:06',
         'HLA-DQA1*05:05-DQB1*03:01', 'HLA-DQA1*05:05-DQB1*03:02', 'HLA-DQA1*05:05-DQB1*03:03', 'HLA-DQA1*05:05-DQB1*03:04',
         'HLA-DQA1*05:05-DQB1*03:05', 'HLA-DQA1*05:05-DQB1*03:06',
         'HLA-DQA1*05:05-DQB1*03:07', 'HLA-DQA1*05:05-DQB1*03:08', 'HLA-DQA1*05:05-DQB1*03:09', 'HLA-DQA1*05:05-DQB1*03:10',
         'HLA-DQA1*05:05-DQB1*03:11', 'HLA-DQA1*05:05-DQB1*03:12',
         'HLA-DQA1*05:05-DQB1*03:13', 'HLA-DQA1*05:05-DQB1*03:14', 'HLA-DQA1*05:05-DQB1*03:15', 'HLA-DQA1*05:05-DQB1*03:16',
         'HLA-DQA1*05:05-DQB1*03:17', 'HLA-DQA1*05:05-DQB1*03:18',
         'HLA-DQA1*05:05-DQB1*03:19', 'HLA-DQA1*05:05-DQB1*03:20', 'HLA-DQA1*05:05-DQB1*03:21', 'HLA-DQA1*05:05-DQB1*03:22',
         'HLA-DQA1*05:05-DQB1*03:23', 'HLA-DQA1*05:05-DQB1*03:24',
         'HLA-DQA1*05:05-DQB1*03:25', 'HLA-DQA1*05:05-DQB1*03:26', 'HLA-DQA1*05:05-DQB1*03:27', 'HLA-DQA1*05:05-DQB1*03:28',
         'HLA-DQA1*05:05-DQB1*03:29', 'HLA-DQA1*05:05-DQB1*03:30',
         'HLA-DQA1*05:05-DQB1*03:31', 'HLA-DQA1*05:05-DQB1*03:32', 'HLA-DQA1*05:05-DQB1*03:33', 'HLA-DQA1*05:05-DQB1*03:34',
         'HLA-DQA1*05:05-DQB1*03:35', 'HLA-DQA1*05:05-DQB1*03:36',
         'HLA-DQA1*05:05-DQB1*03:37', 'HLA-DQA1*05:05-DQB1*03:38', 'HLA-DQA1*05:05-DQB1*04:01', 'HLA-DQA1*05:05-DQB1*04:02',
         'HLA-DQA1*05:05-DQB1*04:03', 'HLA-DQA1*05:05-DQB1*04:04',
         'HLA-DQA1*05:05-DQB1*04:05', 'HLA-DQA1*05:05-DQB1*04:06', 'HLA-DQA1*05:05-DQB1*04:07', 'HLA-DQA1*05:05-DQB1*04:08',
         'HLA-DQA1*05:05-DQB1*05:01', 'HLA-DQA1*05:05-DQB1*05:02',
         'HLA-DQA1*05:05-DQB1*05:03', 'HLA-DQA1*05:05-DQB1*05:05', 'HLA-DQA1*05:05-DQB1*05:06', 'HLA-DQA1*05:05-DQB1*05:07',
         'HLA-DQA1*05:05-DQB1*05:08', 'HLA-DQA1*05:05-DQB1*05:09',
         'HLA-DQA1*05:05-DQB1*05:10', 'HLA-DQA1*05:05-DQB1*05:11', 'HLA-DQA1*05:05-DQB1*05:12', 'HLA-DQA1*05:05-DQB1*05:13',
         'HLA-DQA1*05:05-DQB1*05:14', 'HLA-DQA1*05:05-DQB1*06:01',
         'HLA-DQA1*05:05-DQB1*06:02', 'HLA-DQA1*05:05-DQB1*06:03', 'HLA-DQA1*05:05-DQB1*06:04', 'HLA-DQA1*05:05-DQB1*06:07',
         'HLA-DQA1*05:05-DQB1*06:08', 'HLA-DQA1*05:05-DQB1*06:09',
         'HLA-DQA1*05:05-DQB1*06:10', 'HLA-DQA1*05:05-DQB1*06:11', 'HLA-DQA1*05:05-DQB1*06:12', 'HLA-DQA1*05:05-DQB1*06:14',
         'HLA-DQA1*05:05-DQB1*06:15', 'HLA-DQA1*05:05-DQB1*06:16',
         'HLA-DQA1*05:05-DQB1*06:17', 'HLA-DQA1*05:05-DQB1*06:18', 'HLA-DQA1*05:05-DQB1*06:19', 'HLA-DQA1*05:05-DQB1*06:21',
         'HLA-DQA1*05:05-DQB1*06:22', 'HLA-DQA1*05:05-DQB1*06:23',
         'HLA-DQA1*05:05-DQB1*06:24', 'HLA-DQA1*05:05-DQB1*06:25', 'HLA-DQA1*05:05-DQB1*06:27', 'HLA-DQA1*05:05-DQB1*06:28',
         'HLA-DQA1*05:05-DQB1*06:29', 'HLA-DQA1*05:05-DQB1*06:30',
         'HLA-DQA1*05:05-DQB1*06:31', 'HLA-DQA1*05:05-DQB1*06:32', 'HLA-DQA1*05:05-DQB1*06:33', 'HLA-DQA1*05:05-DQB1*06:34',
         'HLA-DQA1*05:05-DQB1*06:35', 'HLA-DQA1*05:05-DQB1*06:36',
         'HLA-DQA1*05:05-DQB1*06:37', 'HLA-DQA1*05:05-DQB1*06:38', 'HLA-DQA1*05:05-DQB1*06:39', 'HLA-DQA1*05:05-DQB1*06:40',
         'HLA-DQA1*05:05-DQB1*06:41', 'HLA-DQA1*05:05-DQB1*06:42',
         'HLA-DQA1*05:05-DQB1*06:43', 'HLA-DQA1*05:05-DQB1*06:44', 'HLA-DQA1*05:06-DQB1*02:01', 'HLA-DQA1*05:06-DQB1*02:02',
         'HLA-DQA1*05:06-DQB1*02:03', 'HLA-DQA1*05:06-DQB1*02:04',
         'HLA-DQA1*05:06-DQB1*02:05', 'HLA-DQA1*05:06-DQB1*02:06', 'HLA-DQA1*05:06-DQB1*03:01', 'HLA-DQA1*05:06-DQB1*03:02',
         'HLA-DQA1*05:06-DQB1*03:03', 'HLA-DQA1*05:06-DQB1*03:04',
         'HLA-DQA1*05:06-DQB1*03:05', 'HLA-DQA1*05:06-DQB1*03:06', 'HLA-DQA1*05:06-DQB1*03:07', 'HLA-DQA1*05:06-DQB1*03:08',
         'HLA-DQA1*05:06-DQB1*03:09', 'HLA-DQA1*05:06-DQB1*03:10',
         'HLA-DQA1*05:06-DQB1*03:11', 'HLA-DQA1*05:06-DQB1*03:12', 'HLA-DQA1*05:06-DQB1*03:13', 'HLA-DQA1*05:06-DQB1*03:14',
         'HLA-DQA1*05:06-DQB1*03:15', 'HLA-DQA1*05:06-DQB1*03:16',
         'HLA-DQA1*05:06-DQB1*03:17', 'HLA-DQA1*05:06-DQB1*03:18', 'HLA-DQA1*05:06-DQB1*03:19', 'HLA-DQA1*05:06-DQB1*03:20',
         'HLA-DQA1*05:06-DQB1*03:21', 'HLA-DQA1*05:06-DQB1*03:22',
         'HLA-DQA1*05:06-DQB1*03:23', 'HLA-DQA1*05:06-DQB1*03:24', 'HLA-DQA1*05:06-DQB1*03:25', 'HLA-DQA1*05:06-DQB1*03:26',
         'HLA-DQA1*05:06-DQB1*03:27', 'HLA-DQA1*05:06-DQB1*03:28',
         'HLA-DQA1*05:06-DQB1*03:29', 'HLA-DQA1*05:06-DQB1*03:30', 'HLA-DQA1*05:06-DQB1*03:31', 'HLA-DQA1*05:06-DQB1*03:32',
         'HLA-DQA1*05:06-DQB1*03:33', 'HLA-DQA1*05:06-DQB1*03:34',
         'HLA-DQA1*05:06-DQB1*03:35', 'HLA-DQA1*05:06-DQB1*03:36', 'HLA-DQA1*05:06-DQB1*03:37', 'HLA-DQA1*05:06-DQB1*03:38',
         'HLA-DQA1*05:06-DQB1*04:01', 'HLA-DQA1*05:06-DQB1*04:02',
         'HLA-DQA1*05:06-DQB1*04:03', 'HLA-DQA1*05:06-DQB1*04:04', 'HLA-DQA1*05:06-DQB1*04:05', 'HLA-DQA1*05:06-DQB1*04:06',
         'HLA-DQA1*05:06-DQB1*04:07', 'HLA-DQA1*05:06-DQB1*04:08',
         'HLA-DQA1*05:06-DQB1*05:01', 'HLA-DQA1*05:06-DQB1*05:02', 'HLA-DQA1*05:06-DQB1*05:03', 'HLA-DQA1*05:06-DQB1*05:05',
         'HLA-DQA1*05:06-DQB1*05:06', 'HLA-DQA1*05:06-DQB1*05:07',
         'HLA-DQA1*05:06-DQB1*05:08', 'HLA-DQA1*05:06-DQB1*05:09', 'HLA-DQA1*05:06-DQB1*05:10', 'HLA-DQA1*05:06-DQB1*05:11',
         'HLA-DQA1*05:06-DQB1*05:12', 'HLA-DQA1*05:06-DQB1*05:13',
         'HLA-DQA1*05:06-DQB1*05:14', 'HLA-DQA1*05:06-DQB1*06:01', 'HLA-DQA1*05:06-DQB1*06:02', 'HLA-DQA1*05:06-DQB1*06:03',
         'HLA-DQA1*05:06-DQB1*06:04', 'HLA-DQA1*05:06-DQB1*06:07',
         'HLA-DQA1*05:06-DQB1*06:08', 'HLA-DQA1*05:06-DQB1*06:09', 'HLA-DQA1*05:06-DQB1*06:10', 'HLA-DQA1*05:06-DQB1*06:11',
         'HLA-DQA1*05:06-DQB1*06:12', 'HLA-DQA1*05:06-DQB1*06:14',
         'HLA-DQA1*05:06-DQB1*06:15', 'HLA-DQA1*05:06-DQB1*06:16', 'HLA-DQA1*05:06-DQB1*06:17', 'HLA-DQA1*05:06-DQB1*06:18',
         'HLA-DQA1*05:06-DQB1*06:19', 'HLA-DQA1*05:06-DQB1*06:21',
         'HLA-DQA1*05:06-DQB1*06:22', 'HLA-DQA1*05:06-DQB1*06:23', 'HLA-DQA1*05:06-DQB1*06:24', 'HLA-DQA1*05:06-DQB1*06:25',
         'HLA-DQA1*05:06-DQB1*06:27', 'HLA-DQA1*05:06-DQB1*06:28',
         'HLA-DQA1*05:06-DQB1*06:29', 'HLA-DQA1*05:06-DQB1*06:30', 'HLA-DQA1*05:06-DQB1*06:31', 'HLA-DQA1*05:06-DQB1*06:32',
         'HLA-DQA1*05:06-DQB1*06:33', 'HLA-DQA1*05:06-DQB1*06:34',
         'HLA-DQA1*05:06-DQB1*06:35', 'HLA-DQA1*05:06-DQB1*06:36', 'HLA-DQA1*05:06-DQB1*06:37', 'HLA-DQA1*05:06-DQB1*06:38',
         'HLA-DQA1*05:06-DQB1*06:39', 'HLA-DQA1*05:06-DQB1*06:40',
         'HLA-DQA1*05:06-DQB1*06:41', 'HLA-DQA1*05:06-DQB1*06:42', 'HLA-DQA1*05:06-DQB1*06:43', 'HLA-DQA1*05:06-DQB1*06:44',
         'HLA-DQA1*05:07-DQB1*02:01', 'HLA-DQA1*05:07-DQB1*02:02',
         'HLA-DQA1*05:07-DQB1*02:03', 'HLA-DQA1*05:07-DQB1*02:04', 'HLA-DQA1*05:07-DQB1*02:05', 'HLA-DQA1*05:07-DQB1*02:06',
         'HLA-DQA1*05:07-DQB1*03:01', 'HLA-DQA1*05:07-DQB1*03:02',
         'HLA-DQA1*05:07-DQB1*03:03', 'HLA-DQA1*05:07-DQB1*03:04', 'HLA-DQA1*05:07-DQB1*03:05', 'HLA-DQA1*05:07-DQB1*03:06',
         'HLA-DQA1*05:07-DQB1*03:07', 'HLA-DQA1*05:07-DQB1*03:08',
         'HLA-DQA1*05:07-DQB1*03:09', 'HLA-DQA1*05:07-DQB1*03:10', 'HLA-DQA1*05:07-DQB1*03:11', 'HLA-DQA1*05:07-DQB1*03:12',
         'HLA-DQA1*05:07-DQB1*03:13', 'HLA-DQA1*05:07-DQB1*03:14',
         'HLA-DQA1*05:07-DQB1*03:15', 'HLA-DQA1*05:07-DQB1*03:16', 'HLA-DQA1*05:07-DQB1*03:17', 'HLA-DQA1*05:07-DQB1*03:18',
         'HLA-DQA1*05:07-DQB1*03:19', 'HLA-DQA1*05:07-DQB1*03:20',
         'HLA-DQA1*05:07-DQB1*03:21', 'HLA-DQA1*05:07-DQB1*03:22', 'HLA-DQA1*05:07-DQB1*03:23', 'HLA-DQA1*05:07-DQB1*03:24',
         'HLA-DQA1*05:07-DQB1*03:25', 'HLA-DQA1*05:07-DQB1*03:26',
         'HLA-DQA1*05:07-DQB1*03:27', 'HLA-DQA1*05:07-DQB1*03:28', 'HLA-DQA1*05:07-DQB1*03:29', 'HLA-DQA1*05:07-DQB1*03:30',
         'HLA-DQA1*05:07-DQB1*03:31', 'HLA-DQA1*05:07-DQB1*03:32',
         'HLA-DQA1*05:07-DQB1*03:33', 'HLA-DQA1*05:07-DQB1*03:34', 'HLA-DQA1*05:07-DQB1*03:35', 'HLA-DQA1*05:07-DQB1*03:36',
         'HLA-DQA1*05:07-DQB1*03:37', 'HLA-DQA1*05:07-DQB1*03:38',
         'HLA-DQA1*05:07-DQB1*04:01', 'HLA-DQA1*05:07-DQB1*04:02', 'HLA-DQA1*05:07-DQB1*04:03', 'HLA-DQA1*05:07-DQB1*04:04',
         'HLA-DQA1*05:07-DQB1*04:05', 'HLA-DQA1*05:07-DQB1*04:06',
         'HLA-DQA1*05:07-DQB1*04:07', 'HLA-DQA1*05:07-DQB1*04:08', 'HLA-DQA1*05:07-DQB1*05:01', 'HLA-DQA1*05:07-DQB1*05:02',
         'HLA-DQA1*05:07-DQB1*05:03', 'HLA-DQA1*05:07-DQB1*05:05',
         'HLA-DQA1*05:07-DQB1*05:06', 'HLA-DQA1*05:07-DQB1*05:07', 'HLA-DQA1*05:07-DQB1*05:08', 'HLA-DQA1*05:07-DQB1*05:09',
         'HLA-DQA1*05:07-DQB1*05:10', 'HLA-DQA1*05:07-DQB1*05:11',
         'HLA-DQA1*05:07-DQB1*05:12', 'HLA-DQA1*05:07-DQB1*05:13', 'HLA-DQA1*05:07-DQB1*05:14', 'HLA-DQA1*05:07-DQB1*06:01',
         'HLA-DQA1*05:07-DQB1*06:02', 'HLA-DQA1*05:07-DQB1*06:03',
         'HLA-DQA1*05:07-DQB1*06:04', 'HLA-DQA1*05:07-DQB1*06:07', 'HLA-DQA1*05:07-DQB1*06:08', 'HLA-DQA1*05:07-DQB1*06:09',
         'HLA-DQA1*05:07-DQB1*06:10', 'HLA-DQA1*05:07-DQB1*06:11',
         'HLA-DQA1*05:07-DQB1*06:12', 'HLA-DQA1*05:07-DQB1*06:14', 'HLA-DQA1*05:07-DQB1*06:15', 'HLA-DQA1*05:07-DQB1*06:16',
         'HLA-DQA1*05:07-DQB1*06:17', 'HLA-DQA1*05:07-DQB1*06:18',
         'HLA-DQA1*05:07-DQB1*06:19', 'HLA-DQA1*05:07-DQB1*06:21', 'HLA-DQA1*05:07-DQB1*06:22', 'HLA-DQA1*05:07-DQB1*06:23',
         'HLA-DQA1*05:07-DQB1*06:24', 'HLA-DQA1*05:07-DQB1*06:25',
         'HLA-DQA1*05:07-DQB1*06:27', 'HLA-DQA1*05:07-DQB1*06:28', 'HLA-DQA1*05:07-DQB1*06:29', 'HLA-DQA1*05:07-DQB1*06:30',
         'HLA-DQA1*05:07-DQB1*06:31', 'HLA-DQA1*05:07-DQB1*06:32',
         'HLA-DQA1*05:07-DQB1*06:33', 'HLA-DQA1*05:07-DQB1*06:34', 'HLA-DQA1*05:07-DQB1*06:35', 'HLA-DQA1*05:07-DQB1*06:36',
         'HLA-DQA1*05:07-DQB1*06:37', 'HLA-DQA1*05:07-DQB1*06:38',
         'HLA-DQA1*05:07-DQB1*06:39', 'HLA-DQA1*05:07-DQB1*06:40', 'HLA-DQA1*05:07-DQB1*06:41', 'HLA-DQA1*05:07-DQB1*06:42',
         'HLA-DQA1*05:07-DQB1*06:43', 'HLA-DQA1*05:07-DQB1*06:44',
         'HLA-DQA1*05:08-DQB1*02:01', 'HLA-DQA1*05:08-DQB1*02:02', 'HLA-DQA1*05:08-DQB1*02:03', 'HLA-DQA1*05:08-DQB1*02:04',
         'HLA-DQA1*05:08-DQB1*02:05', 'HLA-DQA1*05:08-DQB1*02:06',
         'HLA-DQA1*05:08-DQB1*03:01', 'HLA-DQA1*05:08-DQB1*03:02', 'HLA-DQA1*05:08-DQB1*03:03', 'HLA-DQA1*05:08-DQB1*03:04',
         'HLA-DQA1*05:08-DQB1*03:05', 'HLA-DQA1*05:08-DQB1*03:06',
         'HLA-DQA1*05:08-DQB1*03:07', 'HLA-DQA1*05:08-DQB1*03:08', 'HLA-DQA1*05:08-DQB1*03:09', 'HLA-DQA1*05:08-DQB1*03:10',
         'HLA-DQA1*05:08-DQB1*03:11', 'HLA-DQA1*05:08-DQB1*03:12',
         'HLA-DQA1*05:08-DQB1*03:13', 'HLA-DQA1*05:08-DQB1*03:14', 'HLA-DQA1*05:08-DQB1*03:15', 'HLA-DQA1*05:08-DQB1*03:16',
         'HLA-DQA1*05:08-DQB1*03:17', 'HLA-DQA1*05:08-DQB1*03:18',
         'HLA-DQA1*05:08-DQB1*03:19', 'HLA-DQA1*05:08-DQB1*03:20', 'HLA-DQA1*05:08-DQB1*03:21', 'HLA-DQA1*05:08-DQB1*03:22',
         'HLA-DQA1*05:08-DQB1*03:23', 'HLA-DQA1*05:08-DQB1*03:24',
         'HLA-DQA1*05:08-DQB1*03:25', 'HLA-DQA1*05:08-DQB1*03:26', 'HLA-DQA1*05:08-DQB1*03:27', 'HLA-DQA1*05:08-DQB1*03:28',
         'HLA-DQA1*05:08-DQB1*03:29', 'HLA-DQA1*05:08-DQB1*03:30',
         'HLA-DQA1*05:08-DQB1*03:31', 'HLA-DQA1*05:08-DQB1*03:32', 'HLA-DQA1*05:08-DQB1*03:33', 'HLA-DQA1*05:08-DQB1*03:34',
         'HLA-DQA1*05:08-DQB1*03:35', 'HLA-DQA1*05:08-DQB1*03:36',
         'HLA-DQA1*05:08-DQB1*03:37', 'HLA-DQA1*05:08-DQB1*03:38', 'HLA-DQA1*05:08-DQB1*04:01', 'HLA-DQA1*05:08-DQB1*04:02',
         'HLA-DQA1*05:08-DQB1*04:03', 'HLA-DQA1*05:08-DQB1*04:04',
         'HLA-DQA1*05:08-DQB1*04:05', 'HLA-DQA1*05:08-DQB1*04:06', 'HLA-DQA1*05:08-DQB1*04:07', 'HLA-DQA1*05:08-DQB1*04:08',
         'HLA-DQA1*05:08-DQB1*05:01', 'HLA-DQA1*05:08-DQB1*05:02',
         'HLA-DQA1*05:08-DQB1*05:03', 'HLA-DQA1*05:08-DQB1*05:05', 'HLA-DQA1*05:08-DQB1*05:06', 'HLA-DQA1*05:08-DQB1*05:07',
         'HLA-DQA1*05:08-DQB1*05:08', 'HLA-DQA1*05:08-DQB1*05:09',
         'HLA-DQA1*05:08-DQB1*05:10', 'HLA-DQA1*05:08-DQB1*05:11', 'HLA-DQA1*05:08-DQB1*05:12', 'HLA-DQA1*05:08-DQB1*05:13',
         'HLA-DQA1*05:08-DQB1*05:14', 'HLA-DQA1*05:08-DQB1*06:01',
         'HLA-DQA1*05:08-DQB1*06:02', 'HLA-DQA1*05:08-DQB1*06:03', 'HLA-DQA1*05:08-DQB1*06:04', 'HLA-DQA1*05:08-DQB1*06:07',
         'HLA-DQA1*05:08-DQB1*06:08', 'HLA-DQA1*05:08-DQB1*06:09',
         'HLA-DQA1*05:08-DQB1*06:10', 'HLA-DQA1*05:08-DQB1*06:11', 'HLA-DQA1*05:08-DQB1*06:12', 'HLA-DQA1*05:08-DQB1*06:14',
         'HLA-DQA1*05:08-DQB1*06:15', 'HLA-DQA1*05:08-DQB1*06:16',
         'HLA-DQA1*05:08-DQB1*06:17', 'HLA-DQA1*05:08-DQB1*06:18', 'HLA-DQA1*05:08-DQB1*06:19', 'HLA-DQA1*05:08-DQB1*06:21',
         'HLA-DQA1*05:08-DQB1*06:22', 'HLA-DQA1*05:08-DQB1*06:23',
         'HLA-DQA1*05:08-DQB1*06:24', 'HLA-DQA1*05:08-DQB1*06:25', 'HLA-DQA1*05:08-DQB1*06:27', 'HLA-DQA1*05:08-DQB1*06:28',
         'HLA-DQA1*05:08-DQB1*06:29', 'HLA-DQA1*05:08-DQB1*06:30',
         'HLA-DQA1*05:08-DQB1*06:31', 'HLA-DQA1*05:08-DQB1*06:32', 'HLA-DQA1*05:08-DQB1*06:33', 'HLA-DQA1*05:08-DQB1*06:34',
         'HLA-DQA1*05:08-DQB1*06:35', 'HLA-DQA1*05:08-DQB1*06:36',
         'HLA-DQA1*05:08-DQB1*06:37', 'HLA-DQA1*05:08-DQB1*06:38', 'HLA-DQA1*05:08-DQB1*06:39', 'HLA-DQA1*05:08-DQB1*06:40',
         'HLA-DQA1*05:08-DQB1*06:41', 'HLA-DQA1*05:08-DQB1*06:42',
         'HLA-DQA1*05:08-DQB1*06:43', 'HLA-DQA1*05:08-DQB1*06:44', 'HLA-DQA1*05:09-DQB1*02:01', 'HLA-DQA1*05:09-DQB1*02:02',
         'HLA-DQA1*05:09-DQB1*02:03', 'HLA-DQA1*05:09-DQB1*02:04',
         'HLA-DQA1*05:09-DQB1*02:05', 'HLA-DQA1*05:09-DQB1*02:06', 'HLA-DQA1*05:09-DQB1*03:01', 'HLA-DQA1*05:09-DQB1*03:02',
         'HLA-DQA1*05:09-DQB1*03:03', 'HLA-DQA1*05:09-DQB1*03:04',
         'HLA-DQA1*05:09-DQB1*03:05', 'HLA-DQA1*05:09-DQB1*03:06', 'HLA-DQA1*05:09-DQB1*03:07', 'HLA-DQA1*05:09-DQB1*03:08',
         'HLA-DQA1*05:09-DQB1*03:09', 'HLA-DQA1*05:09-DQB1*03:10',
         'HLA-DQA1*05:09-DQB1*03:11', 'HLA-DQA1*05:09-DQB1*03:12', 'HLA-DQA1*05:09-DQB1*03:13', 'HLA-DQA1*05:09-DQB1*03:14',
         'HLA-DQA1*05:09-DQB1*03:15', 'HLA-DQA1*05:09-DQB1*03:16',
         'HLA-DQA1*05:09-DQB1*03:17', 'HLA-DQA1*05:09-DQB1*03:18', 'HLA-DQA1*05:09-DQB1*03:19', 'HLA-DQA1*05:09-DQB1*03:20',
         'HLA-DQA1*05:09-DQB1*03:21', 'HLA-DQA1*05:09-DQB1*03:22',
         'HLA-DQA1*05:09-DQB1*03:23', 'HLA-DQA1*05:09-DQB1*03:24', 'HLA-DQA1*05:09-DQB1*03:25', 'HLA-DQA1*05:09-DQB1*03:26',
         'HLA-DQA1*05:09-DQB1*03:27', 'HLA-DQA1*05:09-DQB1*03:28',
         'HLA-DQA1*05:09-DQB1*03:29', 'HLA-DQA1*05:09-DQB1*03:30', 'HLA-DQA1*05:09-DQB1*03:31', 'HLA-DQA1*05:09-DQB1*03:32',
         'HLA-DQA1*05:09-DQB1*03:33', 'HLA-DQA1*05:09-DQB1*03:34',
         'HLA-DQA1*05:09-DQB1*03:35', 'HLA-DQA1*05:09-DQB1*03:36', 'HLA-DQA1*05:09-DQB1*03:37', 'HLA-DQA1*05:09-DQB1*03:38',
         'HLA-DQA1*05:09-DQB1*04:01', 'HLA-DQA1*05:09-DQB1*04:02',
         'HLA-DQA1*05:09-DQB1*04:03', 'HLA-DQA1*05:09-DQB1*04:04', 'HLA-DQA1*05:09-DQB1*04:05', 'HLA-DQA1*05:09-DQB1*04:06',
         'HLA-DQA1*05:09-DQB1*04:07', 'HLA-DQA1*05:09-DQB1*04:08',
         'HLA-DQA1*05:09-DQB1*05:01', 'HLA-DQA1*05:09-DQB1*05:02', 'HLA-DQA1*05:09-DQB1*05:03', 'HLA-DQA1*05:09-DQB1*05:05',
         'HLA-DQA1*05:09-DQB1*05:06', 'HLA-DQA1*05:09-DQB1*05:07',
         'HLA-DQA1*05:09-DQB1*05:08', 'HLA-DQA1*05:09-DQB1*05:09', 'HLA-DQA1*05:09-DQB1*05:10', 'HLA-DQA1*05:09-DQB1*05:11',
         'HLA-DQA1*05:09-DQB1*05:12', 'HLA-DQA1*05:09-DQB1*05:13',
         'HLA-DQA1*05:09-DQB1*05:14', 'HLA-DQA1*05:09-DQB1*06:01', 'HLA-DQA1*05:09-DQB1*06:02', 'HLA-DQA1*05:09-DQB1*06:03',
         'HLA-DQA1*05:09-DQB1*06:04', 'HLA-DQA1*05:09-DQB1*06:07',
         'HLA-DQA1*05:09-DQB1*06:08', 'HLA-DQA1*05:09-DQB1*06:09', 'HLA-DQA1*05:09-DQB1*06:10', 'HLA-DQA1*05:09-DQB1*06:11',
         'HLA-DQA1*05:09-DQB1*06:12', 'HLA-DQA1*05:09-DQB1*06:14',
         'HLA-DQA1*05:09-DQB1*06:15', 'HLA-DQA1*05:09-DQB1*06:16', 'HLA-DQA1*05:09-DQB1*06:17', 'HLA-DQA1*05:09-DQB1*06:18',
         'HLA-DQA1*05:09-DQB1*06:19', 'HLA-DQA1*05:09-DQB1*06:21',
         'HLA-DQA1*05:09-DQB1*06:22', 'HLA-DQA1*05:09-DQB1*06:23', 'HLA-DQA1*05:09-DQB1*06:24', 'HLA-DQA1*05:09-DQB1*06:25',
         'HLA-DQA1*05:09-DQB1*06:27', 'HLA-DQA1*05:09-DQB1*06:28',
         'HLA-DQA1*05:09-DQB1*06:29', 'HLA-DQA1*05:09-DQB1*06:30', 'HLA-DQA1*05:09-DQB1*06:31', 'HLA-DQA1*05:09-DQB1*06:32',
         'HLA-DQA1*05:09-DQB1*06:33', 'HLA-DQA1*05:09-DQB1*06:34',
         'HLA-DQA1*05:09-DQB1*06:35', 'HLA-DQA1*05:09-DQB1*06:36', 'HLA-DQA1*05:09-DQB1*06:37', 'HLA-DQA1*05:09-DQB1*06:38',
         'HLA-DQA1*05:09-DQB1*06:39', 'HLA-DQA1*05:09-DQB1*06:40',
         'HLA-DQA1*05:09-DQB1*06:41', 'HLA-DQA1*05:09-DQB1*06:42', 'HLA-DQA1*05:09-DQB1*06:43', 'HLA-DQA1*05:09-DQB1*06:44',
         'HLA-DQA1*05:10-DQB1*02:01', 'HLA-DQA1*05:10-DQB1*02:02',
         'HLA-DQA1*05:10-DQB1*02:03', 'HLA-DQA1*05:10-DQB1*02:04', 'HLA-DQA1*05:10-DQB1*02:05', 'HLA-DQA1*05:10-DQB1*02:06',
         'HLA-DQA1*05:10-DQB1*03:01', 'HLA-DQA1*05:10-DQB1*03:02',
         'HLA-DQA1*05:10-DQB1*03:03', 'HLA-DQA1*05:10-DQB1*03:04', 'HLA-DQA1*05:10-DQB1*03:05', 'HLA-DQA1*05:10-DQB1*03:06',
         'HLA-DQA1*05:10-DQB1*03:07', 'HLA-DQA1*05:10-DQB1*03:08',
         'HLA-DQA1*05:10-DQB1*03:09', 'HLA-DQA1*05:10-DQB1*03:10', 'HLA-DQA1*05:10-DQB1*03:11', 'HLA-DQA1*05:10-DQB1*03:12',
         'HLA-DQA1*05:10-DQB1*03:13', 'HLA-DQA1*05:10-DQB1*03:14',
         'HLA-DQA1*05:10-DQB1*03:15', 'HLA-DQA1*05:10-DQB1*03:16', 'HLA-DQA1*05:10-DQB1*03:17', 'HLA-DQA1*05:10-DQB1*03:18',
         'HLA-DQA1*05:10-DQB1*03:19', 'HLA-DQA1*05:10-DQB1*03:20',
         'HLA-DQA1*05:10-DQB1*03:21', 'HLA-DQA1*05:10-DQB1*03:22', 'HLA-DQA1*05:10-DQB1*03:23', 'HLA-DQA1*05:10-DQB1*03:24',
         'HLA-DQA1*05:10-DQB1*03:25', 'HLA-DQA1*05:10-DQB1*03:26',
         'HLA-DQA1*05:10-DQB1*03:27', 'HLA-DQA1*05:10-DQB1*03:28', 'HLA-DQA1*05:10-DQB1*03:29', 'HLA-DQA1*05:10-DQB1*03:30',
         'HLA-DQA1*05:10-DQB1*03:31', 'HLA-DQA1*05:10-DQB1*03:32',
         'HLA-DQA1*05:10-DQB1*03:33', 'HLA-DQA1*05:10-DQB1*03:34', 'HLA-DQA1*05:10-DQB1*03:35', 'HLA-DQA1*05:10-DQB1*03:36',
         'HLA-DQA1*05:10-DQB1*03:37', 'HLA-DQA1*05:10-DQB1*03:38',
         'HLA-DQA1*05:10-DQB1*04:01', 'HLA-DQA1*05:10-DQB1*04:02', 'HLA-DQA1*05:10-DQB1*04:03', 'HLA-DQA1*05:10-DQB1*04:04',
         'HLA-DQA1*05:10-DQB1*04:05', 'HLA-DQA1*05:10-DQB1*04:06',
         'HLA-DQA1*05:10-DQB1*04:07', 'HLA-DQA1*05:10-DQB1*04:08', 'HLA-DQA1*05:10-DQB1*05:01', 'HLA-DQA1*05:10-DQB1*05:02',
         'HLA-DQA1*05:10-DQB1*05:03', 'HLA-DQA1*05:10-DQB1*05:05',
         'HLA-DQA1*05:10-DQB1*05:06', 'HLA-DQA1*05:10-DQB1*05:07', 'HLA-DQA1*05:10-DQB1*05:08', 'HLA-DQA1*05:10-DQB1*05:09',
         'HLA-DQA1*05:10-DQB1*05:10', 'HLA-DQA1*05:10-DQB1*05:11',
         'HLA-DQA1*05:10-DQB1*05:12', 'HLA-DQA1*05:10-DQB1*05:13', 'HLA-DQA1*05:10-DQB1*05:14', 'HLA-DQA1*05:10-DQB1*06:01',
         'HLA-DQA1*05:10-DQB1*06:02', 'HLA-DQA1*05:10-DQB1*06:03',
         'HLA-DQA1*05:10-DQB1*06:04', 'HLA-DQA1*05:10-DQB1*06:07', 'HLA-DQA1*05:10-DQB1*06:08', 'HLA-DQA1*05:10-DQB1*06:09',
         'HLA-DQA1*05:10-DQB1*06:10', 'HLA-DQA1*05:10-DQB1*06:11',
         'HLA-DQA1*05:10-DQB1*06:12', 'HLA-DQA1*05:10-DQB1*06:14', 'HLA-DQA1*05:10-DQB1*06:15', 'HLA-DQA1*05:10-DQB1*06:16',
         'HLA-DQA1*05:10-DQB1*06:17', 'HLA-DQA1*05:10-DQB1*06:18',
         'HLA-DQA1*05:10-DQB1*06:19', 'HLA-DQA1*05:10-DQB1*06:21', 'HLA-DQA1*05:10-DQB1*06:22', 'HLA-DQA1*05:10-DQB1*06:23',
         'HLA-DQA1*05:10-DQB1*06:24', 'HLA-DQA1*05:10-DQB1*06:25',
         'HLA-DQA1*05:10-DQB1*06:27', 'HLA-DQA1*05:10-DQB1*06:28', 'HLA-DQA1*05:10-DQB1*06:29', 'HLA-DQA1*05:10-DQB1*06:30',
         'HLA-DQA1*05:10-DQB1*06:31', 'HLA-DQA1*05:10-DQB1*06:32',
         'HLA-DQA1*05:10-DQB1*06:33', 'HLA-DQA1*05:10-DQB1*06:34', 'HLA-DQA1*05:10-DQB1*06:35', 'HLA-DQA1*05:10-DQB1*06:36',
         'HLA-DQA1*05:10-DQB1*06:37', 'HLA-DQA1*05:10-DQB1*06:38',
         'HLA-DQA1*05:10-DQB1*06:39', 'HLA-DQA1*05:10-DQB1*06:40', 'HLA-DQA1*05:10-DQB1*06:41', 'HLA-DQA1*05:10-DQB1*06:42',
         'HLA-DQA1*05:10-DQB1*06:43', 'HLA-DQA1*05:10-DQB1*06:44',
         'HLA-DQA1*05:11-DQB1*02:01', 'HLA-DQA1*05:11-DQB1*02:02', 'HLA-DQA1*05:11-DQB1*02:03', 'HLA-DQA1*05:11-DQB1*02:04',
         'HLA-DQA1*05:11-DQB1*02:05', 'HLA-DQA1*05:11-DQB1*02:06',
         'HLA-DQA1*05:11-DQB1*03:01', 'HLA-DQA1*05:11-DQB1*03:02', 'HLA-DQA1*05:11-DQB1*03:03', 'HLA-DQA1*05:11-DQB1*03:04',
         'HLA-DQA1*05:11-DQB1*03:05', 'HLA-DQA1*05:11-DQB1*03:06',
         'HLA-DQA1*05:11-DQB1*03:07', 'HLA-DQA1*05:11-DQB1*03:08', 'HLA-DQA1*05:11-DQB1*03:09', 'HLA-DQA1*05:11-DQB1*03:10',
         'HLA-DQA1*05:11-DQB1*03:11', 'HLA-DQA1*05:11-DQB1*03:12',
         'HLA-DQA1*05:11-DQB1*03:13', 'HLA-DQA1*05:11-DQB1*03:14', 'HLA-DQA1*05:11-DQB1*03:15', 'HLA-DQA1*05:11-DQB1*03:16',
         'HLA-DQA1*05:11-DQB1*03:17', 'HLA-DQA1*05:11-DQB1*03:18',
         'HLA-DQA1*05:11-DQB1*03:19', 'HLA-DQA1*05:11-DQB1*03:20', 'HLA-DQA1*05:11-DQB1*03:21', 'HLA-DQA1*05:11-DQB1*03:22',
         'HLA-DQA1*05:11-DQB1*03:23', 'HLA-DQA1*05:11-DQB1*03:24',
         'HLA-DQA1*05:11-DQB1*03:25', 'HLA-DQA1*05:11-DQB1*03:26', 'HLA-DQA1*05:11-DQB1*03:27', 'HLA-DQA1*05:11-DQB1*03:28',
         'HLA-DQA1*05:11-DQB1*03:29', 'HLA-DQA1*05:11-DQB1*03:30',
         'HLA-DQA1*05:11-DQB1*03:31', 'HLA-DQA1*05:11-DQB1*03:32', 'HLA-DQA1*05:11-DQB1*03:33', 'HLA-DQA1*05:11-DQB1*03:34',
         'HLA-DQA1*05:11-DQB1*03:35', 'HLA-DQA1*05:11-DQB1*03:36',
         'HLA-DQA1*05:11-DQB1*03:37', 'HLA-DQA1*05:11-DQB1*03:38', 'HLA-DQA1*05:11-DQB1*04:01', 'HLA-DQA1*05:11-DQB1*04:02',
         'HLA-DQA1*05:11-DQB1*04:03', 'HLA-DQA1*05:11-DQB1*04:04',
         'HLA-DQA1*05:11-DQB1*04:05', 'HLA-DQA1*05:11-DQB1*04:06', 'HLA-DQA1*05:11-DQB1*04:07', 'HLA-DQA1*05:11-DQB1*04:08',
         'HLA-DQA1*05:11-DQB1*05:01', 'HLA-DQA1*05:11-DQB1*05:02',
         'HLA-DQA1*05:11-DQB1*05:03', 'HLA-DQA1*05:11-DQB1*05:05', 'HLA-DQA1*05:11-DQB1*05:06', 'HLA-DQA1*05:11-DQB1*05:07',
         'HLA-DQA1*05:11-DQB1*05:08', 'HLA-DQA1*05:11-DQB1*05:09',
         'HLA-DQA1*05:11-DQB1*05:10', 'HLA-DQA1*05:11-DQB1*05:11', 'HLA-DQA1*05:11-DQB1*05:12', 'HLA-DQA1*05:11-DQB1*05:13',
         'HLA-DQA1*05:11-DQB1*05:14', 'HLA-DQA1*05:11-DQB1*06:01',
         'HLA-DQA1*05:11-DQB1*06:02', 'HLA-DQA1*05:11-DQB1*06:03', 'HLA-DQA1*05:11-DQB1*06:04', 'HLA-DQA1*05:11-DQB1*06:07',
         'HLA-DQA1*05:11-DQB1*06:08', 'HLA-DQA1*05:11-DQB1*06:09',
         'HLA-DQA1*05:11-DQB1*06:10', 'HLA-DQA1*05:11-DQB1*06:11', 'HLA-DQA1*05:11-DQB1*06:12', 'HLA-DQA1*05:11-DQB1*06:14',
         'HLA-DQA1*05:11-DQB1*06:15', 'HLA-DQA1*05:11-DQB1*06:16',
         'HLA-DQA1*05:11-DQB1*06:17', 'HLA-DQA1*05:11-DQB1*06:18', 'HLA-DQA1*05:11-DQB1*06:19', 'HLA-DQA1*05:11-DQB1*06:21',
         'HLA-DQA1*05:11-DQB1*06:22', 'HLA-DQA1*05:11-DQB1*06:23',
         'HLA-DQA1*05:11-DQB1*06:24', 'HLA-DQA1*05:11-DQB1*06:25', 'HLA-DQA1*05:11-DQB1*06:27', 'HLA-DQA1*05:11-DQB1*06:28',
         'HLA-DQA1*05:11-DQB1*06:29', 'HLA-DQA1*05:11-DQB1*06:30',
         'HLA-DQA1*05:11-DQB1*06:31', 'HLA-DQA1*05:11-DQB1*06:32', 'HLA-DQA1*05:11-DQB1*06:33', 'HLA-DQA1*05:11-DQB1*06:34',
         'HLA-DQA1*05:11-DQB1*06:35', 'HLA-DQA1*05:11-DQB1*06:36',
         'HLA-DQA1*05:11-DQB1*06:37', 'HLA-DQA1*05:11-DQB1*06:38', 'HLA-DQA1*05:11-DQB1*06:39', 'HLA-DQA1*05:11-DQB1*06:40',
         'HLA-DQA1*05:11-DQB1*06:41', 'HLA-DQA1*05:11-DQB1*06:42',
         'HLA-DQA1*05:11-DQB1*06:43', 'HLA-DQA1*05:11-DQB1*06:44', 'HLA-DQA1*06:01-DQB1*02:01', 'HLA-DQA1*06:01-DQB1*02:02',
         'HLA-DQA1*06:01-DQB1*02:03', 'HLA-DQA1*06:01-DQB1*02:04',
         'HLA-DQA1*06:01-DQB1*02:05', 'HLA-DQA1*06:01-DQB1*02:06', 'HLA-DQA1*06:01-DQB1*03:01', 'HLA-DQA1*06:01-DQB1*03:02',
         'HLA-DQA1*06:01-DQB1*03:03', 'HLA-DQA1*06:01-DQB1*03:04',
         'HLA-DQA1*06:01-DQB1*03:05', 'HLA-DQA1*06:01-DQB1*03:06', 'HLA-DQA1*06:01-DQB1*03:07', 'HLA-DQA1*06:01-DQB1*03:08',
         'HLA-DQA1*06:01-DQB1*03:09', 'HLA-DQA1*06:01-DQB1*03:10',
         'HLA-DQA1*06:01-DQB1*03:11', 'HLA-DQA1*06:01-DQB1*03:12', 'HLA-DQA1*06:01-DQB1*03:13', 'HLA-DQA1*06:01-DQB1*03:14',
         'HLA-DQA1*06:01-DQB1*03:15', 'HLA-DQA1*06:01-DQB1*03:16',
         'HLA-DQA1*06:01-DQB1*03:17', 'HLA-DQA1*06:01-DQB1*03:18', 'HLA-DQA1*06:01-DQB1*03:19', 'HLA-DQA1*06:01-DQB1*03:20',
         'HLA-DQA1*06:01-DQB1*03:21', 'HLA-DQA1*06:01-DQB1*03:22',
         'HLA-DQA1*06:01-DQB1*03:23', 'HLA-DQA1*06:01-DQB1*03:24', 'HLA-DQA1*06:01-DQB1*03:25', 'HLA-DQA1*06:01-DQB1*03:26',
         'HLA-DQA1*06:01-DQB1*03:27', 'HLA-DQA1*06:01-DQB1*03:28',
         'HLA-DQA1*06:01-DQB1*03:29', 'HLA-DQA1*06:01-DQB1*03:30', 'HLA-DQA1*06:01-DQB1*03:31', 'HLA-DQA1*06:01-DQB1*03:32',
         'HLA-DQA1*06:01-DQB1*03:33', 'HLA-DQA1*06:01-DQB1*03:34',
         'HLA-DQA1*06:01-DQB1*03:35', 'HLA-DQA1*06:01-DQB1*03:36', 'HLA-DQA1*06:01-DQB1*03:37', 'HLA-DQA1*06:01-DQB1*03:38',
         'HLA-DQA1*06:01-DQB1*04:01', 'HLA-DQA1*06:01-DQB1*04:02',
         'HLA-DQA1*06:01-DQB1*04:03', 'HLA-DQA1*06:01-DQB1*04:04', 'HLA-DQA1*06:01-DQB1*04:05', 'HLA-DQA1*06:01-DQB1*04:06',
         'HLA-DQA1*06:01-DQB1*04:07', 'HLA-DQA1*06:01-DQB1*04:08',
         'HLA-DQA1*06:01-DQB1*05:01', 'HLA-DQA1*06:01-DQB1*05:02', 'HLA-DQA1*06:01-DQB1*05:03', 'HLA-DQA1*06:01-DQB1*05:05',
         'HLA-DQA1*06:01-DQB1*05:06', 'HLA-DQA1*06:01-DQB1*05:07',
         'HLA-DQA1*06:01-DQB1*05:08', 'HLA-DQA1*06:01-DQB1*05:09', 'HLA-DQA1*06:01-DQB1*05:10', 'HLA-DQA1*06:01-DQB1*05:11',
         'HLA-DQA1*06:01-DQB1*05:12', 'HLA-DQA1*06:01-DQB1*05:13',
         'HLA-DQA1*06:01-DQB1*05:14', 'HLA-DQA1*06:01-DQB1*06:01', 'HLA-DQA1*06:01-DQB1*06:02', 'HLA-DQA1*06:01-DQB1*06:03',
         'HLA-DQA1*06:01-DQB1*06:04', 'HLA-DQA1*06:01-DQB1*06:07',
         'HLA-DQA1*06:01-DQB1*06:08', 'HLA-DQA1*06:01-DQB1*06:09', 'HLA-DQA1*06:01-DQB1*06:10', 'HLA-DQA1*06:01-DQB1*06:11',
         'HLA-DQA1*06:01-DQB1*06:12', 'HLA-DQA1*06:01-DQB1*06:14',
         'HLA-DQA1*06:01-DQB1*06:15', 'HLA-DQA1*06:01-DQB1*06:16', 'HLA-DQA1*06:01-DQB1*06:17', 'HLA-DQA1*06:01-DQB1*06:18',
         'HLA-DQA1*06:01-DQB1*06:19', 'HLA-DQA1*06:01-DQB1*06:21',
         'HLA-DQA1*06:01-DQB1*06:22', 'HLA-DQA1*06:01-DQB1*06:23', 'HLA-DQA1*06:01-DQB1*06:24', 'HLA-DQA1*06:01-DQB1*06:25',
         'HLA-DQA1*06:01-DQB1*06:27', 'HLA-DQA1*06:01-DQB1*06:28',
         'HLA-DQA1*06:01-DQB1*06:29', 'HLA-DQA1*06:01-DQB1*06:30', 'HLA-DQA1*06:01-DQB1*06:31', 'HLA-DQA1*06:01-DQB1*06:32',
         'HLA-DQA1*06:01-DQB1*06:33', 'HLA-DQA1*06:01-DQB1*06:34',
         'HLA-DQA1*06:01-DQB1*06:35', 'HLA-DQA1*06:01-DQB1*06:36', 'HLA-DQA1*06:01-DQB1*06:37', 'HLA-DQA1*06:01-DQB1*06:38',
         'HLA-DQA1*06:01-DQB1*06:39', 'HLA-DQA1*06:01-DQB1*06:40',
         'HLA-DQA1*06:01-DQB1*06:41', 'HLA-DQA1*06:01-DQB1*06:42', 'HLA-DQA1*06:01-DQB1*06:43', 'HLA-DQA1*06:01-DQB1*06:44',
         'HLA-DQA1*06:02-DQB1*02:01', 'HLA-DQA1*06:02-DQB1*02:02',
         'HLA-DQA1*06:02-DQB1*02:03', 'HLA-DQA1*06:02-DQB1*02:04', 'HLA-DQA1*06:02-DQB1*02:05', 'HLA-DQA1*06:02-DQB1*02:06',
         'HLA-DQA1*06:02-DQB1*03:01', 'HLA-DQA1*06:02-DQB1*03:02',
         'HLA-DQA1*06:02-DQB1*03:03', 'HLA-DQA1*06:02-DQB1*03:04', 'HLA-DQA1*06:02-DQB1*03:05', 'HLA-DQA1*06:02-DQB1*03:06',
         'HLA-DQA1*06:02-DQB1*03:07', 'HLA-DQA1*06:02-DQB1*03:08',
         'HLA-DQA1*06:02-DQB1*03:09', 'HLA-DQA1*06:02-DQB1*03:10', 'HLA-DQA1*06:02-DQB1*03:11', 'HLA-DQA1*06:02-DQB1*03:12',
         'HLA-DQA1*06:02-DQB1*03:13', 'HLA-DQA1*06:02-DQB1*03:14',
         'HLA-DQA1*06:02-DQB1*03:15', 'HLA-DQA1*06:02-DQB1*03:16', 'HLA-DQA1*06:02-DQB1*03:17', 'HLA-DQA1*06:02-DQB1*03:18',
         'HLA-DQA1*06:02-DQB1*03:19', 'HLA-DQA1*06:02-DQB1*03:20',
         'HLA-DQA1*06:02-DQB1*03:21', 'HLA-DQA1*06:02-DQB1*03:22', 'HLA-DQA1*06:02-DQB1*03:23', 'HLA-DQA1*06:02-DQB1*03:24',
         'HLA-DQA1*06:02-DQB1*03:25', 'HLA-DQA1*06:02-DQB1*03:26',
         'HLA-DQA1*06:02-DQB1*03:27', 'HLA-DQA1*06:02-DQB1*03:28', 'HLA-DQA1*06:02-DQB1*03:29', 'HLA-DQA1*06:02-DQB1*03:30',
         'HLA-DQA1*06:02-DQB1*03:31', 'HLA-DQA1*06:02-DQB1*03:32',
         'HLA-DQA1*06:02-DQB1*03:33', 'HLA-DQA1*06:02-DQB1*03:34', 'HLA-DQA1*06:02-DQB1*03:35', 'HLA-DQA1*06:02-DQB1*03:36',
         'HLA-DQA1*06:02-DQB1*03:37', 'HLA-DQA1*06:02-DQB1*03:38',
         'HLA-DQA1*06:02-DQB1*04:01', 'HLA-DQA1*06:02-DQB1*04:02', 'HLA-DQA1*06:02-DQB1*04:03', 'HLA-DQA1*06:02-DQB1*04:04',
         'HLA-DQA1*06:02-DQB1*04:05', 'HLA-DQA1*06:02-DQB1*04:06',
         'HLA-DQA1*06:02-DQB1*04:07', 'HLA-DQA1*06:02-DQB1*04:08', 'HLA-DQA1*06:02-DQB1*05:01', 'HLA-DQA1*06:02-DQB1*05:02',
         'HLA-DQA1*06:02-DQB1*05:03', 'HLA-DQA1*06:02-DQB1*05:05',
         'HLA-DQA1*06:02-DQB1*05:06', 'HLA-DQA1*06:02-DQB1*05:07', 'HLA-DQA1*06:02-DQB1*05:08', 'HLA-DQA1*06:02-DQB1*05:09',
         'HLA-DQA1*06:02-DQB1*05:10', 'HLA-DQA1*06:02-DQB1*05:11',
         'HLA-DQA1*06:02-DQB1*05:12', 'HLA-DQA1*06:02-DQB1*05:13', 'HLA-DQA1*06:02-DQB1*05:14', 'HLA-DQA1*06:02-DQB1*06:01',
         'HLA-DQA1*06:02-DQB1*06:02', 'HLA-DQA1*06:02-DQB1*06:03',
         'HLA-DQA1*06:02-DQB1*06:04', 'HLA-DQA1*06:02-DQB1*06:07', 'HLA-DQA1*06:02-DQB1*06:08', 'HLA-DQA1*06:02-DQB1*06:09',
         'HLA-DQA1*06:02-DQB1*06:10', 'HLA-DQA1*06:02-DQB1*06:11',
         'HLA-DQA1*06:02-DQB1*06:12', 'HLA-DQA1*06:02-DQB1*06:14', 'HLA-DQA1*06:02-DQB1*06:15', 'HLA-DQA1*06:02-DQB1*06:16',
         'HLA-DQA1*06:02-DQB1*06:17', 'HLA-DQA1*06:02-DQB1*06:18',
         'HLA-DQA1*06:02-DQB1*06:19', 'HLA-DQA1*06:02-DQB1*06:21', 'HLA-DQA1*06:02-DQB1*06:22', 'HLA-DQA1*06:02-DQB1*06:23',
         'HLA-DQA1*06:02-DQB1*06:24', 'HLA-DQA1*06:02-DQB1*06:25',
         'HLA-DQA1*06:02-DQB1*06:27', 'HLA-DQA1*06:02-DQB1*06:28', 'HLA-DQA1*06:02-DQB1*06:29', 'HLA-DQA1*06:02-DQB1*06:30',
         'HLA-DQA1*06:02-DQB1*06:31', 'HLA-DQA1*06:02-DQB1*06:32',
         'HLA-DQA1*06:02-DQB1*06:33', 'HLA-DQA1*06:02-DQB1*06:34', 'HLA-DQA1*06:02-DQB1*06:35', 'HLA-DQA1*06:02-DQB1*06:36',
         'HLA-DQA1*06:02-DQB1*06:37', 'HLA-DQA1*06:02-DQB1*06:38',
         'HLA-DQA1*06:02-DQB1*06:39', 'HLA-DQA1*06:02-DQB1*06:40', 'HLA-DQA1*06:02-DQB1*06:41', 'HLA-DQA1*06:02-DQB1*06:42',
         'HLA-DQA1*06:02-DQB1*06:43', 'HLA-DQA1*06:02-DQB1*06:44',
         'H-2-Iad', 'H-2-Iab'])

    __version = "3.1"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
 
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_3_1]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_3_1 + i * Offset.NETMHCIIPAN_3_1])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_3_1 + i * Offset.NETMHCIIPAN_3_1])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result

class NetMHCIIpan_4_0(NetMHCIIpan_3_1):
    """
    Implementation of NetMHCIIpan 4.0 adapter.

    .. note::

        Reynisson B, Barra C, Kaabinejadian S, Hildebrand WH, Peters B, Nielsen M (2020). Improved prediction of MHC II antigen presentation
        through integration and motif deconvolution of mass spectrometry MHC eluted ligand data.
        Immunogenetics, 1-10.
    """

    __supported_length = frozenset([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    __name = "netmhcIIpan"
    __command = "netMHCIIpan -f {peptides} -inptype 1 -a {alleles} {options} -xls -xlsfile {out}"
    __alleles = frozenset(['HLA-DRB1*01:01', 'HLA-DRB1*01:02', 'HLA-DRB1*01:03', 'HLA-DRB1*01:04', 'HLA-DRB1*01:05',
'HLA-DRB1*01:06', 'HLA-DRB1*01:07', 'HLA-DRB1*01:08', 'HLA-DRB1*01:09', 'HLA-DRB1*01:10',
'HLA-DRB1*01:11', 'HLA-DRB1*01:12', 'HLA-DRB1*01:13', 'HLA-DRB1*01:14', 'HLA-DRB1*01:15',
'HLA-DRB1*01:16', 'HLA-DRB1*01:17', 'HLA-DRB1*01:18', 'HLA-DRB1*01:19', 'HLA-DRB1*01:20',
'HLA-DRB1*01:21', 'HLA-DRB1*01:22', 'HLA-DRB1*01:23', 'HLA-DRB1*01:24', 'HLA-DRB1*01:25',
'HLA-DRB1*01:26', 'HLA-DRB1*01:27', 'HLA-DRB1*01:28', 'HLA-DRB1*01:29', 'HLA-DRB1*01:30',
'HLA-DRB1*01:31', 'HLA-DRB1*01:32', 'HLA-DRB1*03:01', 'HLA-DRB1*03:02', 'HLA-DRB1*03:03',
'HLA-DRB1*03:04', 'HLA-DRB1*03:05', 'HLA-DRB1*03:06', 'HLA-DRB1*03:07', 'HLA-DRB1*03:08',
'HLA-DRB1*03:10', 'HLA-DRB1*03:11', 'HLA-DRB1*03:13', 'HLA-DRB1*03:14', 'HLA-DRB1*03:15',
'HLA-DRB1*03:17', 'HLA-DRB1*03:18', 'HLA-DRB1*03:19', 'HLA-DRB1*03:20', 'HLA-DRB1*03:21',
'HLA-DRB1*03:22', 'HLA-DRB1*03:23', 'HLA-DRB1*03:24', 'HLA-DRB1*03:25', 'HLA-DRB1*03:26',
'HLA-DRB1*03:27', 'HLA-DRB1*03:28', 'HLA-DRB1*03:29', 'HLA-DRB1*03:30', 'HLA-DRB1*03:31',
'HLA-DRB1*03:32', 'HLA-DRB1*03:33', 'HLA-DRB1*03:34', 'HLA-DRB1*03:35', 'HLA-DRB1*03:36',
'HLA-DRB1*03:37', 'HLA-DRB1*03:38', 'HLA-DRB1*03:39', 'HLA-DRB1*03:40', 'HLA-DRB1*03:41',
'HLA-DRB1*03:42', 'HLA-DRB1*03:43', 'HLA-DRB1*03:44', 'HLA-DRB1*03:45', 'HLA-DRB1*03:46',
'HLA-DRB1*03:47', 'HLA-DRB1*03:48', 'HLA-DRB1*03:49', 'HLA-DRB1*03:50', 'HLA-DRB1*03:51',
'HLA-DRB1*03:52', 'HLA-DRB1*03:53', 'HLA-DRB1*03:54', 'HLA-DRB1*03:55', 'HLA-DRB1*04:01',
'HLA-DRB1*04:02', 'HLA-DRB1*04:03', 'HLA-DRB1*04:04', 'HLA-DRB1*04:05', 'HLA-DRB1*04:06',
'HLA-DRB1*04:07', 'HLA-DRB1*04:08', 'HLA-DRB1*04:09', 'HLA-DRB1*04:10', 'HLA-DRB1*04:11',
'HLA-DRB1*04:12', 'HLA-DRB1*04:13', 'HLA-DRB1*04:14', 'HLA-DRB1*04:15', 'HLA-DRB1*04:16',
'HLA-DRB1*04:17', 'HLA-DRB1*04:18', 'HLA-DRB1*04:19', 'HLA-DRB1*04:21', 'HLA-DRB1*04:22',
'HLA-DRB1*04:23', 'HLA-DRB1*04:24', 'HLA-DRB1*04:26', 'HLA-DRB1*04:27', 'HLA-DRB1*04:28',
'HLA-DRB1*04:29', 'HLA-DRB1*04:30', 'HLA-DRB1*04:31', 'HLA-DRB1*04:33', 'HLA-DRB1*04:34',
'HLA-DRB1*04:35', 'HLA-DRB1*04:36', 'HLA-DRB1*04:37', 'HLA-DRB1*04:38', 'HLA-DRB1*04:39',
'HLA-DRB1*04:40', 'HLA-DRB1*04:41', 'HLA-DRB1*04:42', 'HLA-DRB1*04:43', 'HLA-DRB1*04:44',
'HLA-DRB1*04:45', 'HLA-DRB1*04:46', 'HLA-DRB1*04:47', 'HLA-DRB1*04:48', 'HLA-DRB1*04:49',
'HLA-DRB1*04:50', 'HLA-DRB1*04:51', 'HLA-DRB1*04:52', 'HLA-DRB1*04:53', 'HLA-DRB1*04:54',
'HLA-DRB1*04:55', 'HLA-DRB1*04:56', 'HLA-DRB1*04:57', 'HLA-DRB1*04:58', 'HLA-DRB1*04:59',
'HLA-DRB1*04:60', 'HLA-DRB1*04:61', 'HLA-DRB1*04:62', 'HLA-DRB1*04:63', 'HLA-DRB1*04:64',
'HLA-DRB1*04:65', 'HLA-DRB1*04:66', 'HLA-DRB1*04:67', 'HLA-DRB1*04:68', 'HLA-DRB1*04:69',
'HLA-DRB1*04:70', 'HLA-DRB1*04:71', 'HLA-DRB1*04:72', 'HLA-DRB1*04:73', 'HLA-DRB1*04:74',
'HLA-DRB1*04:75', 'HLA-DRB1*04:76', 'HLA-DRB1*04:77', 'HLA-DRB1*04:78', 'HLA-DRB1*04:79',
'HLA-DRB1*04:80', 'HLA-DRB1*04:82', 'HLA-DRB1*04:83', 'HLA-DRB1*04:84', 'HLA-DRB1*04:85',
'HLA-DRB1*04:86', 'HLA-DRB1*04:87', 'HLA-DRB1*04:88', 'HLA-DRB1*04:89', 'HLA-DRB1*04:91',
'HLA-DRB1*07:01', 'HLA-DRB1*07:03', 'HLA-DRB1*07:04', 'HLA-DRB1*07:05', 'HLA-DRB1*07:06',
'HLA-DRB1*07:07', 'HLA-DRB1*07:08', 'HLA-DRB1*07:09', 'HLA-DRB1*07:11', 'HLA-DRB1*07:12',
'HLA-DRB1*07:13', 'HLA-DRB1*07:14', 'HLA-DRB1*07:15', 'HLA-DRB1*07:16', 'HLA-DRB1*07:17',
'HLA-DRB1*07:19', 'HLA-DRB1*08:01', 'HLA-DRB1*08:02', 'HLA-DRB1*08:03', 'HLA-DRB1*08:04',
'HLA-DRB1*08:05', 'HLA-DRB1*08:06', 'HLA-DRB1*08:07', 'HLA-DRB1*08:08', 'HLA-DRB1*08:09',
'HLA-DRB1*08:10', 'HLA-DRB1*08:11', 'HLA-DRB1*08:12', 'HLA-DRB1*08:13', 'HLA-DRB1*08:14',
'HLA-DRB1*08:15', 'HLA-DRB1*08:16', 'HLA-DRB1*08:18', 'HLA-DRB1*08:19', 'HLA-DRB1*08:20',
'HLA-DRB1*08:21', 'HLA-DRB1*08:22', 'HLA-DRB1*08:23', 'HLA-DRB1*08:24', 'HLA-DRB1*08:25',
'HLA-DRB1*08:26', 'HLA-DRB1*08:27', 'HLA-DRB1*08:28', 'HLA-DRB1*08:29', 'HLA-DRB1*08:30',
'HLA-DRB1*08:31', 'HLA-DRB1*08:32', 'HLA-DRB1*08:33', 'HLA-DRB1*08:34', 'HLA-DRB1*08:35',
'HLA-DRB1*08:36', 'HLA-DRB1*08:37', 'HLA-DRB1*08:38', 'HLA-DRB1*08:39', 'HLA-DRB1*08:40',
'HLA-DRB1*09:01', 'HLA-DRB1*09:02', 'HLA-DRB1*09:03', 'HLA-DRB1*09:04', 'HLA-DRB1*09:05',
'HLA-DRB1*09:06', 'HLA-DRB1*09:07', 'HLA-DRB1*09:08', 'HLA-DRB1*09:09', 'HLA-DRB1*10:01',
'HLA-DRB1*10:02', 'HLA-DRB1*10:03', 'HLA-DRB1*11:01', 'HLA-DRB1*11:02', 'HLA-DRB1*11:03',
'HLA-DRB1*11:04', 'HLA-DRB1*11:05', 'HLA-DRB1*11:06', 'HLA-DRB1*11:07', 'HLA-DRB1*11:08',
'HLA-DRB1*11:09', 'HLA-DRB1*11:10', 'HLA-DRB1*11:11', 'HLA-DRB1*11:12', 'HLA-DRB1*11:13',
'HLA-DRB1*11:14', 'HLA-DRB1*11:15', 'HLA-DRB1*11:16', 'HLA-DRB1*11:17', 'HLA-DRB1*11:18',
'HLA-DRB1*11:19', 'HLA-DRB1*11:20', 'HLA-DRB1*11:21', 'HLA-DRB1*11:24', 'HLA-DRB1*11:25',
'HLA-DRB1*11:27', 'HLA-DRB1*11:28', 'HLA-DRB1*11:29', 'HLA-DRB1*11:30', 'HLA-DRB1*11:31',
'HLA-DRB1*11:32', 'HLA-DRB1*11:33', 'HLA-DRB1*11:34', 'HLA-DRB1*11:35', 'HLA-DRB1*11:36',
'HLA-DRB1*11:37', 'HLA-DRB1*11:38', 'HLA-DRB1*11:39', 'HLA-DRB1*11:41', 'HLA-DRB1*11:42',
'HLA-DRB1*11:43', 'HLA-DRB1*11:44', 'HLA-DRB1*11:45', 'HLA-DRB1*11:46', 'HLA-DRB1*11:47',
'HLA-DRB1*11:48', 'HLA-DRB1*11:49', 'HLA-DRB1*11:50', 'HLA-DRB1*11:51', 'HLA-DRB1*11:52',
'HLA-DRB1*11:53', 'HLA-DRB1*11:54', 'HLA-DRB1*11:55', 'HLA-DRB1*11:56', 'HLA-DRB1*11:57',
'HLA-DRB1*11:58', 'HLA-DRB1*11:59', 'HLA-DRB1*11:60', 'HLA-DRB1*11:61', 'HLA-DRB1*11:62',
'HLA-DRB1*11:63', 'HLA-DRB1*11:64', 'HLA-DRB1*11:65', 'HLA-DRB1*11:66', 'HLA-DRB1*11:67',
'HLA-DRB1*11:68', 'HLA-DRB1*11:69', 'HLA-DRB1*11:70', 'HLA-DRB1*11:72', 'HLA-DRB1*11:73',
'HLA-DRB1*11:74', 'HLA-DRB1*11:75', 'HLA-DRB1*11:76', 'HLA-DRB1*11:77', 'HLA-DRB1*11:78',
'HLA-DRB1*11:79', 'HLA-DRB1*11:80', 'HLA-DRB1*11:81', 'HLA-DRB1*11:82', 'HLA-DRB1*11:83',
'HLA-DRB1*11:84', 'HLA-DRB1*11:85', 'HLA-DRB1*11:86', 'HLA-DRB1*11:87', 'HLA-DRB1*11:88',
'HLA-DRB1*11:89', 'HLA-DRB1*11:90', 'HLA-DRB1*11:91', 'HLA-DRB1*11:92', 'HLA-DRB1*11:93',
'HLA-DRB1*11:94', 'HLA-DRB1*11:95', 'HLA-DRB1*11:96', 'HLA-DRB1*12:01', 'HLA-DRB1*12:02',
'HLA-DRB1*12:03', 'HLA-DRB1*12:04', 'HLA-DRB1*12:05', 'HLA-DRB1*12:06', 'HLA-DRB1*12:07',
'HLA-DRB1*12:08', 'HLA-DRB1*12:09', 'HLA-DRB1*12:10', 'HLA-DRB1*12:11', 'HLA-DRB1*12:12',
'HLA-DRB1*12:13', 'HLA-DRB1*12:14', 'HLA-DRB1*12:15', 'HLA-DRB1*12:16', 'HLA-DRB1*12:17',
'HLA-DRB1*12:18', 'HLA-DRB1*12:19', 'HLA-DRB1*12:20', 'HLA-DRB1*12:21', 'HLA-DRB1*12:22',
'HLA-DRB1*12:23', 'HLA-DRB1*13:01', 'HLA-DRB1*13:02', 'HLA-DRB1*13:03', 'HLA-DRB1*13:04',
'HLA-DRB1*13:05', 'HLA-DRB1*13:06', 'HLA-DRB1*13:07', 'HLA-DRB1*13:08', 'HLA-DRB1*13:09',
'HLA-DRB1*13:10', 'HLA-DRB1*13:100', 'HLA-DRB1*13:101', 'HLA-DRB1*13:11', 'HLA-DRB1*13:12',
'HLA-DRB1*13:13', 'HLA-DRB1*13:14', 'HLA-DRB1*13:15', 'HLA-DRB1*13:16', 'HLA-DRB1*13:17',
'HLA-DRB1*13:18', 'HLA-DRB1*13:19', 'HLA-DRB1*13:20', 'HLA-DRB1*13:21', 'HLA-DRB1*13:22',
'HLA-DRB1*13:23', 'HLA-DRB1*13:24', 'HLA-DRB1*13:26', 'HLA-DRB1*13:27', 'HLA-DRB1*13:29',
'HLA-DRB1*13:30', 'HLA-DRB1*13:31', 'HLA-DRB1*13:32', 'HLA-DRB1*13:33', 'HLA-DRB1*13:34',
'HLA-DRB1*13:35', 'HLA-DRB1*13:36', 'HLA-DRB1*13:37', 'HLA-DRB1*13:38', 'HLA-DRB1*13:39',
'HLA-DRB1*13:41', 'HLA-DRB1*13:42', 'HLA-DRB1*13:43', 'HLA-DRB1*13:44', 'HLA-DRB1*13:46',
'HLA-DRB1*13:47', 'HLA-DRB1*13:48', 'HLA-DRB1*13:49', 'HLA-DRB1*13:50', 'HLA-DRB1*13:51',
'HLA-DRB1*13:52', 'HLA-DRB1*13:53', 'HLA-DRB1*13:54', 'HLA-DRB1*13:55', 'HLA-DRB1*13:56',
'HLA-DRB1*13:57', 'HLA-DRB1*13:58', 'HLA-DRB1*13:59', 'HLA-DRB1*13:60', 'HLA-DRB1*13:61',
'HLA-DRB1*13:62', 'HLA-DRB1*13:63', 'HLA-DRB1*13:64', 'HLA-DRB1*13:65', 'HLA-DRB1*13:66',
'HLA-DRB1*13:67', 'HLA-DRB1*13:68', 'HLA-DRB1*13:69', 'HLA-DRB1*13:70', 'HLA-DRB1*13:71',
'HLA-DRB1*13:72', 'HLA-DRB1*13:73', 'HLA-DRB1*13:74', 'HLA-DRB1*13:75', 'HLA-DRB1*13:76',
'HLA-DRB1*13:77', 'HLA-DRB1*13:78', 'HLA-DRB1*13:79', 'HLA-DRB1*13:80', 'HLA-DRB1*13:81',
'HLA-DRB1*13:82', 'HLA-DRB1*13:83', 'HLA-DRB1*13:84', 'HLA-DRB1*13:85', 'HLA-DRB1*13:86',
'HLA-DRB1*13:87', 'HLA-DRB1*13:88', 'HLA-DRB1*13:89', 'HLA-DRB1*13:90', 'HLA-DRB1*13:91',
'HLA-DRB1*13:92', 'HLA-DRB1*13:93', 'HLA-DRB1*13:94', 'HLA-DRB1*13:95', 'HLA-DRB1*13:96',
'HLA-DRB1*13:97', 'HLA-DRB1*13:98', 'HLA-DRB1*13:99', 'HLA-DRB1*14:01', 'HLA-DRB1*14:02',
'HLA-DRB1*14:03', 'HLA-DRB1*14:04', 'HLA-DRB1*14:05', 'HLA-DRB1*14:06', 'HLA-DRB1*14:07',
'HLA-DRB1*14:08', 'HLA-DRB1*14:09', 'HLA-DRB1*14:10', 'HLA-DRB1*14:11', 'HLA-DRB1*14:12',
'HLA-DRB1*14:13', 'HLA-DRB1*14:14', 'HLA-DRB1*14:15', 'HLA-DRB1*14:16', 'HLA-DRB1*14:17',
'HLA-DRB1*14:18', 'HLA-DRB1*14:19', 'HLA-DRB1*14:20', 'HLA-DRB1*14:21', 'HLA-DRB1*14:22',
'HLA-DRB1*14:23', 'HLA-DRB1*14:24', 'HLA-DRB1*14:25', 'HLA-DRB1*14:26', 'HLA-DRB1*14:27',
'HLA-DRB1*14:28', 'HLA-DRB1*14:29', 'HLA-DRB1*14:30', 'HLA-DRB1*14:31', 'HLA-DRB1*14:32',
'HLA-DRB1*14:33', 'HLA-DRB1*14:34', 'HLA-DRB1*14:35', 'HLA-DRB1*14:36', 'HLA-DRB1*14:37',
'HLA-DRB1*14:38', 'HLA-DRB1*14:39', 'HLA-DRB1*14:40', 'HLA-DRB1*14:41', 'HLA-DRB1*14:42',
'HLA-DRB1*14:43', 'HLA-DRB1*14:44', 'HLA-DRB1*14:45', 'HLA-DRB1*14:46', 'HLA-DRB1*14:47',
'HLA-DRB1*14:48', 'HLA-DRB1*14:49', 'HLA-DRB1*14:50', 'HLA-DRB1*14:51', 'HLA-DRB1*14:52',
'HLA-DRB1*14:53', 'HLA-DRB1*14:54', 'HLA-DRB1*14:55', 'HLA-DRB1*14:56', 'HLA-DRB1*14:57',
'HLA-DRB1*14:58', 'HLA-DRB1*14:59', 'HLA-DRB1*14:60', 'HLA-DRB1*14:61', 'HLA-DRB1*14:62',
'HLA-DRB1*14:63', 'HLA-DRB1*14:64', 'HLA-DRB1*14:65', 'HLA-DRB1*14:67', 'HLA-DRB1*14:68',
'HLA-DRB1*14:69', 'HLA-DRB1*14:70', 'HLA-DRB1*14:71', 'HLA-DRB1*14:72', 'HLA-DRB1*14:73',
'HLA-DRB1*14:74', 'HLA-DRB1*14:75', 'HLA-DRB1*14:76', 'HLA-DRB1*14:77', 'HLA-DRB1*14:78',
'HLA-DRB1*14:79', 'HLA-DRB1*14:80', 'HLA-DRB1*14:81', 'HLA-DRB1*14:82', 'HLA-DRB1*14:83',
'HLA-DRB1*14:84', 'HLA-DRB1*14:85', 'HLA-DRB1*14:86', 'HLA-DRB1*14:87', 'HLA-DRB1*14:88',
'HLA-DRB1*14:89', 'HLA-DRB1*14:90', 'HLA-DRB1*14:91', 'HLA-DRB1*14:93', 'HLA-DRB1*14:94',
'HLA-DRB1*14:95', 'HLA-DRB1*14:96', 'HLA-DRB1*14:97', 'HLA-DRB1*14:98', 'HLA-DRB1*14:99',
'HLA-DRB1*15:01', 'HLA-DRB1*15:02', 'HLA-DRB1*15:03', 'HLA-DRB1*15:04', 'HLA-DRB1*15:05',
'HLA-DRB1*15:06', 'HLA-DRB1*15:07', 'HLA-DRB1*15:08', 'HLA-DRB1*15:09', 'HLA-DRB1*15:10',
'HLA-DRB1*15:11', 'HLA-DRB1*15:12', 'HLA-DRB1*15:13', 'HLA-DRB1*15:14', 'HLA-DRB1*15:15',
'HLA-DRB1*15:16', 'HLA-DRB1*15:18', 'HLA-DRB1*15:19', 'HLA-DRB1*15:20', 'HLA-DRB1*15:21',
'HLA-DRB1*15:22', 'HLA-DRB1*15:23', 'HLA-DRB1*15:24', 'HLA-DRB1*15:25', 'HLA-DRB1*15:26',
'HLA-DRB1*15:27', 'HLA-DRB1*15:28', 'HLA-DRB1*15:29', 'HLA-DRB1*15:30', 'HLA-DRB1*15:31',
'HLA-DRB1*15:32', 'HLA-DRB1*15:33', 'HLA-DRB1*15:34', 'HLA-DRB1*15:35', 'HLA-DRB1*15:36',
'HLA-DRB1*15:37', 'HLA-DRB1*15:38', 'HLA-DRB1*15:39', 'HLA-DRB1*15:40', 'HLA-DRB1*15:41',
'HLA-DRB1*15:42', 'HLA-DRB1*15:43', 'HLA-DRB1*15:44', 'HLA-DRB1*15:45', 'HLA-DRB1*15:46',
'HLA-DRB1*15:47', 'HLA-DRB1*15:48', 'HLA-DRB1*15:49', 'HLA-DRB1*16:01', 'HLA-DRB1*16:02',
'HLA-DRB1*16:03', 'HLA-DRB1*16:04', 'HLA-DRB1*16:05', 'HLA-DRB1*16:07', 'HLA-DRB1*16:08',
'HLA-DRB1*16:09', 'HLA-DRB1*16:10', 'HLA-DRB1*16:11', 'HLA-DRB1*16:12', 'HLA-DRB1*16:14',
'HLA-DRB1*16:15', 'HLA-DRB1*16:16', 'HLA-DRB3*01:01', 'HLA-DRB3*01:04', 'HLA-DRB3*01:05',
'HLA-DRB3*01:08', 'HLA-DRB3*01:09', 'HLA-DRB3*01:11', 'HLA-DRB3*01:12', 'HLA-DRB3*01:13',
'HLA-DRB3*01:14', 'HLA-DRB3*02:01', 'HLA-DRB3*02:02', 'HLA-DRB3*02:04', 'HLA-DRB3*02:05',
'HLA-DRB3*02:09', 'HLA-DRB3*02:10', 'HLA-DRB3*02:11', 'HLA-DRB3*02:12', 'HLA-DRB3*02:13',
'HLA-DRB3*02:14', 'HLA-DRB3*02:15', 'HLA-DRB3*02:16', 'HLA-DRB3*02:17', 'HLA-DRB3*02:18',
'HLA-DRB3*02:19', 'HLA-DRB3*02:20', 'HLA-DRB3*02:21', 'HLA-DRB3*02:22', 'HLA-DRB3*02:23',
'HLA-DRB3*02:24', 'HLA-DRB3*02:25', 'HLA-DRB3*03:01', 'HLA-DRB3*03:03', 'HLA-DRB4*01:01',
'HLA-DRB4*01:03', 'HLA-DRB4*01:04', 'HLA-DRB4*01:06', 'HLA-DRB4*01:07', 'HLA-DRB4*01:08',
'HLA-DRB5*01:01', 'HLA-DRB5*01:02', 'HLA-DRB5*01:03', 'HLA-DRB5*01:04', 'HLA-DRB5*01:05',
'HLA-DRB5*01:06', 'HLA-DRB5*01:08N', 'HLA-DRB5*01:11', 'HLA-DRB5*01:12', 'HLA-DRB5*01:13',
'HLA-DRB5*01:14', 'HLA-DRB5*02:02', 'HLA-DRB5*02:03', 'HLA-DRB5*02:04', 'HLA-DRB5*02:05',
'HLA-DPA1*01:03-DPB1*01:01', 'HLA-DPA1*01:03-DPB1*02:01', 'HLA-DPA1*01:03-DPB1*02:02', 'HLA-DPA1*01:03-DPB1*03:01', 'HLA-DPA1*01:03-DPB1*04:01',
'HLA-DPA1*01:03-DPB1*04:02', 'HLA-DPA1*01:03-DPB1*05:01', 'HLA-DPA1*01:03-DPB1*06:01', 'HLA-DPA1*01:03-DPB1*08:01', 'HLA-DPA1*01:03-DPB1*09:01',
'HLA-DPA1*01:03-DPB1*10:001', 'HLA-DPA1*01:03-DPB1*10:01', 'HLA-DPA1*01:03-DPB1*10:101', 'HLA-DPA1*01:03-DPB1*10:201', 'HLA-DPA1*01:03-DPB1*10:301',
'HLA-DPA1*01:03-DPB1*10:401', 'HLA-DPA1*01:03-DPB1*10:501', 'HLA-DPA1*01:03-DPB1*10:601', 'HLA-DPA1*01:03-DPB1*10:701', 'HLA-DPA1*01:03-DPB1*10:801',
'HLA-DPA1*01:03-DPB1*10:901', 'HLA-DPA1*01:03-DPB1*11:001', 'HLA-DPA1*01:03-DPB1*11:01', 'HLA-DPA1*01:03-DPB1*11:101', 'HLA-DPA1*01:03-DPB1*11:201',
'HLA-DPA1*01:03-DPB1*11:301', 'HLA-DPA1*01:03-DPB1*11:401', 'HLA-DPA1*01:03-DPB1*11:501', 'HLA-DPA1*01:03-DPB1*11:601', 'HLA-DPA1*01:03-DPB1*11:701',
'HLA-DPA1*01:03-DPB1*11:801', 'HLA-DPA1*01:03-DPB1*11:901', 'HLA-DPA1*01:03-DPB1*12:101', 'HLA-DPA1*01:03-DPB1*12:201', 'HLA-DPA1*01:03-DPB1*12:301',
'HLA-DPA1*01:03-DPB1*12:401', 'HLA-DPA1*01:03-DPB1*12:501', 'HLA-DPA1*01:03-DPB1*12:601', 'HLA-DPA1*01:03-DPB1*12:701', 'HLA-DPA1*01:03-DPB1*12:801',
'HLA-DPA1*01:03-DPB1*12:901', 'HLA-DPA1*01:03-DPB1*13:001', 'HLA-DPA1*01:03-DPB1*13:01', 'HLA-DPA1*01:03-DPB1*13:101', 'HLA-DPA1*01:03-DPB1*13:201',
'HLA-DPA1*01:03-DPB1*13:301', 'HLA-DPA1*01:03-DPB1*13:401', 'HLA-DPA1*01:03-DPB1*14:01', 'HLA-DPA1*01:03-DPB1*15:01', 'HLA-DPA1*01:03-DPB1*16:01',
'HLA-DPA1*01:03-DPB1*17:01', 'HLA-DPA1*01:03-DPB1*18:01', 'HLA-DPA1*01:03-DPB1*19:01', 'HLA-DPA1*01:03-DPB1*20:01', 'HLA-DPA1*01:03-DPB1*21:01',
'HLA-DPA1*01:03-DPB1*22:01', 'HLA-DPA1*01:03-DPB1*23:01', 'HLA-DPA1*01:03-DPB1*24:01', 'HLA-DPA1*01:03-DPB1*25:01', 'HLA-DPA1*01:03-DPB1*26:01',
'HLA-DPA1*01:03-DPB1*27:01', 'HLA-DPA1*01:03-DPB1*28:01', 'HLA-DPA1*01:03-DPB1*29:01', 'HLA-DPA1*01:03-DPB1*30:01', 'HLA-DPA1*01:03-DPB1*31:01',
'HLA-DPA1*01:03-DPB1*32:01', 'HLA-DPA1*01:03-DPB1*33:01', 'HLA-DPA1*01:03-DPB1*34:01', 'HLA-DPA1*01:03-DPB1*35:01', 'HLA-DPA1*01:03-DPB1*36:01',
'HLA-DPA1*01:03-DPB1*37:01', 'HLA-DPA1*01:03-DPB1*38:01', 'HLA-DPA1*01:03-DPB1*39:01', 'HLA-DPA1*01:03-DPB1*40:01', 'HLA-DPA1*01:03-DPB1*41:01',
'HLA-DPA1*01:03-DPB1*44:01', 'HLA-DPA1*01:03-DPB1*45:01', 'HLA-DPA1*01:03-DPB1*46:01', 'HLA-DPA1*01:03-DPB1*47:01', 'HLA-DPA1*01:03-DPB1*48:01',
'HLA-DPA1*01:03-DPB1*49:01', 'HLA-DPA1*01:03-DPB1*50:01', 'HLA-DPA1*01:03-DPB1*51:01', 'HLA-DPA1*01:03-DPB1*52:01', 'HLA-DPA1*01:03-DPB1*53:01',
'HLA-DPA1*01:03-DPB1*54:01', 'HLA-DPA1*01:03-DPB1*55:01', 'HLA-DPA1*01:03-DPB1*56:01', 'HLA-DPA1*01:03-DPB1*58:01', 'HLA-DPA1*01:03-DPB1*59:01',
'HLA-DPA1*01:03-DPB1*60:01', 'HLA-DPA1*01:03-DPB1*62:01', 'HLA-DPA1*01:03-DPB1*63:01', 'HLA-DPA1*01:03-DPB1*65:01', 'HLA-DPA1*01:03-DPB1*66:01',
'HLA-DPA1*01:03-DPB1*67:01', 'HLA-DPA1*01:03-DPB1*68:01', 'HLA-DPA1*01:03-DPB1*69:01', 'HLA-DPA1*01:03-DPB1*70:01', 'HLA-DPA1*01:03-DPB1*71:01',
'HLA-DPA1*01:03-DPB1*72:01', 'HLA-DPA1*01:03-DPB1*73:01', 'HLA-DPA1*01:03-DPB1*74:01', 'HLA-DPA1*01:03-DPB1*75:01', 'HLA-DPA1*01:03-DPB1*76:01',
'HLA-DPA1*01:03-DPB1*77:01', 'HLA-DPA1*01:03-DPB1*78:01', 'HLA-DPA1*01:03-DPB1*79:01', 'HLA-DPA1*01:03-DPB1*80:01', 'HLA-DPA1*01:03-DPB1*81:01',
'HLA-DPA1*01:03-DPB1*82:01', 'HLA-DPA1*01:03-DPB1*83:01', 'HLA-DPA1*01:03-DPB1*84:01', 'HLA-DPA1*01:03-DPB1*85:01', 'HLA-DPA1*01:03-DPB1*86:01',
'HLA-DPA1*01:03-DPB1*87:01', 'HLA-DPA1*01:03-DPB1*88:01', 'HLA-DPA1*01:03-DPB1*89:01', 'HLA-DPA1*01:03-DPB1*90:01', 'HLA-DPA1*01:03-DPB1*91:01',
'HLA-DPA1*01:03-DPB1*92:01', 'HLA-DPA1*01:03-DPB1*93:01', 'HLA-DPA1*01:03-DPB1*94:01', 'HLA-DPA1*01:03-DPB1*95:01', 'HLA-DPA1*01:03-DPB1*96:01',
'HLA-DPA1*01:03-DPB1*97:01', 'HLA-DPA1*01:03-DPB1*98:01', 'HLA-DPA1*01:03-DPB1*99:01', 'HLA-DPA1*01:04-DPB1*01:01', 'HLA-DPA1*01:04-DPB1*02:01',
'HLA-DPA1*01:04-DPB1*02:02', 'HLA-DPA1*01:04-DPB1*03:01', 'HLA-DPA1*01:04-DPB1*04:01', 'HLA-DPA1*01:04-DPB1*04:02', 'HLA-DPA1*01:04-DPB1*05:01',
'HLA-DPA1*01:04-DPB1*06:01', 'HLA-DPA1*01:04-DPB1*08:01', 'HLA-DPA1*01:04-DPB1*09:01', 'HLA-DPA1*01:04-DPB1*10:001', 'HLA-DPA1*01:04-DPB1*10:01',
'HLA-DPA1*01:04-DPB1*10:101', 'HLA-DPA1*01:04-DPB1*10:201', 'HLA-DPA1*01:04-DPB1*10:301', 'HLA-DPA1*01:04-DPB1*10:401', 'HLA-DPA1*01:04-DPB1*10:501',
'HLA-DPA1*01:04-DPB1*10:601', 'HLA-DPA1*01:04-DPB1*10:701', 'HLA-DPA1*01:04-DPB1*10:801', 'HLA-DPA1*01:04-DPB1*10:901', 'HLA-DPA1*01:04-DPB1*11:001',
'HLA-DPA1*01:04-DPB1*11:01', 'HLA-DPA1*01:04-DPB1*11:101', 'HLA-DPA1*01:04-DPB1*11:201', 'HLA-DPA1*01:04-DPB1*11:301', 'HLA-DPA1*01:04-DPB1*11:401',
'HLA-DPA1*01:04-DPB1*11:501', 'HLA-DPA1*01:04-DPB1*11:601', 'HLA-DPA1*01:04-DPB1*11:701', 'HLA-DPA1*01:04-DPB1*11:801', 'HLA-DPA1*01:04-DPB1*11:901',
'HLA-DPA1*01:04-DPB1*12:101', 'HLA-DPA1*01:04-DPB1*12:201', 'HLA-DPA1*01:04-DPB1*12:301', 'HLA-DPA1*01:04-DPB1*12:401', 'HLA-DPA1*01:04-DPB1*12:501',
'HLA-DPA1*01:04-DPB1*12:601', 'HLA-DPA1*01:04-DPB1*12:701', 'HLA-DPA1*01:04-DPB1*12:801', 'HLA-DPA1*01:04-DPB1*12:901', 'HLA-DPA1*01:04-DPB1*13:001',
'HLA-DPA1*01:04-DPB1*13:01', 'HLA-DPA1*01:04-DPB1*13:101', 'HLA-DPA1*01:04-DPB1*13:201', 'HLA-DPA1*01:04-DPB1*13:301', 'HLA-DPA1*01:04-DPB1*13:401',
'HLA-DPA1*01:04-DPB1*14:01', 'HLA-DPA1*01:04-DPB1*15:01', 'HLA-DPA1*01:04-DPB1*16:01', 'HLA-DPA1*01:04-DPB1*17:01', 'HLA-DPA1*01:04-DPB1*18:01',
'HLA-DPA1*01:04-DPB1*19:01', 'HLA-DPA1*01:04-DPB1*20:01', 'HLA-DPA1*01:04-DPB1*21:01', 'HLA-DPA1*01:04-DPB1*22:01', 'HLA-DPA1*01:04-DPB1*23:01',
'HLA-DPA1*01:04-DPB1*24:01', 'HLA-DPA1*01:04-DPB1*25:01', 'HLA-DPA1*01:04-DPB1*26:01', 'HLA-DPA1*01:04-DPB1*27:01', 'HLA-DPA1*01:04-DPB1*28:01',
'HLA-DPA1*01:04-DPB1*29:01', 'HLA-DPA1*01:04-DPB1*30:01', 'HLA-DPA1*01:04-DPB1*31:01', 'HLA-DPA1*01:04-DPB1*32:01', 'HLA-DPA1*01:04-DPB1*33:01',
'HLA-DPA1*01:04-DPB1*34:01', 'HLA-DPA1*01:04-DPB1*35:01', 'HLA-DPA1*01:04-DPB1*36:01', 'HLA-DPA1*01:04-DPB1*37:01', 'HLA-DPA1*01:04-DPB1*38:01',
'HLA-DPA1*01:04-DPB1*39:01', 'HLA-DPA1*01:04-DPB1*40:01', 'HLA-DPA1*01:04-DPB1*41:01', 'HLA-DPA1*01:04-DPB1*44:01', 'HLA-DPA1*01:04-DPB1*45:01',
'HLA-DPA1*01:04-DPB1*46:01', 'HLA-DPA1*01:04-DPB1*47:01', 'HLA-DPA1*01:04-DPB1*48:01', 'HLA-DPA1*01:04-DPB1*49:01', 'HLA-DPA1*01:04-DPB1*50:01',
'HLA-DPA1*01:04-DPB1*51:01', 'HLA-DPA1*01:04-DPB1*52:01', 'HLA-DPA1*01:04-DPB1*53:01', 'HLA-DPA1*01:04-DPB1*54:01', 'HLA-DPA1*01:04-DPB1*55:01',
'HLA-DPA1*01:04-DPB1*56:01', 'HLA-DPA1*01:04-DPB1*58:01', 'HLA-DPA1*01:04-DPB1*59:01', 'HLA-DPA1*01:04-DPB1*60:01', 'HLA-DPA1*01:04-DPB1*62:01',
'HLA-DPA1*01:04-DPB1*63:01', 'HLA-DPA1*01:04-DPB1*65:01', 'HLA-DPA1*01:04-DPB1*66:01', 'HLA-DPA1*01:04-DPB1*67:01', 'HLA-DPA1*01:04-DPB1*68:01',
'HLA-DPA1*01:04-DPB1*69:01', 'HLA-DPA1*01:04-DPB1*70:01', 'HLA-DPA1*01:04-DPB1*71:01', 'HLA-DPA1*01:04-DPB1*72:01', 'HLA-DPA1*01:04-DPB1*73:01',
'HLA-DPA1*01:04-DPB1*74:01', 'HLA-DPA1*01:04-DPB1*75:01', 'HLA-DPA1*01:04-DPB1*76:01', 'HLA-DPA1*01:04-DPB1*77:01', 'HLA-DPA1*01:04-DPB1*78:01',
'HLA-DPA1*01:04-DPB1*79:01', 'HLA-DPA1*01:04-DPB1*80:01', 'HLA-DPA1*01:04-DPB1*81:01', 'HLA-DPA1*01:04-DPB1*82:01', 'HLA-DPA1*01:04-DPB1*83:01',
'HLA-DPA1*01:04-DPB1*84:01', 'HLA-DPA1*01:04-DPB1*85:01', 'HLA-DPA1*01:04-DPB1*86:01', 'HLA-DPA1*01:04-DPB1*87:01', 'HLA-DPA1*01:04-DPB1*88:01',
'HLA-DPA1*01:04-DPB1*89:01', 'HLA-DPA1*01:04-DPB1*90:01', 'HLA-DPA1*01:04-DPB1*91:01', 'HLA-DPA1*01:04-DPB1*92:01', 'HLA-DPA1*01:04-DPB1*93:01',
'HLA-DPA1*01:04-DPB1*94:01', 'HLA-DPA1*01:04-DPB1*95:01', 'HLA-DPA1*01:04-DPB1*96:01', 'HLA-DPA1*01:04-DPB1*97:01', 'HLA-DPA1*01:04-DPB1*98:01',
'HLA-DPA1*01:04-DPB1*99:01', 'HLA-DPA1*01:05-DPB1*01:01', 'HLA-DPA1*01:05-DPB1*02:01', 'HLA-DPA1*01:05-DPB1*02:02', 'HLA-DPA1*01:05-DPB1*03:01',
'HLA-DPA1*01:05-DPB1*04:01', 'HLA-DPA1*01:05-DPB1*04:02', 'HLA-DPA1*01:05-DPB1*05:01', 'HLA-DPA1*01:05-DPB1*06:01', 'HLA-DPA1*01:05-DPB1*08:01',
'HLA-DPA1*01:05-DPB1*09:01', 'HLA-DPA1*01:05-DPB1*10:001', 'HLA-DPA1*01:05-DPB1*10:01', 'HLA-DPA1*01:05-DPB1*10:101', 'HLA-DPA1*01:05-DPB1*10:201',
'HLA-DPA1*01:05-DPB1*10:301', 'HLA-DPA1*01:05-DPB1*10:401', 'HLA-DPA1*01:05-DPB1*10:501', 'HLA-DPA1*01:05-DPB1*10:601', 'HLA-DPA1*01:05-DPB1*10:701',
'HLA-DPA1*01:05-DPB1*10:801', 'HLA-DPA1*01:05-DPB1*10:901', 'HLA-DPA1*01:05-DPB1*11:001', 'HLA-DPA1*01:05-DPB1*11:01', 'HLA-DPA1*01:05-DPB1*11:101',
'HLA-DPA1*01:05-DPB1*11:201', 'HLA-DPA1*01:05-DPB1*11:301', 'HLA-DPA1*01:05-DPB1*11:401', 'HLA-DPA1*01:05-DPB1*11:501', 'HLA-DPA1*01:05-DPB1*11:601',
'HLA-DPA1*01:05-DPB1*11:701', 'HLA-DPA1*01:05-DPB1*11:801', 'HLA-DPA1*01:05-DPB1*11:901', 'HLA-DPA1*01:05-DPB1*12:101', 'HLA-DPA1*01:05-DPB1*12:201',
'HLA-DPA1*01:05-DPB1*12:301', 'HLA-DPA1*01:05-DPB1*12:401', 'HLA-DPA1*01:05-DPB1*12:501', 'HLA-DPA1*01:05-DPB1*12:601', 'HLA-DPA1*01:05-DPB1*12:701',
'HLA-DPA1*01:05-DPB1*12:801', 'HLA-DPA1*01:05-DPB1*12:901', 'HLA-DPA1*01:05-DPB1*13:001', 'HLA-DPA1*01:05-DPB1*13:01', 'HLA-DPA1*01:05-DPB1*13:101',
'HLA-DPA1*01:05-DPB1*13:201', 'HLA-DPA1*01:05-DPB1*13:301', 'HLA-DPA1*01:05-DPB1*13:401', 'HLA-DPA1*01:05-DPB1*14:01', 'HLA-DPA1*01:05-DPB1*15:01',
'HLA-DPA1*01:05-DPB1*16:01', 'HLA-DPA1*01:05-DPB1*17:01', 'HLA-DPA1*01:05-DPB1*18:01', 'HLA-DPA1*01:05-DPB1*19:01', 'HLA-DPA1*01:05-DPB1*20:01',
'HLA-DPA1*01:05-DPB1*21:01', 'HLA-DPA1*01:05-DPB1*22:01', 'HLA-DPA1*01:05-DPB1*23:01', 'HLA-DPA1*01:05-DPB1*24:01', 'HLA-DPA1*01:05-DPB1*25:01',
'HLA-DPA1*01:05-DPB1*26:01', 'HLA-DPA1*01:05-DPB1*27:01', 'HLA-DPA1*01:05-DPB1*28:01', 'HLA-DPA1*01:05-DPB1*29:01', 'HLA-DPA1*01:05-DPB1*30:01',
'HLA-DPA1*01:05-DPB1*31:01', 'HLA-DPA1*01:05-DPB1*32:01', 'HLA-DPA1*01:05-DPB1*33:01', 'HLA-DPA1*01:05-DPB1*34:01', 'HLA-DPA1*01:05-DPB1*35:01',
'HLA-DPA1*01:05-DPB1*36:01', 'HLA-DPA1*01:05-DPB1*37:01', 'HLA-DPA1*01:05-DPB1*38:01', 'HLA-DPA1*01:05-DPB1*39:01', 'HLA-DPA1*01:05-DPB1*40:01',
'HLA-DPA1*01:05-DPB1*41:01', 'HLA-DPA1*01:05-DPB1*44:01', 'HLA-DPA1*01:05-DPB1*45:01', 'HLA-DPA1*01:05-DPB1*46:01', 'HLA-DPA1*01:05-DPB1*47:01',
'HLA-DPA1*01:05-DPB1*48:01', 'HLA-DPA1*01:05-DPB1*49:01', 'HLA-DPA1*01:05-DPB1*50:01', 'HLA-DPA1*01:05-DPB1*51:01', 'HLA-DPA1*01:05-DPB1*52:01',
'HLA-DPA1*01:05-DPB1*53:01', 'HLA-DPA1*01:05-DPB1*54:01', 'HLA-DPA1*01:05-DPB1*55:01', 'HLA-DPA1*01:05-DPB1*56:01', 'HLA-DPA1*01:05-DPB1*58:01',
'HLA-DPA1*01:05-DPB1*59:01', 'HLA-DPA1*01:05-DPB1*60:01', 'HLA-DPA1*01:05-DPB1*62:01', 'HLA-DPA1*01:05-DPB1*63:01', 'HLA-DPA1*01:05-DPB1*65:01',
'HLA-DPA1*01:05-DPB1*66:01', 'HLA-DPA1*01:05-DPB1*67:01', 'HLA-DPA1*01:05-DPB1*68:01', 'HLA-DPA1*01:05-DPB1*69:01', 'HLA-DPA1*01:05-DPB1*70:01',
'HLA-DPA1*01:05-DPB1*71:01', 'HLA-DPA1*01:05-DPB1*72:01', 'HLA-DPA1*01:05-DPB1*73:01', 'HLA-DPA1*01:05-DPB1*74:01', 'HLA-DPA1*01:05-DPB1*75:01',
'HLA-DPA1*01:05-DPB1*76:01', 'HLA-DPA1*01:05-DPB1*77:01', 'HLA-DPA1*01:05-DPB1*78:01', 'HLA-DPA1*01:05-DPB1*79:01', 'HLA-DPA1*01:05-DPB1*80:01',
'HLA-DPA1*01:05-DPB1*81:01', 'HLA-DPA1*01:05-DPB1*82:01', 'HLA-DPA1*01:05-DPB1*83:01', 'HLA-DPA1*01:05-DPB1*84:01', 'HLA-DPA1*01:05-DPB1*85:01',
'HLA-DPA1*01:05-DPB1*86:01', 'HLA-DPA1*01:05-DPB1*87:01', 'HLA-DPA1*01:05-DPB1*88:01', 'HLA-DPA1*01:05-DPB1*89:01', 'HLA-DPA1*01:05-DPB1*90:01',
'HLA-DPA1*01:05-DPB1*91:01', 'HLA-DPA1*01:05-DPB1*92:01', 'HLA-DPA1*01:05-DPB1*93:01', 'HLA-DPA1*01:05-DPB1*94:01', 'HLA-DPA1*01:05-DPB1*95:01',
'HLA-DPA1*01:05-DPB1*96:01', 'HLA-DPA1*01:05-DPB1*97:01', 'HLA-DPA1*01:05-DPB1*98:01', 'HLA-DPA1*01:05-DPB1*99:01', 'HLA-DPA1*01:06-DPB1*01:01',
'HLA-DPA1*01:06-DPB1*02:01', 'HLA-DPA1*01:06-DPB1*02:02', 'HLA-DPA1*01:06-DPB1*03:01', 'HLA-DPA1*01:06-DPB1*04:01', 'HLA-DPA1*01:06-DPB1*04:02',
'HLA-DPA1*01:06-DPB1*05:01', 'HLA-DPA1*01:06-DPB1*06:01', 'HLA-DPA1*01:06-DPB1*08:01', 'HLA-DPA1*01:06-DPB1*09:01', 'HLA-DPA1*01:06-DPB1*10:001',
'HLA-DPA1*01:06-DPB1*10:01', 'HLA-DPA1*01:06-DPB1*10:101', 'HLA-DPA1*01:06-DPB1*10:201', 'HLA-DPA1*01:06-DPB1*10:301', 'HLA-DPA1*01:06-DPB1*10:401',
'HLA-DPA1*01:06-DPB1*10:501', 'HLA-DPA1*01:06-DPB1*10:601', 'HLA-DPA1*01:06-DPB1*10:701', 'HLA-DPA1*01:06-DPB1*10:801', 'HLA-DPA1*01:06-DPB1*10:901',
'HLA-DPA1*01:06-DPB1*11:001', 'HLA-DPA1*01:06-DPB1*11:01', 'HLA-DPA1*01:06-DPB1*11:101', 'HLA-DPA1*01:06-DPB1*11:201', 'HLA-DPA1*01:06-DPB1*11:301',
'HLA-DPA1*01:06-DPB1*11:401', 'HLA-DPA1*01:06-DPB1*11:501', 'HLA-DPA1*01:06-DPB1*11:601', 'HLA-DPA1*01:06-DPB1*11:701', 'HLA-DPA1*01:06-DPB1*11:801',
'HLA-DPA1*01:06-DPB1*11:901', 'HLA-DPA1*01:06-DPB1*12:101', 'HLA-DPA1*01:06-DPB1*12:201', 'HLA-DPA1*01:06-DPB1*12:301', 'HLA-DPA1*01:06-DPB1*12:401',
'HLA-DPA1*01:06-DPB1*12:501', 'HLA-DPA1*01:06-DPB1*12:601', 'HLA-DPA1*01:06-DPB1*12:701', 'HLA-DPA1*01:06-DPB1*12:801', 'HLA-DPA1*01:06-DPB1*12:901',
'HLA-DPA1*01:06-DPB1*13:001', 'HLA-DPA1*01:06-DPB1*13:01', 'HLA-DPA1*01:06-DPB1*13:101', 'HLA-DPA1*01:06-DPB1*13:201', 'HLA-DPA1*01:06-DPB1*13:301',
'HLA-DPA1*01:06-DPB1*13:401', 'HLA-DPA1*01:06-DPB1*14:01', 'HLA-DPA1*01:06-DPB1*15:01', 'HLA-DPA1*01:06-DPB1*16:01', 'HLA-DPA1*01:06-DPB1*17:01',
'HLA-DPA1*01:06-DPB1*18:01', 'HLA-DPA1*01:06-DPB1*19:01', 'HLA-DPA1*01:06-DPB1*20:01', 'HLA-DPA1*01:06-DPB1*21:01', 'HLA-DPA1*01:06-DPB1*22:01',
'HLA-DPA1*01:06-DPB1*23:01', 'HLA-DPA1*01:06-DPB1*24:01', 'HLA-DPA1*01:06-DPB1*25:01', 'HLA-DPA1*01:06-DPB1*26:01', 'HLA-DPA1*01:06-DPB1*27:01',
'HLA-DPA1*01:06-DPB1*28:01', 'HLA-DPA1*01:06-DPB1*29:01', 'HLA-DPA1*01:06-DPB1*30:01', 'HLA-DPA1*01:06-DPB1*31:01', 'HLA-DPA1*01:06-DPB1*32:01',
'HLA-DPA1*01:06-DPB1*33:01', 'HLA-DPA1*01:06-DPB1*34:01', 'HLA-DPA1*01:06-DPB1*35:01', 'HLA-DPA1*01:06-DPB1*36:01', 'HLA-DPA1*01:06-DPB1*37:01',
'HLA-DPA1*01:06-DPB1*38:01', 'HLA-DPA1*01:06-DPB1*39:01', 'HLA-DPA1*01:06-DPB1*40:01', 'HLA-DPA1*01:06-DPB1*41:01', 'HLA-DPA1*01:06-DPB1*44:01',
'HLA-DPA1*01:06-DPB1*45:01', 'HLA-DPA1*01:06-DPB1*46:01', 'HLA-DPA1*01:06-DPB1*47:01', 'HLA-DPA1*01:06-DPB1*48:01', 'HLA-DPA1*01:06-DPB1*49:01',
'HLA-DPA1*01:06-DPB1*50:01', 'HLA-DPA1*01:06-DPB1*51:01', 'HLA-DPA1*01:06-DPB1*52:01', 'HLA-DPA1*01:06-DPB1*53:01', 'HLA-DPA1*01:06-DPB1*54:01',
'HLA-DPA1*01:06-DPB1*55:01', 'HLA-DPA1*01:06-DPB1*56:01', 'HLA-DPA1*01:06-DPB1*58:01', 'HLA-DPA1*01:06-DPB1*59:01', 'HLA-DPA1*01:06-DPB1*60:01',
'HLA-DPA1*01:06-DPB1*62:01', 'HLA-DPA1*01:06-DPB1*63:01', 'HLA-DPA1*01:06-DPB1*65:01', 'HLA-DPA1*01:06-DPB1*66:01', 'HLA-DPA1*01:06-DPB1*67:01',
'HLA-DPA1*01:06-DPB1*68:01', 'HLA-DPA1*01:06-DPB1*69:01', 'HLA-DPA1*01:06-DPB1*70:01', 'HLA-DPA1*01:06-DPB1*71:01', 'HLA-DPA1*01:06-DPB1*72:01',
'HLA-DPA1*01:06-DPB1*73:01', 'HLA-DPA1*01:06-DPB1*74:01', 'HLA-DPA1*01:06-DPB1*75:01', 'HLA-DPA1*01:06-DPB1*76:01', 'HLA-DPA1*01:06-DPB1*77:01',
'HLA-DPA1*01:06-DPB1*78:01', 'HLA-DPA1*01:06-DPB1*79:01', 'HLA-DPA1*01:06-DPB1*80:01', 'HLA-DPA1*01:06-DPB1*81:01', 'HLA-DPA1*01:06-DPB1*82:01',
'HLA-DPA1*01:06-DPB1*83:01', 'HLA-DPA1*01:06-DPB1*84:01', 'HLA-DPA1*01:06-DPB1*85:01', 'HLA-DPA1*01:06-DPB1*86:01', 'HLA-DPA1*01:06-DPB1*87:01',
'HLA-DPA1*01:06-DPB1*88:01', 'HLA-DPA1*01:06-DPB1*89:01', 'HLA-DPA1*01:06-DPB1*90:01', 'HLA-DPA1*01:06-DPB1*91:01', 'HLA-DPA1*01:06-DPB1*92:01',
'HLA-DPA1*01:06-DPB1*93:01', 'HLA-DPA1*01:06-DPB1*94:01', 'HLA-DPA1*01:06-DPB1*95:01', 'HLA-DPA1*01:06-DPB1*96:01', 'HLA-DPA1*01:06-DPB1*97:01',
'HLA-DPA1*01:06-DPB1*98:01', 'HLA-DPA1*01:06-DPB1*99:01', 'HLA-DPA1*01:07-DPB1*01:01', 'HLA-DPA1*01:07-DPB1*02:01', 'HLA-DPA1*01:07-DPB1*02:02',
'HLA-DPA1*01:07-DPB1*03:01', 'HLA-DPA1*01:07-DPB1*04:01', 'HLA-DPA1*01:07-DPB1*04:02', 'HLA-DPA1*01:07-DPB1*05:01', 'HLA-DPA1*01:07-DPB1*06:01',
'HLA-DPA1*01:07-DPB1*08:01', 'HLA-DPA1*01:07-DPB1*09:01', 'HLA-DPA1*01:07-DPB1*10:001', 'HLA-DPA1*01:07-DPB1*10:01', 'HLA-DPA1*01:07-DPB1*10:101',
'HLA-DPA1*01:07-DPB1*10:201', 'HLA-DPA1*01:07-DPB1*10:301', 'HLA-DPA1*01:07-DPB1*10:401', 'HLA-DPA1*01:07-DPB1*10:501', 'HLA-DPA1*01:07-DPB1*10:601',
'HLA-DPA1*01:07-DPB1*10:701', 'HLA-DPA1*01:07-DPB1*10:801', 'HLA-DPA1*01:07-DPB1*10:901', 'HLA-DPA1*01:07-DPB1*11:001', 'HLA-DPA1*01:07-DPB1*11:01',
'HLA-DPA1*01:07-DPB1*11:101', 'HLA-DPA1*01:07-DPB1*11:201', 'HLA-DPA1*01:07-DPB1*11:301', 'HLA-DPA1*01:07-DPB1*11:401', 'HLA-DPA1*01:07-DPB1*11:501',
'HLA-DPA1*01:07-DPB1*11:601', 'HLA-DPA1*01:07-DPB1*11:701', 'HLA-DPA1*01:07-DPB1*11:801', 'HLA-DPA1*01:07-DPB1*11:901', 'HLA-DPA1*01:07-DPB1*12:101',
'HLA-DPA1*01:07-DPB1*12:201', 'HLA-DPA1*01:07-DPB1*12:301', 'HLA-DPA1*01:07-DPB1*12:401', 'HLA-DPA1*01:07-DPB1*12:501', 'HLA-DPA1*01:07-DPB1*12:601',
'HLA-DPA1*01:07-DPB1*12:701', 'HLA-DPA1*01:07-DPB1*12:801', 'HLA-DPA1*01:07-DPB1*12:901', 'HLA-DPA1*01:07-DPB1*13:001', 'HLA-DPA1*01:07-DPB1*13:01',
'HLA-DPA1*01:07-DPB1*13:101', 'HLA-DPA1*01:07-DPB1*13:201', 'HLA-DPA1*01:07-DPB1*13:301', 'HLA-DPA1*01:07-DPB1*13:401', 'HLA-DPA1*01:07-DPB1*14:01',
'HLA-DPA1*01:07-DPB1*15:01', 'HLA-DPA1*01:07-DPB1*16:01', 'HLA-DPA1*01:07-DPB1*17:01', 'HLA-DPA1*01:07-DPB1*18:01', 'HLA-DPA1*01:07-DPB1*19:01',
'HLA-DPA1*01:07-DPB1*20:01', 'HLA-DPA1*01:07-DPB1*21:01', 'HLA-DPA1*01:07-DPB1*22:01', 'HLA-DPA1*01:07-DPB1*23:01', 'HLA-DPA1*01:07-DPB1*24:01',
'HLA-DPA1*01:07-DPB1*25:01', 'HLA-DPA1*01:07-DPB1*26:01', 'HLA-DPA1*01:07-DPB1*27:01', 'HLA-DPA1*01:07-DPB1*28:01', 'HLA-DPA1*01:07-DPB1*29:01',
'HLA-DPA1*01:07-DPB1*30:01', 'HLA-DPA1*01:07-DPB1*31:01', 'HLA-DPA1*01:07-DPB1*32:01', 'HLA-DPA1*01:07-DPB1*33:01', 'HLA-DPA1*01:07-DPB1*34:01',
'HLA-DPA1*01:07-DPB1*35:01', 'HLA-DPA1*01:07-DPB1*36:01', 'HLA-DPA1*01:07-DPB1*37:01', 'HLA-DPA1*01:07-DPB1*38:01', 'HLA-DPA1*01:07-DPB1*39:01',
'HLA-DPA1*01:07-DPB1*40:01', 'HLA-DPA1*01:07-DPB1*41:01', 'HLA-DPA1*01:07-DPB1*44:01', 'HLA-DPA1*01:07-DPB1*45:01', 'HLA-DPA1*01:07-DPB1*46:01',
'HLA-DPA1*01:07-DPB1*47:01', 'HLA-DPA1*01:07-DPB1*48:01', 'HLA-DPA1*01:07-DPB1*49:01', 'HLA-DPA1*01:07-DPB1*50:01', 'HLA-DPA1*01:07-DPB1*51:01',
'HLA-DPA1*01:07-DPB1*52:01', 'HLA-DPA1*01:07-DPB1*53:01', 'HLA-DPA1*01:07-DPB1*54:01', 'HLA-DPA1*01:07-DPB1*55:01', 'HLA-DPA1*01:07-DPB1*56:01',
'HLA-DPA1*01:07-DPB1*58:01', 'HLA-DPA1*01:07-DPB1*59:01', 'HLA-DPA1*01:07-DPB1*60:01', 'HLA-DPA1*01:07-DPB1*62:01', 'HLA-DPA1*01:07-DPB1*63:01',
'HLA-DPA1*01:07-DPB1*65:01', 'HLA-DPA1*01:07-DPB1*66:01', 'HLA-DPA1*01:07-DPB1*67:01', 'HLA-DPA1*01:07-DPB1*68:01', 'HLA-DPA1*01:07-DPB1*69:01',
'HLA-DPA1*01:07-DPB1*70:01', 'HLA-DPA1*01:07-DPB1*71:01', 'HLA-DPA1*01:07-DPB1*72:01', 'HLA-DPA1*01:07-DPB1*73:01', 'HLA-DPA1*01:07-DPB1*74:01',
'HLA-DPA1*01:07-DPB1*75:01', 'HLA-DPA1*01:07-DPB1*76:01', 'HLA-DPA1*01:07-DPB1*77:01', 'HLA-DPA1*01:07-DPB1*78:01', 'HLA-DPA1*01:07-DPB1*79:01',
'HLA-DPA1*01:07-DPB1*80:01', 'HLA-DPA1*01:07-DPB1*81:01', 'HLA-DPA1*01:07-DPB1*82:01', 'HLA-DPA1*01:07-DPB1*83:01', 'HLA-DPA1*01:07-DPB1*84:01',
'HLA-DPA1*01:07-DPB1*85:01', 'HLA-DPA1*01:07-DPB1*86:01', 'HLA-DPA1*01:07-DPB1*87:01', 'HLA-DPA1*01:07-DPB1*88:01', 'HLA-DPA1*01:07-DPB1*89:01',
'HLA-DPA1*01:07-DPB1*90:01', 'HLA-DPA1*01:07-DPB1*91:01', 'HLA-DPA1*01:07-DPB1*92:01', 'HLA-DPA1*01:07-DPB1*93:01', 'HLA-DPA1*01:07-DPB1*94:01',
'HLA-DPA1*01:07-DPB1*95:01', 'HLA-DPA1*01:07-DPB1*96:01', 'HLA-DPA1*01:07-DPB1*97:01', 'HLA-DPA1*01:07-DPB1*98:01', 'HLA-DPA1*01:07-DPB1*99:01',
'HLA-DPA1*01:08-DPB1*01:01', 'HLA-DPA1*01:08-DPB1*02:01', 'HLA-DPA1*01:08-DPB1*02:02', 'HLA-DPA1*01:08-DPB1*03:01', 'HLA-DPA1*01:08-DPB1*04:01',
'HLA-DPA1*01:08-DPB1*04:02', 'HLA-DPA1*01:08-DPB1*05:01', 'HLA-DPA1*01:08-DPB1*06:01', 'HLA-DPA1*01:08-DPB1*08:01', 'HLA-DPA1*01:08-DPB1*09:01',
'HLA-DPA1*01:08-DPB1*10:001', 'HLA-DPA1*01:08-DPB1*10:01', 'HLA-DPA1*01:08-DPB1*10:101', 'HLA-DPA1*01:08-DPB1*10:201', 'HLA-DPA1*01:08-DPB1*10:301',
'HLA-DPA1*01:08-DPB1*10:401', 'HLA-DPA1*01:08-DPB1*10:501', 'HLA-DPA1*01:08-DPB1*10:601', 'HLA-DPA1*01:08-DPB1*10:701', 'HLA-DPA1*01:08-DPB1*10:801',
'HLA-DPA1*01:08-DPB1*10:901', 'HLA-DPA1*01:08-DPB1*11:001', 'HLA-DPA1*01:08-DPB1*11:01', 'HLA-DPA1*01:08-DPB1*11:101', 'HLA-DPA1*01:08-DPB1*11:201',
'HLA-DPA1*01:08-DPB1*11:301', 'HLA-DPA1*01:08-DPB1*11:401', 'HLA-DPA1*01:08-DPB1*11:501', 'HLA-DPA1*01:08-DPB1*11:601', 'HLA-DPA1*01:08-DPB1*11:701',
'HLA-DPA1*01:08-DPB1*11:801', 'HLA-DPA1*01:08-DPB1*11:901', 'HLA-DPA1*01:08-DPB1*12:101', 'HLA-DPA1*01:08-DPB1*12:201', 'HLA-DPA1*01:08-DPB1*12:301',
'HLA-DPA1*01:08-DPB1*12:401', 'HLA-DPA1*01:08-DPB1*12:501', 'HLA-DPA1*01:08-DPB1*12:601', 'HLA-DPA1*01:08-DPB1*12:701', 'HLA-DPA1*01:08-DPB1*12:801',
'HLA-DPA1*01:08-DPB1*12:901', 'HLA-DPA1*01:08-DPB1*13:001', 'HLA-DPA1*01:08-DPB1*13:01', 'HLA-DPA1*01:08-DPB1*13:101', 'HLA-DPA1*01:08-DPB1*13:201',
'HLA-DPA1*01:08-DPB1*13:301', 'HLA-DPA1*01:08-DPB1*13:401', 'HLA-DPA1*01:08-DPB1*14:01', 'HLA-DPA1*01:08-DPB1*15:01', 'HLA-DPA1*01:08-DPB1*16:01',
'HLA-DPA1*01:08-DPB1*17:01', 'HLA-DPA1*01:08-DPB1*18:01', 'HLA-DPA1*01:08-DPB1*19:01', 'HLA-DPA1*01:08-DPB1*20:01', 'HLA-DPA1*01:08-DPB1*21:01',
'HLA-DPA1*01:08-DPB1*22:01', 'HLA-DPA1*01:08-DPB1*23:01', 'HLA-DPA1*01:08-DPB1*24:01', 'HLA-DPA1*01:08-DPB1*25:01', 'HLA-DPA1*01:08-DPB1*26:01',
'HLA-DPA1*01:08-DPB1*27:01', 'HLA-DPA1*01:08-DPB1*28:01', 'HLA-DPA1*01:08-DPB1*29:01', 'HLA-DPA1*01:08-DPB1*30:01', 'HLA-DPA1*01:08-DPB1*31:01',
'HLA-DPA1*01:08-DPB1*32:01', 'HLA-DPA1*01:08-DPB1*33:01', 'HLA-DPA1*01:08-DPB1*34:01', 'HLA-DPA1*01:08-DPB1*35:01', 'HLA-DPA1*01:08-DPB1*36:01',
'HLA-DPA1*01:08-DPB1*37:01', 'HLA-DPA1*01:08-DPB1*38:01', 'HLA-DPA1*01:08-DPB1*39:01', 'HLA-DPA1*01:08-DPB1*40:01', 'HLA-DPA1*01:08-DPB1*41:01',
'HLA-DPA1*01:08-DPB1*44:01', 'HLA-DPA1*01:08-DPB1*45:01', 'HLA-DPA1*01:08-DPB1*46:01', 'HLA-DPA1*01:08-DPB1*47:01', 'HLA-DPA1*01:08-DPB1*48:01',
'HLA-DPA1*01:08-DPB1*49:01', 'HLA-DPA1*01:08-DPB1*50:01', 'HLA-DPA1*01:08-DPB1*51:01', 'HLA-DPA1*01:08-DPB1*52:01', 'HLA-DPA1*01:08-DPB1*53:01',
'HLA-DPA1*01:08-DPB1*54:01', 'HLA-DPA1*01:08-DPB1*55:01', 'HLA-DPA1*01:08-DPB1*56:01', 'HLA-DPA1*01:08-DPB1*58:01', 'HLA-DPA1*01:08-DPB1*59:01',
'HLA-DPA1*01:08-DPB1*60:01', 'HLA-DPA1*01:08-DPB1*62:01', 'HLA-DPA1*01:08-DPB1*63:01', 'HLA-DPA1*01:08-DPB1*65:01', 'HLA-DPA1*01:08-DPB1*66:01',
'HLA-DPA1*01:08-DPB1*67:01', 'HLA-DPA1*01:08-DPB1*68:01', 'HLA-DPA1*01:08-DPB1*69:01', 'HLA-DPA1*01:08-DPB1*70:01', 'HLA-DPA1*01:08-DPB1*71:01',
'HLA-DPA1*01:08-DPB1*72:01', 'HLA-DPA1*01:08-DPB1*73:01', 'HLA-DPA1*01:08-DPB1*74:01', 'HLA-DPA1*01:08-DPB1*75:01', 'HLA-DPA1*01:08-DPB1*76:01',
'HLA-DPA1*01:08-DPB1*77:01', 'HLA-DPA1*01:08-DPB1*78:01', 'HLA-DPA1*01:08-DPB1*79:01', 'HLA-DPA1*01:08-DPB1*80:01', 'HLA-DPA1*01:08-DPB1*81:01',
'HLA-DPA1*01:08-DPB1*82:01', 'HLA-DPA1*01:08-DPB1*83:01', 'HLA-DPA1*01:08-DPB1*84:01', 'HLA-DPA1*01:08-DPB1*85:01', 'HLA-DPA1*01:08-DPB1*86:01',
'HLA-DPA1*01:08-DPB1*87:01', 'HLA-DPA1*01:08-DPB1*88:01', 'HLA-DPA1*01:08-DPB1*89:01', 'HLA-DPA1*01:08-DPB1*90:01', 'HLA-DPA1*01:08-DPB1*91:01',
'HLA-DPA1*01:08-DPB1*92:01', 'HLA-DPA1*01:08-DPB1*93:01', 'HLA-DPA1*01:08-DPB1*94:01', 'HLA-DPA1*01:08-DPB1*95:01', 'HLA-DPA1*01:08-DPB1*96:01',
'HLA-DPA1*01:08-DPB1*97:01', 'HLA-DPA1*01:08-DPB1*98:01', 'HLA-DPA1*01:08-DPB1*99:01', 'HLA-DPA1*01:09-DPB1*01:01', 'HLA-DPA1*01:09-DPB1*02:01',
'HLA-DPA1*01:09-DPB1*02:02', 'HLA-DPA1*01:09-DPB1*03:01', 'HLA-DPA1*01:09-DPB1*04:01', 'HLA-DPA1*01:09-DPB1*04:02', 'HLA-DPA1*01:09-DPB1*05:01',
'HLA-DPA1*01:09-DPB1*06:01', 'HLA-DPA1*01:09-DPB1*08:01', 'HLA-DPA1*01:09-DPB1*09:01', 'HLA-DPA1*01:09-DPB1*10:001', 'HLA-DPA1*01:09-DPB1*10:01',
'HLA-DPA1*01:09-DPB1*10:101', 'HLA-DPA1*01:09-DPB1*10:201', 'HLA-DPA1*01:09-DPB1*10:301', 'HLA-DPA1*01:09-DPB1*10:401', 'HLA-DPA1*01:09-DPB1*10:501',
'HLA-DPA1*01:09-DPB1*10:601', 'HLA-DPA1*01:09-DPB1*10:701', 'HLA-DPA1*01:09-DPB1*10:801', 'HLA-DPA1*01:09-DPB1*10:901', 'HLA-DPA1*01:09-DPB1*11:001',
'HLA-DPA1*01:09-DPB1*11:01', 'HLA-DPA1*01:09-DPB1*11:101', 'HLA-DPA1*01:09-DPB1*11:201', 'HLA-DPA1*01:09-DPB1*11:301', 'HLA-DPA1*01:09-DPB1*11:401',
'HLA-DPA1*01:09-DPB1*11:501', 'HLA-DPA1*01:09-DPB1*11:601', 'HLA-DPA1*01:09-DPB1*11:701', 'HLA-DPA1*01:09-DPB1*11:801', 'HLA-DPA1*01:09-DPB1*11:901',
'HLA-DPA1*01:09-DPB1*12:101', 'HLA-DPA1*01:09-DPB1*12:201', 'HLA-DPA1*01:09-DPB1*12:301', 'HLA-DPA1*01:09-DPB1*12:401', 'HLA-DPA1*01:09-DPB1*12:501',
'HLA-DPA1*01:09-DPB1*12:601', 'HLA-DPA1*01:09-DPB1*12:701', 'HLA-DPA1*01:09-DPB1*12:801', 'HLA-DPA1*01:09-DPB1*12:901', 'HLA-DPA1*01:09-DPB1*13:001',
'HLA-DPA1*01:09-DPB1*13:01', 'HLA-DPA1*01:09-DPB1*13:101', 'HLA-DPA1*01:09-DPB1*13:201', 'HLA-DPA1*01:09-DPB1*13:301', 'HLA-DPA1*01:09-DPB1*13:401',
'HLA-DPA1*01:09-DPB1*14:01', 'HLA-DPA1*01:09-DPB1*15:01', 'HLA-DPA1*01:09-DPB1*16:01', 'HLA-DPA1*01:09-DPB1*17:01', 'HLA-DPA1*01:09-DPB1*18:01',
'HLA-DPA1*01:09-DPB1*19:01', 'HLA-DPA1*01:09-DPB1*20:01', 'HLA-DPA1*01:09-DPB1*21:01', 'HLA-DPA1*01:09-DPB1*22:01', 'HLA-DPA1*01:09-DPB1*23:01',
'HLA-DPA1*01:09-DPB1*24:01', 'HLA-DPA1*01:09-DPB1*25:01', 'HLA-DPA1*01:09-DPB1*26:01', 'HLA-DPA1*01:09-DPB1*27:01', 'HLA-DPA1*01:09-DPB1*28:01',
'HLA-DPA1*01:09-DPB1*29:01', 'HLA-DPA1*01:09-DPB1*30:01', 'HLA-DPA1*01:09-DPB1*31:01', 'HLA-DPA1*01:09-DPB1*32:01', 'HLA-DPA1*01:09-DPB1*33:01',
'HLA-DPA1*01:09-DPB1*34:01', 'HLA-DPA1*01:09-DPB1*35:01', 'HLA-DPA1*01:09-DPB1*36:01', 'HLA-DPA1*01:09-DPB1*37:01', 'HLA-DPA1*01:09-DPB1*38:01',
'HLA-DPA1*01:09-DPB1*39:01', 'HLA-DPA1*01:09-DPB1*40:01', 'HLA-DPA1*01:09-DPB1*41:01', 'HLA-DPA1*01:09-DPB1*44:01', 'HLA-DPA1*01:09-DPB1*45:01',
'HLA-DPA1*01:09-DPB1*46:01', 'HLA-DPA1*01:09-DPB1*47:01', 'HLA-DPA1*01:09-DPB1*48:01', 'HLA-DPA1*01:09-DPB1*49:01', 'HLA-DPA1*01:09-DPB1*50:01',
'HLA-DPA1*01:09-DPB1*51:01', 'HLA-DPA1*01:09-DPB1*52:01', 'HLA-DPA1*01:09-DPB1*53:01', 'HLA-DPA1*01:09-DPB1*54:01', 'HLA-DPA1*01:09-DPB1*55:01',
'HLA-DPA1*01:09-DPB1*56:01', 'HLA-DPA1*01:09-DPB1*58:01', 'HLA-DPA1*01:09-DPB1*59:01', 'HLA-DPA1*01:09-DPB1*60:01', 'HLA-DPA1*01:09-DPB1*62:01',
'HLA-DPA1*01:09-DPB1*63:01', 'HLA-DPA1*01:09-DPB1*65:01', 'HLA-DPA1*01:09-DPB1*66:01', 'HLA-DPA1*01:09-DPB1*67:01', 'HLA-DPA1*01:09-DPB1*68:01',
'HLA-DPA1*01:09-DPB1*69:01', 'HLA-DPA1*01:09-DPB1*70:01', 'HLA-DPA1*01:09-DPB1*71:01', 'HLA-DPA1*01:09-DPB1*72:01', 'HLA-DPA1*01:09-DPB1*73:01',
'HLA-DPA1*01:09-DPB1*74:01', 'HLA-DPA1*01:09-DPB1*75:01', 'HLA-DPA1*01:09-DPB1*76:01', 'HLA-DPA1*01:09-DPB1*77:01', 'HLA-DPA1*01:09-DPB1*78:01',
'HLA-DPA1*01:09-DPB1*79:01', 'HLA-DPA1*01:09-DPB1*80:01', 'HLA-DPA1*01:09-DPB1*81:01', 'HLA-DPA1*01:09-DPB1*82:01', 'HLA-DPA1*01:09-DPB1*83:01',
'HLA-DPA1*01:09-DPB1*84:01', 'HLA-DPA1*01:09-DPB1*85:01', 'HLA-DPA1*01:09-DPB1*86:01', 'HLA-DPA1*01:09-DPB1*87:01', 'HLA-DPA1*01:09-DPB1*88:01',
'HLA-DPA1*01:09-DPB1*89:01', 'HLA-DPA1*01:09-DPB1*90:01', 'HLA-DPA1*01:09-DPB1*91:01', 'HLA-DPA1*01:09-DPB1*92:01', 'HLA-DPA1*01:09-DPB1*93:01',
'HLA-DPA1*01:09-DPB1*94:01', 'HLA-DPA1*01:09-DPB1*95:01', 'HLA-DPA1*01:09-DPB1*96:01', 'HLA-DPA1*01:09-DPB1*97:01', 'HLA-DPA1*01:09-DPB1*98:01',
'HLA-DPA1*01:09-DPB1*99:01', 'HLA-DPA1*01:10-DPB1*01:01', 'HLA-DPA1*01:10-DPB1*02:01', 'HLA-DPA1*01:10-DPB1*02:02', 'HLA-DPA1*01:10-DPB1*03:01',
'HLA-DPA1*01:10-DPB1*04:01', 'HLA-DPA1*01:10-DPB1*04:02', 'HLA-DPA1*01:10-DPB1*05:01', 'HLA-DPA1*01:10-DPB1*06:01', 'HLA-DPA1*01:10-DPB1*08:01',
'HLA-DPA1*01:10-DPB1*09:01', 'HLA-DPA1*01:10-DPB1*10:001', 'HLA-DPA1*01:10-DPB1*10:01', 'HLA-DPA1*01:10-DPB1*10:101', 'HLA-DPA1*01:10-DPB1*10:201',
'HLA-DPA1*01:10-DPB1*10:301', 'HLA-DPA1*01:10-DPB1*10:401', 'HLA-DPA1*01:10-DPB1*10:501', 'HLA-DPA1*01:10-DPB1*10:601', 'HLA-DPA1*01:10-DPB1*10:701',
'HLA-DPA1*01:10-DPB1*10:801', 'HLA-DPA1*01:10-DPB1*10:901', 'HLA-DPA1*01:10-DPB1*11:001', 'HLA-DPA1*01:10-DPB1*11:01', 'HLA-DPA1*01:10-DPB1*11:101',
'HLA-DPA1*01:10-DPB1*11:201', 'HLA-DPA1*01:10-DPB1*11:301', 'HLA-DPA1*01:10-DPB1*11:401', 'HLA-DPA1*01:10-DPB1*11:501', 'HLA-DPA1*01:10-DPB1*11:601',
'HLA-DPA1*01:10-DPB1*11:701', 'HLA-DPA1*01:10-DPB1*11:801', 'HLA-DPA1*01:10-DPB1*11:901', 'HLA-DPA1*01:10-DPB1*12:101', 'HLA-DPA1*01:10-DPB1*12:201',
'HLA-DPA1*01:10-DPB1*12:301', 'HLA-DPA1*01:10-DPB1*12:401', 'HLA-DPA1*01:10-DPB1*12:501', 'HLA-DPA1*01:10-DPB1*12:601', 'HLA-DPA1*01:10-DPB1*12:701',
'HLA-DPA1*01:10-DPB1*12:801', 'HLA-DPA1*01:10-DPB1*12:901', 'HLA-DPA1*01:10-DPB1*13:001', 'HLA-DPA1*01:10-DPB1*13:01', 'HLA-DPA1*01:10-DPB1*13:101',
'HLA-DPA1*01:10-DPB1*13:201', 'HLA-DPA1*01:10-DPB1*13:301', 'HLA-DPA1*01:10-DPB1*13:401', 'HLA-DPA1*01:10-DPB1*14:01', 'HLA-DPA1*01:10-DPB1*15:01',
'HLA-DPA1*01:10-DPB1*16:01', 'HLA-DPA1*01:10-DPB1*17:01', 'HLA-DPA1*01:10-DPB1*18:01', 'HLA-DPA1*01:10-DPB1*19:01', 'HLA-DPA1*01:10-DPB1*20:01',
'HLA-DPA1*01:10-DPB1*21:01', 'HLA-DPA1*01:10-DPB1*22:01', 'HLA-DPA1*01:10-DPB1*23:01', 'HLA-DPA1*01:10-DPB1*24:01', 'HLA-DPA1*01:10-DPB1*25:01',
'HLA-DPA1*01:10-DPB1*26:01', 'HLA-DPA1*01:10-DPB1*27:01', 'HLA-DPA1*01:10-DPB1*28:01', 'HLA-DPA1*01:10-DPB1*29:01', 'HLA-DPA1*01:10-DPB1*30:01',
'HLA-DPA1*01:10-DPB1*31:01', 'HLA-DPA1*01:10-DPB1*32:01', 'HLA-DPA1*01:10-DPB1*33:01', 'HLA-DPA1*01:10-DPB1*34:01', 'HLA-DPA1*01:10-DPB1*35:01',
'HLA-DPA1*01:10-DPB1*36:01', 'HLA-DPA1*01:10-DPB1*37:01', 'HLA-DPA1*01:10-DPB1*38:01', 'HLA-DPA1*01:10-DPB1*39:01', 'HLA-DPA1*01:10-DPB1*40:01',
'HLA-DPA1*01:10-DPB1*41:01', 'HLA-DPA1*01:10-DPB1*44:01', 'HLA-DPA1*01:10-DPB1*45:01', 'HLA-DPA1*01:10-DPB1*46:01', 'HLA-DPA1*01:10-DPB1*47:01',
'HLA-DPA1*01:10-DPB1*48:01', 'HLA-DPA1*01:10-DPB1*49:01', 'HLA-DPA1*01:10-DPB1*50:01', 'HLA-DPA1*01:10-DPB1*51:01', 'HLA-DPA1*01:10-DPB1*52:01',
'HLA-DPA1*01:10-DPB1*53:01', 'HLA-DPA1*01:10-DPB1*54:01', 'HLA-DPA1*01:10-DPB1*55:01', 'HLA-DPA1*01:10-DPB1*56:01', 'HLA-DPA1*01:10-DPB1*58:01',
'HLA-DPA1*01:10-DPB1*59:01', 'HLA-DPA1*01:10-DPB1*60:01', 'HLA-DPA1*01:10-DPB1*62:01', 'HLA-DPA1*01:10-DPB1*63:01', 'HLA-DPA1*01:10-DPB1*65:01',
'HLA-DPA1*01:10-DPB1*66:01', 'HLA-DPA1*01:10-DPB1*67:01', 'HLA-DPA1*01:10-DPB1*68:01', 'HLA-DPA1*01:10-DPB1*69:01', 'HLA-DPA1*01:10-DPB1*70:01',
'HLA-DPA1*01:10-DPB1*71:01', 'HLA-DPA1*01:10-DPB1*72:01', 'HLA-DPA1*01:10-DPB1*73:01', 'HLA-DPA1*01:10-DPB1*74:01', 'HLA-DPA1*01:10-DPB1*75:01',
'HLA-DPA1*01:10-DPB1*76:01', 'HLA-DPA1*01:10-DPB1*77:01', 'HLA-DPA1*01:10-DPB1*78:01', 'HLA-DPA1*01:10-DPB1*79:01', 'HLA-DPA1*01:10-DPB1*80:01',
'HLA-DPA1*01:10-DPB1*81:01', 'HLA-DPA1*01:10-DPB1*82:01', 'HLA-DPA1*01:10-DPB1*83:01', 'HLA-DPA1*01:10-DPB1*84:01', 'HLA-DPA1*01:10-DPB1*85:01',
'HLA-DPA1*01:10-DPB1*86:01', 'HLA-DPA1*01:10-DPB1*87:01', 'HLA-DPA1*01:10-DPB1*88:01', 'HLA-DPA1*01:10-DPB1*89:01', 'HLA-DPA1*01:10-DPB1*90:01',
'HLA-DPA1*01:10-DPB1*91:01', 'HLA-DPA1*01:10-DPB1*92:01', 'HLA-DPA1*01:10-DPB1*93:01', 'HLA-DPA1*01:10-DPB1*94:01', 'HLA-DPA1*01:10-DPB1*95:01',
'HLA-DPA1*01:10-DPB1*96:01', 'HLA-DPA1*01:10-DPB1*97:01', 'HLA-DPA1*01:10-DPB1*98:01', 'HLA-DPA1*01:10-DPB1*99:01', 'HLA-DPA1*02:01-DPB1*01:01',
'HLA-DPA1*02:01-DPB1*02:01', 'HLA-DPA1*02:01-DPB1*02:02', 'HLA-DPA1*02:01-DPB1*03:01', 'HLA-DPA1*02:01-DPB1*04:01', 'HLA-DPA1*02:01-DPB1*04:02',
'HLA-DPA1*02:01-DPB1*05:01', 'HLA-DPA1*02:01-DPB1*06:01', 'HLA-DPA1*02:01-DPB1*08:01', 'HLA-DPA1*02:01-DPB1*09:01', 'HLA-DPA1*02:01-DPB1*10:001',
'HLA-DPA1*02:01-DPB1*10:01', 'HLA-DPA1*02:01-DPB1*10:101', 'HLA-DPA1*02:01-DPB1*10:201', 'HLA-DPA1*02:01-DPB1*10:301', 'HLA-DPA1*02:01-DPB1*10:401',
'HLA-DPA1*02:01-DPB1*10:501', 'HLA-DPA1*02:01-DPB1*10:601', 'HLA-DPA1*02:01-DPB1*10:701', 'HLA-DPA1*02:01-DPB1*10:801', 'HLA-DPA1*02:01-DPB1*10:901',
'HLA-DPA1*02:01-DPB1*11:001', 'HLA-DPA1*02:01-DPB1*11:01', 'HLA-DPA1*02:01-DPB1*11:101', 'HLA-DPA1*02:01-DPB1*11:201', 'HLA-DPA1*02:01-DPB1*11:301',
'HLA-DPA1*02:01-DPB1*11:401', 'HLA-DPA1*02:01-DPB1*11:501', 'HLA-DPA1*02:01-DPB1*11:601', 'HLA-DPA1*02:01-DPB1*11:701', 'HLA-DPA1*02:01-DPB1*11:801',
'HLA-DPA1*02:01-DPB1*11:901', 'HLA-DPA1*02:01-DPB1*12:101', 'HLA-DPA1*02:01-DPB1*12:201', 'HLA-DPA1*02:01-DPB1*12:301', 'HLA-DPA1*02:01-DPB1*12:401',
'HLA-DPA1*02:01-DPB1*12:501', 'HLA-DPA1*02:01-DPB1*12:601', 'HLA-DPA1*02:01-DPB1*12:701', 'HLA-DPA1*02:01-DPB1*12:801', 'HLA-DPA1*02:01-DPB1*12:901',
'HLA-DPA1*02:01-DPB1*13:001', 'HLA-DPA1*02:01-DPB1*13:01', 'HLA-DPA1*02:01-DPB1*13:101', 'HLA-DPA1*02:01-DPB1*13:201', 'HLA-DPA1*02:01-DPB1*13:301',
'HLA-DPA1*02:01-DPB1*13:401', 'HLA-DPA1*02:01-DPB1*14:01', 'HLA-DPA1*02:01-DPB1*15:01', 'HLA-DPA1*02:01-DPB1*16:01', 'HLA-DPA1*02:01-DPB1*17:01',
'HLA-DPA1*02:01-DPB1*18:01', 'HLA-DPA1*02:01-DPB1*19:01', 'HLA-DPA1*02:01-DPB1*20:01', 'HLA-DPA1*02:01-DPB1*21:01', 'HLA-DPA1*02:01-DPB1*22:01',
'HLA-DPA1*02:01-DPB1*23:01', 'HLA-DPA1*02:01-DPB1*24:01', 'HLA-DPA1*02:01-DPB1*25:01', 'HLA-DPA1*02:01-DPB1*26:01', 'HLA-DPA1*02:01-DPB1*27:01',
'HLA-DPA1*02:01-DPB1*28:01', 'HLA-DPA1*02:01-DPB1*29:01', 'HLA-DPA1*02:01-DPB1*30:01', 'HLA-DPA1*02:01-DPB1*31:01', 'HLA-DPA1*02:01-DPB1*32:01',
'HLA-DPA1*02:01-DPB1*33:01', 'HLA-DPA1*02:01-DPB1*34:01', 'HLA-DPA1*02:01-DPB1*35:01', 'HLA-DPA1*02:01-DPB1*36:01', 'HLA-DPA1*02:01-DPB1*37:01',
'HLA-DPA1*02:01-DPB1*38:01', 'HLA-DPA1*02:01-DPB1*39:01', 'HLA-DPA1*02:01-DPB1*40:01', 'HLA-DPA1*02:01-DPB1*41:01', 'HLA-DPA1*02:01-DPB1*44:01',
'HLA-DPA1*02:01-DPB1*45:01', 'HLA-DPA1*02:01-DPB1*46:01', 'HLA-DPA1*02:01-DPB1*47:01', 'HLA-DPA1*02:01-DPB1*48:01', 'HLA-DPA1*02:01-DPB1*49:01',
'HLA-DPA1*02:01-DPB1*50:01', 'HLA-DPA1*02:01-DPB1*51:01', 'HLA-DPA1*02:01-DPB1*52:01', 'HLA-DPA1*02:01-DPB1*53:01', 'HLA-DPA1*02:01-DPB1*54:01',
'HLA-DPA1*02:01-DPB1*55:01', 'HLA-DPA1*02:01-DPB1*56:01', 'HLA-DPA1*02:01-DPB1*58:01', 'HLA-DPA1*02:01-DPB1*59:01', 'HLA-DPA1*02:01-DPB1*60:01',
'HLA-DPA1*02:01-DPB1*62:01', 'HLA-DPA1*02:01-DPB1*63:01', 'HLA-DPA1*02:01-DPB1*65:01', 'HLA-DPA1*02:01-DPB1*66:01', 'HLA-DPA1*02:01-DPB1*67:01',
'HLA-DPA1*02:01-DPB1*68:01', 'HLA-DPA1*02:01-DPB1*69:01', 'HLA-DPA1*02:01-DPB1*70:01', 'HLA-DPA1*02:01-DPB1*71:01', 'HLA-DPA1*02:01-DPB1*72:01',
'HLA-DPA1*02:01-DPB1*73:01', 'HLA-DPA1*02:01-DPB1*74:01', 'HLA-DPA1*02:01-DPB1*75:01', 'HLA-DPA1*02:01-DPB1*76:01', 'HLA-DPA1*02:01-DPB1*77:01',
'HLA-DPA1*02:01-DPB1*78:01', 'HLA-DPA1*02:01-DPB1*79:01', 'HLA-DPA1*02:01-DPB1*80:01', 'HLA-DPA1*02:01-DPB1*81:01', 'HLA-DPA1*02:01-DPB1*82:01',
'HLA-DPA1*02:01-DPB1*83:01', 'HLA-DPA1*02:01-DPB1*84:01', 'HLA-DPA1*02:01-DPB1*85:01', 'HLA-DPA1*02:01-DPB1*86:01', 'HLA-DPA1*02:01-DPB1*87:01',
'HLA-DPA1*02:01-DPB1*88:01', 'HLA-DPA1*02:01-DPB1*89:01', 'HLA-DPA1*02:01-DPB1*90:01', 'HLA-DPA1*02:01-DPB1*91:01', 'HLA-DPA1*02:01-DPB1*92:01',
'HLA-DPA1*02:01-DPB1*93:01', 'HLA-DPA1*02:01-DPB1*94:01', 'HLA-DPA1*02:01-DPB1*95:01', 'HLA-DPA1*02:01-DPB1*96:01', 'HLA-DPA1*02:01-DPB1*97:01',
'HLA-DPA1*02:01-DPB1*98:01', 'HLA-DPA1*02:01-DPB1*99:01', 'HLA-DPA1*02:02-DPB1*01:01', 'HLA-DPA1*02:02-DPB1*02:01', 'HLA-DPA1*02:02-DPB1*02:02',
'HLA-DPA1*02:02-DPB1*03:01', 'HLA-DPA1*02:02-DPB1*04:01', 'HLA-DPA1*02:02-DPB1*04:02', 'HLA-DPA1*02:02-DPB1*05:01', 'HLA-DPA1*02:02-DPB1*06:01',
'HLA-DPA1*02:02-DPB1*08:01', 'HLA-DPA1*02:02-DPB1*09:01', 'HLA-DPA1*02:02-DPB1*10:001', 'HLA-DPA1*02:02-DPB1*10:01', 'HLA-DPA1*02:02-DPB1*10:101',
'HLA-DPA1*02:02-DPB1*10:201', 'HLA-DPA1*02:02-DPB1*10:301', 'HLA-DPA1*02:02-DPB1*10:401', 'HLA-DPA1*02:02-DPB1*10:501', 'HLA-DPA1*02:02-DPB1*10:601',
'HLA-DPA1*02:02-DPB1*10:701', 'HLA-DPA1*02:02-DPB1*10:801', 'HLA-DPA1*02:02-DPB1*10:901', 'HLA-DPA1*02:02-DPB1*11:001', 'HLA-DPA1*02:02-DPB1*11:01',
'HLA-DPA1*02:02-DPB1*11:101', 'HLA-DPA1*02:02-DPB1*11:201', 'HLA-DPA1*02:02-DPB1*11:301', 'HLA-DPA1*02:02-DPB1*11:401', 'HLA-DPA1*02:02-DPB1*11:501',
'HLA-DPA1*02:02-DPB1*11:601', 'HLA-DPA1*02:02-DPB1*11:701', 'HLA-DPA1*02:02-DPB1*11:801', 'HLA-DPA1*02:02-DPB1*11:901', 'HLA-DPA1*02:02-DPB1*12:101',
'HLA-DPA1*02:02-DPB1*12:201', 'HLA-DPA1*02:02-DPB1*12:301', 'HLA-DPA1*02:02-DPB1*12:401', 'HLA-DPA1*02:02-DPB1*12:501', 'HLA-DPA1*02:02-DPB1*12:601',
'HLA-DPA1*02:02-DPB1*12:701', 'HLA-DPA1*02:02-DPB1*12:801', 'HLA-DPA1*02:02-DPB1*12:901', 'HLA-DPA1*02:02-DPB1*13:001', 'HLA-DPA1*02:02-DPB1*13:01',
'HLA-DPA1*02:02-DPB1*13:101', 'HLA-DPA1*02:02-DPB1*13:201', 'HLA-DPA1*02:02-DPB1*13:301', 'HLA-DPA1*02:02-DPB1*13:401', 'HLA-DPA1*02:02-DPB1*14:01',
'HLA-DPA1*02:02-DPB1*15:01', 'HLA-DPA1*02:02-DPB1*16:01', 'HLA-DPA1*02:02-DPB1*17:01', 'HLA-DPA1*02:02-DPB1*18:01', 'HLA-DPA1*02:02-DPB1*19:01',
'HLA-DPA1*02:02-DPB1*20:01', 'HLA-DPA1*02:02-DPB1*21:01', 'HLA-DPA1*02:02-DPB1*22:01', 'HLA-DPA1*02:02-DPB1*23:01', 'HLA-DPA1*02:02-DPB1*24:01',
'HLA-DPA1*02:02-DPB1*25:01', 'HLA-DPA1*02:02-DPB1*26:01', 'HLA-DPA1*02:02-DPB1*27:01', 'HLA-DPA1*02:02-DPB1*28:01', 'HLA-DPA1*02:02-DPB1*29:01',
'HLA-DPA1*02:02-DPB1*30:01', 'HLA-DPA1*02:02-DPB1*31:01', 'HLA-DPA1*02:02-DPB1*32:01', 'HLA-DPA1*02:02-DPB1*33:01', 'HLA-DPA1*02:02-DPB1*34:01',
'HLA-DPA1*02:02-DPB1*35:01', 'HLA-DPA1*02:02-DPB1*36:01', 'HLA-DPA1*02:02-DPB1*37:01', 'HLA-DPA1*02:02-DPB1*38:01', 'HLA-DPA1*02:02-DPB1*39:01',
'HLA-DPA1*02:02-DPB1*40:01', 'HLA-DPA1*02:02-DPB1*41:01', 'HLA-DPA1*02:02-DPB1*44:01', 'HLA-DPA1*02:02-DPB1*45:01', 'HLA-DPA1*02:02-DPB1*46:01',
'HLA-DPA1*02:02-DPB1*47:01', 'HLA-DPA1*02:02-DPB1*48:01', 'HLA-DPA1*02:02-DPB1*49:01', 'HLA-DPA1*02:02-DPB1*50:01', 'HLA-DPA1*02:02-DPB1*51:01',
'HLA-DPA1*02:02-DPB1*52:01', 'HLA-DPA1*02:02-DPB1*53:01', 'HLA-DPA1*02:02-DPB1*54:01', 'HLA-DPA1*02:02-DPB1*55:01', 'HLA-DPA1*02:02-DPB1*56:01',
'HLA-DPA1*02:02-DPB1*58:01', 'HLA-DPA1*02:02-DPB1*59:01', 'HLA-DPA1*02:02-DPB1*60:01', 'HLA-DPA1*02:02-DPB1*62:01', 'HLA-DPA1*02:02-DPB1*63:01',
'HLA-DPA1*02:02-DPB1*65:01', 'HLA-DPA1*02:02-DPB1*66:01', 'HLA-DPA1*02:02-DPB1*67:01', 'HLA-DPA1*02:02-DPB1*68:01', 'HLA-DPA1*02:02-DPB1*69:01',
'HLA-DPA1*02:02-DPB1*70:01', 'HLA-DPA1*02:02-DPB1*71:01', 'HLA-DPA1*02:02-DPB1*72:01', 'HLA-DPA1*02:02-DPB1*73:01', 'HLA-DPA1*02:02-DPB1*74:01',
'HLA-DPA1*02:02-DPB1*75:01', 'HLA-DPA1*02:02-DPB1*76:01', 'HLA-DPA1*02:02-DPB1*77:01', 'HLA-DPA1*02:02-DPB1*78:01', 'HLA-DPA1*02:02-DPB1*79:01',
'HLA-DPA1*02:02-DPB1*80:01', 'HLA-DPA1*02:02-DPB1*81:01', 'HLA-DPA1*02:02-DPB1*82:01', 'HLA-DPA1*02:02-DPB1*83:01', 'HLA-DPA1*02:02-DPB1*84:01',
'HLA-DPA1*02:02-DPB1*85:01', 'HLA-DPA1*02:02-DPB1*86:01', 'HLA-DPA1*02:02-DPB1*87:01', 'HLA-DPA1*02:02-DPB1*88:01', 'HLA-DPA1*02:02-DPB1*89:01',
'HLA-DPA1*02:02-DPB1*90:01', 'HLA-DPA1*02:02-DPB1*91:01', 'HLA-DPA1*02:02-DPB1*92:01', 'HLA-DPA1*02:02-DPB1*93:01', 'HLA-DPA1*02:02-DPB1*94:01',
'HLA-DPA1*02:02-DPB1*95:01', 'HLA-DPA1*02:02-DPB1*96:01', 'HLA-DPA1*02:02-DPB1*97:01', 'HLA-DPA1*02:02-DPB1*98:01', 'HLA-DPA1*02:02-DPB1*99:01',
'HLA-DPA1*02:03-DPB1*01:01', 'HLA-DPA1*02:03-DPB1*02:01', 'HLA-DPA1*02:03-DPB1*02:02', 'HLA-DPA1*02:03-DPB1*03:01', 'HLA-DPA1*02:03-DPB1*04:01',
'HLA-DPA1*02:03-DPB1*04:02', 'HLA-DPA1*02:03-DPB1*05:01', 'HLA-DPA1*02:03-DPB1*06:01', 'HLA-DPA1*02:03-DPB1*08:01', 'HLA-DPA1*02:03-DPB1*09:01',
'HLA-DPA1*02:03-DPB1*10:001', 'HLA-DPA1*02:03-DPB1*10:01', 'HLA-DPA1*02:03-DPB1*10:101', 'HLA-DPA1*02:03-DPB1*10:201', 'HLA-DPA1*02:03-DPB1*10:301',
'HLA-DPA1*02:03-DPB1*10:401', 'HLA-DPA1*02:03-DPB1*10:501', 'HLA-DPA1*02:03-DPB1*10:601', 'HLA-DPA1*02:03-DPB1*10:701', 'HLA-DPA1*02:03-DPB1*10:801',
'HLA-DPA1*02:03-DPB1*10:901', 'HLA-DPA1*02:03-DPB1*11:001', 'HLA-DPA1*02:03-DPB1*11:01', 'HLA-DPA1*02:03-DPB1*11:101', 'HLA-DPA1*02:03-DPB1*11:201',
'HLA-DPA1*02:03-DPB1*11:301', 'HLA-DPA1*02:03-DPB1*11:401', 'HLA-DPA1*02:03-DPB1*11:501', 'HLA-DPA1*02:03-DPB1*11:601', 'HLA-DPA1*02:03-DPB1*11:701',
'HLA-DPA1*02:03-DPB1*11:801', 'HLA-DPA1*02:03-DPB1*11:901', 'HLA-DPA1*02:03-DPB1*12:101', 'HLA-DPA1*02:03-DPB1*12:201', 'HLA-DPA1*02:03-DPB1*12:301',
'HLA-DPA1*02:03-DPB1*12:401', 'HLA-DPA1*02:03-DPB1*12:501', 'HLA-DPA1*02:03-DPB1*12:601', 'HLA-DPA1*02:03-DPB1*12:701', 'HLA-DPA1*02:03-DPB1*12:801',
'HLA-DPA1*02:03-DPB1*12:901', 'HLA-DPA1*02:03-DPB1*13:001', 'HLA-DPA1*02:03-DPB1*13:01', 'HLA-DPA1*02:03-DPB1*13:101', 'HLA-DPA1*02:03-DPB1*13:201',
'HLA-DPA1*02:03-DPB1*13:301', 'HLA-DPA1*02:03-DPB1*13:401', 'HLA-DPA1*02:03-DPB1*14:01', 'HLA-DPA1*02:03-DPB1*15:01', 'HLA-DPA1*02:03-DPB1*16:01',
'HLA-DPA1*02:03-DPB1*17:01', 'HLA-DPA1*02:03-DPB1*18:01', 'HLA-DPA1*02:03-DPB1*19:01', 'HLA-DPA1*02:03-DPB1*20:01', 'HLA-DPA1*02:03-DPB1*21:01',
'HLA-DPA1*02:03-DPB1*22:01', 'HLA-DPA1*02:03-DPB1*23:01', 'HLA-DPA1*02:03-DPB1*24:01', 'HLA-DPA1*02:03-DPB1*25:01', 'HLA-DPA1*02:03-DPB1*26:01',
'HLA-DPA1*02:03-DPB1*27:01', 'HLA-DPA1*02:03-DPB1*28:01', 'HLA-DPA1*02:03-DPB1*29:01', 'HLA-DPA1*02:03-DPB1*30:01', 'HLA-DPA1*02:03-DPB1*31:01',
'HLA-DPA1*02:03-DPB1*32:01', 'HLA-DPA1*02:03-DPB1*33:01', 'HLA-DPA1*02:03-DPB1*34:01', 'HLA-DPA1*02:03-DPB1*35:01', 'HLA-DPA1*02:03-DPB1*36:01',
'HLA-DPA1*02:03-DPB1*37:01', 'HLA-DPA1*02:03-DPB1*38:01', 'HLA-DPA1*02:03-DPB1*39:01', 'HLA-DPA1*02:03-DPB1*40:01', 'HLA-DPA1*02:03-DPB1*41:01',
'HLA-DPA1*02:03-DPB1*44:01', 'HLA-DPA1*02:03-DPB1*45:01', 'HLA-DPA1*02:03-DPB1*46:01', 'HLA-DPA1*02:03-DPB1*47:01', 'HLA-DPA1*02:03-DPB1*48:01',
'HLA-DPA1*02:03-DPB1*49:01', 'HLA-DPA1*02:03-DPB1*50:01', 'HLA-DPA1*02:03-DPB1*51:01', 'HLA-DPA1*02:03-DPB1*52:01', 'HLA-DPA1*02:03-DPB1*53:01',
'HLA-DPA1*02:03-DPB1*54:01', 'HLA-DPA1*02:03-DPB1*55:01', 'HLA-DPA1*02:03-DPB1*56:01', 'HLA-DPA1*02:03-DPB1*58:01', 'HLA-DPA1*02:03-DPB1*59:01',
'HLA-DPA1*02:03-DPB1*60:01', 'HLA-DPA1*02:03-DPB1*62:01', 'HLA-DPA1*02:03-DPB1*63:01', 'HLA-DPA1*02:03-DPB1*65:01', 'HLA-DPA1*02:03-DPB1*66:01',
'HLA-DPA1*02:03-DPB1*67:01', 'HLA-DPA1*02:03-DPB1*68:01', 'HLA-DPA1*02:03-DPB1*69:01', 'HLA-DPA1*02:03-DPB1*70:01', 'HLA-DPA1*02:03-DPB1*71:01',
'HLA-DPA1*02:03-DPB1*72:01', 'HLA-DPA1*02:03-DPB1*73:01', 'HLA-DPA1*02:03-DPB1*74:01', 'HLA-DPA1*02:03-DPB1*75:01', 'HLA-DPA1*02:03-DPB1*76:01',
'HLA-DPA1*02:03-DPB1*77:01', 'HLA-DPA1*02:03-DPB1*78:01', 'HLA-DPA1*02:03-DPB1*79:01', 'HLA-DPA1*02:03-DPB1*80:01', 'HLA-DPA1*02:03-DPB1*81:01',
'HLA-DPA1*02:03-DPB1*82:01', 'HLA-DPA1*02:03-DPB1*83:01', 'HLA-DPA1*02:03-DPB1*84:01', 'HLA-DPA1*02:03-DPB1*85:01', 'HLA-DPA1*02:03-DPB1*86:01',
'HLA-DPA1*02:03-DPB1*87:01', 'HLA-DPA1*02:03-DPB1*88:01', 'HLA-DPA1*02:03-DPB1*89:01', 'HLA-DPA1*02:03-DPB1*90:01', 'HLA-DPA1*02:03-DPB1*91:01',
'HLA-DPA1*02:03-DPB1*92:01', 'HLA-DPA1*02:03-DPB1*93:01', 'HLA-DPA1*02:03-DPB1*94:01', 'HLA-DPA1*02:03-DPB1*95:01', 'HLA-DPA1*02:03-DPB1*96:01',
'HLA-DPA1*02:03-DPB1*97:01', 'HLA-DPA1*02:03-DPB1*98:01', 'HLA-DPA1*02:03-DPB1*99:01', 'HLA-DPA1*02:04-DPB1*01:01', 'HLA-DPA1*02:04-DPB1*02:01',
'HLA-DPA1*02:04-DPB1*02:02', 'HLA-DPA1*02:04-DPB1*03:01', 'HLA-DPA1*02:04-DPB1*04:01', 'HLA-DPA1*02:04-DPB1*04:02', 'HLA-DPA1*02:04-DPB1*05:01',
'HLA-DPA1*02:04-DPB1*06:01', 'HLA-DPA1*02:04-DPB1*08:01', 'HLA-DPA1*02:04-DPB1*09:01', 'HLA-DPA1*02:04-DPB1*10:001', 'HLA-DPA1*02:04-DPB1*10:01',
'HLA-DPA1*02:04-DPB1*10:101', 'HLA-DPA1*02:04-DPB1*10:201', 'HLA-DPA1*02:04-DPB1*10:301', 'HLA-DPA1*02:04-DPB1*10:401', 'HLA-DPA1*02:04-DPB1*10:501',
'HLA-DPA1*02:04-DPB1*10:601', 'HLA-DPA1*02:04-DPB1*10:701', 'HLA-DPA1*02:04-DPB1*10:801', 'HLA-DPA1*02:04-DPB1*10:901', 'HLA-DPA1*02:04-DPB1*11:001',
'HLA-DPA1*02:04-DPB1*11:01', 'HLA-DPA1*02:04-DPB1*11:101', 'HLA-DPA1*02:04-DPB1*11:201', 'HLA-DPA1*02:04-DPB1*11:301', 'HLA-DPA1*02:04-DPB1*11:401',
'HLA-DPA1*02:04-DPB1*11:501', 'HLA-DPA1*02:04-DPB1*11:601', 'HLA-DPA1*02:04-DPB1*11:701', 'HLA-DPA1*02:04-DPB1*11:801', 'HLA-DPA1*02:04-DPB1*11:901',
'HLA-DPA1*02:04-DPB1*12:101', 'HLA-DPA1*02:04-DPB1*12:201', 'HLA-DPA1*02:04-DPB1*12:301', 'HLA-DPA1*02:04-DPB1*12:401', 'HLA-DPA1*02:04-DPB1*12:501',
'HLA-DPA1*02:04-DPB1*12:601', 'HLA-DPA1*02:04-DPB1*12:701', 'HLA-DPA1*02:04-DPB1*12:801', 'HLA-DPA1*02:04-DPB1*12:901', 'HLA-DPA1*02:04-DPB1*13:001',
'HLA-DPA1*02:04-DPB1*13:01', 'HLA-DPA1*02:04-DPB1*13:101', 'HLA-DPA1*02:04-DPB1*13:201', 'HLA-DPA1*02:04-DPB1*13:301', 'HLA-DPA1*02:04-DPB1*13:401',
'HLA-DPA1*02:04-DPB1*14:01', 'HLA-DPA1*02:04-DPB1*15:01', 'HLA-DPA1*02:04-DPB1*16:01', 'HLA-DPA1*02:04-DPB1*17:01', 'HLA-DPA1*02:04-DPB1*18:01',
'HLA-DPA1*02:04-DPB1*19:01', 'HLA-DPA1*02:04-DPB1*20:01', 'HLA-DPA1*02:04-DPB1*21:01', 'HLA-DPA1*02:04-DPB1*22:01', 'HLA-DPA1*02:04-DPB1*23:01',
'HLA-DPA1*02:04-DPB1*24:01', 'HLA-DPA1*02:04-DPB1*25:01', 'HLA-DPA1*02:04-DPB1*26:01', 'HLA-DPA1*02:04-DPB1*27:01', 'HLA-DPA1*02:04-DPB1*28:01',
'HLA-DPA1*02:04-DPB1*29:01', 'HLA-DPA1*02:04-DPB1*30:01', 'HLA-DPA1*02:04-DPB1*31:01', 'HLA-DPA1*02:04-DPB1*32:01', 'HLA-DPA1*02:04-DPB1*33:01',
'HLA-DPA1*02:04-DPB1*34:01', 'HLA-DPA1*02:04-DPB1*35:01', 'HLA-DPA1*02:04-DPB1*36:01', 'HLA-DPA1*02:04-DPB1*37:01', 'HLA-DPA1*02:04-DPB1*38:01',
'HLA-DPA1*02:04-DPB1*39:01', 'HLA-DPA1*02:04-DPB1*40:01', 'HLA-DPA1*02:04-DPB1*41:01', 'HLA-DPA1*02:04-DPB1*44:01', 'HLA-DPA1*02:04-DPB1*45:01',
'HLA-DPA1*02:04-DPB1*46:01', 'HLA-DPA1*02:04-DPB1*47:01', 'HLA-DPA1*02:04-DPB1*48:01', 'HLA-DPA1*02:04-DPB1*49:01', 'HLA-DPA1*02:04-DPB1*50:01',
'HLA-DPA1*02:04-DPB1*51:01', 'HLA-DPA1*02:04-DPB1*52:01', 'HLA-DPA1*02:04-DPB1*53:01', 'HLA-DPA1*02:04-DPB1*54:01', 'HLA-DPA1*02:04-DPB1*55:01',
'HLA-DPA1*02:04-DPB1*56:01', 'HLA-DPA1*02:04-DPB1*58:01', 'HLA-DPA1*02:04-DPB1*59:01', 'HLA-DPA1*02:04-DPB1*60:01', 'HLA-DPA1*02:04-DPB1*62:01',
'HLA-DPA1*02:04-DPB1*63:01', 'HLA-DPA1*02:04-DPB1*65:01', 'HLA-DPA1*02:04-DPB1*66:01', 'HLA-DPA1*02:04-DPB1*67:01', 'HLA-DPA1*02:04-DPB1*68:01',
'HLA-DPA1*02:04-DPB1*69:01', 'HLA-DPA1*02:04-DPB1*70:01', 'HLA-DPA1*02:04-DPB1*71:01', 'HLA-DPA1*02:04-DPB1*72:01', 'HLA-DPA1*02:04-DPB1*73:01',
'HLA-DPA1*02:04-DPB1*74:01', 'HLA-DPA1*02:04-DPB1*75:01', 'HLA-DPA1*02:04-DPB1*76:01', 'HLA-DPA1*02:04-DPB1*77:01', 'HLA-DPA1*02:04-DPB1*78:01',
'HLA-DPA1*02:04-DPB1*79:01', 'HLA-DPA1*02:04-DPB1*80:01', 'HLA-DPA1*02:04-DPB1*81:01', 'HLA-DPA1*02:04-DPB1*82:01', 'HLA-DPA1*02:04-DPB1*83:01',
'HLA-DPA1*02:04-DPB1*84:01', 'HLA-DPA1*02:04-DPB1*85:01', 'HLA-DPA1*02:04-DPB1*86:01', 'HLA-DPA1*02:04-DPB1*87:01', 'HLA-DPA1*02:04-DPB1*88:01',
'HLA-DPA1*02:04-DPB1*89:01', 'HLA-DPA1*02:04-DPB1*90:01', 'HLA-DPA1*02:04-DPB1*91:01', 'HLA-DPA1*02:04-DPB1*92:01', 'HLA-DPA1*02:04-DPB1*93:01',
'HLA-DPA1*02:04-DPB1*94:01', 'HLA-DPA1*02:04-DPB1*95:01', 'HLA-DPA1*02:04-DPB1*96:01', 'HLA-DPA1*02:04-DPB1*97:01', 'HLA-DPA1*02:04-DPB1*98:01',
'HLA-DPA1*02:04-DPB1*99:01', 'HLA-DPA1*03:01-DPB1*01:01', 'HLA-DPA1*03:01-DPB1*02:01', 'HLA-DPA1*03:01-DPB1*02:02', 'HLA-DPA1*03:01-DPB1*03:01',
'HLA-DPA1*03:01-DPB1*04:01', 'HLA-DPA1*03:01-DPB1*04:02', 'HLA-DPA1*03:01-DPB1*05:01', 'HLA-DPA1*03:01-DPB1*06:01', 'HLA-DPA1*03:01-DPB1*08:01',
'HLA-DPA1*03:01-DPB1*09:01', 'HLA-DPA1*03:01-DPB1*10:001', 'HLA-DPA1*03:01-DPB1*10:01', 'HLA-DPA1*03:01-DPB1*10:101', 'HLA-DPA1*03:01-DPB1*10:201',
'HLA-DPA1*03:01-DPB1*10:301', 'HLA-DPA1*03:01-DPB1*10:401', 'HLA-DPA1*03:01-DPB1*10:501', 'HLA-DPA1*03:01-DPB1*10:601', 'HLA-DPA1*03:01-DPB1*10:701',
'HLA-DPA1*03:01-DPB1*10:801', 'HLA-DPA1*03:01-DPB1*10:901', 'HLA-DPA1*03:01-DPB1*11:001', 'HLA-DPA1*03:01-DPB1*11:01', 'HLA-DPA1*03:01-DPB1*11:101',
'HLA-DPA1*03:01-DPB1*11:201', 'HLA-DPA1*03:01-DPB1*11:301', 'HLA-DPA1*03:01-DPB1*11:401', 'HLA-DPA1*03:01-DPB1*11:501', 'HLA-DPA1*03:01-DPB1*11:601',
'HLA-DPA1*03:01-DPB1*11:701', 'HLA-DPA1*03:01-DPB1*11:801', 'HLA-DPA1*03:01-DPB1*11:901', 'HLA-DPA1*03:01-DPB1*12:101', 'HLA-DPA1*03:01-DPB1*12:201',
'HLA-DPA1*03:01-DPB1*12:301', 'HLA-DPA1*03:01-DPB1*12:401', 'HLA-DPA1*03:01-DPB1*12:501', 'HLA-DPA1*03:01-DPB1*12:601', 'HLA-DPA1*03:01-DPB1*12:701',
'HLA-DPA1*03:01-DPB1*12:801', 'HLA-DPA1*03:01-DPB1*12:901', 'HLA-DPA1*03:01-DPB1*13:001', 'HLA-DPA1*03:01-DPB1*13:01', 'HLA-DPA1*03:01-DPB1*13:101',
'HLA-DPA1*03:01-DPB1*13:201', 'HLA-DPA1*03:01-DPB1*13:301', 'HLA-DPA1*03:01-DPB1*13:401', 'HLA-DPA1*03:01-DPB1*14:01', 'HLA-DPA1*03:01-DPB1*15:01',
'HLA-DPA1*03:01-DPB1*16:01', 'HLA-DPA1*03:01-DPB1*17:01', 'HLA-DPA1*03:01-DPB1*18:01', 'HLA-DPA1*03:01-DPB1*19:01', 'HLA-DPA1*03:01-DPB1*20:01',
'HLA-DPA1*03:01-DPB1*21:01', 'HLA-DPA1*03:01-DPB1*22:01', 'HLA-DPA1*03:01-DPB1*23:01', 'HLA-DPA1*03:01-DPB1*24:01', 'HLA-DPA1*03:01-DPB1*25:01',
'HLA-DPA1*03:01-DPB1*26:01', 'HLA-DPA1*03:01-DPB1*27:01', 'HLA-DPA1*03:01-DPB1*28:01', 'HLA-DPA1*03:01-DPB1*29:01', 'HLA-DPA1*03:01-DPB1*30:01',
'HLA-DPA1*03:01-DPB1*31:01', 'HLA-DPA1*03:01-DPB1*32:01', 'HLA-DPA1*03:01-DPB1*33:01', 'HLA-DPA1*03:01-DPB1*34:01', 'HLA-DPA1*03:01-DPB1*35:01',
'HLA-DPA1*03:01-DPB1*36:01', 'HLA-DPA1*03:01-DPB1*37:01', 'HLA-DPA1*03:01-DPB1*38:01', 'HLA-DPA1*03:01-DPB1*39:01', 'HLA-DPA1*03:01-DPB1*40:01',
'HLA-DPA1*03:01-DPB1*41:01', 'HLA-DPA1*03:01-DPB1*44:01', 'HLA-DPA1*03:01-DPB1*45:01', 'HLA-DPA1*03:01-DPB1*46:01', 'HLA-DPA1*03:01-DPB1*47:01',
'HLA-DPA1*03:01-DPB1*48:01', 'HLA-DPA1*03:01-DPB1*49:01', 'HLA-DPA1*03:01-DPB1*50:01', 'HLA-DPA1*03:01-DPB1*51:01', 'HLA-DPA1*03:01-DPB1*52:01',
'HLA-DPA1*03:01-DPB1*53:01', 'HLA-DPA1*03:01-DPB1*54:01', 'HLA-DPA1*03:01-DPB1*55:01', 'HLA-DPA1*03:01-DPB1*56:01', 'HLA-DPA1*03:01-DPB1*58:01',
'HLA-DPA1*03:01-DPB1*59:01', 'HLA-DPA1*03:01-DPB1*60:01', 'HLA-DPA1*03:01-DPB1*62:01', 'HLA-DPA1*03:01-DPB1*63:01', 'HLA-DPA1*03:01-DPB1*65:01',
'HLA-DPA1*03:01-DPB1*66:01', 'HLA-DPA1*03:01-DPB1*67:01', 'HLA-DPA1*03:01-DPB1*68:01', 'HLA-DPA1*03:01-DPB1*69:01', 'HLA-DPA1*03:01-DPB1*70:01',
'HLA-DPA1*03:01-DPB1*71:01', 'HLA-DPA1*03:01-DPB1*72:01', 'HLA-DPA1*03:01-DPB1*73:01', 'HLA-DPA1*03:01-DPB1*74:01', 'HLA-DPA1*03:01-DPB1*75:01',
'HLA-DPA1*03:01-DPB1*76:01', 'HLA-DPA1*03:01-DPB1*77:01', 'HLA-DPA1*03:01-DPB1*78:01', 'HLA-DPA1*03:01-DPB1*79:01', 'HLA-DPA1*03:01-DPB1*80:01',
'HLA-DPA1*03:01-DPB1*81:01', 'HLA-DPA1*03:01-DPB1*82:01', 'HLA-DPA1*03:01-DPB1*83:01', 'HLA-DPA1*03:01-DPB1*84:01', 'HLA-DPA1*03:01-DPB1*85:01',
'HLA-DPA1*03:01-DPB1*86:01', 'HLA-DPA1*03:01-DPB1*87:01', 'HLA-DPA1*03:01-DPB1*88:01', 'HLA-DPA1*03:01-DPB1*89:01', 'HLA-DPA1*03:01-DPB1*90:01',
'HLA-DPA1*03:01-DPB1*91:01', 'HLA-DPA1*03:01-DPB1*92:01', 'HLA-DPA1*03:01-DPB1*93:01', 'HLA-DPA1*03:01-DPB1*94:01', 'HLA-DPA1*03:01-DPB1*95:01',
'HLA-DPA1*03:01-DPB1*96:01', 'HLA-DPA1*03:01-DPB1*97:01', 'HLA-DPA1*03:01-DPB1*98:01', 'HLA-DPA1*03:01-DPB1*99:01', 'HLA-DPA1*03:02-DPB1*01:01',
'HLA-DPA1*03:02-DPB1*02:01', 'HLA-DPA1*03:02-DPB1*02:02', 'HLA-DPA1*03:02-DPB1*03:01', 'HLA-DPA1*03:02-DPB1*04:01', 'HLA-DPA1*03:02-DPB1*04:02',
'HLA-DPA1*03:02-DPB1*05:01', 'HLA-DPA1*03:02-DPB1*06:01', 'HLA-DPA1*03:02-DPB1*08:01', 'HLA-DPA1*03:02-DPB1*09:01', 'HLA-DPA1*03:02-DPB1*10:001',
'HLA-DPA1*03:02-DPB1*10:01', 'HLA-DPA1*03:02-DPB1*10:101', 'HLA-DPA1*03:02-DPB1*10:201', 'HLA-DPA1*03:02-DPB1*10:301', 'HLA-DPA1*03:02-DPB1*10:401',
'HLA-DPA1*03:02-DPB1*10:501', 'HLA-DPA1*03:02-DPB1*10:601', 'HLA-DPA1*03:02-DPB1*10:701', 'HLA-DPA1*03:02-DPB1*10:801', 'HLA-DPA1*03:02-DPB1*10:901',
'HLA-DPA1*03:02-DPB1*11:001', 'HLA-DPA1*03:02-DPB1*11:01', 'HLA-DPA1*03:02-DPB1*11:101', 'HLA-DPA1*03:02-DPB1*11:201', 'HLA-DPA1*03:02-DPB1*11:301',
'HLA-DPA1*03:02-DPB1*11:401', 'HLA-DPA1*03:02-DPB1*11:501', 'HLA-DPA1*03:02-DPB1*11:601', 'HLA-DPA1*03:02-DPB1*11:701', 'HLA-DPA1*03:02-DPB1*11:801',
'HLA-DPA1*03:02-DPB1*11:901', 'HLA-DPA1*03:02-DPB1*12:101', 'HLA-DPA1*03:02-DPB1*12:201', 'HLA-DPA1*03:02-DPB1*12:301', 'HLA-DPA1*03:02-DPB1*12:401',
'HLA-DPA1*03:02-DPB1*12:501', 'HLA-DPA1*03:02-DPB1*12:601', 'HLA-DPA1*03:02-DPB1*12:701', 'HLA-DPA1*03:02-DPB1*12:801', 'HLA-DPA1*03:02-DPB1*12:901',
'HLA-DPA1*03:02-DPB1*13:001', 'HLA-DPA1*03:02-DPB1*13:01', 'HLA-DPA1*03:02-DPB1*13:101', 'HLA-DPA1*03:02-DPB1*13:201', 'HLA-DPA1*03:02-DPB1*13:301',
'HLA-DPA1*03:02-DPB1*13:401', 'HLA-DPA1*03:02-DPB1*14:01', 'HLA-DPA1*03:02-DPB1*15:01', 'HLA-DPA1*03:02-DPB1*16:01', 'HLA-DPA1*03:02-DPB1*17:01',
'HLA-DPA1*03:02-DPB1*18:01', 'HLA-DPA1*03:02-DPB1*19:01', 'HLA-DPA1*03:02-DPB1*20:01', 'HLA-DPA1*03:02-DPB1*21:01', 'HLA-DPA1*03:02-DPB1*22:01',
'HLA-DPA1*03:02-DPB1*23:01', 'HLA-DPA1*03:02-DPB1*24:01', 'HLA-DPA1*03:02-DPB1*25:01', 'HLA-DPA1*03:02-DPB1*26:01', 'HLA-DPA1*03:02-DPB1*27:01',
'HLA-DPA1*03:02-DPB1*28:01', 'HLA-DPA1*03:02-DPB1*29:01', 'HLA-DPA1*03:02-DPB1*30:01', 'HLA-DPA1*03:02-DPB1*31:01', 'HLA-DPA1*03:02-DPB1*32:01',
'HLA-DPA1*03:02-DPB1*33:01', 'HLA-DPA1*03:02-DPB1*34:01', 'HLA-DPA1*03:02-DPB1*35:01', 'HLA-DPA1*03:02-DPB1*36:01', 'HLA-DPA1*03:02-DPB1*37:01',
'HLA-DPA1*03:02-DPB1*38:01', 'HLA-DPA1*03:02-DPB1*39:01', 'HLA-DPA1*03:02-DPB1*40:01', 'HLA-DPA1*03:02-DPB1*41:01', 'HLA-DPA1*03:02-DPB1*44:01',
'HLA-DPA1*03:02-DPB1*45:01', 'HLA-DPA1*03:02-DPB1*46:01', 'HLA-DPA1*03:02-DPB1*47:01', 'HLA-DPA1*03:02-DPB1*48:01', 'HLA-DPA1*03:02-DPB1*49:01',
'HLA-DPA1*03:02-DPB1*50:01', 'HLA-DPA1*03:02-DPB1*51:01', 'HLA-DPA1*03:02-DPB1*52:01', 'HLA-DPA1*03:02-DPB1*53:01', 'HLA-DPA1*03:02-DPB1*54:01',
'HLA-DPA1*03:02-DPB1*55:01', 'HLA-DPA1*03:02-DPB1*56:01', 'HLA-DPA1*03:02-DPB1*58:01', 'HLA-DPA1*03:02-DPB1*59:01', 'HLA-DPA1*03:02-DPB1*60:01',
'HLA-DPA1*03:02-DPB1*62:01', 'HLA-DPA1*03:02-DPB1*63:01', 'HLA-DPA1*03:02-DPB1*65:01', 'HLA-DPA1*03:02-DPB1*66:01', 'HLA-DPA1*03:02-DPB1*67:01',
'HLA-DPA1*03:02-DPB1*68:01', 'HLA-DPA1*03:02-DPB1*69:01', 'HLA-DPA1*03:02-DPB1*70:01', 'HLA-DPA1*03:02-DPB1*71:01', 'HLA-DPA1*03:02-DPB1*72:01',
'HLA-DPA1*03:02-DPB1*73:01', 'HLA-DPA1*03:02-DPB1*74:01', 'HLA-DPA1*03:02-DPB1*75:01', 'HLA-DPA1*03:02-DPB1*76:01', 'HLA-DPA1*03:02-DPB1*77:01',
'HLA-DPA1*03:02-DPB1*78:01', 'HLA-DPA1*03:02-DPB1*79:01', 'HLA-DPA1*03:02-DPB1*80:01', 'HLA-DPA1*03:02-DPB1*81:01', 'HLA-DPA1*03:02-DPB1*82:01',
'HLA-DPA1*03:02-DPB1*83:01', 'HLA-DPA1*03:02-DPB1*84:01', 'HLA-DPA1*03:02-DPB1*85:01', 'HLA-DPA1*03:02-DPB1*86:01', 'HLA-DPA1*03:02-DPB1*87:01',
'HLA-DPA1*03:02-DPB1*88:01', 'HLA-DPA1*03:02-DPB1*89:01', 'HLA-DPA1*03:02-DPB1*90:01', 'HLA-DPA1*03:02-DPB1*91:01', 'HLA-DPA1*03:02-DPB1*92:01',
'HLA-DPA1*03:02-DPB1*93:01', 'HLA-DPA1*03:02-DPB1*94:01', 'HLA-DPA1*03:02-DPB1*95:01', 'HLA-DPA1*03:02-DPB1*96:01', 'HLA-DPA1*03:02-DPB1*97:01',
'HLA-DPA1*03:02-DPB1*98:01', 'HLA-DPA1*03:02-DPB1*99:01', 'HLA-DPA1*03:03-DPB1*01:01', 'HLA-DPA1*03:03-DPB1*02:01', 'HLA-DPA1*03:03-DPB1*02:02',
'HLA-DPA1*03:03-DPB1*03:01', 'HLA-DPA1*03:03-DPB1*04:01', 'HLA-DPA1*03:03-DPB1*04:02', 'HLA-DPA1*03:03-DPB1*05:01', 'HLA-DPA1*03:03-DPB1*06:01',
'HLA-DPA1*03:03-DPB1*08:01', 'HLA-DPA1*03:03-DPB1*09:01', 'HLA-DPA1*03:03-DPB1*10:001', 'HLA-DPA1*03:03-DPB1*10:01', 'HLA-DPA1*03:03-DPB1*10:101',
'HLA-DPA1*03:03-DPB1*10:201', 'HLA-DPA1*03:03-DPB1*10:301', 'HLA-DPA1*03:03-DPB1*10:401', 'HLA-DPA1*03:03-DPB1*10:501', 'HLA-DPA1*03:03-DPB1*10:601',
'HLA-DPA1*03:03-DPB1*10:701', 'HLA-DPA1*03:03-DPB1*10:801', 'HLA-DPA1*03:03-DPB1*10:901', 'HLA-DPA1*03:03-DPB1*11:001', 'HLA-DPA1*03:03-DPB1*11:01',
'HLA-DPA1*03:03-DPB1*11:101', 'HLA-DPA1*03:03-DPB1*11:201', 'HLA-DPA1*03:03-DPB1*11:301', 'HLA-DPA1*03:03-DPB1*11:401', 'HLA-DPA1*03:03-DPB1*11:501',
'HLA-DPA1*03:03-DPB1*11:601', 'HLA-DPA1*03:03-DPB1*11:701', 'HLA-DPA1*03:03-DPB1*11:801', 'HLA-DPA1*03:03-DPB1*11:901', 'HLA-DPA1*03:03-DPB1*12:101',
'HLA-DPA1*03:03-DPB1*12:201', 'HLA-DPA1*03:03-DPB1*12:301', 'HLA-DPA1*03:03-DPB1*12:401', 'HLA-DPA1*03:03-DPB1*12:501', 'HLA-DPA1*03:03-DPB1*12:601',
'HLA-DPA1*03:03-DPB1*12:701', 'HLA-DPA1*03:03-DPB1*12:801', 'HLA-DPA1*03:03-DPB1*12:901', 'HLA-DPA1*03:03-DPB1*13:001', 'HLA-DPA1*03:03-DPB1*13:01',
'HLA-DPA1*03:03-DPB1*13:101', 'HLA-DPA1*03:03-DPB1*13:201', 'HLA-DPA1*03:03-DPB1*13:301', 'HLA-DPA1*03:03-DPB1*13:401', 'HLA-DPA1*03:03-DPB1*14:01',
'HLA-DPA1*03:03-DPB1*15:01', 'HLA-DPA1*03:03-DPB1*16:01', 'HLA-DPA1*03:03-DPB1*17:01', 'HLA-DPA1*03:03-DPB1*18:01', 'HLA-DPA1*03:03-DPB1*19:01',
'HLA-DPA1*03:03-DPB1*20:01', 'HLA-DPA1*03:03-DPB1*21:01', 'HLA-DPA1*03:03-DPB1*22:01', 'HLA-DPA1*03:03-DPB1*23:01', 'HLA-DPA1*03:03-DPB1*24:01',
'HLA-DPA1*03:03-DPB1*25:01', 'HLA-DPA1*03:03-DPB1*26:01', 'HLA-DPA1*03:03-DPB1*27:01', 'HLA-DPA1*03:03-DPB1*28:01', 'HLA-DPA1*03:03-DPB1*29:01',
'HLA-DPA1*03:03-DPB1*30:01', 'HLA-DPA1*03:03-DPB1*31:01', 'HLA-DPA1*03:03-DPB1*32:01', 'HLA-DPA1*03:03-DPB1*33:01', 'HLA-DPA1*03:03-DPB1*34:01',
'HLA-DPA1*03:03-DPB1*35:01', 'HLA-DPA1*03:03-DPB1*36:01', 'HLA-DPA1*03:03-DPB1*37:01', 'HLA-DPA1*03:03-DPB1*38:01', 'HLA-DPA1*03:03-DPB1*39:01',
'HLA-DPA1*03:03-DPB1*40:01', 'HLA-DPA1*03:03-DPB1*41:01', 'HLA-DPA1*03:03-DPB1*44:01', 'HLA-DPA1*03:03-DPB1*45:01', 'HLA-DPA1*03:03-DPB1*46:01',
'HLA-DPA1*03:03-DPB1*47:01', 'HLA-DPA1*03:03-DPB1*48:01', 'HLA-DPA1*03:03-DPB1*49:01', 'HLA-DPA1*03:03-DPB1*50:01', 'HLA-DPA1*03:03-DPB1*51:01',
'HLA-DPA1*03:03-DPB1*52:01', 'HLA-DPA1*03:03-DPB1*53:01', 'HLA-DPA1*03:03-DPB1*54:01', 'HLA-DPA1*03:03-DPB1*55:01', 'HLA-DPA1*03:03-DPB1*56:01',
'HLA-DPA1*03:03-DPB1*58:01', 'HLA-DPA1*03:03-DPB1*59:01', 'HLA-DPA1*03:03-DPB1*60:01', 'HLA-DPA1*03:03-DPB1*62:01', 'HLA-DPA1*03:03-DPB1*63:01',
'HLA-DPA1*03:03-DPB1*65:01', 'HLA-DPA1*03:03-DPB1*66:01', 'HLA-DPA1*03:03-DPB1*67:01', 'HLA-DPA1*03:03-DPB1*68:01', 'HLA-DPA1*03:03-DPB1*69:01',
'HLA-DPA1*03:03-DPB1*70:01', 'HLA-DPA1*03:03-DPB1*71:01', 'HLA-DPA1*03:03-DPB1*72:01', 'HLA-DPA1*03:03-DPB1*73:01', 'HLA-DPA1*03:03-DPB1*74:01',
'HLA-DPA1*03:03-DPB1*75:01', 'HLA-DPA1*03:03-DPB1*76:01', 'HLA-DPA1*03:03-DPB1*77:01', 'HLA-DPA1*03:03-DPB1*78:01', 'HLA-DPA1*03:03-DPB1*79:01',
'HLA-DPA1*03:03-DPB1*80:01', 'HLA-DPA1*03:03-DPB1*81:01', 'HLA-DPA1*03:03-DPB1*82:01', 'HLA-DPA1*03:03-DPB1*83:01', 'HLA-DPA1*03:03-DPB1*84:01',
'HLA-DPA1*03:03-DPB1*85:01', 'HLA-DPA1*03:03-DPB1*86:01', 'HLA-DPA1*03:03-DPB1*87:01', 'HLA-DPA1*03:03-DPB1*88:01', 'HLA-DPA1*03:03-DPB1*89:01',
'HLA-DPA1*03:03-DPB1*90:01', 'HLA-DPA1*03:03-DPB1*91:01', 'HLA-DPA1*03:03-DPB1*92:01', 'HLA-DPA1*03:03-DPB1*93:01', 'HLA-DPA1*03:03-DPB1*94:01',
'HLA-DPA1*03:03-DPB1*95:01', 'HLA-DPA1*03:03-DPB1*96:01', 'HLA-DPA1*03:03-DPB1*97:01', 'HLA-DPA1*03:03-DPB1*98:01', 'HLA-DPA1*03:03-DPB1*99:01',
'HLA-DPA1*04:01-DPB1*01:01', 'HLA-DPA1*04:01-DPB1*02:01', 'HLA-DPA1*04:01-DPB1*02:02', 'HLA-DPA1*04:01-DPB1*03:01', 'HLA-DPA1*04:01-DPB1*04:01',
'HLA-DPA1*04:01-DPB1*04:02', 'HLA-DPA1*04:01-DPB1*05:01', 'HLA-DPA1*04:01-DPB1*06:01', 'HLA-DPA1*04:01-DPB1*08:01', 'HLA-DPA1*04:01-DPB1*09:01',
'HLA-DPA1*04:01-DPB1*10:001', 'HLA-DPA1*04:01-DPB1*10:01', 'HLA-DPA1*04:01-DPB1*10:101', 'HLA-DPA1*04:01-DPB1*10:201', 'HLA-DPA1*04:01-DPB1*10:301',
'HLA-DPA1*04:01-DPB1*10:401', 'HLA-DPA1*04:01-DPB1*10:501', 'HLA-DPA1*04:01-DPB1*10:601', 'HLA-DPA1*04:01-DPB1*10:701', 'HLA-DPA1*04:01-DPB1*10:801',
'HLA-DPA1*04:01-DPB1*10:901', 'HLA-DPA1*04:01-DPB1*11:001', 'HLA-DPA1*04:01-DPB1*11:01', 'HLA-DPA1*04:01-DPB1*11:101', 'HLA-DPA1*04:01-DPB1*11:201',
'HLA-DPA1*04:01-DPB1*11:301', 'HLA-DPA1*04:01-DPB1*11:401', 'HLA-DPA1*04:01-DPB1*11:501', 'HLA-DPA1*04:01-DPB1*11:601', 'HLA-DPA1*04:01-DPB1*11:701',
'HLA-DPA1*04:01-DPB1*11:801', 'HLA-DPA1*04:01-DPB1*11:901', 'HLA-DPA1*04:01-DPB1*12:101', 'HLA-DPA1*04:01-DPB1*12:201', 'HLA-DPA1*04:01-DPB1*12:301',
'HLA-DPA1*04:01-DPB1*12:401', 'HLA-DPA1*04:01-DPB1*12:501', 'HLA-DPA1*04:01-DPB1*12:601', 'HLA-DPA1*04:01-DPB1*12:701', 'HLA-DPA1*04:01-DPB1*12:801',
'HLA-DPA1*04:01-DPB1*12:901', 'HLA-DPA1*04:01-DPB1*13:001', 'HLA-DPA1*04:01-DPB1*13:01', 'HLA-DPA1*04:01-DPB1*13:101', 'HLA-DPA1*04:01-DPB1*13:201',
'HLA-DPA1*04:01-DPB1*13:301', 'HLA-DPA1*04:01-DPB1*13:401', 'HLA-DPA1*04:01-DPB1*14:01', 'HLA-DPA1*04:01-DPB1*15:01', 'HLA-DPA1*04:01-DPB1*16:01',
'HLA-DPA1*04:01-DPB1*17:01', 'HLA-DPA1*04:01-DPB1*18:01', 'HLA-DPA1*04:01-DPB1*19:01', 'HLA-DPA1*04:01-DPB1*20:01', 'HLA-DPA1*04:01-DPB1*21:01',
'HLA-DPA1*04:01-DPB1*22:01', 'HLA-DPA1*04:01-DPB1*23:01', 'HLA-DPA1*04:01-DPB1*24:01', 'HLA-DPA1*04:01-DPB1*25:01', 'HLA-DPA1*04:01-DPB1*26:01',
'HLA-DPA1*04:01-DPB1*27:01', 'HLA-DPA1*04:01-DPB1*28:01', 'HLA-DPA1*04:01-DPB1*29:01', 'HLA-DPA1*04:01-DPB1*30:01', 'HLA-DPA1*04:01-DPB1*31:01',
'HLA-DPA1*04:01-DPB1*32:01', 'HLA-DPA1*04:01-DPB1*33:01', 'HLA-DPA1*04:01-DPB1*34:01', 'HLA-DPA1*04:01-DPB1*35:01', 'HLA-DPA1*04:01-DPB1*36:01',
'HLA-DPA1*04:01-DPB1*37:01', 'HLA-DPA1*04:01-DPB1*38:01', 'HLA-DPA1*04:01-DPB1*39:01', 'HLA-DPA1*04:01-DPB1*40:01', 'HLA-DPA1*04:01-DPB1*41:01',
'HLA-DPA1*04:01-DPB1*44:01', 'HLA-DPA1*04:01-DPB1*45:01', 'HLA-DPA1*04:01-DPB1*46:01', 'HLA-DPA1*04:01-DPB1*47:01', 'HLA-DPA1*04:01-DPB1*48:01',
'HLA-DPA1*04:01-DPB1*49:01', 'HLA-DPA1*04:01-DPB1*50:01', 'HLA-DPA1*04:01-DPB1*51:01', 'HLA-DPA1*04:01-DPB1*52:01', 'HLA-DPA1*04:01-DPB1*53:01',
'HLA-DPA1*04:01-DPB1*54:01', 'HLA-DPA1*04:01-DPB1*55:01', 'HLA-DPA1*04:01-DPB1*56:01', 'HLA-DPA1*04:01-DPB1*58:01', 'HLA-DPA1*04:01-DPB1*59:01',
'HLA-DPA1*04:01-DPB1*60:01', 'HLA-DPA1*04:01-DPB1*62:01', 'HLA-DPA1*04:01-DPB1*63:01', 'HLA-DPA1*04:01-DPB1*65:01', 'HLA-DPA1*04:01-DPB1*66:01',
'HLA-DPA1*04:01-DPB1*67:01', 'HLA-DPA1*04:01-DPB1*68:01', 'HLA-DPA1*04:01-DPB1*69:01', 'HLA-DPA1*04:01-DPB1*70:01', 'HLA-DPA1*04:01-DPB1*71:01',
'HLA-DPA1*04:01-DPB1*72:01', 'HLA-DPA1*04:01-DPB1*73:01', 'HLA-DPA1*04:01-DPB1*74:01', 'HLA-DPA1*04:01-DPB1*75:01', 'HLA-DPA1*04:01-DPB1*76:01',
'HLA-DPA1*04:01-DPB1*77:01', 'HLA-DPA1*04:01-DPB1*78:01', 'HLA-DPA1*04:01-DPB1*79:01', 'HLA-DPA1*04:01-DPB1*80:01', 'HLA-DPA1*04:01-DPB1*81:01',
'HLA-DPA1*04:01-DPB1*82:01', 'HLA-DPA1*04:01-DPB1*83:01', 'HLA-DPA1*04:01-DPB1*84:01', 'HLA-DPA1*04:01-DPB1*85:01', 'HLA-DPA1*04:01-DPB1*86:01',
'HLA-DPA1*04:01-DPB1*87:01', 'HLA-DPA1*04:01-DPB1*88:01', 'HLA-DPA1*04:01-DPB1*89:01', 'HLA-DPA1*04:01-DPB1*90:01', 'HLA-DPA1*04:01-DPB1*91:01',
'HLA-DPA1*04:01-DPB1*92:01', 'HLA-DPA1*04:01-DPB1*93:01', 'HLA-DPA1*04:01-DPB1*94:01', 'HLA-DPA1*04:01-DPB1*95:01', 'HLA-DPA1*04:01-DPB1*96:01',
'HLA-DPA1*04:01-DPB1*97:01', 'HLA-DPA1*04:01-DPB1*98:01', 'HLA-DPA1*04:01-DPB1*99:01', 'HLA-DQA1*01:01-DQB1*02:01', 'HLA-DQA1*01:01-DQB1*02:02',
'HLA-DQA1*01:01-DQB1*02:03', 'HLA-DQA1*01:01-DQB1*02:04', 'HLA-DQA1*01:01-DQB1*02:05', 'HLA-DQA1*01:01-DQB1*02:06', 'HLA-DQA1*01:01-DQB1*03:01',
'HLA-DQA1*01:01-DQB1*03:02', 'HLA-DQA1*01:01-DQB1*03:03', 'HLA-DQA1*01:01-DQB1*03:04', 'HLA-DQA1*01:01-DQB1*03:05', 'HLA-DQA1*01:01-DQB1*03:06',
'HLA-DQA1*01:01-DQB1*03:07', 'HLA-DQA1*01:01-DQB1*03:08', 'HLA-DQA1*01:01-DQB1*03:09', 'HLA-DQA1*01:01-DQB1*03:10', 'HLA-DQA1*01:01-DQB1*03:11',
'HLA-DQA1*01:01-DQB1*03:12', 'HLA-DQA1*01:01-DQB1*03:13', 'HLA-DQA1*01:01-DQB1*03:14', 'HLA-DQA1*01:01-DQB1*03:15', 'HLA-DQA1*01:01-DQB1*03:16',
'HLA-DQA1*01:01-DQB1*03:17', 'HLA-DQA1*01:01-DQB1*03:18', 'HLA-DQA1*01:01-DQB1*03:19', 'HLA-DQA1*01:01-DQB1*03:20', 'HLA-DQA1*01:01-DQB1*03:21',
'HLA-DQA1*01:01-DQB1*03:22', 'HLA-DQA1*01:01-DQB1*03:23', 'HLA-DQA1*01:01-DQB1*03:24', 'HLA-DQA1*01:01-DQB1*03:25', 'HLA-DQA1*01:01-DQB1*03:26',
'HLA-DQA1*01:01-DQB1*03:27', 'HLA-DQA1*01:01-DQB1*03:28', 'HLA-DQA1*01:01-DQB1*03:29', 'HLA-DQA1*01:01-DQB1*03:30', 'HLA-DQA1*01:01-DQB1*03:31',
'HLA-DQA1*01:01-DQB1*03:32', 'HLA-DQA1*01:01-DQB1*03:33', 'HLA-DQA1*01:01-DQB1*03:34', 'HLA-DQA1*01:01-DQB1*03:35', 'HLA-DQA1*01:01-DQB1*03:36',
'HLA-DQA1*01:01-DQB1*03:37', 'HLA-DQA1*01:01-DQB1*03:38', 'HLA-DQA1*01:01-DQB1*04:01', 'HLA-DQA1*01:01-DQB1*04:02', 'HLA-DQA1*01:01-DQB1*04:03',
'HLA-DQA1*01:01-DQB1*04:04', 'HLA-DQA1*01:01-DQB1*04:05', 'HLA-DQA1*01:01-DQB1*04:06', 'HLA-DQA1*01:01-DQB1*04:07', 'HLA-DQA1*01:01-DQB1*04:08',
'HLA-DQA1*01:01-DQB1*05:01', 'HLA-DQA1*01:01-DQB1*05:02', 'HLA-DQA1*01:01-DQB1*05:03', 'HLA-DQA1*01:01-DQB1*05:05', 'HLA-DQA1*01:01-DQB1*05:06',
'HLA-DQA1*01:01-DQB1*05:07', 'HLA-DQA1*01:01-DQB1*05:08', 'HLA-DQA1*01:01-DQB1*05:09', 'HLA-DQA1*01:01-DQB1*05:10', 'HLA-DQA1*01:01-DQB1*05:11',
'HLA-DQA1*01:01-DQB1*05:12', 'HLA-DQA1*01:01-DQB1*05:13', 'HLA-DQA1*01:01-DQB1*05:14', 'HLA-DQA1*01:01-DQB1*06:01', 'HLA-DQA1*01:01-DQB1*06:02',
'HLA-DQA1*01:01-DQB1*06:03', 'HLA-DQA1*01:01-DQB1*06:04', 'HLA-DQA1*01:01-DQB1*06:07', 'HLA-DQA1*01:01-DQB1*06:08', 'HLA-DQA1*01:01-DQB1*06:09',
'HLA-DQA1*01:01-DQB1*06:10', 'HLA-DQA1*01:01-DQB1*06:11', 'HLA-DQA1*01:01-DQB1*06:12', 'HLA-DQA1*01:01-DQB1*06:14', 'HLA-DQA1*01:01-DQB1*06:15',
'HLA-DQA1*01:01-DQB1*06:16', 'HLA-DQA1*01:01-DQB1*06:17', 'HLA-DQA1*01:01-DQB1*06:18', 'HLA-DQA1*01:01-DQB1*06:19', 'HLA-DQA1*01:01-DQB1*06:21',
'HLA-DQA1*01:01-DQB1*06:22', 'HLA-DQA1*01:01-DQB1*06:23', 'HLA-DQA1*01:01-DQB1*06:24', 'HLA-DQA1*01:01-DQB1*06:25', 'HLA-DQA1*01:01-DQB1*06:27',
'HLA-DQA1*01:01-DQB1*06:28', 'HLA-DQA1*01:01-DQB1*06:29', 'HLA-DQA1*01:01-DQB1*06:30', 'HLA-DQA1*01:01-DQB1*06:31', 'HLA-DQA1*01:01-DQB1*06:32',
'HLA-DQA1*01:01-DQB1*06:33', 'HLA-DQA1*01:01-DQB1*06:34', 'HLA-DQA1*01:01-DQB1*06:35', 'HLA-DQA1*01:01-DQB1*06:36', 'HLA-DQA1*01:01-DQB1*06:37',
'HLA-DQA1*01:01-DQB1*06:38', 'HLA-DQA1*01:01-DQB1*06:39', 'HLA-DQA1*01:01-DQB1*06:40', 'HLA-DQA1*01:01-DQB1*06:41', 'HLA-DQA1*01:01-DQB1*06:42',
'HLA-DQA1*01:01-DQB1*06:43', 'HLA-DQA1*01:01-DQB1*06:44', 'HLA-DQA1*01:02-DQB1*02:01', 'HLA-DQA1*01:02-DQB1*02:02', 'HLA-DQA1*01:02-DQB1*02:03',
'HLA-DQA1*01:02-DQB1*02:04', 'HLA-DQA1*01:02-DQB1*02:05', 'HLA-DQA1*01:02-DQB1*02:06', 'HLA-DQA1*01:02-DQB1*03:01', 'HLA-DQA1*01:02-DQB1*03:02',
'HLA-DQA1*01:02-DQB1*03:03', 'HLA-DQA1*01:02-DQB1*03:04', 'HLA-DQA1*01:02-DQB1*03:05', 'HLA-DQA1*01:02-DQB1*03:06', 'HLA-DQA1*01:02-DQB1*03:07',
'HLA-DQA1*01:02-DQB1*03:08', 'HLA-DQA1*01:02-DQB1*03:09', 'HLA-DQA1*01:02-DQB1*03:10', 'HLA-DQA1*01:02-DQB1*03:11', 'HLA-DQA1*01:02-DQB1*03:12',
'HLA-DQA1*01:02-DQB1*03:13', 'HLA-DQA1*01:02-DQB1*03:14', 'HLA-DQA1*01:02-DQB1*03:15', 'HLA-DQA1*01:02-DQB1*03:16', 'HLA-DQA1*01:02-DQB1*03:17',
'HLA-DQA1*01:02-DQB1*03:18', 'HLA-DQA1*01:02-DQB1*03:19', 'HLA-DQA1*01:02-DQB1*03:20', 'HLA-DQA1*01:02-DQB1*03:21', 'HLA-DQA1*01:02-DQB1*03:22',
'HLA-DQA1*01:02-DQB1*03:23', 'HLA-DQA1*01:02-DQB1*03:24', 'HLA-DQA1*01:02-DQB1*03:25', 'HLA-DQA1*01:02-DQB1*03:26', 'HLA-DQA1*01:02-DQB1*03:27',
'HLA-DQA1*01:02-DQB1*03:28', 'HLA-DQA1*01:02-DQB1*03:29', 'HLA-DQA1*01:02-DQB1*03:30', 'HLA-DQA1*01:02-DQB1*03:31', 'HLA-DQA1*01:02-DQB1*03:32',
'HLA-DQA1*01:02-DQB1*03:33', 'HLA-DQA1*01:02-DQB1*03:34', 'HLA-DQA1*01:02-DQB1*03:35', 'HLA-DQA1*01:02-DQB1*03:36', 'HLA-DQA1*01:02-DQB1*03:37',
'HLA-DQA1*01:02-DQB1*03:38', 'HLA-DQA1*01:02-DQB1*04:01', 'HLA-DQA1*01:02-DQB1*04:02', 'HLA-DQA1*01:02-DQB1*04:03', 'HLA-DQA1*01:02-DQB1*04:04',
'HLA-DQA1*01:02-DQB1*04:05', 'HLA-DQA1*01:02-DQB1*04:06', 'HLA-DQA1*01:02-DQB1*04:07', 'HLA-DQA1*01:02-DQB1*04:08', 'HLA-DQA1*01:02-DQB1*05:01',
'HLA-DQA1*01:02-DQB1*05:02', 'HLA-DQA1*01:02-DQB1*05:03', 'HLA-DQA1*01:02-DQB1*05:05', 'HLA-DQA1*01:02-DQB1*05:06', 'HLA-DQA1*01:02-DQB1*05:07',
'HLA-DQA1*01:02-DQB1*05:08', 'HLA-DQA1*01:02-DQB1*05:09', 'HLA-DQA1*01:02-DQB1*05:10', 'HLA-DQA1*01:02-DQB1*05:11', 'HLA-DQA1*01:02-DQB1*05:12',
'HLA-DQA1*01:02-DQB1*05:13', 'HLA-DQA1*01:02-DQB1*05:14', 'HLA-DQA1*01:02-DQB1*06:01', 'HLA-DQA1*01:02-DQB1*06:02', 'HLA-DQA1*01:02-DQB1*06:03',
'HLA-DQA1*01:02-DQB1*06:04', 'HLA-DQA1*01:02-DQB1*06:07', 'HLA-DQA1*01:02-DQB1*06:08', 'HLA-DQA1*01:02-DQB1*06:09', 'HLA-DQA1*01:02-DQB1*06:10',
'HLA-DQA1*01:02-DQB1*06:11', 'HLA-DQA1*01:02-DQB1*06:12', 'HLA-DQA1*01:02-DQB1*06:14', 'HLA-DQA1*01:02-DQB1*06:15', 'HLA-DQA1*01:02-DQB1*06:16',
'HLA-DQA1*01:02-DQB1*06:17', 'HLA-DQA1*01:02-DQB1*06:18', 'HLA-DQA1*01:02-DQB1*06:19', 'HLA-DQA1*01:02-DQB1*06:21', 'HLA-DQA1*01:02-DQB1*06:22',
'HLA-DQA1*01:02-DQB1*06:23', 'HLA-DQA1*01:02-DQB1*06:24', 'HLA-DQA1*01:02-DQB1*06:25', 'HLA-DQA1*01:02-DQB1*06:27', 'HLA-DQA1*01:02-DQB1*06:28',
'HLA-DQA1*01:02-DQB1*06:29', 'HLA-DQA1*01:02-DQB1*06:30', 'HLA-DQA1*01:02-DQB1*06:31', 'HLA-DQA1*01:02-DQB1*06:32', 'HLA-DQA1*01:02-DQB1*06:33',
'HLA-DQA1*01:02-DQB1*06:34', 'HLA-DQA1*01:02-DQB1*06:35', 'HLA-DQA1*01:02-DQB1*06:36', 'HLA-DQA1*01:02-DQB1*06:37', 'HLA-DQA1*01:02-DQB1*06:38',
'HLA-DQA1*01:02-DQB1*06:39', 'HLA-DQA1*01:02-DQB1*06:40', 'HLA-DQA1*01:02-DQB1*06:41', 'HLA-DQA1*01:02-DQB1*06:42', 'HLA-DQA1*01:02-DQB1*06:43',
'HLA-DQA1*01:02-DQB1*06:44', 'HLA-DQA1*01:03-DQB1*02:01', 'HLA-DQA1*01:03-DQB1*02:02', 'HLA-DQA1*01:03-DQB1*02:03', 'HLA-DQA1*01:03-DQB1*02:04',
'HLA-DQA1*01:03-DQB1*02:05', 'HLA-DQA1*01:03-DQB1*02:06', 'HLA-DQA1*01:03-DQB1*03:01', 'HLA-DQA1*01:03-DQB1*03:02', 'HLA-DQA1*01:03-DQB1*03:03',
'HLA-DQA1*01:03-DQB1*03:04', 'HLA-DQA1*01:03-DQB1*03:05', 'HLA-DQA1*01:03-DQB1*03:06', 'HLA-DQA1*01:03-DQB1*03:07', 'HLA-DQA1*01:03-DQB1*03:08',
'HLA-DQA1*01:03-DQB1*03:09', 'HLA-DQA1*01:03-DQB1*03:10', 'HLA-DQA1*01:03-DQB1*03:11', 'HLA-DQA1*01:03-DQB1*03:12', 'HLA-DQA1*01:03-DQB1*03:13',
'HLA-DQA1*01:03-DQB1*03:14', 'HLA-DQA1*01:03-DQB1*03:15', 'HLA-DQA1*01:03-DQB1*03:16', 'HLA-DQA1*01:03-DQB1*03:17', 'HLA-DQA1*01:03-DQB1*03:18',
'HLA-DQA1*01:03-DQB1*03:19', 'HLA-DQA1*01:03-DQB1*03:20', 'HLA-DQA1*01:03-DQB1*03:21', 'HLA-DQA1*01:03-DQB1*03:22', 'HLA-DQA1*01:03-DQB1*03:23',
'HLA-DQA1*01:03-DQB1*03:24', 'HLA-DQA1*01:03-DQB1*03:25', 'HLA-DQA1*01:03-DQB1*03:26', 'HLA-DQA1*01:03-DQB1*03:27', 'HLA-DQA1*01:03-DQB1*03:28',
'HLA-DQA1*01:03-DQB1*03:29', 'HLA-DQA1*01:03-DQB1*03:30', 'HLA-DQA1*01:03-DQB1*03:31', 'HLA-DQA1*01:03-DQB1*03:32', 'HLA-DQA1*01:03-DQB1*03:33',
'HLA-DQA1*01:03-DQB1*03:34', 'HLA-DQA1*01:03-DQB1*03:35', 'HLA-DQA1*01:03-DQB1*03:36', 'HLA-DQA1*01:03-DQB1*03:37', 'HLA-DQA1*01:03-DQB1*03:38',
'HLA-DQA1*01:03-DQB1*04:01', 'HLA-DQA1*01:03-DQB1*04:02', 'HLA-DQA1*01:03-DQB1*04:03', 'HLA-DQA1*01:03-DQB1*04:04', 'HLA-DQA1*01:03-DQB1*04:05',
'HLA-DQA1*01:03-DQB1*04:06', 'HLA-DQA1*01:03-DQB1*04:07', 'HLA-DQA1*01:03-DQB1*04:08', 'HLA-DQA1*01:03-DQB1*05:01', 'HLA-DQA1*01:03-DQB1*05:02',
'HLA-DQA1*01:03-DQB1*05:03', 'HLA-DQA1*01:03-DQB1*05:05', 'HLA-DQA1*01:03-DQB1*05:06', 'HLA-DQA1*01:03-DQB1*05:07', 'HLA-DQA1*01:03-DQB1*05:08',
'HLA-DQA1*01:03-DQB1*05:09', 'HLA-DQA1*01:03-DQB1*05:10', 'HLA-DQA1*01:03-DQB1*05:11', 'HLA-DQA1*01:03-DQB1*05:12', 'HLA-DQA1*01:03-DQB1*05:13',
'HLA-DQA1*01:03-DQB1*05:14', 'HLA-DQA1*01:03-DQB1*06:01', 'HLA-DQA1*01:03-DQB1*06:02', 'HLA-DQA1*01:03-DQB1*06:03', 'HLA-DQA1*01:03-DQB1*06:04',
'HLA-DQA1*01:03-DQB1*06:07', 'HLA-DQA1*01:03-DQB1*06:08', 'HLA-DQA1*01:03-DQB1*06:09', 'HLA-DQA1*01:03-DQB1*06:10', 'HLA-DQA1*01:03-DQB1*06:11',
'HLA-DQA1*01:03-DQB1*06:12', 'HLA-DQA1*01:03-DQB1*06:14', 'HLA-DQA1*01:03-DQB1*06:15', 'HLA-DQA1*01:03-DQB1*06:16', 'HLA-DQA1*01:03-DQB1*06:17',
'HLA-DQA1*01:03-DQB1*06:18', 'HLA-DQA1*01:03-DQB1*06:19', 'HLA-DQA1*01:03-DQB1*06:21', 'HLA-DQA1*01:03-DQB1*06:22', 'HLA-DQA1*01:03-DQB1*06:23',
'HLA-DQA1*01:03-DQB1*06:24', 'HLA-DQA1*01:03-DQB1*06:25', 'HLA-DQA1*01:03-DQB1*06:27', 'HLA-DQA1*01:03-DQB1*06:28', 'HLA-DQA1*01:03-DQB1*06:29',
'HLA-DQA1*01:03-DQB1*06:30', 'HLA-DQA1*01:03-DQB1*06:31', 'HLA-DQA1*01:03-DQB1*06:32', 'HLA-DQA1*01:03-DQB1*06:33', 'HLA-DQA1*01:03-DQB1*06:34',
'HLA-DQA1*01:03-DQB1*06:35', 'HLA-DQA1*01:03-DQB1*06:36', 'HLA-DQA1*01:03-DQB1*06:37', 'HLA-DQA1*01:03-DQB1*06:38', 'HLA-DQA1*01:03-DQB1*06:39',
'HLA-DQA1*01:03-DQB1*06:40', 'HLA-DQA1*01:03-DQB1*06:41', 'HLA-DQA1*01:03-DQB1*06:42', 'HLA-DQA1*01:03-DQB1*06:43', 'HLA-DQA1*01:03-DQB1*06:44',
'HLA-DQA1*01:04-DQB1*02:01', 'HLA-DQA1*01:04-DQB1*02:02', 'HLA-DQA1*01:04-DQB1*02:03', 'HLA-DQA1*01:04-DQB1*02:04', 'HLA-DQA1*01:04-DQB1*02:05',
'HLA-DQA1*01:04-DQB1*02:06', 'HLA-DQA1*01:04-DQB1*03:01', 'HLA-DQA1*01:04-DQB1*03:02', 'HLA-DQA1*01:04-DQB1*03:03', 'HLA-DQA1*01:04-DQB1*03:04',
'HLA-DQA1*01:04-DQB1*03:05', 'HLA-DQA1*01:04-DQB1*03:06', 'HLA-DQA1*01:04-DQB1*03:07', 'HLA-DQA1*01:04-DQB1*03:08', 'HLA-DQA1*01:04-DQB1*03:09',
'HLA-DQA1*01:04-DQB1*03:10', 'HLA-DQA1*01:04-DQB1*03:11', 'HLA-DQA1*01:04-DQB1*03:12', 'HLA-DQA1*01:04-DQB1*03:13', 'HLA-DQA1*01:04-DQB1*03:14',
'HLA-DQA1*01:04-DQB1*03:15', 'HLA-DQA1*01:04-DQB1*03:16', 'HLA-DQA1*01:04-DQB1*03:17', 'HLA-DQA1*01:04-DQB1*03:18', 'HLA-DQA1*01:04-DQB1*03:19',
'HLA-DQA1*01:04-DQB1*03:20', 'HLA-DQA1*01:04-DQB1*03:21', 'HLA-DQA1*01:04-DQB1*03:22', 'HLA-DQA1*01:04-DQB1*03:23', 'HLA-DQA1*01:04-DQB1*03:24',
'HLA-DQA1*01:04-DQB1*03:25', 'HLA-DQA1*01:04-DQB1*03:26', 'HLA-DQA1*01:04-DQB1*03:27', 'HLA-DQA1*01:04-DQB1*03:28', 'HLA-DQA1*01:04-DQB1*03:29',
'HLA-DQA1*01:04-DQB1*03:30', 'HLA-DQA1*01:04-DQB1*03:31', 'HLA-DQA1*01:04-DQB1*03:32', 'HLA-DQA1*01:04-DQB1*03:33', 'HLA-DQA1*01:04-DQB1*03:34',
'HLA-DQA1*01:04-DQB1*03:35', 'HLA-DQA1*01:04-DQB1*03:36', 'HLA-DQA1*01:04-DQB1*03:37', 'HLA-DQA1*01:04-DQB1*03:38', 'HLA-DQA1*01:04-DQB1*04:01',
'HLA-DQA1*01:04-DQB1*04:02', 'HLA-DQA1*01:04-DQB1*04:03', 'HLA-DQA1*01:04-DQB1*04:04', 'HLA-DQA1*01:04-DQB1*04:05', 'HLA-DQA1*01:04-DQB1*04:06',
'HLA-DQA1*01:04-DQB1*04:07', 'HLA-DQA1*01:04-DQB1*04:08', 'HLA-DQA1*01:04-DQB1*05:01', 'HLA-DQA1*01:04-DQB1*05:02', 'HLA-DQA1*01:04-DQB1*05:03',
'HLA-DQA1*01:04-DQB1*05:05', 'HLA-DQA1*01:04-DQB1*05:06', 'HLA-DQA1*01:04-DQB1*05:07', 'HLA-DQA1*01:04-DQB1*05:08', 'HLA-DQA1*01:04-DQB1*05:09',
'HLA-DQA1*01:04-DQB1*05:10', 'HLA-DQA1*01:04-DQB1*05:11', 'HLA-DQA1*01:04-DQB1*05:12', 'HLA-DQA1*01:04-DQB1*05:13', 'HLA-DQA1*01:04-DQB1*05:14',
'HLA-DQA1*01:04-DQB1*06:01', 'HLA-DQA1*01:04-DQB1*06:02', 'HLA-DQA1*01:04-DQB1*06:03', 'HLA-DQA1*01:04-DQB1*06:04', 'HLA-DQA1*01:04-DQB1*06:07',
'HLA-DQA1*01:04-DQB1*06:08', 'HLA-DQA1*01:04-DQB1*06:09', 'HLA-DQA1*01:04-DQB1*06:10', 'HLA-DQA1*01:04-DQB1*06:11', 'HLA-DQA1*01:04-DQB1*06:12',
'HLA-DQA1*01:04-DQB1*06:14', 'HLA-DQA1*01:04-DQB1*06:15', 'HLA-DQA1*01:04-DQB1*06:16', 'HLA-DQA1*01:04-DQB1*06:17', 'HLA-DQA1*01:04-DQB1*06:18',
'HLA-DQA1*01:04-DQB1*06:19', 'HLA-DQA1*01:04-DQB1*06:21', 'HLA-DQA1*01:04-DQB1*06:22', 'HLA-DQA1*01:04-DQB1*06:23', 'HLA-DQA1*01:04-DQB1*06:24',
'HLA-DQA1*01:04-DQB1*06:25', 'HLA-DQA1*01:04-DQB1*06:27', 'HLA-DQA1*01:04-DQB1*06:28', 'HLA-DQA1*01:04-DQB1*06:29', 'HLA-DQA1*01:04-DQB1*06:30',
'HLA-DQA1*01:04-DQB1*06:31', 'HLA-DQA1*01:04-DQB1*06:32', 'HLA-DQA1*01:04-DQB1*06:33', 'HLA-DQA1*01:04-DQB1*06:34', 'HLA-DQA1*01:04-DQB1*06:35',
'HLA-DQA1*01:04-DQB1*06:36', 'HLA-DQA1*01:04-DQB1*06:37', 'HLA-DQA1*01:04-DQB1*06:38', 'HLA-DQA1*01:04-DQB1*06:39', 'HLA-DQA1*01:04-DQB1*06:40',
'HLA-DQA1*01:04-DQB1*06:41', 'HLA-DQA1*01:04-DQB1*06:42', 'HLA-DQA1*01:04-DQB1*06:43', 'HLA-DQA1*01:04-DQB1*06:44', 'HLA-DQA1*01:05-DQB1*02:01',
'HLA-DQA1*01:05-DQB1*02:02', 'HLA-DQA1*01:05-DQB1*02:03', 'HLA-DQA1*01:05-DQB1*02:04', 'HLA-DQA1*01:05-DQB1*02:05', 'HLA-DQA1*01:05-DQB1*02:06',
'HLA-DQA1*01:05-DQB1*03:01', 'HLA-DQA1*01:05-DQB1*03:02', 'HLA-DQA1*01:05-DQB1*03:03', 'HLA-DQA1*01:05-DQB1*03:04', 'HLA-DQA1*01:05-DQB1*03:05',
'HLA-DQA1*01:05-DQB1*03:06', 'HLA-DQA1*01:05-DQB1*03:07', 'HLA-DQA1*01:05-DQB1*03:08', 'HLA-DQA1*01:05-DQB1*03:09', 'HLA-DQA1*01:05-DQB1*03:10',
'HLA-DQA1*01:05-DQB1*03:11', 'HLA-DQA1*01:05-DQB1*03:12', 'HLA-DQA1*01:05-DQB1*03:13', 'HLA-DQA1*01:05-DQB1*03:14', 'HLA-DQA1*01:05-DQB1*03:15',
'HLA-DQA1*01:05-DQB1*03:16', 'HLA-DQA1*01:05-DQB1*03:17', 'HLA-DQA1*01:05-DQB1*03:18', 'HLA-DQA1*01:05-DQB1*03:19', 'HLA-DQA1*01:05-DQB1*03:20',
'HLA-DQA1*01:05-DQB1*03:21', 'HLA-DQA1*01:05-DQB1*03:22', 'HLA-DQA1*01:05-DQB1*03:23', 'HLA-DQA1*01:05-DQB1*03:24', 'HLA-DQA1*01:05-DQB1*03:25',
'HLA-DQA1*01:05-DQB1*03:26', 'HLA-DQA1*01:05-DQB1*03:27', 'HLA-DQA1*01:05-DQB1*03:28', 'HLA-DQA1*01:05-DQB1*03:29', 'HLA-DQA1*01:05-DQB1*03:30',
'HLA-DQA1*01:05-DQB1*03:31', 'HLA-DQA1*01:05-DQB1*03:32', 'HLA-DQA1*01:05-DQB1*03:33', 'HLA-DQA1*01:05-DQB1*03:34', 'HLA-DQA1*01:05-DQB1*03:35',
'HLA-DQA1*01:05-DQB1*03:36', 'HLA-DQA1*01:05-DQB1*03:37', 'HLA-DQA1*01:05-DQB1*03:38', 'HLA-DQA1*01:05-DQB1*04:01', 'HLA-DQA1*01:05-DQB1*04:02',
'HLA-DQA1*01:05-DQB1*04:03', 'HLA-DQA1*01:05-DQB1*04:04', 'HLA-DQA1*01:05-DQB1*04:05', 'HLA-DQA1*01:05-DQB1*04:06', 'HLA-DQA1*01:05-DQB1*04:07',
'HLA-DQA1*01:05-DQB1*04:08', 'HLA-DQA1*01:05-DQB1*05:01', 'HLA-DQA1*01:05-DQB1*05:02', 'HLA-DQA1*01:05-DQB1*05:03', 'HLA-DQA1*01:05-DQB1*05:05',
'HLA-DQA1*01:05-DQB1*05:06', 'HLA-DQA1*01:05-DQB1*05:07', 'HLA-DQA1*01:05-DQB1*05:08', 'HLA-DQA1*01:05-DQB1*05:09', 'HLA-DQA1*01:05-DQB1*05:10',
'HLA-DQA1*01:05-DQB1*05:11', 'HLA-DQA1*01:05-DQB1*05:12', 'HLA-DQA1*01:05-DQB1*05:13', 'HLA-DQA1*01:05-DQB1*05:14', 'HLA-DQA1*01:05-DQB1*06:01',
'HLA-DQA1*01:05-DQB1*06:02', 'HLA-DQA1*01:05-DQB1*06:03', 'HLA-DQA1*01:05-DQB1*06:04', 'HLA-DQA1*01:05-DQB1*06:07', 'HLA-DQA1*01:05-DQB1*06:08',
'HLA-DQA1*01:05-DQB1*06:09', 'HLA-DQA1*01:05-DQB1*06:10', 'HLA-DQA1*01:05-DQB1*06:11', 'HLA-DQA1*01:05-DQB1*06:12', 'HLA-DQA1*01:05-DQB1*06:14',
'HLA-DQA1*01:05-DQB1*06:15', 'HLA-DQA1*01:05-DQB1*06:16', 'HLA-DQA1*01:05-DQB1*06:17', 'HLA-DQA1*01:05-DQB1*06:18', 'HLA-DQA1*01:05-DQB1*06:19',
'HLA-DQA1*01:05-DQB1*06:21', 'HLA-DQA1*01:05-DQB1*06:22', 'HLA-DQA1*01:05-DQB1*06:23', 'HLA-DQA1*01:05-DQB1*06:24', 'HLA-DQA1*01:05-DQB1*06:25',
'HLA-DQA1*01:05-DQB1*06:27', 'HLA-DQA1*01:05-DQB1*06:28', 'HLA-DQA1*01:05-DQB1*06:29', 'HLA-DQA1*01:05-DQB1*06:30', 'HLA-DQA1*01:05-DQB1*06:31',
'HLA-DQA1*01:05-DQB1*06:32', 'HLA-DQA1*01:05-DQB1*06:33', 'HLA-DQA1*01:05-DQB1*06:34', 'HLA-DQA1*01:05-DQB1*06:35', 'HLA-DQA1*01:05-DQB1*06:36',
'HLA-DQA1*01:05-DQB1*06:37', 'HLA-DQA1*01:05-DQB1*06:38', 'HLA-DQA1*01:05-DQB1*06:39', 'HLA-DQA1*01:05-DQB1*06:40', 'HLA-DQA1*01:05-DQB1*06:41',
'HLA-DQA1*01:05-DQB1*06:42', 'HLA-DQA1*01:05-DQB1*06:43', 'HLA-DQA1*01:05-DQB1*06:44', 'HLA-DQA1*01:06-DQB1*02:01', 'HLA-DQA1*01:06-DQB1*02:02',
'HLA-DQA1*01:06-DQB1*02:03', 'HLA-DQA1*01:06-DQB1*02:04', 'HLA-DQA1*01:06-DQB1*02:05', 'HLA-DQA1*01:06-DQB1*02:06', 'HLA-DQA1*01:06-DQB1*03:01',
'HLA-DQA1*01:06-DQB1*03:02', 'HLA-DQA1*01:06-DQB1*03:03', 'HLA-DQA1*01:06-DQB1*03:04', 'HLA-DQA1*01:06-DQB1*03:05', 'HLA-DQA1*01:06-DQB1*03:06',
'HLA-DQA1*01:06-DQB1*03:07', 'HLA-DQA1*01:06-DQB1*03:08', 'HLA-DQA1*01:06-DQB1*03:09', 'HLA-DQA1*01:06-DQB1*03:10', 'HLA-DQA1*01:06-DQB1*03:11',
'HLA-DQA1*01:06-DQB1*03:12', 'HLA-DQA1*01:06-DQB1*03:13', 'HLA-DQA1*01:06-DQB1*03:14', 'HLA-DQA1*01:06-DQB1*03:15', 'HLA-DQA1*01:06-DQB1*03:16',
'HLA-DQA1*01:06-DQB1*03:17', 'HLA-DQA1*01:06-DQB1*03:18', 'HLA-DQA1*01:06-DQB1*03:19', 'HLA-DQA1*01:06-DQB1*03:20', 'HLA-DQA1*01:06-DQB1*03:21',
'HLA-DQA1*01:06-DQB1*03:22', 'HLA-DQA1*01:06-DQB1*03:23', 'HLA-DQA1*01:06-DQB1*03:24', 'HLA-DQA1*01:06-DQB1*03:25', 'HLA-DQA1*01:06-DQB1*03:26',
'HLA-DQA1*01:06-DQB1*03:27', 'HLA-DQA1*01:06-DQB1*03:28', 'HLA-DQA1*01:06-DQB1*03:29', 'HLA-DQA1*01:06-DQB1*03:30', 'HLA-DQA1*01:06-DQB1*03:31',
'HLA-DQA1*01:06-DQB1*03:32', 'HLA-DQA1*01:06-DQB1*03:33', 'HLA-DQA1*01:06-DQB1*03:34', 'HLA-DQA1*01:06-DQB1*03:35', 'HLA-DQA1*01:06-DQB1*03:36',
'HLA-DQA1*01:06-DQB1*03:37', 'HLA-DQA1*01:06-DQB1*03:38', 'HLA-DQA1*01:06-DQB1*04:01', 'HLA-DQA1*01:06-DQB1*04:02', 'HLA-DQA1*01:06-DQB1*04:03',
'HLA-DQA1*01:06-DQB1*04:04', 'HLA-DQA1*01:06-DQB1*04:05', 'HLA-DQA1*01:06-DQB1*04:06', 'HLA-DQA1*01:06-DQB1*04:07', 'HLA-DQA1*01:06-DQB1*04:08',
'HLA-DQA1*01:06-DQB1*05:01', 'HLA-DQA1*01:06-DQB1*05:02', 'HLA-DQA1*01:06-DQB1*05:03', 'HLA-DQA1*01:06-DQB1*05:05', 'HLA-DQA1*01:06-DQB1*05:06',
'HLA-DQA1*01:06-DQB1*05:07', 'HLA-DQA1*01:06-DQB1*05:08', 'HLA-DQA1*01:06-DQB1*05:09', 'HLA-DQA1*01:06-DQB1*05:10', 'HLA-DQA1*01:06-DQB1*05:11',
'HLA-DQA1*01:06-DQB1*05:12', 'HLA-DQA1*01:06-DQB1*05:13', 'HLA-DQA1*01:06-DQB1*05:14', 'HLA-DQA1*01:06-DQB1*06:01', 'HLA-DQA1*01:06-DQB1*06:02',
'HLA-DQA1*01:06-DQB1*06:03', 'HLA-DQA1*01:06-DQB1*06:04', 'HLA-DQA1*01:06-DQB1*06:07', 'HLA-DQA1*01:06-DQB1*06:08', 'HLA-DQA1*01:06-DQB1*06:09',
'HLA-DQA1*01:06-DQB1*06:10', 'HLA-DQA1*01:06-DQB1*06:11', 'HLA-DQA1*01:06-DQB1*06:12', 'HLA-DQA1*01:06-DQB1*06:14', 'HLA-DQA1*01:06-DQB1*06:15',
'HLA-DQA1*01:06-DQB1*06:16', 'HLA-DQA1*01:06-DQB1*06:17', 'HLA-DQA1*01:06-DQB1*06:18', 'HLA-DQA1*01:06-DQB1*06:19', 'HLA-DQA1*01:06-DQB1*06:21',
'HLA-DQA1*01:06-DQB1*06:22', 'HLA-DQA1*01:06-DQB1*06:23', 'HLA-DQA1*01:06-DQB1*06:24', 'HLA-DQA1*01:06-DQB1*06:25', 'HLA-DQA1*01:06-DQB1*06:27',
'HLA-DQA1*01:06-DQB1*06:28', 'HLA-DQA1*01:06-DQB1*06:29', 'HLA-DQA1*01:06-DQB1*06:30', 'HLA-DQA1*01:06-DQB1*06:31', 'HLA-DQA1*01:06-DQB1*06:32',
'HLA-DQA1*01:06-DQB1*06:33', 'HLA-DQA1*01:06-DQB1*06:34', 'HLA-DQA1*01:06-DQB1*06:35', 'HLA-DQA1*01:06-DQB1*06:36', 'HLA-DQA1*01:06-DQB1*06:37',
'HLA-DQA1*01:06-DQB1*06:38', 'HLA-DQA1*01:06-DQB1*06:39', 'HLA-DQA1*01:06-DQB1*06:40', 'HLA-DQA1*01:06-DQB1*06:41', 'HLA-DQA1*01:06-DQB1*06:42',
'HLA-DQA1*01:06-DQB1*06:43', 'HLA-DQA1*01:06-DQB1*06:44', 'HLA-DQA1*01:07-DQB1*02:01', 'HLA-DQA1*01:07-DQB1*02:02', 'HLA-DQA1*01:07-DQB1*02:03',
'HLA-DQA1*01:07-DQB1*02:04', 'HLA-DQA1*01:07-DQB1*02:05', 'HLA-DQA1*01:07-DQB1*02:06', 'HLA-DQA1*01:07-DQB1*03:01', 'HLA-DQA1*01:07-DQB1*03:02',
'HLA-DQA1*01:07-DQB1*03:03', 'HLA-DQA1*01:07-DQB1*03:04', 'HLA-DQA1*01:07-DQB1*03:05', 'HLA-DQA1*01:07-DQB1*03:06', 'HLA-DQA1*01:07-DQB1*03:07',
'HLA-DQA1*01:07-DQB1*03:08', 'HLA-DQA1*01:07-DQB1*03:09', 'HLA-DQA1*01:07-DQB1*03:10', 'HLA-DQA1*01:07-DQB1*03:11', 'HLA-DQA1*01:07-DQB1*03:12',
'HLA-DQA1*01:07-DQB1*03:13', 'HLA-DQA1*01:07-DQB1*03:14', 'HLA-DQA1*01:07-DQB1*03:15', 'HLA-DQA1*01:07-DQB1*03:16', 'HLA-DQA1*01:07-DQB1*03:17',
'HLA-DQA1*01:07-DQB1*03:18', 'HLA-DQA1*01:07-DQB1*03:19', 'HLA-DQA1*01:07-DQB1*03:20', 'HLA-DQA1*01:07-DQB1*03:21', 'HLA-DQA1*01:07-DQB1*03:22',
'HLA-DQA1*01:07-DQB1*03:23', 'HLA-DQA1*01:07-DQB1*03:24', 'HLA-DQA1*01:07-DQB1*03:25', 'HLA-DQA1*01:07-DQB1*03:26', 'HLA-DQA1*01:07-DQB1*03:27',
'HLA-DQA1*01:07-DQB1*03:28', 'HLA-DQA1*01:07-DQB1*03:29', 'HLA-DQA1*01:07-DQB1*03:30', 'HLA-DQA1*01:07-DQB1*03:31', 'HLA-DQA1*01:07-DQB1*03:32',
'HLA-DQA1*01:07-DQB1*03:33', 'HLA-DQA1*01:07-DQB1*03:34', 'HLA-DQA1*01:07-DQB1*03:35', 'HLA-DQA1*01:07-DQB1*03:36', 'HLA-DQA1*01:07-DQB1*03:37',
'HLA-DQA1*01:07-DQB1*03:38', 'HLA-DQA1*01:07-DQB1*04:01', 'HLA-DQA1*01:07-DQB1*04:02', 'HLA-DQA1*01:07-DQB1*04:03', 'HLA-DQA1*01:07-DQB1*04:04',
'HLA-DQA1*01:07-DQB1*04:05', 'HLA-DQA1*01:07-DQB1*04:06', 'HLA-DQA1*01:07-DQB1*04:07', 'HLA-DQA1*01:07-DQB1*04:08', 'HLA-DQA1*01:07-DQB1*05:01',
'HLA-DQA1*01:07-DQB1*05:02', 'HLA-DQA1*01:07-DQB1*05:03', 'HLA-DQA1*01:07-DQB1*05:05', 'HLA-DQA1*01:07-DQB1*05:06', 'HLA-DQA1*01:07-DQB1*05:07',
'HLA-DQA1*01:07-DQB1*05:08', 'HLA-DQA1*01:07-DQB1*05:09', 'HLA-DQA1*01:07-DQB1*05:10', 'HLA-DQA1*01:07-DQB1*05:11', 'HLA-DQA1*01:07-DQB1*05:12',
'HLA-DQA1*01:07-DQB1*05:13', 'HLA-DQA1*01:07-DQB1*05:14', 'HLA-DQA1*01:07-DQB1*06:01', 'HLA-DQA1*01:07-DQB1*06:02', 'HLA-DQA1*01:07-DQB1*06:03',
'HLA-DQA1*01:07-DQB1*06:04', 'HLA-DQA1*01:07-DQB1*06:07', 'HLA-DQA1*01:07-DQB1*06:08', 'HLA-DQA1*01:07-DQB1*06:09', 'HLA-DQA1*01:07-DQB1*06:10',
'HLA-DQA1*01:07-DQB1*06:11', 'HLA-DQA1*01:07-DQB1*06:12', 'HLA-DQA1*01:07-DQB1*06:14', 'HLA-DQA1*01:07-DQB1*06:15', 'HLA-DQA1*01:07-DQB1*06:16',
'HLA-DQA1*01:07-DQB1*06:17', 'HLA-DQA1*01:07-DQB1*06:18', 'HLA-DQA1*01:07-DQB1*06:19', 'HLA-DQA1*01:07-DQB1*06:21', 'HLA-DQA1*01:07-DQB1*06:22',
'HLA-DQA1*01:07-DQB1*06:23', 'HLA-DQA1*01:07-DQB1*06:24', 'HLA-DQA1*01:07-DQB1*06:25', 'HLA-DQA1*01:07-DQB1*06:27', 'HLA-DQA1*01:07-DQB1*06:28',
'HLA-DQA1*01:07-DQB1*06:29', 'HLA-DQA1*01:07-DQB1*06:30', 'HLA-DQA1*01:07-DQB1*06:31', 'HLA-DQA1*01:07-DQB1*06:32', 'HLA-DQA1*01:07-DQB1*06:33',
'HLA-DQA1*01:07-DQB1*06:34', 'HLA-DQA1*01:07-DQB1*06:35', 'HLA-DQA1*01:07-DQB1*06:36', 'HLA-DQA1*01:07-DQB1*06:37', 'HLA-DQA1*01:07-DQB1*06:38',
'HLA-DQA1*01:07-DQB1*06:39', 'HLA-DQA1*01:07-DQB1*06:40', 'HLA-DQA1*01:07-DQB1*06:41', 'HLA-DQA1*01:07-DQB1*06:42', 'HLA-DQA1*01:07-DQB1*06:43',
'HLA-DQA1*01:07-DQB1*06:44', 'HLA-DQA1*01:08-DQB1*02:01', 'HLA-DQA1*01:08-DQB1*02:02', 'HLA-DQA1*01:08-DQB1*02:03', 'HLA-DQA1*01:08-DQB1*02:04',
'HLA-DQA1*01:08-DQB1*02:05', 'HLA-DQA1*01:08-DQB1*02:06', 'HLA-DQA1*01:08-DQB1*03:01', 'HLA-DQA1*01:08-DQB1*03:02', 'HLA-DQA1*01:08-DQB1*03:03',
'HLA-DQA1*01:08-DQB1*03:04', 'HLA-DQA1*01:08-DQB1*03:05', 'HLA-DQA1*01:08-DQB1*03:06', 'HLA-DQA1*01:08-DQB1*03:07', 'HLA-DQA1*01:08-DQB1*03:08',
'HLA-DQA1*01:08-DQB1*03:09', 'HLA-DQA1*01:08-DQB1*03:10', 'HLA-DQA1*01:08-DQB1*03:11', 'HLA-DQA1*01:08-DQB1*03:12', 'HLA-DQA1*01:08-DQB1*03:13',
'HLA-DQA1*01:08-DQB1*03:14', 'HLA-DQA1*01:08-DQB1*03:15', 'HLA-DQA1*01:08-DQB1*03:16', 'HLA-DQA1*01:08-DQB1*03:17', 'HLA-DQA1*01:08-DQB1*03:18',
'HLA-DQA1*01:08-DQB1*03:19', 'HLA-DQA1*01:08-DQB1*03:20', 'HLA-DQA1*01:08-DQB1*03:21', 'HLA-DQA1*01:08-DQB1*03:22', 'HLA-DQA1*01:08-DQB1*03:23',
'HLA-DQA1*01:08-DQB1*03:24', 'HLA-DQA1*01:08-DQB1*03:25', 'HLA-DQA1*01:08-DQB1*03:26', 'HLA-DQA1*01:08-DQB1*03:27', 'HLA-DQA1*01:08-DQB1*03:28',
'HLA-DQA1*01:08-DQB1*03:29', 'HLA-DQA1*01:08-DQB1*03:30', 'HLA-DQA1*01:08-DQB1*03:31', 'HLA-DQA1*01:08-DQB1*03:32', 'HLA-DQA1*01:08-DQB1*03:33',
'HLA-DQA1*01:08-DQB1*03:34', 'HLA-DQA1*01:08-DQB1*03:35', 'HLA-DQA1*01:08-DQB1*03:36', 'HLA-DQA1*01:08-DQB1*03:37', 'HLA-DQA1*01:08-DQB1*03:38',
'HLA-DQA1*01:08-DQB1*04:01', 'HLA-DQA1*01:08-DQB1*04:02', 'HLA-DQA1*01:08-DQB1*04:03', 'HLA-DQA1*01:08-DQB1*04:04', 'HLA-DQA1*01:08-DQB1*04:05',
'HLA-DQA1*01:08-DQB1*04:06', 'HLA-DQA1*01:08-DQB1*04:07', 'HLA-DQA1*01:08-DQB1*04:08', 'HLA-DQA1*01:08-DQB1*05:01', 'HLA-DQA1*01:08-DQB1*05:02',
'HLA-DQA1*01:08-DQB1*05:03', 'HLA-DQA1*01:08-DQB1*05:05', 'HLA-DQA1*01:08-DQB1*05:06', 'HLA-DQA1*01:08-DQB1*05:07', 'HLA-DQA1*01:08-DQB1*05:08',
'HLA-DQA1*01:08-DQB1*05:09', 'HLA-DQA1*01:08-DQB1*05:10', 'HLA-DQA1*01:08-DQB1*05:11', 'HLA-DQA1*01:08-DQB1*05:12', 'HLA-DQA1*01:08-DQB1*05:13',
'HLA-DQA1*01:08-DQB1*05:14', 'HLA-DQA1*01:08-DQB1*06:01', 'HLA-DQA1*01:08-DQB1*06:02', 'HLA-DQA1*01:08-DQB1*06:03', 'HLA-DQA1*01:08-DQB1*06:04',
'HLA-DQA1*01:08-DQB1*06:07', 'HLA-DQA1*01:08-DQB1*06:08', 'HLA-DQA1*01:08-DQB1*06:09', 'HLA-DQA1*01:08-DQB1*06:10', 'HLA-DQA1*01:08-DQB1*06:11',
'HLA-DQA1*01:08-DQB1*06:12', 'HLA-DQA1*01:08-DQB1*06:14', 'HLA-DQA1*01:08-DQB1*06:15', 'HLA-DQA1*01:08-DQB1*06:16', 'HLA-DQA1*01:08-DQB1*06:17',
'HLA-DQA1*01:08-DQB1*06:18', 'HLA-DQA1*01:08-DQB1*06:19', 'HLA-DQA1*01:08-DQB1*06:21', 'HLA-DQA1*01:08-DQB1*06:22', 'HLA-DQA1*01:08-DQB1*06:23',
'HLA-DQA1*01:08-DQB1*06:24', 'HLA-DQA1*01:08-DQB1*06:25', 'HLA-DQA1*01:08-DQB1*06:27', 'HLA-DQA1*01:08-DQB1*06:28', 'HLA-DQA1*01:08-DQB1*06:29',
'HLA-DQA1*01:08-DQB1*06:30', 'HLA-DQA1*01:08-DQB1*06:31', 'HLA-DQA1*01:08-DQB1*06:32', 'HLA-DQA1*01:08-DQB1*06:33', 'HLA-DQA1*01:08-DQB1*06:34',
'HLA-DQA1*01:08-DQB1*06:35', 'HLA-DQA1*01:08-DQB1*06:36', 'HLA-DQA1*01:08-DQB1*06:37', 'HLA-DQA1*01:08-DQB1*06:38', 'HLA-DQA1*01:08-DQB1*06:39',
'HLA-DQA1*01:08-DQB1*06:40', 'HLA-DQA1*01:08-DQB1*06:41', 'HLA-DQA1*01:08-DQB1*06:42', 'HLA-DQA1*01:08-DQB1*06:43', 'HLA-DQA1*01:08-DQB1*06:44',
'HLA-DQA1*01:09-DQB1*02:01', 'HLA-DQA1*01:09-DQB1*02:02', 'HLA-DQA1*01:09-DQB1*02:03', 'HLA-DQA1*01:09-DQB1*02:04', 'HLA-DQA1*01:09-DQB1*02:05',
'HLA-DQA1*01:09-DQB1*02:06', 'HLA-DQA1*01:09-DQB1*03:01', 'HLA-DQA1*01:09-DQB1*03:02', 'HLA-DQA1*01:09-DQB1*03:03', 'HLA-DQA1*01:09-DQB1*03:04',
'HLA-DQA1*01:09-DQB1*03:05', 'HLA-DQA1*01:09-DQB1*03:06', 'HLA-DQA1*01:09-DQB1*03:07', 'HLA-DQA1*01:09-DQB1*03:08', 'HLA-DQA1*01:09-DQB1*03:09',
'HLA-DQA1*01:09-DQB1*03:10', 'HLA-DQA1*01:09-DQB1*03:11', 'HLA-DQA1*01:09-DQB1*03:12', 'HLA-DQA1*01:09-DQB1*03:13', 'HLA-DQA1*01:09-DQB1*03:14',
'HLA-DQA1*01:09-DQB1*03:15', 'HLA-DQA1*01:09-DQB1*03:16', 'HLA-DQA1*01:09-DQB1*03:17', 'HLA-DQA1*01:09-DQB1*03:18', 'HLA-DQA1*01:09-DQB1*03:19',
'HLA-DQA1*01:09-DQB1*03:20', 'HLA-DQA1*01:09-DQB1*03:21', 'HLA-DQA1*01:09-DQB1*03:22', 'HLA-DQA1*01:09-DQB1*03:23', 'HLA-DQA1*01:09-DQB1*03:24',
'HLA-DQA1*01:09-DQB1*03:25', 'HLA-DQA1*01:09-DQB1*03:26', 'HLA-DQA1*01:09-DQB1*03:27', 'HLA-DQA1*01:09-DQB1*03:28', 'HLA-DQA1*01:09-DQB1*03:29',
'HLA-DQA1*01:09-DQB1*03:30', 'HLA-DQA1*01:09-DQB1*03:31', 'HLA-DQA1*01:09-DQB1*03:32', 'HLA-DQA1*01:09-DQB1*03:33', 'HLA-DQA1*01:09-DQB1*03:34',
'HLA-DQA1*01:09-DQB1*03:35', 'HLA-DQA1*01:09-DQB1*03:36', 'HLA-DQA1*01:09-DQB1*03:37', 'HLA-DQA1*01:09-DQB1*03:38', 'HLA-DQA1*01:09-DQB1*04:01',
'HLA-DQA1*01:09-DQB1*04:02', 'HLA-DQA1*01:09-DQB1*04:03', 'HLA-DQA1*01:09-DQB1*04:04', 'HLA-DQA1*01:09-DQB1*04:05', 'HLA-DQA1*01:09-DQB1*04:06',
'HLA-DQA1*01:09-DQB1*04:07', 'HLA-DQA1*01:09-DQB1*04:08', 'HLA-DQA1*01:09-DQB1*05:01', 'HLA-DQA1*01:09-DQB1*05:02', 'HLA-DQA1*01:09-DQB1*05:03',
'HLA-DQA1*01:09-DQB1*05:05', 'HLA-DQA1*01:09-DQB1*05:06', 'HLA-DQA1*01:09-DQB1*05:07', 'HLA-DQA1*01:09-DQB1*05:08', 'HLA-DQA1*01:09-DQB1*05:09',
'HLA-DQA1*01:09-DQB1*05:10', 'HLA-DQA1*01:09-DQB1*05:11', 'HLA-DQA1*01:09-DQB1*05:12', 'HLA-DQA1*01:09-DQB1*05:13', 'HLA-DQA1*01:09-DQB1*05:14',
'HLA-DQA1*01:09-DQB1*06:01', 'HLA-DQA1*01:09-DQB1*06:02', 'HLA-DQA1*01:09-DQB1*06:03', 'HLA-DQA1*01:09-DQB1*06:04', 'HLA-DQA1*01:09-DQB1*06:07',
'HLA-DQA1*01:09-DQB1*06:08', 'HLA-DQA1*01:09-DQB1*06:09', 'HLA-DQA1*01:09-DQB1*06:10', 'HLA-DQA1*01:09-DQB1*06:11', 'HLA-DQA1*01:09-DQB1*06:12',
'HLA-DQA1*01:09-DQB1*06:14', 'HLA-DQA1*01:09-DQB1*06:15', 'HLA-DQA1*01:09-DQB1*06:16', 'HLA-DQA1*01:09-DQB1*06:17', 'HLA-DQA1*01:09-DQB1*06:18',
'HLA-DQA1*01:09-DQB1*06:19', 'HLA-DQA1*01:09-DQB1*06:21', 'HLA-DQA1*01:09-DQB1*06:22', 'HLA-DQA1*01:09-DQB1*06:23', 'HLA-DQA1*01:09-DQB1*06:24',
'HLA-DQA1*01:09-DQB1*06:25', 'HLA-DQA1*01:09-DQB1*06:27', 'HLA-DQA1*01:09-DQB1*06:28', 'HLA-DQA1*01:09-DQB1*06:29', 'HLA-DQA1*01:09-DQB1*06:30',
'HLA-DQA1*01:09-DQB1*06:31', 'HLA-DQA1*01:09-DQB1*06:32', 'HLA-DQA1*01:09-DQB1*06:33', 'HLA-DQA1*01:09-DQB1*06:34', 'HLA-DQA1*01:09-DQB1*06:35',
'HLA-DQA1*01:09-DQB1*06:36', 'HLA-DQA1*01:09-DQB1*06:37', 'HLA-DQA1*01:09-DQB1*06:38', 'HLA-DQA1*01:09-DQB1*06:39', 'HLA-DQA1*01:09-DQB1*06:40',
'HLA-DQA1*01:09-DQB1*06:41', 'HLA-DQA1*01:09-DQB1*06:42', 'HLA-DQA1*01:09-DQB1*06:43', 'HLA-DQA1*01:09-DQB1*06:44', 'HLA-DQA1*02:01-DQB1*02:01',
'HLA-DQA1*02:01-DQB1*02:02', 'HLA-DQA1*02:01-DQB1*02:03', 'HLA-DQA1*02:01-DQB1*02:04', 'HLA-DQA1*02:01-DQB1*02:05', 'HLA-DQA1*02:01-DQB1*02:06',
'HLA-DQA1*02:01-DQB1*03:01', 'HLA-DQA1*02:01-DQB1*03:02', 'HLA-DQA1*02:01-DQB1*03:03', 'HLA-DQA1*02:01-DQB1*03:04', 'HLA-DQA1*02:01-DQB1*03:05',
'HLA-DQA1*02:01-DQB1*03:06', 'HLA-DQA1*02:01-DQB1*03:07', 'HLA-DQA1*02:01-DQB1*03:08', 'HLA-DQA1*02:01-DQB1*03:09', 'HLA-DQA1*02:01-DQB1*03:10',
'HLA-DQA1*02:01-DQB1*03:11', 'HLA-DQA1*02:01-DQB1*03:12', 'HLA-DQA1*02:01-DQB1*03:13', 'HLA-DQA1*02:01-DQB1*03:14', 'HLA-DQA1*02:01-DQB1*03:15',
'HLA-DQA1*02:01-DQB1*03:16', 'HLA-DQA1*02:01-DQB1*03:17', 'HLA-DQA1*02:01-DQB1*03:18', 'HLA-DQA1*02:01-DQB1*03:19', 'HLA-DQA1*02:01-DQB1*03:20',
'HLA-DQA1*02:01-DQB1*03:21', 'HLA-DQA1*02:01-DQB1*03:22', 'HLA-DQA1*02:01-DQB1*03:23', 'HLA-DQA1*02:01-DQB1*03:24', 'HLA-DQA1*02:01-DQB1*03:25',
'HLA-DQA1*02:01-DQB1*03:26', 'HLA-DQA1*02:01-DQB1*03:27', 'HLA-DQA1*02:01-DQB1*03:28', 'HLA-DQA1*02:01-DQB1*03:29', 'HLA-DQA1*02:01-DQB1*03:30',
'HLA-DQA1*02:01-DQB1*03:31', 'HLA-DQA1*02:01-DQB1*03:32', 'HLA-DQA1*02:01-DQB1*03:33', 'HLA-DQA1*02:01-DQB1*03:34', 'HLA-DQA1*02:01-DQB1*03:35',
'HLA-DQA1*02:01-DQB1*03:36', 'HLA-DQA1*02:01-DQB1*03:37', 'HLA-DQA1*02:01-DQB1*03:38', 'HLA-DQA1*02:01-DQB1*04:01', 'HLA-DQA1*02:01-DQB1*04:02',
'HLA-DQA1*02:01-DQB1*04:03', 'HLA-DQA1*02:01-DQB1*04:04', 'HLA-DQA1*02:01-DQB1*04:05', 'HLA-DQA1*02:01-DQB1*04:06', 'HLA-DQA1*02:01-DQB1*04:07',
'HLA-DQA1*02:01-DQB1*04:08', 'HLA-DQA1*02:01-DQB1*05:01', 'HLA-DQA1*02:01-DQB1*05:02', 'HLA-DQA1*02:01-DQB1*05:03', 'HLA-DQA1*02:01-DQB1*05:05',
'HLA-DQA1*02:01-DQB1*05:06', 'HLA-DQA1*02:01-DQB1*05:07', 'HLA-DQA1*02:01-DQB1*05:08', 'HLA-DQA1*02:01-DQB1*05:09', 'HLA-DQA1*02:01-DQB1*05:10',
'HLA-DQA1*02:01-DQB1*05:11', 'HLA-DQA1*02:01-DQB1*05:12', 'HLA-DQA1*02:01-DQB1*05:13', 'HLA-DQA1*02:01-DQB1*05:14', 'HLA-DQA1*02:01-DQB1*06:01',
'HLA-DQA1*02:01-DQB1*06:02', 'HLA-DQA1*02:01-DQB1*06:03', 'HLA-DQA1*02:01-DQB1*06:04', 'HLA-DQA1*02:01-DQB1*06:07', 'HLA-DQA1*02:01-DQB1*06:08',
'HLA-DQA1*02:01-DQB1*06:09', 'HLA-DQA1*02:01-DQB1*06:10', 'HLA-DQA1*02:01-DQB1*06:11', 'HLA-DQA1*02:01-DQB1*06:12', 'HLA-DQA1*02:01-DQB1*06:14',
'HLA-DQA1*02:01-DQB1*06:15', 'HLA-DQA1*02:01-DQB1*06:16', 'HLA-DQA1*02:01-DQB1*06:17', 'HLA-DQA1*02:01-DQB1*06:18', 'HLA-DQA1*02:01-DQB1*06:19',
'HLA-DQA1*02:01-DQB1*06:21', 'HLA-DQA1*02:01-DQB1*06:22', 'HLA-DQA1*02:01-DQB1*06:23', 'HLA-DQA1*02:01-DQB1*06:24', 'HLA-DQA1*02:01-DQB1*06:25',
'HLA-DQA1*02:01-DQB1*06:27', 'HLA-DQA1*02:01-DQB1*06:28', 'HLA-DQA1*02:01-DQB1*06:29', 'HLA-DQA1*02:01-DQB1*06:30', 'HLA-DQA1*02:01-DQB1*06:31',
'HLA-DQA1*02:01-DQB1*06:32', 'HLA-DQA1*02:01-DQB1*06:33', 'HLA-DQA1*02:01-DQB1*06:34', 'HLA-DQA1*02:01-DQB1*06:35', 'HLA-DQA1*02:01-DQB1*06:36',
'HLA-DQA1*02:01-DQB1*06:37', 'HLA-DQA1*02:01-DQB1*06:38', 'HLA-DQA1*02:01-DQB1*06:39', 'HLA-DQA1*02:01-DQB1*06:40', 'HLA-DQA1*02:01-DQB1*06:41',
'HLA-DQA1*02:01-DQB1*06:42', 'HLA-DQA1*02:01-DQB1*06:43', 'HLA-DQA1*02:01-DQB1*06:44', 'HLA-DQA1*03:01-DQB1*02:01', 'HLA-DQA1*03:01-DQB1*02:02',
'HLA-DQA1*03:01-DQB1*02:03', 'HLA-DQA1*03:01-DQB1*02:04', 'HLA-DQA1*03:01-DQB1*02:05', 'HLA-DQA1*03:01-DQB1*02:06', 'HLA-DQA1*03:01-DQB1*03:01',
'HLA-DQA1*03:01-DQB1*03:02', 'HLA-DQA1*03:01-DQB1*03:03', 'HLA-DQA1*03:01-DQB1*03:04', 'HLA-DQA1*03:01-DQB1*03:05', 'HLA-DQA1*03:01-DQB1*03:06',
'HLA-DQA1*03:01-DQB1*03:07', 'HLA-DQA1*03:01-DQB1*03:08', 'HLA-DQA1*03:01-DQB1*03:09', 'HLA-DQA1*03:01-DQB1*03:10', 'HLA-DQA1*03:01-DQB1*03:11',
'HLA-DQA1*03:01-DQB1*03:12', 'HLA-DQA1*03:01-DQB1*03:13', 'HLA-DQA1*03:01-DQB1*03:14', 'HLA-DQA1*03:01-DQB1*03:15', 'HLA-DQA1*03:01-DQB1*03:16',
'HLA-DQA1*03:01-DQB1*03:17', 'HLA-DQA1*03:01-DQB1*03:18', 'HLA-DQA1*03:01-DQB1*03:19', 'HLA-DQA1*03:01-DQB1*03:20', 'HLA-DQA1*03:01-DQB1*03:21',
'HLA-DQA1*03:01-DQB1*03:22', 'HLA-DQA1*03:01-DQB1*03:23', 'HLA-DQA1*03:01-DQB1*03:24', 'HLA-DQA1*03:01-DQB1*03:25', 'HLA-DQA1*03:01-DQB1*03:26',
'HLA-DQA1*03:01-DQB1*03:27', 'HLA-DQA1*03:01-DQB1*03:28', 'HLA-DQA1*03:01-DQB1*03:29', 'HLA-DQA1*03:01-DQB1*03:30', 'HLA-DQA1*03:01-DQB1*03:31',
'HLA-DQA1*03:01-DQB1*03:32', 'HLA-DQA1*03:01-DQB1*03:33', 'HLA-DQA1*03:01-DQB1*03:34', 'HLA-DQA1*03:01-DQB1*03:35', 'HLA-DQA1*03:01-DQB1*03:36',
'HLA-DQA1*03:01-DQB1*03:37', 'HLA-DQA1*03:01-DQB1*03:38', 'HLA-DQA1*03:01-DQB1*04:01', 'HLA-DQA1*03:01-DQB1*04:02', 'HLA-DQA1*03:01-DQB1*04:03',
'HLA-DQA1*03:01-DQB1*04:04', 'HLA-DQA1*03:01-DQB1*04:05', 'HLA-DQA1*03:01-DQB1*04:06', 'HLA-DQA1*03:01-DQB1*04:07', 'HLA-DQA1*03:01-DQB1*04:08',
'HLA-DQA1*03:01-DQB1*05:01', 'HLA-DQA1*03:01-DQB1*05:02', 'HLA-DQA1*03:01-DQB1*05:03', 'HLA-DQA1*03:01-DQB1*05:05', 'HLA-DQA1*03:01-DQB1*05:06',
'HLA-DQA1*03:01-DQB1*05:07', 'HLA-DQA1*03:01-DQB1*05:08', 'HLA-DQA1*03:01-DQB1*05:09', 'HLA-DQA1*03:01-DQB1*05:10', 'HLA-DQA1*03:01-DQB1*05:11',
'HLA-DQA1*03:01-DQB1*05:12', 'HLA-DQA1*03:01-DQB1*05:13', 'HLA-DQA1*03:01-DQB1*05:14', 'HLA-DQA1*03:01-DQB1*06:01', 'HLA-DQA1*03:01-DQB1*06:02',
'HLA-DQA1*03:01-DQB1*06:03', 'HLA-DQA1*03:01-DQB1*06:04', 'HLA-DQA1*03:01-DQB1*06:07', 'HLA-DQA1*03:01-DQB1*06:08', 'HLA-DQA1*03:01-DQB1*06:09',
'HLA-DQA1*03:01-DQB1*06:10', 'HLA-DQA1*03:01-DQB1*06:11', 'HLA-DQA1*03:01-DQB1*06:12', 'HLA-DQA1*03:01-DQB1*06:14', 'HLA-DQA1*03:01-DQB1*06:15',
'HLA-DQA1*03:01-DQB1*06:16', 'HLA-DQA1*03:01-DQB1*06:17', 'HLA-DQA1*03:01-DQB1*06:18', 'HLA-DQA1*03:01-DQB1*06:19', 'HLA-DQA1*03:01-DQB1*06:21',
'HLA-DQA1*03:01-DQB1*06:22', 'HLA-DQA1*03:01-DQB1*06:23', 'HLA-DQA1*03:01-DQB1*06:24', 'HLA-DQA1*03:01-DQB1*06:25', 'HLA-DQA1*03:01-DQB1*06:27',
'HLA-DQA1*03:01-DQB1*06:28', 'HLA-DQA1*03:01-DQB1*06:29', 'HLA-DQA1*03:01-DQB1*06:30', 'HLA-DQA1*03:01-DQB1*06:31', 'HLA-DQA1*03:01-DQB1*06:32',
'HLA-DQA1*03:01-DQB1*06:33', 'HLA-DQA1*03:01-DQB1*06:34', 'HLA-DQA1*03:01-DQB1*06:35', 'HLA-DQA1*03:01-DQB1*06:36', 'HLA-DQA1*03:01-DQB1*06:37',
'HLA-DQA1*03:01-DQB1*06:38', 'HLA-DQA1*03:01-DQB1*06:39', 'HLA-DQA1*03:01-DQB1*06:40', 'HLA-DQA1*03:01-DQB1*06:41', 'HLA-DQA1*03:01-DQB1*06:42',
'HLA-DQA1*03:01-DQB1*06:43', 'HLA-DQA1*03:01-DQB1*06:44', 'HLA-DQA1*03:02-DQB1*02:01', 'HLA-DQA1*03:02-DQB1*02:02', 'HLA-DQA1*03:02-DQB1*02:03',
'HLA-DQA1*03:02-DQB1*02:04', 'HLA-DQA1*03:02-DQB1*02:05', 'HLA-DQA1*03:02-DQB1*02:06', 'HLA-DQA1*03:02-DQB1*03:01', 'HLA-DQA1*03:02-DQB1*03:02',
'HLA-DQA1*03:02-DQB1*03:03', 'HLA-DQA1*03:02-DQB1*03:04', 'HLA-DQA1*03:02-DQB1*03:05', 'HLA-DQA1*03:02-DQB1*03:06', 'HLA-DQA1*03:02-DQB1*03:07',
'HLA-DQA1*03:02-DQB1*03:08', 'HLA-DQA1*03:02-DQB1*03:09', 'HLA-DQA1*03:02-DQB1*03:10', 'HLA-DQA1*03:02-DQB1*03:11', 'HLA-DQA1*03:02-DQB1*03:12',
'HLA-DQA1*03:02-DQB1*03:13', 'HLA-DQA1*03:02-DQB1*03:14', 'HLA-DQA1*03:02-DQB1*03:15', 'HLA-DQA1*03:02-DQB1*03:16', 'HLA-DQA1*03:02-DQB1*03:17',
'HLA-DQA1*03:02-DQB1*03:18', 'HLA-DQA1*03:02-DQB1*03:19', 'HLA-DQA1*03:02-DQB1*03:20', 'HLA-DQA1*03:02-DQB1*03:21', 'HLA-DQA1*03:02-DQB1*03:22',
'HLA-DQA1*03:02-DQB1*03:23', 'HLA-DQA1*03:02-DQB1*03:24', 'HLA-DQA1*03:02-DQB1*03:25', 'HLA-DQA1*03:02-DQB1*03:26', 'HLA-DQA1*03:02-DQB1*03:27',
'HLA-DQA1*03:02-DQB1*03:28', 'HLA-DQA1*03:02-DQB1*03:29', 'HLA-DQA1*03:02-DQB1*03:30', 'HLA-DQA1*03:02-DQB1*03:31', 'HLA-DQA1*03:02-DQB1*03:32',
'HLA-DQA1*03:02-DQB1*03:33', 'HLA-DQA1*03:02-DQB1*03:34', 'HLA-DQA1*03:02-DQB1*03:35', 'HLA-DQA1*03:02-DQB1*03:36', 'HLA-DQA1*03:02-DQB1*03:37',
'HLA-DQA1*03:02-DQB1*03:38', 'HLA-DQA1*03:02-DQB1*04:01', 'HLA-DQA1*03:02-DQB1*04:02', 'HLA-DQA1*03:02-DQB1*04:03', 'HLA-DQA1*03:02-DQB1*04:04',
'HLA-DQA1*03:02-DQB1*04:05', 'HLA-DQA1*03:02-DQB1*04:06', 'HLA-DQA1*03:02-DQB1*04:07', 'HLA-DQA1*03:02-DQB1*04:08', 'HLA-DQA1*03:02-DQB1*05:01',
'HLA-DQA1*03:02-DQB1*05:02', 'HLA-DQA1*03:02-DQB1*05:03', 'HLA-DQA1*03:02-DQB1*05:05', 'HLA-DQA1*03:02-DQB1*05:06', 'HLA-DQA1*03:02-DQB1*05:07',
'HLA-DQA1*03:02-DQB1*05:08', 'HLA-DQA1*03:02-DQB1*05:09', 'HLA-DQA1*03:02-DQB1*05:10', 'HLA-DQA1*03:02-DQB1*05:11', 'HLA-DQA1*03:02-DQB1*05:12',
'HLA-DQA1*03:02-DQB1*05:13', 'HLA-DQA1*03:02-DQB1*05:14', 'HLA-DQA1*03:02-DQB1*06:01', 'HLA-DQA1*03:02-DQB1*06:02', 'HLA-DQA1*03:02-DQB1*06:03',
'HLA-DQA1*03:02-DQB1*06:04', 'HLA-DQA1*03:02-DQB1*06:07', 'HLA-DQA1*03:02-DQB1*06:08', 'HLA-DQA1*03:02-DQB1*06:09', 'HLA-DQA1*03:02-DQB1*06:10',
'HLA-DQA1*03:02-DQB1*06:11', 'HLA-DQA1*03:02-DQB1*06:12', 'HLA-DQA1*03:02-DQB1*06:14', 'HLA-DQA1*03:02-DQB1*06:15', 'HLA-DQA1*03:02-DQB1*06:16',
'HLA-DQA1*03:02-DQB1*06:17', 'HLA-DQA1*03:02-DQB1*06:18', 'HLA-DQA1*03:02-DQB1*06:19', 'HLA-DQA1*03:02-DQB1*06:21', 'HLA-DQA1*03:02-DQB1*06:22',
'HLA-DQA1*03:02-DQB1*06:23', 'HLA-DQA1*03:02-DQB1*06:24', 'HLA-DQA1*03:02-DQB1*06:25', 'HLA-DQA1*03:02-DQB1*06:27', 'HLA-DQA1*03:02-DQB1*06:28',
'HLA-DQA1*03:02-DQB1*06:29', 'HLA-DQA1*03:02-DQB1*06:30', 'HLA-DQA1*03:02-DQB1*06:31', 'HLA-DQA1*03:02-DQB1*06:32', 'HLA-DQA1*03:02-DQB1*06:33',
'HLA-DQA1*03:02-DQB1*06:34', 'HLA-DQA1*03:02-DQB1*06:35', 'HLA-DQA1*03:02-DQB1*06:36', 'HLA-DQA1*03:02-DQB1*06:37', 'HLA-DQA1*03:02-DQB1*06:38',
'HLA-DQA1*03:02-DQB1*06:39', 'HLA-DQA1*03:02-DQB1*06:40', 'HLA-DQA1*03:02-DQB1*06:41', 'HLA-DQA1*03:02-DQB1*06:42', 'HLA-DQA1*03:02-DQB1*06:43',
'HLA-DQA1*03:02-DQB1*06:44', 'HLA-DQA1*03:03-DQB1*02:01', 'HLA-DQA1*03:03-DQB1*02:02', 'HLA-DQA1*03:03-DQB1*02:03', 'HLA-DQA1*03:03-DQB1*02:04',
'HLA-DQA1*03:03-DQB1*02:05', 'HLA-DQA1*03:03-DQB1*02:06', 'HLA-DQA1*03:03-DQB1*03:01', 'HLA-DQA1*03:03-DQB1*03:02', 'HLA-DQA1*03:03-DQB1*03:03',
'HLA-DQA1*03:03-DQB1*03:04', 'HLA-DQA1*03:03-DQB1*03:05', 'HLA-DQA1*03:03-DQB1*03:06', 'HLA-DQA1*03:03-DQB1*03:07', 'HLA-DQA1*03:03-DQB1*03:08',
'HLA-DQA1*03:03-DQB1*03:09', 'HLA-DQA1*03:03-DQB1*03:10', 'HLA-DQA1*03:03-DQB1*03:11', 'HLA-DQA1*03:03-DQB1*03:12', 'HLA-DQA1*03:03-DQB1*03:13',
'HLA-DQA1*03:03-DQB1*03:14', 'HLA-DQA1*03:03-DQB1*03:15', 'HLA-DQA1*03:03-DQB1*03:16', 'HLA-DQA1*03:03-DQB1*03:17', 'HLA-DQA1*03:03-DQB1*03:18',
'HLA-DQA1*03:03-DQB1*03:19', 'HLA-DQA1*03:03-DQB1*03:20', 'HLA-DQA1*03:03-DQB1*03:21', 'HLA-DQA1*03:03-DQB1*03:22', 'HLA-DQA1*03:03-DQB1*03:23',
'HLA-DQA1*03:03-DQB1*03:24', 'HLA-DQA1*03:03-DQB1*03:25', 'HLA-DQA1*03:03-DQB1*03:26', 'HLA-DQA1*03:03-DQB1*03:27', 'HLA-DQA1*03:03-DQB1*03:28',
'HLA-DQA1*03:03-DQB1*03:29', 'HLA-DQA1*03:03-DQB1*03:30', 'HLA-DQA1*03:03-DQB1*03:31', 'HLA-DQA1*03:03-DQB1*03:32', 'HLA-DQA1*03:03-DQB1*03:33',
'HLA-DQA1*03:03-DQB1*03:34', 'HLA-DQA1*03:03-DQB1*03:35', 'HLA-DQA1*03:03-DQB1*03:36', 'HLA-DQA1*03:03-DQB1*03:37', 'HLA-DQA1*03:03-DQB1*03:38',
'HLA-DQA1*03:03-DQB1*04:01', 'HLA-DQA1*03:03-DQB1*04:02', 'HLA-DQA1*03:03-DQB1*04:03', 'HLA-DQA1*03:03-DQB1*04:04', 'HLA-DQA1*03:03-DQB1*04:05',
'HLA-DQA1*03:03-DQB1*04:06', 'HLA-DQA1*03:03-DQB1*04:07', 'HLA-DQA1*03:03-DQB1*04:08', 'HLA-DQA1*03:03-DQB1*05:01', 'HLA-DQA1*03:03-DQB1*05:02',
'HLA-DQA1*03:03-DQB1*05:03', 'HLA-DQA1*03:03-DQB1*05:05', 'HLA-DQA1*03:03-DQB1*05:06', 'HLA-DQA1*03:03-DQB1*05:07', 'HLA-DQA1*03:03-DQB1*05:08',
'HLA-DQA1*03:03-DQB1*05:09', 'HLA-DQA1*03:03-DQB1*05:10', 'HLA-DQA1*03:03-DQB1*05:11', 'HLA-DQA1*03:03-DQB1*05:12', 'HLA-DQA1*03:03-DQB1*05:13',
'HLA-DQA1*03:03-DQB1*05:14', 'HLA-DQA1*03:03-DQB1*06:01', 'HLA-DQA1*03:03-DQB1*06:02', 'HLA-DQA1*03:03-DQB1*06:03', 'HLA-DQA1*03:03-DQB1*06:04',
'HLA-DQA1*03:03-DQB1*06:07', 'HLA-DQA1*03:03-DQB1*06:08', 'HLA-DQA1*03:03-DQB1*06:09', 'HLA-DQA1*03:03-DQB1*06:10', 'HLA-DQA1*03:03-DQB1*06:11',
'HLA-DQA1*03:03-DQB1*06:12', 'HLA-DQA1*03:03-DQB1*06:14', 'HLA-DQA1*03:03-DQB1*06:15', 'HLA-DQA1*03:03-DQB1*06:16', 'HLA-DQA1*03:03-DQB1*06:17',
'HLA-DQA1*03:03-DQB1*06:18', 'HLA-DQA1*03:03-DQB1*06:19', 'HLA-DQA1*03:03-DQB1*06:21', 'HLA-DQA1*03:03-DQB1*06:22', 'HLA-DQA1*03:03-DQB1*06:23',
'HLA-DQA1*03:03-DQB1*06:24', 'HLA-DQA1*03:03-DQB1*06:25', 'HLA-DQA1*03:03-DQB1*06:27', 'HLA-DQA1*03:03-DQB1*06:28', 'HLA-DQA1*03:03-DQB1*06:29',
'HLA-DQA1*03:03-DQB1*06:30', 'HLA-DQA1*03:03-DQB1*06:31', 'HLA-DQA1*03:03-DQB1*06:32', 'HLA-DQA1*03:03-DQB1*06:33', 'HLA-DQA1*03:03-DQB1*06:34',
'HLA-DQA1*03:03-DQB1*06:35', 'HLA-DQA1*03:03-DQB1*06:36', 'HLA-DQA1*03:03-DQB1*06:37', 'HLA-DQA1*03:03-DQB1*06:38', 'HLA-DQA1*03:03-DQB1*06:39',
'HLA-DQA1*03:03-DQB1*06:40', 'HLA-DQA1*03:03-DQB1*06:41', 'HLA-DQA1*03:03-DQB1*06:42', 'HLA-DQA1*03:03-DQB1*06:43', 'HLA-DQA1*03:03-DQB1*06:44',
'HLA-DQA1*04:01-DQB1*02:01', 'HLA-DQA1*04:01-DQB1*02:02', 'HLA-DQA1*04:01-DQB1*02:03', 'HLA-DQA1*04:01-DQB1*02:04', 'HLA-DQA1*04:01-DQB1*02:05',
'HLA-DQA1*04:01-DQB1*02:06', 'HLA-DQA1*04:01-DQB1*03:01', 'HLA-DQA1*04:01-DQB1*03:02', 'HLA-DQA1*04:01-DQB1*03:03', 'HLA-DQA1*04:01-DQB1*03:04',
'HLA-DQA1*04:01-DQB1*03:05', 'HLA-DQA1*04:01-DQB1*03:06', 'HLA-DQA1*04:01-DQB1*03:07', 'HLA-DQA1*04:01-DQB1*03:08', 'HLA-DQA1*04:01-DQB1*03:09',
'HLA-DQA1*04:01-DQB1*03:10', 'HLA-DQA1*04:01-DQB1*03:11', 'HLA-DQA1*04:01-DQB1*03:12', 'HLA-DQA1*04:01-DQB1*03:13', 'HLA-DQA1*04:01-DQB1*03:14',
'HLA-DQA1*04:01-DQB1*03:15', 'HLA-DQA1*04:01-DQB1*03:16', 'HLA-DQA1*04:01-DQB1*03:17', 'HLA-DQA1*04:01-DQB1*03:18', 'HLA-DQA1*04:01-DQB1*03:19',
'HLA-DQA1*04:01-DQB1*03:20', 'HLA-DQA1*04:01-DQB1*03:21', 'HLA-DQA1*04:01-DQB1*03:22', 'HLA-DQA1*04:01-DQB1*03:23', 'HLA-DQA1*04:01-DQB1*03:24',
'HLA-DQA1*04:01-DQB1*03:25', 'HLA-DQA1*04:01-DQB1*03:26', 'HLA-DQA1*04:01-DQB1*03:27', 'HLA-DQA1*04:01-DQB1*03:28', 'HLA-DQA1*04:01-DQB1*03:29',
'HLA-DQA1*04:01-DQB1*03:30', 'HLA-DQA1*04:01-DQB1*03:31', 'HLA-DQA1*04:01-DQB1*03:32', 'HLA-DQA1*04:01-DQB1*03:33', 'HLA-DQA1*04:01-DQB1*03:34',
'HLA-DQA1*04:01-DQB1*03:35', 'HLA-DQA1*04:01-DQB1*03:36', 'HLA-DQA1*04:01-DQB1*03:37', 'HLA-DQA1*04:01-DQB1*03:38', 'HLA-DQA1*04:01-DQB1*04:01',
'HLA-DQA1*04:01-DQB1*04:02', 'HLA-DQA1*04:01-DQB1*04:03', 'HLA-DQA1*04:01-DQB1*04:04', 'HLA-DQA1*04:01-DQB1*04:05', 'HLA-DQA1*04:01-DQB1*04:06',
'HLA-DQA1*04:01-DQB1*04:07', 'HLA-DQA1*04:01-DQB1*04:08', 'HLA-DQA1*04:01-DQB1*05:01', 'HLA-DQA1*04:01-DQB1*05:02', 'HLA-DQA1*04:01-DQB1*05:03',
'HLA-DQA1*04:01-DQB1*05:05', 'HLA-DQA1*04:01-DQB1*05:06', 'HLA-DQA1*04:01-DQB1*05:07', 'HLA-DQA1*04:01-DQB1*05:08', 'HLA-DQA1*04:01-DQB1*05:09',
'HLA-DQA1*04:01-DQB1*05:10', 'HLA-DQA1*04:01-DQB1*05:11', 'HLA-DQA1*04:01-DQB1*05:12', 'HLA-DQA1*04:01-DQB1*05:13', 'HLA-DQA1*04:01-DQB1*05:14',
'HLA-DQA1*04:01-DQB1*06:01', 'HLA-DQA1*04:01-DQB1*06:02', 'HLA-DQA1*04:01-DQB1*06:03', 'HLA-DQA1*04:01-DQB1*06:04', 'HLA-DQA1*04:01-DQB1*06:07',
'HLA-DQA1*04:01-DQB1*06:08', 'HLA-DQA1*04:01-DQB1*06:09', 'HLA-DQA1*04:01-DQB1*06:10', 'HLA-DQA1*04:01-DQB1*06:11', 'HLA-DQA1*04:01-DQB1*06:12',
'HLA-DQA1*04:01-DQB1*06:14', 'HLA-DQA1*04:01-DQB1*06:15', 'HLA-DQA1*04:01-DQB1*06:16', 'HLA-DQA1*04:01-DQB1*06:17', 'HLA-DQA1*04:01-DQB1*06:18',
'HLA-DQA1*04:01-DQB1*06:19', 'HLA-DQA1*04:01-DQB1*06:21', 'HLA-DQA1*04:01-DQB1*06:22', 'HLA-DQA1*04:01-DQB1*06:23', 'HLA-DQA1*04:01-DQB1*06:24',
'HLA-DQA1*04:01-DQB1*06:25', 'HLA-DQA1*04:01-DQB1*06:27', 'HLA-DQA1*04:01-DQB1*06:28', 'HLA-DQA1*04:01-DQB1*06:29', 'HLA-DQA1*04:01-DQB1*06:30',
'HLA-DQA1*04:01-DQB1*06:31', 'HLA-DQA1*04:01-DQB1*06:32', 'HLA-DQA1*04:01-DQB1*06:33', 'HLA-DQA1*04:01-DQB1*06:34', 'HLA-DQA1*04:01-DQB1*06:35',
'HLA-DQA1*04:01-DQB1*06:36', 'HLA-DQA1*04:01-DQB1*06:37', 'HLA-DQA1*04:01-DQB1*06:38', 'HLA-DQA1*04:01-DQB1*06:39', 'HLA-DQA1*04:01-DQB1*06:40',
'HLA-DQA1*04:01-DQB1*06:41', 'HLA-DQA1*04:01-DQB1*06:42', 'HLA-DQA1*04:01-DQB1*06:43', 'HLA-DQA1*04:01-DQB1*06:44', 'HLA-DQA1*04:02-DQB1*02:01',
'HLA-DQA1*04:02-DQB1*02:02', 'HLA-DQA1*04:02-DQB1*02:03', 'HLA-DQA1*04:02-DQB1*02:04', 'HLA-DQA1*04:02-DQB1*02:05', 'HLA-DQA1*04:02-DQB1*02:06',
'HLA-DQA1*04:02-DQB1*03:01', 'HLA-DQA1*04:02-DQB1*03:02', 'HLA-DQA1*04:02-DQB1*03:03', 'HLA-DQA1*04:02-DQB1*03:04', 'HLA-DQA1*04:02-DQB1*03:05',
'HLA-DQA1*04:02-DQB1*03:06', 'HLA-DQA1*04:02-DQB1*03:07', 'HLA-DQA1*04:02-DQB1*03:08', 'HLA-DQA1*04:02-DQB1*03:09', 'HLA-DQA1*04:02-DQB1*03:10',
'HLA-DQA1*04:02-DQB1*03:11', 'HLA-DQA1*04:02-DQB1*03:12', 'HLA-DQA1*04:02-DQB1*03:13', 'HLA-DQA1*04:02-DQB1*03:14', 'HLA-DQA1*04:02-DQB1*03:15',
'HLA-DQA1*04:02-DQB1*03:16', 'HLA-DQA1*04:02-DQB1*03:17', 'HLA-DQA1*04:02-DQB1*03:18', 'HLA-DQA1*04:02-DQB1*03:19', 'HLA-DQA1*04:02-DQB1*03:20',
'HLA-DQA1*04:02-DQB1*03:21', 'HLA-DQA1*04:02-DQB1*03:22', 'HLA-DQA1*04:02-DQB1*03:23', 'HLA-DQA1*04:02-DQB1*03:24', 'HLA-DQA1*04:02-DQB1*03:25',
'HLA-DQA1*04:02-DQB1*03:26', 'HLA-DQA1*04:02-DQB1*03:27', 'HLA-DQA1*04:02-DQB1*03:28', 'HLA-DQA1*04:02-DQB1*03:29', 'HLA-DQA1*04:02-DQB1*03:30',
'HLA-DQA1*04:02-DQB1*03:31', 'HLA-DQA1*04:02-DQB1*03:32', 'HLA-DQA1*04:02-DQB1*03:33', 'HLA-DQA1*04:02-DQB1*03:34', 'HLA-DQA1*04:02-DQB1*03:35',
'HLA-DQA1*04:02-DQB1*03:36', 'HLA-DQA1*04:02-DQB1*03:37', 'HLA-DQA1*04:02-DQB1*03:38', 'HLA-DQA1*04:02-DQB1*04:01', 'HLA-DQA1*04:02-DQB1*04:02',
'HLA-DQA1*04:02-DQB1*04:03', 'HLA-DQA1*04:02-DQB1*04:04', 'HLA-DQA1*04:02-DQB1*04:05', 'HLA-DQA1*04:02-DQB1*04:06', 'HLA-DQA1*04:02-DQB1*04:07',
'HLA-DQA1*04:02-DQB1*04:08', 'HLA-DQA1*04:02-DQB1*05:01', 'HLA-DQA1*04:02-DQB1*05:02', 'HLA-DQA1*04:02-DQB1*05:03', 'HLA-DQA1*04:02-DQB1*05:05',
'HLA-DQA1*04:02-DQB1*05:06', 'HLA-DQA1*04:02-DQB1*05:07', 'HLA-DQA1*04:02-DQB1*05:08', 'HLA-DQA1*04:02-DQB1*05:09', 'HLA-DQA1*04:02-DQB1*05:10',
'HLA-DQA1*04:02-DQB1*05:11', 'HLA-DQA1*04:02-DQB1*05:12', 'HLA-DQA1*04:02-DQB1*05:13', 'HLA-DQA1*04:02-DQB1*05:14', 'HLA-DQA1*04:02-DQB1*06:01',
'HLA-DQA1*04:02-DQB1*06:02', 'HLA-DQA1*04:02-DQB1*06:03', 'HLA-DQA1*04:02-DQB1*06:04', 'HLA-DQA1*04:02-DQB1*06:07', 'HLA-DQA1*04:02-DQB1*06:08',
'HLA-DQA1*04:02-DQB1*06:09', 'HLA-DQA1*04:02-DQB1*06:10', 'HLA-DQA1*04:02-DQB1*06:11', 'HLA-DQA1*04:02-DQB1*06:12', 'HLA-DQA1*04:02-DQB1*06:14',
'HLA-DQA1*04:02-DQB1*06:15', 'HLA-DQA1*04:02-DQB1*06:16', 'HLA-DQA1*04:02-DQB1*06:17', 'HLA-DQA1*04:02-DQB1*06:18', 'HLA-DQA1*04:02-DQB1*06:19',
'HLA-DQA1*04:02-DQB1*06:21', 'HLA-DQA1*04:02-DQB1*06:22', 'HLA-DQA1*04:02-DQB1*06:23', 'HLA-DQA1*04:02-DQB1*06:24', 'HLA-DQA1*04:02-DQB1*06:25',
'HLA-DQA1*04:02-DQB1*06:27', 'HLA-DQA1*04:02-DQB1*06:28', 'HLA-DQA1*04:02-DQB1*06:29', 'HLA-DQA1*04:02-DQB1*06:30', 'HLA-DQA1*04:02-DQB1*06:31',
'HLA-DQA1*04:02-DQB1*06:32', 'HLA-DQA1*04:02-DQB1*06:33', 'HLA-DQA1*04:02-DQB1*06:34', 'HLA-DQA1*04:02-DQB1*06:35', 'HLA-DQA1*04:02-DQB1*06:36',
'HLA-DQA1*04:02-DQB1*06:37', 'HLA-DQA1*04:02-DQB1*06:38', 'HLA-DQA1*04:02-DQB1*06:39', 'HLA-DQA1*04:02-DQB1*06:40', 'HLA-DQA1*04:02-DQB1*06:41',
'HLA-DQA1*04:02-DQB1*06:42', 'HLA-DQA1*04:02-DQB1*06:43', 'HLA-DQA1*04:02-DQB1*06:44', 'HLA-DQA1*04:04-DQB1*02:01', 'HLA-DQA1*04:04-DQB1*02:02',
'HLA-DQA1*04:04-DQB1*02:03', 'HLA-DQA1*04:04-DQB1*02:04', 'HLA-DQA1*04:04-DQB1*02:05', 'HLA-DQA1*04:04-DQB1*02:06', 'HLA-DQA1*04:04-DQB1*03:01',
'HLA-DQA1*04:04-DQB1*03:02', 'HLA-DQA1*04:04-DQB1*03:03', 'HLA-DQA1*04:04-DQB1*03:04', 'HLA-DQA1*04:04-DQB1*03:05', 'HLA-DQA1*04:04-DQB1*03:06',
'HLA-DQA1*04:04-DQB1*03:07', 'HLA-DQA1*04:04-DQB1*03:08', 'HLA-DQA1*04:04-DQB1*03:09', 'HLA-DQA1*04:04-DQB1*03:10', 'HLA-DQA1*04:04-DQB1*03:11',
'HLA-DQA1*04:04-DQB1*03:12', 'HLA-DQA1*04:04-DQB1*03:13', 'HLA-DQA1*04:04-DQB1*03:14', 'HLA-DQA1*04:04-DQB1*03:15', 'HLA-DQA1*04:04-DQB1*03:16',
'HLA-DQA1*04:04-DQB1*03:17', 'HLA-DQA1*04:04-DQB1*03:18', 'HLA-DQA1*04:04-DQB1*03:19', 'HLA-DQA1*04:04-DQB1*03:20', 'HLA-DQA1*04:04-DQB1*03:21',
'HLA-DQA1*04:04-DQB1*03:22', 'HLA-DQA1*04:04-DQB1*03:23', 'HLA-DQA1*04:04-DQB1*03:24', 'HLA-DQA1*04:04-DQB1*03:25', 'HLA-DQA1*04:04-DQB1*03:26',
'HLA-DQA1*04:04-DQB1*03:27', 'HLA-DQA1*04:04-DQB1*03:28', 'HLA-DQA1*04:04-DQB1*03:29', 'HLA-DQA1*04:04-DQB1*03:30', 'HLA-DQA1*04:04-DQB1*03:31',
'HLA-DQA1*04:04-DQB1*03:32', 'HLA-DQA1*04:04-DQB1*03:33', 'HLA-DQA1*04:04-DQB1*03:34', 'HLA-DQA1*04:04-DQB1*03:35', 'HLA-DQA1*04:04-DQB1*03:36',
'HLA-DQA1*04:04-DQB1*03:37', 'HLA-DQA1*04:04-DQB1*03:38', 'HLA-DQA1*04:04-DQB1*04:01', 'HLA-DQA1*04:04-DQB1*04:02', 'HLA-DQA1*04:04-DQB1*04:03',
'HLA-DQA1*04:04-DQB1*04:04', 'HLA-DQA1*04:04-DQB1*04:05', 'HLA-DQA1*04:04-DQB1*04:06', 'HLA-DQA1*04:04-DQB1*04:07', 'HLA-DQA1*04:04-DQB1*04:08',
'HLA-DQA1*04:04-DQB1*05:01', 'HLA-DQA1*04:04-DQB1*05:02', 'HLA-DQA1*04:04-DQB1*05:03', 'HLA-DQA1*04:04-DQB1*05:05', 'HLA-DQA1*04:04-DQB1*05:06',
'HLA-DQA1*04:04-DQB1*05:07', 'HLA-DQA1*04:04-DQB1*05:08', 'HLA-DQA1*04:04-DQB1*05:09', 'HLA-DQA1*04:04-DQB1*05:10', 'HLA-DQA1*04:04-DQB1*05:11',
'HLA-DQA1*04:04-DQB1*05:12', 'HLA-DQA1*04:04-DQB1*05:13', 'HLA-DQA1*04:04-DQB1*05:14', 'HLA-DQA1*04:04-DQB1*06:01', 'HLA-DQA1*04:04-DQB1*06:02',
'HLA-DQA1*04:04-DQB1*06:03', 'HLA-DQA1*04:04-DQB1*06:04', 'HLA-DQA1*04:04-DQB1*06:07', 'HLA-DQA1*04:04-DQB1*06:08', 'HLA-DQA1*04:04-DQB1*06:09',
'HLA-DQA1*04:04-DQB1*06:10', 'HLA-DQA1*04:04-DQB1*06:11', 'HLA-DQA1*04:04-DQB1*06:12', 'HLA-DQA1*04:04-DQB1*06:14', 'HLA-DQA1*04:04-DQB1*06:15',
'HLA-DQA1*04:04-DQB1*06:16', 'HLA-DQA1*04:04-DQB1*06:17', 'HLA-DQA1*04:04-DQB1*06:18', 'HLA-DQA1*04:04-DQB1*06:19', 'HLA-DQA1*04:04-DQB1*06:21',
'HLA-DQA1*04:04-DQB1*06:22', 'HLA-DQA1*04:04-DQB1*06:23', 'HLA-DQA1*04:04-DQB1*06:24', 'HLA-DQA1*04:04-DQB1*06:25', 'HLA-DQA1*04:04-DQB1*06:27',
'HLA-DQA1*04:04-DQB1*06:28', 'HLA-DQA1*04:04-DQB1*06:29', 'HLA-DQA1*04:04-DQB1*06:30', 'HLA-DQA1*04:04-DQB1*06:31', 'HLA-DQA1*04:04-DQB1*06:32',
'HLA-DQA1*04:04-DQB1*06:33', 'HLA-DQA1*04:04-DQB1*06:34', 'HLA-DQA1*04:04-DQB1*06:35', 'HLA-DQA1*04:04-DQB1*06:36', 'HLA-DQA1*04:04-DQB1*06:37',
'HLA-DQA1*04:04-DQB1*06:38', 'HLA-DQA1*04:04-DQB1*06:39', 'HLA-DQA1*04:04-DQB1*06:40', 'HLA-DQA1*04:04-DQB1*06:41', 'HLA-DQA1*04:04-DQB1*06:42',
'HLA-DQA1*04:04-DQB1*06:43', 'HLA-DQA1*04:04-DQB1*06:44', 'HLA-DQA1*05:01-DQB1*02:01', 'HLA-DQA1*05:01-DQB1*02:02', 'HLA-DQA1*05:01-DQB1*02:03',
'HLA-DQA1*05:01-DQB1*02:04', 'HLA-DQA1*05:01-DQB1*02:05', 'HLA-DQA1*05:01-DQB1*02:06', 'HLA-DQA1*05:01-DQB1*03:01', 'HLA-DQA1*05:01-DQB1*03:02',
'HLA-DQA1*05:01-DQB1*03:03', 'HLA-DQA1*05:01-DQB1*03:04', 'HLA-DQA1*05:01-DQB1*03:05', 'HLA-DQA1*05:01-DQB1*03:06', 'HLA-DQA1*05:01-DQB1*03:07',
'HLA-DQA1*05:01-DQB1*03:08', 'HLA-DQA1*05:01-DQB1*03:09', 'HLA-DQA1*05:01-DQB1*03:10', 'HLA-DQA1*05:01-DQB1*03:11', 'HLA-DQA1*05:01-DQB1*03:12',
'HLA-DQA1*05:01-DQB1*03:13', 'HLA-DQA1*05:01-DQB1*03:14', 'HLA-DQA1*05:01-DQB1*03:15', 'HLA-DQA1*05:01-DQB1*03:16', 'HLA-DQA1*05:01-DQB1*03:17',
'HLA-DQA1*05:01-DQB1*03:18', 'HLA-DQA1*05:01-DQB1*03:19', 'HLA-DQA1*05:01-DQB1*03:20', 'HLA-DQA1*05:01-DQB1*03:21', 'HLA-DQA1*05:01-DQB1*03:22',
'HLA-DQA1*05:01-DQB1*03:23', 'HLA-DQA1*05:01-DQB1*03:24', 'HLA-DQA1*05:01-DQB1*03:25', 'HLA-DQA1*05:01-DQB1*03:26', 'HLA-DQA1*05:01-DQB1*03:27',
'HLA-DQA1*05:01-DQB1*03:28', 'HLA-DQA1*05:01-DQB1*03:29', 'HLA-DQA1*05:01-DQB1*03:30', 'HLA-DQA1*05:01-DQB1*03:31', 'HLA-DQA1*05:01-DQB1*03:32',
'HLA-DQA1*05:01-DQB1*03:33', 'HLA-DQA1*05:01-DQB1*03:34', 'HLA-DQA1*05:01-DQB1*03:35', 'HLA-DQA1*05:01-DQB1*03:36', 'HLA-DQA1*05:01-DQB1*03:37',
'HLA-DQA1*05:01-DQB1*03:38', 'HLA-DQA1*05:01-DQB1*04:01', 'HLA-DQA1*05:01-DQB1*04:02', 'HLA-DQA1*05:01-DQB1*04:03', 'HLA-DQA1*05:01-DQB1*04:04',
'HLA-DQA1*05:01-DQB1*04:05', 'HLA-DQA1*05:01-DQB1*04:06', 'HLA-DQA1*05:01-DQB1*04:07', 'HLA-DQA1*05:01-DQB1*04:08', 'HLA-DQA1*05:01-DQB1*05:01',
'HLA-DQA1*05:01-DQB1*05:02', 'HLA-DQA1*05:01-DQB1*05:03', 'HLA-DQA1*05:01-DQB1*05:05', 'HLA-DQA1*05:01-DQB1*05:06', 'HLA-DQA1*05:01-DQB1*05:07',
'HLA-DQA1*05:01-DQB1*05:08', 'HLA-DQA1*05:01-DQB1*05:09', 'HLA-DQA1*05:01-DQB1*05:10', 'HLA-DQA1*05:01-DQB1*05:11', 'HLA-DQA1*05:01-DQB1*05:12',
'HLA-DQA1*05:01-DQB1*05:13', 'HLA-DQA1*05:01-DQB1*05:14', 'HLA-DQA1*05:01-DQB1*06:01', 'HLA-DQA1*05:01-DQB1*06:02', 'HLA-DQA1*05:01-DQB1*06:03',
'HLA-DQA1*05:01-DQB1*06:04', 'HLA-DQA1*05:01-DQB1*06:07', 'HLA-DQA1*05:01-DQB1*06:08', 'HLA-DQA1*05:01-DQB1*06:09', 'HLA-DQA1*05:01-DQB1*06:10',
'HLA-DQA1*05:01-DQB1*06:11', 'HLA-DQA1*05:01-DQB1*06:12', 'HLA-DQA1*05:01-DQB1*06:14', 'HLA-DQA1*05:01-DQB1*06:15', 'HLA-DQA1*05:01-DQB1*06:16',
'HLA-DQA1*05:01-DQB1*06:17', 'HLA-DQA1*05:01-DQB1*06:18', 'HLA-DQA1*05:01-DQB1*06:19', 'HLA-DQA1*05:01-DQB1*06:21', 'HLA-DQA1*05:01-DQB1*06:22',
'HLA-DQA1*05:01-DQB1*06:23', 'HLA-DQA1*05:01-DQB1*06:24', 'HLA-DQA1*05:01-DQB1*06:25', 'HLA-DQA1*05:01-DQB1*06:27', 'HLA-DQA1*05:01-DQB1*06:28',
'HLA-DQA1*05:01-DQB1*06:29', 'HLA-DQA1*05:01-DQB1*06:30', 'HLA-DQA1*05:01-DQB1*06:31', 'HLA-DQA1*05:01-DQB1*06:32', 'HLA-DQA1*05:01-DQB1*06:33',
'HLA-DQA1*05:01-DQB1*06:34', 'HLA-DQA1*05:01-DQB1*06:35', 'HLA-DQA1*05:01-DQB1*06:36', 'HLA-DQA1*05:01-DQB1*06:37', 'HLA-DQA1*05:01-DQB1*06:38',
'HLA-DQA1*05:01-DQB1*06:39', 'HLA-DQA1*05:01-DQB1*06:40', 'HLA-DQA1*05:01-DQB1*06:41', 'HLA-DQA1*05:01-DQB1*06:42', 'HLA-DQA1*05:01-DQB1*06:43',
'HLA-DQA1*05:01-DQB1*06:44', 'HLA-DQA1*05:03-DQB1*02:01', 'HLA-DQA1*05:03-DQB1*02:02', 'HLA-DQA1*05:03-DQB1*02:03', 'HLA-DQA1*05:03-DQB1*02:04',
'HLA-DQA1*05:03-DQB1*02:05', 'HLA-DQA1*05:03-DQB1*02:06', 'HLA-DQA1*05:03-DQB1*03:01', 'HLA-DQA1*05:03-DQB1*03:02', 'HLA-DQA1*05:03-DQB1*03:03',
'HLA-DQA1*05:03-DQB1*03:04', 'HLA-DQA1*05:03-DQB1*03:05', 'HLA-DQA1*05:03-DQB1*03:06', 'HLA-DQA1*05:03-DQB1*03:07', 'HLA-DQA1*05:03-DQB1*03:08',
'HLA-DQA1*05:03-DQB1*03:09', 'HLA-DQA1*05:03-DQB1*03:10', 'HLA-DQA1*05:03-DQB1*03:11', 'HLA-DQA1*05:03-DQB1*03:12', 'HLA-DQA1*05:03-DQB1*03:13',
'HLA-DQA1*05:03-DQB1*03:14', 'HLA-DQA1*05:03-DQB1*03:15', 'HLA-DQA1*05:03-DQB1*03:16', 'HLA-DQA1*05:03-DQB1*03:17', 'HLA-DQA1*05:03-DQB1*03:18',
'HLA-DQA1*05:03-DQB1*03:19', 'HLA-DQA1*05:03-DQB1*03:20', 'HLA-DQA1*05:03-DQB1*03:21', 'HLA-DQA1*05:03-DQB1*03:22', 'HLA-DQA1*05:03-DQB1*03:23',
'HLA-DQA1*05:03-DQB1*03:24', 'HLA-DQA1*05:03-DQB1*03:25', 'HLA-DQA1*05:03-DQB1*03:26', 'HLA-DQA1*05:03-DQB1*03:27', 'HLA-DQA1*05:03-DQB1*03:28',
'HLA-DQA1*05:03-DQB1*03:29', 'HLA-DQA1*05:03-DQB1*03:30', 'HLA-DQA1*05:03-DQB1*03:31', 'HLA-DQA1*05:03-DQB1*03:32', 'HLA-DQA1*05:03-DQB1*03:33',
'HLA-DQA1*05:03-DQB1*03:34', 'HLA-DQA1*05:03-DQB1*03:35', 'HLA-DQA1*05:03-DQB1*03:36', 'HLA-DQA1*05:03-DQB1*03:37', 'HLA-DQA1*05:03-DQB1*03:38',
'HLA-DQA1*05:03-DQB1*04:01', 'HLA-DQA1*05:03-DQB1*04:02', 'HLA-DQA1*05:03-DQB1*04:03', 'HLA-DQA1*05:03-DQB1*04:04', 'HLA-DQA1*05:03-DQB1*04:05',
'HLA-DQA1*05:03-DQB1*04:06', 'HLA-DQA1*05:03-DQB1*04:07', 'HLA-DQA1*05:03-DQB1*04:08', 'HLA-DQA1*05:03-DQB1*05:01', 'HLA-DQA1*05:03-DQB1*05:02',
'HLA-DQA1*05:03-DQB1*05:03', 'HLA-DQA1*05:03-DQB1*05:05', 'HLA-DQA1*05:03-DQB1*05:06', 'HLA-DQA1*05:03-DQB1*05:07', 'HLA-DQA1*05:03-DQB1*05:08',
'HLA-DQA1*05:03-DQB1*05:09', 'HLA-DQA1*05:03-DQB1*05:10', 'HLA-DQA1*05:03-DQB1*05:11', 'HLA-DQA1*05:03-DQB1*05:12', 'HLA-DQA1*05:03-DQB1*05:13',
'HLA-DQA1*05:03-DQB1*05:14', 'HLA-DQA1*05:03-DQB1*06:01', 'HLA-DQA1*05:03-DQB1*06:02', 'HLA-DQA1*05:03-DQB1*06:03', 'HLA-DQA1*05:03-DQB1*06:04',
'HLA-DQA1*05:03-DQB1*06:07', 'HLA-DQA1*05:03-DQB1*06:08', 'HLA-DQA1*05:03-DQB1*06:09', 'HLA-DQA1*05:03-DQB1*06:10', 'HLA-DQA1*05:03-DQB1*06:11',
'HLA-DQA1*05:03-DQB1*06:12', 'HLA-DQA1*05:03-DQB1*06:14', 'HLA-DQA1*05:03-DQB1*06:15', 'HLA-DQA1*05:03-DQB1*06:16', 'HLA-DQA1*05:03-DQB1*06:17',
'HLA-DQA1*05:03-DQB1*06:18', 'HLA-DQA1*05:03-DQB1*06:19', 'HLA-DQA1*05:03-DQB1*06:21', 'HLA-DQA1*05:03-DQB1*06:22', 'HLA-DQA1*05:03-DQB1*06:23',
'HLA-DQA1*05:03-DQB1*06:24', 'HLA-DQA1*05:03-DQB1*06:25', 'HLA-DQA1*05:03-DQB1*06:27', 'HLA-DQA1*05:03-DQB1*06:28', 'HLA-DQA1*05:03-DQB1*06:29',
'HLA-DQA1*05:03-DQB1*06:30', 'HLA-DQA1*05:03-DQB1*06:31', 'HLA-DQA1*05:03-DQB1*06:32', 'HLA-DQA1*05:03-DQB1*06:33', 'HLA-DQA1*05:03-DQB1*06:34',
'HLA-DQA1*05:03-DQB1*06:35', 'HLA-DQA1*05:03-DQB1*06:36', 'HLA-DQA1*05:03-DQB1*06:37', 'HLA-DQA1*05:03-DQB1*06:38', 'HLA-DQA1*05:03-DQB1*06:39',
'HLA-DQA1*05:03-DQB1*06:40', 'HLA-DQA1*05:03-DQB1*06:41', 'HLA-DQA1*05:03-DQB1*06:42', 'HLA-DQA1*05:03-DQB1*06:43', 'HLA-DQA1*05:03-DQB1*06:44',
'HLA-DQA1*05:04-DQB1*02:01', 'HLA-DQA1*05:04-DQB1*02:02', 'HLA-DQA1*05:04-DQB1*02:03', 'HLA-DQA1*05:04-DQB1*02:04', 'HLA-DQA1*05:04-DQB1*02:05',
'HLA-DQA1*05:04-DQB1*02:06', 'HLA-DQA1*05:04-DQB1*03:01', 'HLA-DQA1*05:04-DQB1*03:02', 'HLA-DQA1*05:04-DQB1*03:03', 'HLA-DQA1*05:04-DQB1*03:04',
'HLA-DQA1*05:04-DQB1*03:05', 'HLA-DQA1*05:04-DQB1*03:06', 'HLA-DQA1*05:04-DQB1*03:07', 'HLA-DQA1*05:04-DQB1*03:08', 'HLA-DQA1*05:04-DQB1*03:09',
'HLA-DQA1*05:04-DQB1*03:10', 'HLA-DQA1*05:04-DQB1*03:11', 'HLA-DQA1*05:04-DQB1*03:12', 'HLA-DQA1*05:04-DQB1*03:13', 'HLA-DQA1*05:04-DQB1*03:14',
'HLA-DQA1*05:04-DQB1*03:15', 'HLA-DQA1*05:04-DQB1*03:16', 'HLA-DQA1*05:04-DQB1*03:17', 'HLA-DQA1*05:04-DQB1*03:18', 'HLA-DQA1*05:04-DQB1*03:19',
'HLA-DQA1*05:04-DQB1*03:20', 'HLA-DQA1*05:04-DQB1*03:21', 'HLA-DQA1*05:04-DQB1*03:22', 'HLA-DQA1*05:04-DQB1*03:23', 'HLA-DQA1*05:04-DQB1*03:24',
'HLA-DQA1*05:04-DQB1*03:25', 'HLA-DQA1*05:04-DQB1*03:26', 'HLA-DQA1*05:04-DQB1*03:27', 'HLA-DQA1*05:04-DQB1*03:28', 'HLA-DQA1*05:04-DQB1*03:29',
'HLA-DQA1*05:04-DQB1*03:30', 'HLA-DQA1*05:04-DQB1*03:31', 'HLA-DQA1*05:04-DQB1*03:32', 'HLA-DQA1*05:04-DQB1*03:33', 'HLA-DQA1*05:04-DQB1*03:34',
'HLA-DQA1*05:04-DQB1*03:35', 'HLA-DQA1*05:04-DQB1*03:36', 'HLA-DQA1*05:04-DQB1*03:37', 'HLA-DQA1*05:04-DQB1*03:38', 'HLA-DQA1*05:04-DQB1*04:01',
'HLA-DQA1*05:04-DQB1*04:02', 'HLA-DQA1*05:04-DQB1*04:03', 'HLA-DQA1*05:04-DQB1*04:04', 'HLA-DQA1*05:04-DQB1*04:05', 'HLA-DQA1*05:04-DQB1*04:06',
'HLA-DQA1*05:04-DQB1*04:07', 'HLA-DQA1*05:04-DQB1*04:08', 'HLA-DQA1*05:04-DQB1*05:01', 'HLA-DQA1*05:04-DQB1*05:02', 'HLA-DQA1*05:04-DQB1*05:03',
'HLA-DQA1*05:04-DQB1*05:05', 'HLA-DQA1*05:04-DQB1*05:06', 'HLA-DQA1*05:04-DQB1*05:07', 'HLA-DQA1*05:04-DQB1*05:08', 'HLA-DQA1*05:04-DQB1*05:09',
'HLA-DQA1*05:04-DQB1*05:10', 'HLA-DQA1*05:04-DQB1*05:11', 'HLA-DQA1*05:04-DQB1*05:12', 'HLA-DQA1*05:04-DQB1*05:13', 'HLA-DQA1*05:04-DQB1*05:14',
'HLA-DQA1*05:04-DQB1*06:01', 'HLA-DQA1*05:04-DQB1*06:02', 'HLA-DQA1*05:04-DQB1*06:03', 'HLA-DQA1*05:04-DQB1*06:04', 'HLA-DQA1*05:04-DQB1*06:07',
'HLA-DQA1*05:04-DQB1*06:08', 'HLA-DQA1*05:04-DQB1*06:09', 'HLA-DQA1*05:04-DQB1*06:10', 'HLA-DQA1*05:04-DQB1*06:11', 'HLA-DQA1*05:04-DQB1*06:12',
'HLA-DQA1*05:04-DQB1*06:14', 'HLA-DQA1*05:04-DQB1*06:15', 'HLA-DQA1*05:04-DQB1*06:16', 'HLA-DQA1*05:04-DQB1*06:17', 'HLA-DQA1*05:04-DQB1*06:18',
'HLA-DQA1*05:04-DQB1*06:19', 'HLA-DQA1*05:04-DQB1*06:21', 'HLA-DQA1*05:04-DQB1*06:22', 'HLA-DQA1*05:04-DQB1*06:23', 'HLA-DQA1*05:04-DQB1*06:24',
'HLA-DQA1*05:04-DQB1*06:25', 'HLA-DQA1*05:04-DQB1*06:27', 'HLA-DQA1*05:04-DQB1*06:28', 'HLA-DQA1*05:04-DQB1*06:29', 'HLA-DQA1*05:04-DQB1*06:30',
'HLA-DQA1*05:04-DQB1*06:31', 'HLA-DQA1*05:04-DQB1*06:32', 'HLA-DQA1*05:04-DQB1*06:33', 'HLA-DQA1*05:04-DQB1*06:34', 'HLA-DQA1*05:04-DQB1*06:35',
'HLA-DQA1*05:04-DQB1*06:36', 'HLA-DQA1*05:04-DQB1*06:37', 'HLA-DQA1*05:04-DQB1*06:38', 'HLA-DQA1*05:04-DQB1*06:39', 'HLA-DQA1*05:04-DQB1*06:40',
'HLA-DQA1*05:04-DQB1*06:41', 'HLA-DQA1*05:04-DQB1*06:42', 'HLA-DQA1*05:04-DQB1*06:43', 'HLA-DQA1*05:04-DQB1*06:44', 'HLA-DQA1*05:05-DQB1*02:01',
'HLA-DQA1*05:05-DQB1*02:02', 'HLA-DQA1*05:05-DQB1*02:03', 'HLA-DQA1*05:05-DQB1*02:04', 'HLA-DQA1*05:05-DQB1*02:05', 'HLA-DQA1*05:05-DQB1*02:06',
'HLA-DQA1*05:05-DQB1*03:01', 'HLA-DQA1*05:05-DQB1*03:02', 'HLA-DQA1*05:05-DQB1*03:03', 'HLA-DQA1*05:05-DQB1*03:04', 'HLA-DQA1*05:05-DQB1*03:05',
'HLA-DQA1*05:05-DQB1*03:06', 'HLA-DQA1*05:05-DQB1*03:07', 'HLA-DQA1*05:05-DQB1*03:08', 'HLA-DQA1*05:05-DQB1*03:09', 'HLA-DQA1*05:05-DQB1*03:10',
'HLA-DQA1*05:05-DQB1*03:11', 'HLA-DQA1*05:05-DQB1*03:12', 'HLA-DQA1*05:05-DQB1*03:13', 'HLA-DQA1*05:05-DQB1*03:14', 'HLA-DQA1*05:05-DQB1*03:15',
'HLA-DQA1*05:05-DQB1*03:16', 'HLA-DQA1*05:05-DQB1*03:17', 'HLA-DQA1*05:05-DQB1*03:18', 'HLA-DQA1*05:05-DQB1*03:19', 'HLA-DQA1*05:05-DQB1*03:20',
'HLA-DQA1*05:05-DQB1*03:21', 'HLA-DQA1*05:05-DQB1*03:22', 'HLA-DQA1*05:05-DQB1*03:23', 'HLA-DQA1*05:05-DQB1*03:24', 'HLA-DQA1*05:05-DQB1*03:25',
'HLA-DQA1*05:05-DQB1*03:26', 'HLA-DQA1*05:05-DQB1*03:27', 'HLA-DQA1*05:05-DQB1*03:28', 'HLA-DQA1*05:05-DQB1*03:29', 'HLA-DQA1*05:05-DQB1*03:30',
'HLA-DQA1*05:05-DQB1*03:31', 'HLA-DQA1*05:05-DQB1*03:32', 'HLA-DQA1*05:05-DQB1*03:33', 'HLA-DQA1*05:05-DQB1*03:34', 'HLA-DQA1*05:05-DQB1*03:35',
'HLA-DQA1*05:05-DQB1*03:36', 'HLA-DQA1*05:05-DQB1*03:37', 'HLA-DQA1*05:05-DQB1*03:38', 'HLA-DQA1*05:05-DQB1*04:01', 'HLA-DQA1*05:05-DQB1*04:02',
'HLA-DQA1*05:05-DQB1*04:03', 'HLA-DQA1*05:05-DQB1*04:04', 'HLA-DQA1*05:05-DQB1*04:05', 'HLA-DQA1*05:05-DQB1*04:06', 'HLA-DQA1*05:05-DQB1*04:07',
'HLA-DQA1*05:05-DQB1*04:08', 'HLA-DQA1*05:05-DQB1*05:01', 'HLA-DQA1*05:05-DQB1*05:02', 'HLA-DQA1*05:05-DQB1*05:03', 'HLA-DQA1*05:05-DQB1*05:05',
'HLA-DQA1*05:05-DQB1*05:06', 'HLA-DQA1*05:05-DQB1*05:07', 'HLA-DQA1*05:05-DQB1*05:08', 'HLA-DQA1*05:05-DQB1*05:09', 'HLA-DQA1*05:05-DQB1*05:10',
'HLA-DQA1*05:05-DQB1*05:11', 'HLA-DQA1*05:05-DQB1*05:12', 'HLA-DQA1*05:05-DQB1*05:13', 'HLA-DQA1*05:05-DQB1*05:14', 'HLA-DQA1*05:05-DQB1*06:01',
'HLA-DQA1*05:05-DQB1*06:02', 'HLA-DQA1*05:05-DQB1*06:03', 'HLA-DQA1*05:05-DQB1*06:04', 'HLA-DQA1*05:05-DQB1*06:07', 'HLA-DQA1*05:05-DQB1*06:08',
'HLA-DQA1*05:05-DQB1*06:09', 'HLA-DQA1*05:05-DQB1*06:10', 'HLA-DQA1*05:05-DQB1*06:11', 'HLA-DQA1*05:05-DQB1*06:12', 'HLA-DQA1*05:05-DQB1*06:14',
'HLA-DQA1*05:05-DQB1*06:15', 'HLA-DQA1*05:05-DQB1*06:16', 'HLA-DQA1*05:05-DQB1*06:17', 'HLA-DQA1*05:05-DQB1*06:18', 'HLA-DQA1*05:05-DQB1*06:19',
'HLA-DQA1*05:05-DQB1*06:21', 'HLA-DQA1*05:05-DQB1*06:22', 'HLA-DQA1*05:05-DQB1*06:23', 'HLA-DQA1*05:05-DQB1*06:24', 'HLA-DQA1*05:05-DQB1*06:25',
'HLA-DQA1*05:05-DQB1*06:27', 'HLA-DQA1*05:05-DQB1*06:28', 'HLA-DQA1*05:05-DQB1*06:29', 'HLA-DQA1*05:05-DQB1*06:30', 'HLA-DQA1*05:05-DQB1*06:31',
'HLA-DQA1*05:05-DQB1*06:32', 'HLA-DQA1*05:05-DQB1*06:33', 'HLA-DQA1*05:05-DQB1*06:34', 'HLA-DQA1*05:05-DQB1*06:35', 'HLA-DQA1*05:05-DQB1*06:36',
'HLA-DQA1*05:05-DQB1*06:37', 'HLA-DQA1*05:05-DQB1*06:38', 'HLA-DQA1*05:05-DQB1*06:39', 'HLA-DQA1*05:05-DQB1*06:40', 'HLA-DQA1*05:05-DQB1*06:41',
'HLA-DQA1*05:05-DQB1*06:42', 'HLA-DQA1*05:05-DQB1*06:43', 'HLA-DQA1*05:05-DQB1*06:44', 'HLA-DQA1*05:06-DQB1*02:01', 'HLA-DQA1*05:06-DQB1*02:02',
'HLA-DQA1*05:06-DQB1*02:03', 'HLA-DQA1*05:06-DQB1*02:04', 'HLA-DQA1*05:06-DQB1*02:05', 'HLA-DQA1*05:06-DQB1*02:06', 'HLA-DQA1*05:06-DQB1*03:01',
'HLA-DQA1*05:06-DQB1*03:02', 'HLA-DQA1*05:06-DQB1*03:03', 'HLA-DQA1*05:06-DQB1*03:04', 'HLA-DQA1*05:06-DQB1*03:05', 'HLA-DQA1*05:06-DQB1*03:06',
'HLA-DQA1*05:06-DQB1*03:07', 'HLA-DQA1*05:06-DQB1*03:08', 'HLA-DQA1*05:06-DQB1*03:09', 'HLA-DQA1*05:06-DQB1*03:10', 'HLA-DQA1*05:06-DQB1*03:11',
'HLA-DQA1*05:06-DQB1*03:12', 'HLA-DQA1*05:06-DQB1*03:13', 'HLA-DQA1*05:06-DQB1*03:14', 'HLA-DQA1*05:06-DQB1*03:15', 'HLA-DQA1*05:06-DQB1*03:16',
'HLA-DQA1*05:06-DQB1*03:17', 'HLA-DQA1*05:06-DQB1*03:18', 'HLA-DQA1*05:06-DQB1*03:19', 'HLA-DQA1*05:06-DQB1*03:20', 'HLA-DQA1*05:06-DQB1*03:21',
'HLA-DQA1*05:06-DQB1*03:22', 'HLA-DQA1*05:06-DQB1*03:23', 'HLA-DQA1*05:06-DQB1*03:24', 'HLA-DQA1*05:06-DQB1*03:25', 'HLA-DQA1*05:06-DQB1*03:26',
'HLA-DQA1*05:06-DQB1*03:27', 'HLA-DQA1*05:06-DQB1*03:28', 'HLA-DQA1*05:06-DQB1*03:29', 'HLA-DQA1*05:06-DQB1*03:30', 'HLA-DQA1*05:06-DQB1*03:31',
'HLA-DQA1*05:06-DQB1*03:32', 'HLA-DQA1*05:06-DQB1*03:33', 'HLA-DQA1*05:06-DQB1*03:34', 'HLA-DQA1*05:06-DQB1*03:35', 'HLA-DQA1*05:06-DQB1*03:36',
'HLA-DQA1*05:06-DQB1*03:37', 'HLA-DQA1*05:06-DQB1*03:38', 'HLA-DQA1*05:06-DQB1*04:01', 'HLA-DQA1*05:06-DQB1*04:02', 'HLA-DQA1*05:06-DQB1*04:03',
'HLA-DQA1*05:06-DQB1*04:04', 'HLA-DQA1*05:06-DQB1*04:05', 'HLA-DQA1*05:06-DQB1*04:06', 'HLA-DQA1*05:06-DQB1*04:07', 'HLA-DQA1*05:06-DQB1*04:08',
'HLA-DQA1*05:06-DQB1*05:01', 'HLA-DQA1*05:06-DQB1*05:02', 'HLA-DQA1*05:06-DQB1*05:03', 'HLA-DQA1*05:06-DQB1*05:05', 'HLA-DQA1*05:06-DQB1*05:06',
'HLA-DQA1*05:06-DQB1*05:07', 'HLA-DQA1*05:06-DQB1*05:08', 'HLA-DQA1*05:06-DQB1*05:09', 'HLA-DQA1*05:06-DQB1*05:10', 'HLA-DQA1*05:06-DQB1*05:11',
'HLA-DQA1*05:06-DQB1*05:12', 'HLA-DQA1*05:06-DQB1*05:13', 'HLA-DQA1*05:06-DQB1*05:14', 'HLA-DQA1*05:06-DQB1*06:01', 'HLA-DQA1*05:06-DQB1*06:02',
'HLA-DQA1*05:06-DQB1*06:03', 'HLA-DQA1*05:06-DQB1*06:04', 'HLA-DQA1*05:06-DQB1*06:07', 'HLA-DQA1*05:06-DQB1*06:08', 'HLA-DQA1*05:06-DQB1*06:09',
'HLA-DQA1*05:06-DQB1*06:10', 'HLA-DQA1*05:06-DQB1*06:11', 'HLA-DQA1*05:06-DQB1*06:12', 'HLA-DQA1*05:06-DQB1*06:14', 'HLA-DQA1*05:06-DQB1*06:15',
'HLA-DQA1*05:06-DQB1*06:16', 'HLA-DQA1*05:06-DQB1*06:17', 'HLA-DQA1*05:06-DQB1*06:18', 'HLA-DQA1*05:06-DQB1*06:19', 'HLA-DQA1*05:06-DQB1*06:21',
'HLA-DQA1*05:06-DQB1*06:22', 'HLA-DQA1*05:06-DQB1*06:23', 'HLA-DQA1*05:06-DQB1*06:24', 'HLA-DQA1*05:06-DQB1*06:25', 'HLA-DQA1*05:06-DQB1*06:27',
'HLA-DQA1*05:06-DQB1*06:28', 'HLA-DQA1*05:06-DQB1*06:29', 'HLA-DQA1*05:06-DQB1*06:30', 'HLA-DQA1*05:06-DQB1*06:31', 'HLA-DQA1*05:06-DQB1*06:32',
'HLA-DQA1*05:06-DQB1*06:33', 'HLA-DQA1*05:06-DQB1*06:34', 'HLA-DQA1*05:06-DQB1*06:35', 'HLA-DQA1*05:06-DQB1*06:36', 'HLA-DQA1*05:06-DQB1*06:37',
'HLA-DQA1*05:06-DQB1*06:38', 'HLA-DQA1*05:06-DQB1*06:39', 'HLA-DQA1*05:06-DQB1*06:40', 'HLA-DQA1*05:06-DQB1*06:41', 'HLA-DQA1*05:06-DQB1*06:42',
'HLA-DQA1*05:06-DQB1*06:43', 'HLA-DQA1*05:06-DQB1*06:44', 'HLA-DQA1*05:07-DQB1*02:01', 'HLA-DQA1*05:07-DQB1*02:02', 'HLA-DQA1*05:07-DQB1*02:03',
'HLA-DQA1*05:07-DQB1*02:04', 'HLA-DQA1*05:07-DQB1*02:05', 'HLA-DQA1*05:07-DQB1*02:06', 'HLA-DQA1*05:07-DQB1*03:01', 'HLA-DQA1*05:07-DQB1*03:02',
'HLA-DQA1*05:07-DQB1*03:03', 'HLA-DQA1*05:07-DQB1*03:04', 'HLA-DQA1*05:07-DQB1*03:05', 'HLA-DQA1*05:07-DQB1*03:06', 'HLA-DQA1*05:07-DQB1*03:07',
'HLA-DQA1*05:07-DQB1*03:08', 'HLA-DQA1*05:07-DQB1*03:09', 'HLA-DQA1*05:07-DQB1*03:10', 'HLA-DQA1*05:07-DQB1*03:11', 'HLA-DQA1*05:07-DQB1*03:12',
'HLA-DQA1*05:07-DQB1*03:13', 'HLA-DQA1*05:07-DQB1*03:14', 'HLA-DQA1*05:07-DQB1*03:15', 'HLA-DQA1*05:07-DQB1*03:16', 'HLA-DQA1*05:07-DQB1*03:17',
'HLA-DQA1*05:07-DQB1*03:18', 'HLA-DQA1*05:07-DQB1*03:19', 'HLA-DQA1*05:07-DQB1*03:20', 'HLA-DQA1*05:07-DQB1*03:21', 'HLA-DQA1*05:07-DQB1*03:22',
'HLA-DQA1*05:07-DQB1*03:23', 'HLA-DQA1*05:07-DQB1*03:24', 'HLA-DQA1*05:07-DQB1*03:25', 'HLA-DQA1*05:07-DQB1*03:26', 'HLA-DQA1*05:07-DQB1*03:27',
'HLA-DQA1*05:07-DQB1*03:28', 'HLA-DQA1*05:07-DQB1*03:29', 'HLA-DQA1*05:07-DQB1*03:30', 'HLA-DQA1*05:07-DQB1*03:31', 'HLA-DQA1*05:07-DQB1*03:32',
'HLA-DQA1*05:07-DQB1*03:33', 'HLA-DQA1*05:07-DQB1*03:34', 'HLA-DQA1*05:07-DQB1*03:35', 'HLA-DQA1*05:07-DQB1*03:36', 'HLA-DQA1*05:07-DQB1*03:37',
'HLA-DQA1*05:07-DQB1*03:38', 'HLA-DQA1*05:07-DQB1*04:01', 'HLA-DQA1*05:07-DQB1*04:02', 'HLA-DQA1*05:07-DQB1*04:03', 'HLA-DQA1*05:07-DQB1*04:04',
'HLA-DQA1*05:07-DQB1*04:05', 'HLA-DQA1*05:07-DQB1*04:06', 'HLA-DQA1*05:07-DQB1*04:07', 'HLA-DQA1*05:07-DQB1*04:08', 'HLA-DQA1*05:07-DQB1*05:01',
'HLA-DQA1*05:07-DQB1*05:02', 'HLA-DQA1*05:07-DQB1*05:03', 'HLA-DQA1*05:07-DQB1*05:05', 'HLA-DQA1*05:07-DQB1*05:06', 'HLA-DQA1*05:07-DQB1*05:07',
'HLA-DQA1*05:07-DQB1*05:08', 'HLA-DQA1*05:07-DQB1*05:09', 'HLA-DQA1*05:07-DQB1*05:10', 'HLA-DQA1*05:07-DQB1*05:11', 'HLA-DQA1*05:07-DQB1*05:12',
'HLA-DQA1*05:07-DQB1*05:13', 'HLA-DQA1*05:07-DQB1*05:14', 'HLA-DQA1*05:07-DQB1*06:01', 'HLA-DQA1*05:07-DQB1*06:02', 'HLA-DQA1*05:07-DQB1*06:03',
'HLA-DQA1*05:07-DQB1*06:04', 'HLA-DQA1*05:07-DQB1*06:07', 'HLA-DQA1*05:07-DQB1*06:08', 'HLA-DQA1*05:07-DQB1*06:09', 'HLA-DQA1*05:07-DQB1*06:10',
'HLA-DQA1*05:07-DQB1*06:11', 'HLA-DQA1*05:07-DQB1*06:12', 'HLA-DQA1*05:07-DQB1*06:14', 'HLA-DQA1*05:07-DQB1*06:15', 'HLA-DQA1*05:07-DQB1*06:16',
'HLA-DQA1*05:07-DQB1*06:17', 'HLA-DQA1*05:07-DQB1*06:18', 'HLA-DQA1*05:07-DQB1*06:19', 'HLA-DQA1*05:07-DQB1*06:21', 'HLA-DQA1*05:07-DQB1*06:22',
'HLA-DQA1*05:07-DQB1*06:23', 'HLA-DQA1*05:07-DQB1*06:24', 'HLA-DQA1*05:07-DQB1*06:25', 'HLA-DQA1*05:07-DQB1*06:27', 'HLA-DQA1*05:07-DQB1*06:28',
'HLA-DQA1*05:07-DQB1*06:29', 'HLA-DQA1*05:07-DQB1*06:30', 'HLA-DQA1*05:07-DQB1*06:31', 'HLA-DQA1*05:07-DQB1*06:32', 'HLA-DQA1*05:07-DQB1*06:33',
'HLA-DQA1*05:07-DQB1*06:34', 'HLA-DQA1*05:07-DQB1*06:35', 'HLA-DQA1*05:07-DQB1*06:36', 'HLA-DQA1*05:07-DQB1*06:37', 'HLA-DQA1*05:07-DQB1*06:38',
'HLA-DQA1*05:07-DQB1*06:39', 'HLA-DQA1*05:07-DQB1*06:40', 'HLA-DQA1*05:07-DQB1*06:41', 'HLA-DQA1*05:07-DQB1*06:42', 'HLA-DQA1*05:07-DQB1*06:43',
'HLA-DQA1*05:07-DQB1*06:44', 'HLA-DQA1*05:08-DQB1*02:01', 'HLA-DQA1*05:08-DQB1*02:02', 'HLA-DQA1*05:08-DQB1*02:03', 'HLA-DQA1*05:08-DQB1*02:04',
'HLA-DQA1*05:08-DQB1*02:05', 'HLA-DQA1*05:08-DQB1*02:06', 'HLA-DQA1*05:08-DQB1*03:01', 'HLA-DQA1*05:08-DQB1*03:02', 'HLA-DQA1*05:08-DQB1*03:03',
'HLA-DQA1*05:08-DQB1*03:04', 'HLA-DQA1*05:08-DQB1*03:05', 'HLA-DQA1*05:08-DQB1*03:06', 'HLA-DQA1*05:08-DQB1*03:07', 'HLA-DQA1*05:08-DQB1*03:08',
'HLA-DQA1*05:08-DQB1*03:09', 'HLA-DQA1*05:08-DQB1*03:10', 'HLA-DQA1*05:08-DQB1*03:11', 'HLA-DQA1*05:08-DQB1*03:12', 'HLA-DQA1*05:08-DQB1*03:13',
'HLA-DQA1*05:08-DQB1*03:14', 'HLA-DQA1*05:08-DQB1*03:15', 'HLA-DQA1*05:08-DQB1*03:16', 'HLA-DQA1*05:08-DQB1*03:17', 'HLA-DQA1*05:08-DQB1*03:18',
'HLA-DQA1*05:08-DQB1*03:19', 'HLA-DQA1*05:08-DQB1*03:20', 'HLA-DQA1*05:08-DQB1*03:21', 'HLA-DQA1*05:08-DQB1*03:22', 'HLA-DQA1*05:08-DQB1*03:23',
'HLA-DQA1*05:08-DQB1*03:24', 'HLA-DQA1*05:08-DQB1*03:25', 'HLA-DQA1*05:08-DQB1*03:26', 'HLA-DQA1*05:08-DQB1*03:27', 'HLA-DQA1*05:08-DQB1*03:28',
'HLA-DQA1*05:08-DQB1*03:29', 'HLA-DQA1*05:08-DQB1*03:30', 'HLA-DQA1*05:08-DQB1*03:31', 'HLA-DQA1*05:08-DQB1*03:32', 'HLA-DQA1*05:08-DQB1*03:33',
'HLA-DQA1*05:08-DQB1*03:34', 'HLA-DQA1*05:08-DQB1*03:35', 'HLA-DQA1*05:08-DQB1*03:36', 'HLA-DQA1*05:08-DQB1*03:37', 'HLA-DQA1*05:08-DQB1*03:38',
'HLA-DQA1*05:08-DQB1*04:01', 'HLA-DQA1*05:08-DQB1*04:02', 'HLA-DQA1*05:08-DQB1*04:03', 'HLA-DQA1*05:08-DQB1*04:04', 'HLA-DQA1*05:08-DQB1*04:05',
'HLA-DQA1*05:08-DQB1*04:06', 'HLA-DQA1*05:08-DQB1*04:07', 'HLA-DQA1*05:08-DQB1*04:08', 'HLA-DQA1*05:08-DQB1*05:01', 'HLA-DQA1*05:08-DQB1*05:02',
'HLA-DQA1*05:08-DQB1*05:03', 'HLA-DQA1*05:08-DQB1*05:05', 'HLA-DQA1*05:08-DQB1*05:06', 'HLA-DQA1*05:08-DQB1*05:07', 'HLA-DQA1*05:08-DQB1*05:08',
'HLA-DQA1*05:08-DQB1*05:09', 'HLA-DQA1*05:08-DQB1*05:10', 'HLA-DQA1*05:08-DQB1*05:11', 'HLA-DQA1*05:08-DQB1*05:12', 'HLA-DQA1*05:08-DQB1*05:13',
'HLA-DQA1*05:08-DQB1*05:14', 'HLA-DQA1*05:08-DQB1*06:01', 'HLA-DQA1*05:08-DQB1*06:02', 'HLA-DQA1*05:08-DQB1*06:03', 'HLA-DQA1*05:08-DQB1*06:04',
'HLA-DQA1*05:08-DQB1*06:07', 'HLA-DQA1*05:08-DQB1*06:08', 'HLA-DQA1*05:08-DQB1*06:09', 'HLA-DQA1*05:08-DQB1*06:10', 'HLA-DQA1*05:08-DQB1*06:11',
'HLA-DQA1*05:08-DQB1*06:12', 'HLA-DQA1*05:08-DQB1*06:14', 'HLA-DQA1*05:08-DQB1*06:15', 'HLA-DQA1*05:08-DQB1*06:16', 'HLA-DQA1*05:08-DQB1*06:17',
'HLA-DQA1*05:08-DQB1*06:18', 'HLA-DQA1*05:08-DQB1*06:19', 'HLA-DQA1*05:08-DQB1*06:21', 'HLA-DQA1*05:08-DQB1*06:22', 'HLA-DQA1*05:08-DQB1*06:23',
'HLA-DQA1*05:08-DQB1*06:24', 'HLA-DQA1*05:08-DQB1*06:25', 'HLA-DQA1*05:08-DQB1*06:27', 'HLA-DQA1*05:08-DQB1*06:28', 'HLA-DQA1*05:08-DQB1*06:29',
'HLA-DQA1*05:08-DQB1*06:30', 'HLA-DQA1*05:08-DQB1*06:31', 'HLA-DQA1*05:08-DQB1*06:32', 'HLA-DQA1*05:08-DQB1*06:33', 'HLA-DQA1*05:08-DQB1*06:34',
'HLA-DQA1*05:08-DQB1*06:35', 'HLA-DQA1*05:08-DQB1*06:36', 'HLA-DQA1*05:08-DQB1*06:37', 'HLA-DQA1*05:08-DQB1*06:38', 'HLA-DQA1*05:08-DQB1*06:39',
'HLA-DQA1*05:08-DQB1*06:40', 'HLA-DQA1*05:08-DQB1*06:41', 'HLA-DQA1*05:08-DQB1*06:42', 'HLA-DQA1*05:08-DQB1*06:43', 'HLA-DQA1*05:08-DQB1*06:44',
'HLA-DQA1*05:09-DQB1*02:01', 'HLA-DQA1*05:09-DQB1*02:02', 'HLA-DQA1*05:09-DQB1*02:03', 'HLA-DQA1*05:09-DQB1*02:04', 'HLA-DQA1*05:09-DQB1*02:05',
'HLA-DQA1*05:09-DQB1*02:06', 'HLA-DQA1*05:09-DQB1*03:01', 'HLA-DQA1*05:09-DQB1*03:02', 'HLA-DQA1*05:09-DQB1*03:03', 'HLA-DQA1*05:09-DQB1*03:04',
'HLA-DQA1*05:09-DQB1*03:05', 'HLA-DQA1*05:09-DQB1*03:06', 'HLA-DQA1*05:09-DQB1*03:07', 'HLA-DQA1*05:09-DQB1*03:08', 'HLA-DQA1*05:09-DQB1*03:09',
'HLA-DQA1*05:09-DQB1*03:10', 'HLA-DQA1*05:09-DQB1*03:11', 'HLA-DQA1*05:09-DQB1*03:12', 'HLA-DQA1*05:09-DQB1*03:13', 'HLA-DQA1*05:09-DQB1*03:14',
'HLA-DQA1*05:09-DQB1*03:15', 'HLA-DQA1*05:09-DQB1*03:16', 'HLA-DQA1*05:09-DQB1*03:17', 'HLA-DQA1*05:09-DQB1*03:18', 'HLA-DQA1*05:09-DQB1*03:19',
'HLA-DQA1*05:09-DQB1*03:20', 'HLA-DQA1*05:09-DQB1*03:21', 'HLA-DQA1*05:09-DQB1*03:22', 'HLA-DQA1*05:09-DQB1*03:23', 'HLA-DQA1*05:09-DQB1*03:24',
'HLA-DQA1*05:09-DQB1*03:25', 'HLA-DQA1*05:09-DQB1*03:26', 'HLA-DQA1*05:09-DQB1*03:27', 'HLA-DQA1*05:09-DQB1*03:28', 'HLA-DQA1*05:09-DQB1*03:29',
'HLA-DQA1*05:09-DQB1*03:30', 'HLA-DQA1*05:09-DQB1*03:31', 'HLA-DQA1*05:09-DQB1*03:32', 'HLA-DQA1*05:09-DQB1*03:33', 'HLA-DQA1*05:09-DQB1*03:34',
'HLA-DQA1*05:09-DQB1*03:35', 'HLA-DQA1*05:09-DQB1*03:36', 'HLA-DQA1*05:09-DQB1*03:37', 'HLA-DQA1*05:09-DQB1*03:38', 'HLA-DQA1*05:09-DQB1*04:01',
'HLA-DQA1*05:09-DQB1*04:02', 'HLA-DQA1*05:09-DQB1*04:03', 'HLA-DQA1*05:09-DQB1*04:04', 'HLA-DQA1*05:09-DQB1*04:05', 'HLA-DQA1*05:09-DQB1*04:06',
'HLA-DQA1*05:09-DQB1*04:07', 'HLA-DQA1*05:09-DQB1*04:08', 'HLA-DQA1*05:09-DQB1*05:01', 'HLA-DQA1*05:09-DQB1*05:02', 'HLA-DQA1*05:09-DQB1*05:03',
'HLA-DQA1*05:09-DQB1*05:05', 'HLA-DQA1*05:09-DQB1*05:06', 'HLA-DQA1*05:09-DQB1*05:07', 'HLA-DQA1*05:09-DQB1*05:08', 'HLA-DQA1*05:09-DQB1*05:09',
'HLA-DQA1*05:09-DQB1*05:10', 'HLA-DQA1*05:09-DQB1*05:11', 'HLA-DQA1*05:09-DQB1*05:12', 'HLA-DQA1*05:09-DQB1*05:13', 'HLA-DQA1*05:09-DQB1*05:14',
'HLA-DQA1*05:09-DQB1*06:01', 'HLA-DQA1*05:09-DQB1*06:02', 'HLA-DQA1*05:09-DQB1*06:03', 'HLA-DQA1*05:09-DQB1*06:04', 'HLA-DQA1*05:09-DQB1*06:07',
'HLA-DQA1*05:09-DQB1*06:08', 'HLA-DQA1*05:09-DQB1*06:09', 'HLA-DQA1*05:09-DQB1*06:10', 'HLA-DQA1*05:09-DQB1*06:11', 'HLA-DQA1*05:09-DQB1*06:12',
'HLA-DQA1*05:09-DQB1*06:14', 'HLA-DQA1*05:09-DQB1*06:15', 'HLA-DQA1*05:09-DQB1*06:16', 'HLA-DQA1*05:09-DQB1*06:17', 'HLA-DQA1*05:09-DQB1*06:18',
'HLA-DQA1*05:09-DQB1*06:19', 'HLA-DQA1*05:09-DQB1*06:21', 'HLA-DQA1*05:09-DQB1*06:22', 'HLA-DQA1*05:09-DQB1*06:23', 'HLA-DQA1*05:09-DQB1*06:24',
'HLA-DQA1*05:09-DQB1*06:25', 'HLA-DQA1*05:09-DQB1*06:27', 'HLA-DQA1*05:09-DQB1*06:28', 'HLA-DQA1*05:09-DQB1*06:29', 'HLA-DQA1*05:09-DQB1*06:30',
'HLA-DQA1*05:09-DQB1*06:31', 'HLA-DQA1*05:09-DQB1*06:32', 'HLA-DQA1*05:09-DQB1*06:33', 'HLA-DQA1*05:09-DQB1*06:34', 'HLA-DQA1*05:09-DQB1*06:35',
'HLA-DQA1*05:09-DQB1*06:36', 'HLA-DQA1*05:09-DQB1*06:37', 'HLA-DQA1*05:09-DQB1*06:38', 'HLA-DQA1*05:09-DQB1*06:39', 'HLA-DQA1*05:09-DQB1*06:40',
'HLA-DQA1*05:09-DQB1*06:41', 'HLA-DQA1*05:09-DQB1*06:42', 'HLA-DQA1*05:09-DQB1*06:43', 'HLA-DQA1*05:09-DQB1*06:44', 'HLA-DQA1*05:10-DQB1*02:01',
'HLA-DQA1*05:10-DQB1*02:02', 'HLA-DQA1*05:10-DQB1*02:03', 'HLA-DQA1*05:10-DQB1*02:04', 'HLA-DQA1*05:10-DQB1*02:05', 'HLA-DQA1*05:10-DQB1*02:06',
'HLA-DQA1*05:10-DQB1*03:01', 'HLA-DQA1*05:10-DQB1*03:02', 'HLA-DQA1*05:10-DQB1*03:03', 'HLA-DQA1*05:10-DQB1*03:04', 'HLA-DQA1*05:10-DQB1*03:05',
'HLA-DQA1*05:10-DQB1*03:06', 'HLA-DQA1*05:10-DQB1*03:07', 'HLA-DQA1*05:10-DQB1*03:08', 'HLA-DQA1*05:10-DQB1*03:09', 'HLA-DQA1*05:10-DQB1*03:10',
'HLA-DQA1*05:10-DQB1*03:11', 'HLA-DQA1*05:10-DQB1*03:12', 'HLA-DQA1*05:10-DQB1*03:13', 'HLA-DQA1*05:10-DQB1*03:14', 'HLA-DQA1*05:10-DQB1*03:15',
'HLA-DQA1*05:10-DQB1*03:16', 'HLA-DQA1*05:10-DQB1*03:17', 'HLA-DQA1*05:10-DQB1*03:18', 'HLA-DQA1*05:10-DQB1*03:19', 'HLA-DQA1*05:10-DQB1*03:20',
'HLA-DQA1*05:10-DQB1*03:21', 'HLA-DQA1*05:10-DQB1*03:22', 'HLA-DQA1*05:10-DQB1*03:23', 'HLA-DQA1*05:10-DQB1*03:24', 'HLA-DQA1*05:10-DQB1*03:25',
'HLA-DQA1*05:10-DQB1*03:26', 'HLA-DQA1*05:10-DQB1*03:27', 'HLA-DQA1*05:10-DQB1*03:28', 'HLA-DQA1*05:10-DQB1*03:29', 'HLA-DQA1*05:10-DQB1*03:30',
'HLA-DQA1*05:10-DQB1*03:31', 'HLA-DQA1*05:10-DQB1*03:32', 'HLA-DQA1*05:10-DQB1*03:33', 'HLA-DQA1*05:10-DQB1*03:34', 'HLA-DQA1*05:10-DQB1*03:35',
'HLA-DQA1*05:10-DQB1*03:36', 'HLA-DQA1*05:10-DQB1*03:37', 'HLA-DQA1*05:10-DQB1*03:38', 'HLA-DQA1*05:10-DQB1*04:01', 'HLA-DQA1*05:10-DQB1*04:02',
'HLA-DQA1*05:10-DQB1*04:03', 'HLA-DQA1*05:10-DQB1*04:04', 'HLA-DQA1*05:10-DQB1*04:05', 'HLA-DQA1*05:10-DQB1*04:06', 'HLA-DQA1*05:10-DQB1*04:07',
'HLA-DQA1*05:10-DQB1*04:08', 'HLA-DQA1*05:10-DQB1*05:01', 'HLA-DQA1*05:10-DQB1*05:02', 'HLA-DQA1*05:10-DQB1*05:03', 'HLA-DQA1*05:10-DQB1*05:05',
'HLA-DQA1*05:10-DQB1*05:06', 'HLA-DQA1*05:10-DQB1*05:07', 'HLA-DQA1*05:10-DQB1*05:08', 'HLA-DQA1*05:10-DQB1*05:09', 'HLA-DQA1*05:10-DQB1*05:10',
'HLA-DQA1*05:10-DQB1*05:11', 'HLA-DQA1*05:10-DQB1*05:12', 'HLA-DQA1*05:10-DQB1*05:13', 'HLA-DQA1*05:10-DQB1*05:14', 'HLA-DQA1*05:10-DQB1*06:01',
'HLA-DQA1*05:10-DQB1*06:02', 'HLA-DQA1*05:10-DQB1*06:03', 'HLA-DQA1*05:10-DQB1*06:04', 'HLA-DQA1*05:10-DQB1*06:07', 'HLA-DQA1*05:10-DQB1*06:08',
'HLA-DQA1*05:10-DQB1*06:09', 'HLA-DQA1*05:10-DQB1*06:10', 'HLA-DQA1*05:10-DQB1*06:11', 'HLA-DQA1*05:10-DQB1*06:12', 'HLA-DQA1*05:10-DQB1*06:14',
'HLA-DQA1*05:10-DQB1*06:15', 'HLA-DQA1*05:10-DQB1*06:16', 'HLA-DQA1*05:10-DQB1*06:17', 'HLA-DQA1*05:10-DQB1*06:18', 'HLA-DQA1*05:10-DQB1*06:19',
'HLA-DQA1*05:10-DQB1*06:21', 'HLA-DQA1*05:10-DQB1*06:22', 'HLA-DQA1*05:10-DQB1*06:23', 'HLA-DQA1*05:10-DQB1*06:24', 'HLA-DQA1*05:10-DQB1*06:25',
'HLA-DQA1*05:10-DQB1*06:27', 'HLA-DQA1*05:10-DQB1*06:28', 'HLA-DQA1*05:10-DQB1*06:29', 'HLA-DQA1*05:10-DQB1*06:30', 'HLA-DQA1*05:10-DQB1*06:31',
'HLA-DQA1*05:10-DQB1*06:32', 'HLA-DQA1*05:10-DQB1*06:33', 'HLA-DQA1*05:10-DQB1*06:34', 'HLA-DQA1*05:10-DQB1*06:35', 'HLA-DQA1*05:10-DQB1*06:36',
'HLA-DQA1*05:10-DQB1*06:37', 'HLA-DQA1*05:10-DQB1*06:38', 'HLA-DQA1*05:10-DQB1*06:39', 'HLA-DQA1*05:10-DQB1*06:40', 'HLA-DQA1*05:10-DQB1*06:41',
'HLA-DQA1*05:10-DQB1*06:42', 'HLA-DQA1*05:10-DQB1*06:43', 'HLA-DQA1*05:10-DQB1*06:44', 'HLA-DQA1*05:11-DQB1*02:01', 'HLA-DQA1*05:11-DQB1*02:02',
'HLA-DQA1*05:11-DQB1*02:03', 'HLA-DQA1*05:11-DQB1*02:04', 'HLA-DQA1*05:11-DQB1*02:05', 'HLA-DQA1*05:11-DQB1*02:06', 'HLA-DQA1*05:11-DQB1*03:01',
'HLA-DQA1*05:11-DQB1*03:02', 'HLA-DQA1*05:11-DQB1*03:03', 'HLA-DQA1*05:11-DQB1*03:04', 'HLA-DQA1*05:11-DQB1*03:05', 'HLA-DQA1*05:11-DQB1*03:06',
'HLA-DQA1*05:11-DQB1*03:07', 'HLA-DQA1*05:11-DQB1*03:08', 'HLA-DQA1*05:11-DQB1*03:09', 'HLA-DQA1*05:11-DQB1*03:10', 'HLA-DQA1*05:11-DQB1*03:11',
'HLA-DQA1*05:11-DQB1*03:12', 'HLA-DQA1*05:11-DQB1*03:13', 'HLA-DQA1*05:11-DQB1*03:14', 'HLA-DQA1*05:11-DQB1*03:15', 'HLA-DQA1*05:11-DQB1*03:16',
'HLA-DQA1*05:11-DQB1*03:17', 'HLA-DQA1*05:11-DQB1*03:18', 'HLA-DQA1*05:11-DQB1*03:19', 'HLA-DQA1*05:11-DQB1*03:20', 'HLA-DQA1*05:11-DQB1*03:21',
'HLA-DQA1*05:11-DQB1*03:22', 'HLA-DQA1*05:11-DQB1*03:23', 'HLA-DQA1*05:11-DQB1*03:24', 'HLA-DQA1*05:11-DQB1*03:25', 'HLA-DQA1*05:11-DQB1*03:26',
'HLA-DQA1*05:11-DQB1*03:27', 'HLA-DQA1*05:11-DQB1*03:28', 'HLA-DQA1*05:11-DQB1*03:29', 'HLA-DQA1*05:11-DQB1*03:30', 'HLA-DQA1*05:11-DQB1*03:31',
'HLA-DQA1*05:11-DQB1*03:32', 'HLA-DQA1*05:11-DQB1*03:33', 'HLA-DQA1*05:11-DQB1*03:34', 'HLA-DQA1*05:11-DQB1*03:35', 'HLA-DQA1*05:11-DQB1*03:36',
'HLA-DQA1*05:11-DQB1*03:37', 'HLA-DQA1*05:11-DQB1*03:38', 'HLA-DQA1*05:11-DQB1*04:01', 'HLA-DQA1*05:11-DQB1*04:02', 'HLA-DQA1*05:11-DQB1*04:03',
'HLA-DQA1*05:11-DQB1*04:04', 'HLA-DQA1*05:11-DQB1*04:05', 'HLA-DQA1*05:11-DQB1*04:06', 'HLA-DQA1*05:11-DQB1*04:07', 'HLA-DQA1*05:11-DQB1*04:08',
'HLA-DQA1*05:11-DQB1*05:01', 'HLA-DQA1*05:11-DQB1*05:02', 'HLA-DQA1*05:11-DQB1*05:03', 'HLA-DQA1*05:11-DQB1*05:05', 'HLA-DQA1*05:11-DQB1*05:06',
'HLA-DQA1*05:11-DQB1*05:07', 'HLA-DQA1*05:11-DQB1*05:08', 'HLA-DQA1*05:11-DQB1*05:09', 'HLA-DQA1*05:11-DQB1*05:10', 'HLA-DQA1*05:11-DQB1*05:11',
'HLA-DQA1*05:11-DQB1*05:12', 'HLA-DQA1*05:11-DQB1*05:13', 'HLA-DQA1*05:11-DQB1*05:14', 'HLA-DQA1*05:11-DQB1*06:01', 'HLA-DQA1*05:11-DQB1*06:02',
'HLA-DQA1*05:11-DQB1*06:03', 'HLA-DQA1*05:11-DQB1*06:04', 'HLA-DQA1*05:11-DQB1*06:07', 'HLA-DQA1*05:11-DQB1*06:08', 'HLA-DQA1*05:11-DQB1*06:09',
'HLA-DQA1*05:11-DQB1*06:10', 'HLA-DQA1*05:11-DQB1*06:11', 'HLA-DQA1*05:11-DQB1*06:12', 'HLA-DQA1*05:11-DQB1*06:14', 'HLA-DQA1*05:11-DQB1*06:15',
'HLA-DQA1*05:11-DQB1*06:16', 'HLA-DQA1*05:11-DQB1*06:17', 'HLA-DQA1*05:11-DQB1*06:18', 'HLA-DQA1*05:11-DQB1*06:19', 'HLA-DQA1*05:11-DQB1*06:21',
'HLA-DQA1*05:11-DQB1*06:22', 'HLA-DQA1*05:11-DQB1*06:23', 'HLA-DQA1*05:11-DQB1*06:24', 'HLA-DQA1*05:11-DQB1*06:25', 'HLA-DQA1*05:11-DQB1*06:27',
'HLA-DQA1*05:11-DQB1*06:28', 'HLA-DQA1*05:11-DQB1*06:29', 'HLA-DQA1*05:11-DQB1*06:30', 'HLA-DQA1*05:11-DQB1*06:31', 'HLA-DQA1*05:11-DQB1*06:32',
'HLA-DQA1*05:11-DQB1*06:33', 'HLA-DQA1*05:11-DQB1*06:34', 'HLA-DQA1*05:11-DQB1*06:35', 'HLA-DQA1*05:11-DQB1*06:36', 'HLA-DQA1*05:11-DQB1*06:37',
'HLA-DQA1*05:11-DQB1*06:38', 'HLA-DQA1*05:11-DQB1*06:39', 'HLA-DQA1*05:11-DQB1*06:40', 'HLA-DQA1*05:11-DQB1*06:41', 'HLA-DQA1*05:11-DQB1*06:42',
'HLA-DQA1*05:11-DQB1*06:43', 'HLA-DQA1*05:11-DQB1*06:44', 'HLA-DQA1*06:01-DQB1*02:01', 'HLA-DQA1*06:01-DQB1*02:02', 'HLA-DQA1*06:01-DQB1*02:03',
'HLA-DQA1*06:01-DQB1*02:04', 'HLA-DQA1*06:01-DQB1*02:05', 'HLA-DQA1*06:01-DQB1*02:06', 'HLA-DQA1*06:01-DQB1*03:01', 'HLA-DQA1*06:01-DQB1*03:02',
'HLA-DQA1*06:01-DQB1*03:03', 'HLA-DQA1*06:01-DQB1*03:04', 'HLA-DQA1*06:01-DQB1*03:05', 'HLA-DQA1*06:01-DQB1*03:06', 'HLA-DQA1*06:01-DQB1*03:07',
'HLA-DQA1*06:01-DQB1*03:08', 'HLA-DQA1*06:01-DQB1*03:09', 'HLA-DQA1*06:01-DQB1*03:10', 'HLA-DQA1*06:01-DQB1*03:11', 'HLA-DQA1*06:01-DQB1*03:12',
'HLA-DQA1*06:01-DQB1*03:13', 'HLA-DQA1*06:01-DQB1*03:14', 'HLA-DQA1*06:01-DQB1*03:15', 'HLA-DQA1*06:01-DQB1*03:16', 'HLA-DQA1*06:01-DQB1*03:17',
'HLA-DQA1*06:01-DQB1*03:18', 'HLA-DQA1*06:01-DQB1*03:19', 'HLA-DQA1*06:01-DQB1*03:20', 'HLA-DQA1*06:01-DQB1*03:21', 'HLA-DQA1*06:01-DQB1*03:22',
'HLA-DQA1*06:01-DQB1*03:23', 'HLA-DQA1*06:01-DQB1*03:24', 'HLA-DQA1*06:01-DQB1*03:25', 'HLA-DQA1*06:01-DQB1*03:26', 'HLA-DQA1*06:01-DQB1*03:27',
'HLA-DQA1*06:01-DQB1*03:28', 'HLA-DQA1*06:01-DQB1*03:29', 'HLA-DQA1*06:01-DQB1*03:30', 'HLA-DQA1*06:01-DQB1*03:31', 'HLA-DQA1*06:01-DQB1*03:32',
'HLA-DQA1*06:01-DQB1*03:33', 'HLA-DQA1*06:01-DQB1*03:34', 'HLA-DQA1*06:01-DQB1*03:35', 'HLA-DQA1*06:01-DQB1*03:36', 'HLA-DQA1*06:01-DQB1*03:37',
'HLA-DQA1*06:01-DQB1*03:38', 'HLA-DQA1*06:01-DQB1*04:01', 'HLA-DQA1*06:01-DQB1*04:02', 'HLA-DQA1*06:01-DQB1*04:03', 'HLA-DQA1*06:01-DQB1*04:04',
'HLA-DQA1*06:01-DQB1*04:05', 'HLA-DQA1*06:01-DQB1*04:06', 'HLA-DQA1*06:01-DQB1*04:07', 'HLA-DQA1*06:01-DQB1*04:08', 'HLA-DQA1*06:01-DQB1*05:01',
'HLA-DQA1*06:01-DQB1*05:02', 'HLA-DQA1*06:01-DQB1*05:03', 'HLA-DQA1*06:01-DQB1*05:05', 'HLA-DQA1*06:01-DQB1*05:06', 'HLA-DQA1*06:01-DQB1*05:07',
'HLA-DQA1*06:01-DQB1*05:08', 'HLA-DQA1*06:01-DQB1*05:09', 'HLA-DQA1*06:01-DQB1*05:10', 'HLA-DQA1*06:01-DQB1*05:11', 'HLA-DQA1*06:01-DQB1*05:12',
'HLA-DQA1*06:01-DQB1*05:13', 'HLA-DQA1*06:01-DQB1*05:14', 'HLA-DQA1*06:01-DQB1*06:01', 'HLA-DQA1*06:01-DQB1*06:02', 'HLA-DQA1*06:01-DQB1*06:03',
'HLA-DQA1*06:01-DQB1*06:04', 'HLA-DQA1*06:01-DQB1*06:07', 'HLA-DQA1*06:01-DQB1*06:08', 'HLA-DQA1*06:01-DQB1*06:09', 'HLA-DQA1*06:01-DQB1*06:10',
'HLA-DQA1*06:01-DQB1*06:11', 'HLA-DQA1*06:01-DQB1*06:12', 'HLA-DQA1*06:01-DQB1*06:14', 'HLA-DQA1*06:01-DQB1*06:15', 'HLA-DQA1*06:01-DQB1*06:16',
'HLA-DQA1*06:01-DQB1*06:17', 'HLA-DQA1*06:01-DQB1*06:18', 'HLA-DQA1*06:01-DQB1*06:19', 'HLA-DQA1*06:01-DQB1*06:21', 'HLA-DQA1*06:01-DQB1*06:22',
'HLA-DQA1*06:01-DQB1*06:23', 'HLA-DQA1*06:01-DQB1*06:24', 'HLA-DQA1*06:01-DQB1*06:25', 'HLA-DQA1*06:01-DQB1*06:27', 'HLA-DQA1*06:01-DQB1*06:28',
'HLA-DQA1*06:01-DQB1*06:29', 'HLA-DQA1*06:01-DQB1*06:30', 'HLA-DQA1*06:01-DQB1*06:31', 'HLA-DQA1*06:01-DQB1*06:32', 'HLA-DQA1*06:01-DQB1*06:33',
'HLA-DQA1*06:01-DQB1*06:34', 'HLA-DQA1*06:01-DQB1*06:35', 'HLA-DQA1*06:01-DQB1*06:36', 'HLA-DQA1*06:01-DQB1*06:37', 'HLA-DQA1*06:01-DQB1*06:38',
'HLA-DQA1*06:01-DQB1*06:39', 'HLA-DQA1*06:01-DQB1*06:40', 'HLA-DQA1*06:01-DQB1*06:41', 'HLA-DQA1*06:01-DQB1*06:42', 'HLA-DQA1*06:01-DQB1*06:43',
'HLA-DQA1*06:01-DQB1*06:44', 'HLA-DQA1*06:02-DQB1*02:01', 'HLA-DQA1*06:02-DQB1*02:02', 'HLA-DQA1*06:02-DQB1*02:03', 'HLA-DQA1*06:02-DQB1*02:04',
'HLA-DQA1*06:02-DQB1*02:05', 'HLA-DQA1*06:02-DQB1*02:06', 'HLA-DQA1*06:02-DQB1*03:01', 'HLA-DQA1*06:02-DQB1*03:02', 'HLA-DQA1*06:02-DQB1*03:03',
'HLA-DQA1*06:02-DQB1*03:04', 'HLA-DQA1*06:02-DQB1*03:05', 'HLA-DQA1*06:02-DQB1*03:06', 'HLA-DQA1*06:02-DQB1*03:07', 'HLA-DQA1*06:02-DQB1*03:08',
'HLA-DQA1*06:02-DQB1*03:09', 'HLA-DQA1*06:02-DQB1*03:10', 'HLA-DQA1*06:02-DQB1*03:11', 'HLA-DQA1*06:02-DQB1*03:12', 'HLA-DQA1*06:02-DQB1*03:13',
'HLA-DQA1*06:02-DQB1*03:14', 'HLA-DQA1*06:02-DQB1*03:15', 'HLA-DQA1*06:02-DQB1*03:16', 'HLA-DQA1*06:02-DQB1*03:17', 'HLA-DQA1*06:02-DQB1*03:18',
'HLA-DQA1*06:02-DQB1*03:19', 'HLA-DQA1*06:02-DQB1*03:20', 'HLA-DQA1*06:02-DQB1*03:21', 'HLA-DQA1*06:02-DQB1*03:22', 'HLA-DQA1*06:02-DQB1*03:23',
'HLA-DQA1*06:02-DQB1*03:24', 'HLA-DQA1*06:02-DQB1*03:25', 'HLA-DQA1*06:02-DQB1*03:26', 'HLA-DQA1*06:02-DQB1*03:27', 'HLA-DQA1*06:02-DQB1*03:28',
'HLA-DQA1*06:02-DQB1*03:29', 'HLA-DQA1*06:02-DQB1*03:30', 'HLA-DQA1*06:02-DQB1*03:31', 'HLA-DQA1*06:02-DQB1*03:32', 'HLA-DQA1*06:02-DQB1*03:33',
'HLA-DQA1*06:02-DQB1*03:34', 'HLA-DQA1*06:02-DQB1*03:35', 'HLA-DQA1*06:02-DQB1*03:36', 'HLA-DQA1*06:02-DQB1*03:37', 'HLA-DQA1*06:02-DQB1*03:38',
'HLA-DQA1*06:02-DQB1*04:01', 'HLA-DQA1*06:02-DQB1*04:02', 'HLA-DQA1*06:02-DQB1*04:03', 'HLA-DQA1*06:02-DQB1*04:04', 'HLA-DQA1*06:02-DQB1*04:05',
'HLA-DQA1*06:02-DQB1*04:06', 'HLA-DQA1*06:02-DQB1*04:07', 'HLA-DQA1*06:02-DQB1*04:08', 'HLA-DQA1*06:02-DQB1*05:01', 'HLA-DQA1*06:02-DQB1*05:02',
'HLA-DQA1*06:02-DQB1*05:03', 'HLA-DQA1*06:02-DQB1*05:05', 'HLA-DQA1*06:02-DQB1*05:06', 'HLA-DQA1*06:02-DQB1*05:07', 'HLA-DQA1*06:02-DQB1*05:08',
'HLA-DQA1*06:02-DQB1*05:09', 'HLA-DQA1*06:02-DQB1*05:10', 'HLA-DQA1*06:02-DQB1*05:11', 'HLA-DQA1*06:02-DQB1*05:12', 'HLA-DQA1*06:02-DQB1*05:13',
'HLA-DQA1*06:02-DQB1*05:14', 'HLA-DQA1*06:02-DQB1*06:01', 'HLA-DQA1*06:02-DQB1*06:02', 'HLA-DQA1*06:02-DQB1*06:03', 'HLA-DQA1*06:02-DQB1*06:04',
'HLA-DQA1*06:02-DQB1*06:07', 'HLA-DQA1*06:02-DQB1*06:08', 'HLA-DQA1*06:02-DQB1*06:09', 'HLA-DQA1*06:02-DQB1*06:10', 'HLA-DQA1*06:02-DQB1*06:11',
'HLA-DQA1*06:02-DQB1*06:12', 'HLA-DQA1*06:02-DQB1*06:14', 'HLA-DQA1*06:02-DQB1*06:15', 'HLA-DQA1*06:02-DQB1*06:16', 'HLA-DQA1*06:02-DQB1*06:17',
'HLA-DQA1*06:02-DQB1*06:18', 'HLA-DQA1*06:02-DQB1*06:19', 'HLA-DQA1*06:02-DQB1*06:21', 'HLA-DQA1*06:02-DQB1*06:22', 'HLA-DQA1*06:02-DQB1*06:23',
'HLA-DQA1*06:02-DQB1*06:24', 'HLA-DQA1*06:02-DQB1*06:25', 'HLA-DQA1*06:02-DQB1*06:27', 'HLA-DQA1*06:02-DQB1*06:28', 'HLA-DQA1*06:02-DQB1*06:29',
'HLA-DQA1*06:02-DQB1*06:30', 'HLA-DQA1*06:02-DQB1*06:31', 'HLA-DQA1*06:02-DQB1*06:32', 'HLA-DQA1*06:02-DQB1*06:33', 'HLA-DQA1*06:02-DQB1*06:34',
'HLA-DQA1*06:02-DQB1*06:35', 'HLA-DQA1*06:02-DQB1*06:36', 'HLA-DQA1*06:02-DQB1*06:37', 'HLA-DQA1*06:02-DQB1*06:38', 'HLA-DQA1*06:02-DQB1*06:39',
'HLA-DQA1*06:02-DQB1*06:40', 'HLA-DQA1*06:02-DQB1*06:41', 'HLA-DQA1*06:02-DQB1*06:42', 'HLA-DQA1*06:02-DQB1*06:43', 'HLA-DQA1*06:02-DQB1*06:44',
'H-2-Iab', 'H-2-Iad', 'H-2-Iak', 'H-2-Iaq', 'H-2-Ias',
'H-2-Iau', 'H-2-Iad', 'H-2-Iak'])

    __version = "4.0"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """A list of valid :class:`~epytope.Core.Allele.Allele` models"""
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        f = csv.reader(open(file, "r"), delimiter='\t')
        scores = defaultdict(defaultdict)
        ranks = defaultdict(defaultdict)
        alleles = [x for x in set([x for x in next(f) if x != ""])]
        next(f)
        for row in f:
            pep_seq = row[PeptideIndex.NETMHCIIPAN_4_0]
            for i, a in enumerate(alleles):
                scores[a][pep_seq] = float(row[ScoreIndex.NETMHCIIPAN_4_0 + i * Offset.NETMHCIIPAN_4_0])
                ranks[a][pep_seq] = float(row[RankIndex.NETMHCIIPAN_4_0 + i * Offset.NETMHCIIPAN_4_0])
                # Create dictionary with hierarchy: {'Allele1': {'Score': {'Pep1': Score1, 'Pep2': Score2,..}, 'Rank': {'Pep1': RankScore1, 'Pep2': RankScore2,..}}, 'Allele2':...}
        result = {allele: {metric:(list(scores.values())[j] if metric == "Score" else list(ranks.values())[j]) for metric in ["Score", "Rank"]} for j, allele in enumerate(alleles)}

        return result

class PickPocket_1_1(AExternalEpitopePrediction):
    """
    Implementation of PickPocket adapter.

    .. note::

        Zhang, H., Lund, O., & Nielsen, M. (2009). The PickPocket method for predicting binding specificities
        for receptors based on receptor pocket similarities: application to MHC-peptide binding.
        Bioinformatics, 25(10), 1293-1299.

    """
    __name = "pickpocket"
    __supported_length = frozenset([8, 9, 10, 11])
    __command = 'PickPocket -p {peptides} -a {alleles} {options} | grep -v "#" > {out}'
    __supported_alleles = frozenset(['HLA-A*01:01', 'HLA-A*01:02', 'HLA-A*01:03', 'HLA-A*01:06', 'HLA-A*01:07', 'HLA-A*01:08', 'HLA-A*01:09',
                                     'HLA-A*01:10', 'HLA-A*01:12', 'HLA-A*01:13', 'HLA-A*01:14', 'HLA-A*01:17', 'HLA-A*01:19', 'HLA-A*01:20',
                                     'HLA-A*01:21', 'HLA-A*01:23', 'HLA-A*01:24',
                                     'HLA-A*01:25', 'HLA-A*01:26', 'HLA-A*01:28', 'HLA-A*01:29', 'HLA-A*01:30', 'HLA-A*01:32', 'HLA-A*01:33',
                                     'HLA-A*01:35', 'HLA-A*01:36', 'HLA-A*01:37',
                                     'HLA-A*01:38', 'HLA-A*01:39', 'HLA-A*01:40', 'HLA-A*01:41', 'HLA-A*01:42', 'HLA-A*01:43', 'HLA-A*01:44',
                                     'HLA-A*01:45', 'HLA-A*01:46', 'HLA-A*01:47',
                                     'HLA-A*01:48', 'HLA-A*01:49', 'HLA-A*01:50', 'HLA-A*01:51', 'HLA-A*01:54', 'HLA-A*01:55', 'HLA-A*01:58',
                                     'HLA-A*01:59', 'HLA-A*01:60', 'HLA-A*01:61',
                                     'HLA-A*01:62', 'HLA-A*01:63', 'HLA-A*01:64', 'HLA-A*01:65', 'HLA-A*01:66', 'HLA-A*02:01', 'HLA-A*02:02',
                                     'HLA-A*02:03', 'HLA-A*02:04', 'HLA-A*02:05',
                                     'HLA-A*02:06', 'HLA-A*02:07', 'HLA-A*02:08', 'HLA-A*02:09', 'HLA-A*02:10', 'HLA-A*02:11', 'HLA-A*02:12',
                                     'HLA-A*02:13', 'HLA-A*02:14', 'HLA-A*02:16',
                                     'HLA-A*02:17', 'HLA-A*02:18', 'HLA-A*02:19', 'HLA-A*02:20', 'HLA-A*02:21', 'HLA-A*02:22', 'HLA-A*02:24',
                                     'HLA-A*02:25', 'HLA-A*02:26', 'HLA-A*02:27',
                                     'HLA-A*02:28', 'HLA-A*02:29', 'HLA-A*02:30', 'HLA-A*02:31', 'HLA-A*02:33', 'HLA-A*02:34', 'HLA-A*02:35',
                                     'HLA-A*02:36', 'HLA-A*02:37', 'HLA-A*02:38',
                                     'HLA-A*02:39', 'HLA-A*02:40', 'HLA-A*02:41', 'HLA-A*02:42', 'HLA-A*02:44', 'HLA-A*02:45', 'HLA-A*02:46',
                                     'HLA-A*02:47', 'HLA-A*02:48', 'HLA-A*02:49',
                                     'HLA-A*02:50', 'HLA-A*02:51', 'HLA-A*02:52', 'HLA-A*02:54', 'HLA-A*02:55', 'HLA-A*02:56', 'HLA-A*02:57',
                                     'HLA-A*02:58', 'HLA-A*02:59', 'HLA-A*02:60',
                                     'HLA-A*02:61', 'HLA-A*02:62', 'HLA-A*02:63', 'HLA-A*02:64', 'HLA-A*02:65', 'HLA-A*02:66', 'HLA-A*02:67',
                                     'HLA-A*02:68', 'HLA-A*02:69', 'HLA-A*02:70',
                                     'HLA-A*02:71', 'HLA-A*02:72', 'HLA-A*02:73', 'HLA-A*02:74', 'HLA-A*02:75', 'HLA-A*02:76', 'HLA-A*02:77',
                                     'HLA-A*02:78', 'HLA-A*02:79', 'HLA-A*02:80',
                                     'HLA-A*02:81', 'HLA-A*02:84', 'HLA-A*02:85', 'HLA-A*02:86', 'HLA-A*02:87', 'HLA-A*02:89', 'HLA-A*02:90',
                                     'HLA-A*02:91', 'HLA-A*02:92', 'HLA-A*02:93',
                                     'HLA-A*02:95', 'HLA-A*02:96', 'HLA-A*02:97', 'HLA-A*02:99', 'HLA-A*02:101', 'HLA-A*02:102', 'HLA-A*02:103',
                                     'HLA-A*02:104', 'HLA-A*02:105',
                                     'HLA-A*02:106', 'HLA-A*02:107', 'HLA-A*02:108', 'HLA-A*02:109', 'HLA-A*02:110', 'HLA-A*02:111', 'HLA-A*02:112',
                                     'HLA-A*02:114', 'HLA-A*02:115',
                                     'HLA-A*02:116', 'HLA-A*02:117', 'HLA-A*02:118', 'HLA-A*02:119', 'HLA-A*02:120', 'HLA-A*02:121', 'HLA-A*02:122',
                                     'HLA-A*02:123', 'HLA-A*02:124',
                                     'HLA-A*02:126', 'HLA-A*02:127', 'HLA-A*02:128', 'HLA-A*02:129', 'HLA-A*02:130', 'HLA-A*02:131', 'HLA-A*02:132',
                                     'HLA-A*02:133', 'HLA-A*02:134',
                                     'HLA-A*02:135', 'HLA-A*02:136', 'HLA-A*02:137', 'HLA-A*02:138', 'HLA-A*02:139', 'HLA-A*02:140', 'HLA-A*02:141',
                                     'HLA-A*02:142', 'HLA-A*02:143',
                                     'HLA-A*02:144', 'HLA-A*02:145', 'HLA-A*02:146', 'HLA-A*02:147', 'HLA-A*02:148', 'HLA-A*02:149', 'HLA-A*02:150',
                                     'HLA-A*02:151', 'HLA-A*02:152',
                                     'HLA-A*02:153', 'HLA-A*02:154', 'HLA-A*02:155', 'HLA-A*02:156', 'HLA-A*02:157', 'HLA-A*02:158', 'HLA-A*02:159',
                                     'HLA-A*02:160', 'HLA-A*02:161',
                                     'HLA-A*02:162', 'HLA-A*02:163', 'HLA-A*02:164', 'HLA-A*02:165', 'HLA-A*02:166', 'HLA-A*02:167', 'HLA-A*02:168',
                                     'HLA-A*02:169', 'HLA-A*02:170',
                                     'HLA-A*02:171', 'HLA-A*02:172', 'HLA-A*02:173', 'HLA-A*02:174', 'HLA-A*02:175', 'HLA-A*02:176', 'HLA-A*02:177',
                                     'HLA-A*02:178', 'HLA-A*02:179',
                                     'HLA-A*02:180', 'HLA-A*02:181', 'HLA-A*02:182', 'HLA-A*02:183', 'HLA-A*02:184', 'HLA-A*02:185', 'HLA-A*02:186',
                                     'HLA-A*02:187', 'HLA-A*02:188',
                                     'HLA-A*02:189', 'HLA-A*02:190', 'HLA-A*02:191', 'HLA-A*02:192', 'HLA-A*02:193', 'HLA-A*02:194', 'HLA-A*02:195',
                                     'HLA-A*02:196', 'HLA-A*02:197',
                                     'HLA-A*02:198', 'HLA-A*02:199', 'HLA-A*02:200', 'HLA-A*02:201', 'HLA-A*02:202', 'HLA-A*02:203', 'HLA-A*02:204',
                                     'HLA-A*02:205', 'HLA-A*02:206',
                                     'HLA-A*02:207', 'HLA-A*02:208', 'HLA-A*02:209', 'HLA-A*02:210', 'HLA-A*02:211', 'HLA-A*02:212', 'HLA-A*02:213',
                                     'HLA-A*02:214', 'HLA-A*02:215',
                                     'HLA-A*02:216', 'HLA-A*02:217', 'HLA-A*02:218', 'HLA-A*02:219', 'HLA-A*02:220', 'HLA-A*02:221', 'HLA-A*02:224',
                                     'HLA-A*02:228', 'HLA-A*02:229',
                                     'HLA-A*02:230', 'HLA-A*02:231', 'HLA-A*02:232', 'HLA-A*02:233', 'HLA-A*02:234', 'HLA-A*02:235', 'HLA-A*02:236',
                                     'HLA-A*02:237', 'HLA-A*02:238',
                                     'HLA-A*02:239', 'HLA-A*02:240', 'HLA-A*02:241', 'HLA-A*02:242', 'HLA-A*02:243', 'HLA-A*02:244', 'HLA-A*02:245',
                                     'HLA-A*02:246', 'HLA-A*02:247',
                                     'HLA-A*02:248', 'HLA-A*02:249', 'HLA-A*02:251', 'HLA-A*02:252', 'HLA-A*02:253', 'HLA-A*02:254', 'HLA-A*02:255',
                                     'HLA-A*02:256', 'HLA-A*02:257',
                                     'HLA-A*02:258', 'HLA-A*02:259', 'HLA-A*02:260', 'HLA-A*02:261', 'HLA-A*02:262', 'HLA-A*02:263', 'HLA-A*02:264',
                                     'HLA-A*02:265', 'HLA-A*02:266',
                                     'HLA-A*03:01', 'HLA-A*03:02', 'HLA-A*03:04', 'HLA-A*03:05', 'HLA-A*03:06', 'HLA-A*03:07', 'HLA-A*03:08',
                                     'HLA-A*03:09', 'HLA-A*03:10', 'HLA-A*03:12',
                                     'HLA-A*03:13', 'HLA-A*03:14', 'HLA-A*03:15', 'HLA-A*03:16', 'HLA-A*03:17', 'HLA-A*03:18', 'HLA-A*03:19',
                                     'HLA-A*03:20', 'HLA-A*03:22', 'HLA-A*03:23',
                                     'HLA-A*03:24', 'HLA-A*03:25', 'HLA-A*03:26', 'HLA-A*03:27', 'HLA-A*03:28', 'HLA-A*03:29', 'HLA-A*03:30',
                                     'HLA-A*03:31', 'HLA-A*03:32', 'HLA-A*03:33',
                                     'HLA-A*03:34', 'HLA-A*03:35', 'HLA-A*03:37', 'HLA-A*03:38', 'HLA-A*03:39', 'HLA-A*03:40', 'HLA-A*03:41',
                                     'HLA-A*03:42', 'HLA-A*03:43', 'HLA-A*03:44',
                                     'HLA-A*03:45', 'HLA-A*03:46', 'HLA-A*03:47', 'HLA-A*03:48', 'HLA-A*03:49', 'HLA-A*03:50', 'HLA-A*03:51',
                                     'HLA-A*03:52', 'HLA-A*03:53', 'HLA-A*03:54',
                                     'HLA-A*03:55', 'HLA-A*03:56', 'HLA-A*03:57', 'HLA-A*03:58', 'HLA-A*03:59', 'HLA-A*03:60', 'HLA-A*03:61',
                                     'HLA-A*03:62', 'HLA-A*03:63', 'HLA-A*03:64',
                                     'HLA-A*03:65', 'HLA-A*03:66', 'HLA-A*03:67', 'HLA-A*03:70', 'HLA-A*03:71', 'HLA-A*03:72', 'HLA-A*03:73',
                                     'HLA-A*03:74', 'HLA-A*03:75', 'HLA-A*03:76',
                                     'HLA-A*03:77', 'HLA-A*03:78', 'HLA-A*03:79', 'HLA-A*03:80', 'HLA-A*03:81', 'HLA-A*03:82', 'HLA-A*11:01',
                                     'HLA-A*11:02', 'HLA-A*11:03', 'HLA-A*11:04',
                                     'HLA-A*11:05', 'HLA-A*11:06', 'HLA-A*11:07', 'HLA-A*11:08', 'HLA-A*11:09', 'HLA-A*11:10', 'HLA-A*11:11',
                                     'HLA-A*11:12', 'HLA-A*11:13', 'HLA-A*11:14',
                                     'HLA-A*11:15', 'HLA-A*11:16', 'HLA-A*11:17', 'HLA-A*11:18', 'HLA-A*11:19', 'HLA-A*11:20', 'HLA-A*11:22',
                                     'HLA-A*11:23', 'HLA-A*11:24', 'HLA-A*11:25',
                                     'HLA-A*11:26', 'HLA-A*11:27', 'HLA-A*11:29', 'HLA-A*11:30', 'HLA-A*11:31', 'HLA-A*11:32', 'HLA-A*11:33',
                                     'HLA-A*11:34', 'HLA-A*11:35', 'HLA-A*11:36',
                                     'HLA-A*11:37', 'HLA-A*11:38', 'HLA-A*11:39', 'HLA-A*11:40', 'HLA-A*11:41', 'HLA-A*11:42', 'HLA-A*11:43',
                                     'HLA-A*11:44', 'HLA-A*11:45', 'HLA-A*11:46',
                                     'HLA-A*11:47', 'HLA-A*11:48', 'HLA-A*11:49', 'HLA-A*11:51', 'HLA-A*11:53', 'HLA-A*11:54', 'HLA-A*11:55',
                                     'HLA-A*11:56', 'HLA-A*11:57', 'HLA-A*11:58',
                                     'HLA-A*11:59', 'HLA-A*11:60', 'HLA-A*11:61', 'HLA-A*11:62', 'HLA-A*11:63', 'HLA-A*11:64', 'HLA-A*23:01',
                                     'HLA-A*23:02', 'HLA-A*23:03', 'HLA-A*23:04',
                                     'HLA-A*23:05', 'HLA-A*23:06', 'HLA-A*23:09', 'HLA-A*23:10', 'HLA-A*23:12', 'HLA-A*23:13', 'HLA-A*23:14',
                                     'HLA-A*23:15', 'HLA-A*23:16', 'HLA-A*23:17',
                                     'HLA-A*23:18', 'HLA-A*23:20', 'HLA-A*23:21', 'HLA-A*23:22', 'HLA-A*23:23', 'HLA-A*23:24', 'HLA-A*23:25',
                                     'HLA-A*23:26', 'HLA-A*24:02', 'HLA-A*24:03',
                                     'HLA-A*24:04', 'HLA-A*24:05', 'HLA-A*24:06', 'HLA-A*24:07', 'HLA-A*24:08', 'HLA-A*24:10', 'HLA-A*24:13',
                                     'HLA-A*24:14', 'HLA-A*24:15', 'HLA-A*24:17',
                                     'HLA-A*24:18', 'HLA-A*24:19', 'HLA-A*24:20', 'HLA-A*24:21', 'HLA-A*24:22', 'HLA-A*24:23', 'HLA-A*24:24',
                                     'HLA-A*24:25', 'HLA-A*24:26', 'HLA-A*24:27',
                                     'HLA-A*24:28', 'HLA-A*24:29', 'HLA-A*24:30', 'HLA-A*24:31', 'HLA-A*24:32', 'HLA-A*24:33', 'HLA-A*24:34',
                                     'HLA-A*24:35', 'HLA-A*24:37', 'HLA-A*24:38',
                                     'HLA-A*24:39', 'HLA-A*24:41', 'HLA-A*24:42', 'HLA-A*24:43', 'HLA-A*24:44', 'HLA-A*24:46', 'HLA-A*24:47',
                                     'HLA-A*24:49', 'HLA-A*24:50', 'HLA-A*24:51',
                                     'HLA-A*24:52', 'HLA-A*24:53', 'HLA-A*24:54', 'HLA-A*24:55', 'HLA-A*24:56', 'HLA-A*24:57', 'HLA-A*24:58',
                                     'HLA-A*24:59', 'HLA-A*24:61', 'HLA-A*24:62',
                                     'HLA-A*24:63', 'HLA-A*24:64', 'HLA-A*24:66', 'HLA-A*24:67', 'HLA-A*24:68', 'HLA-A*24:69', 'HLA-A*24:70',
                                     'HLA-A*24:71', 'HLA-A*24:72', 'HLA-A*24:73',
                                     'HLA-A*24:74', 'HLA-A*24:75', 'HLA-A*24:76', 'HLA-A*24:77', 'HLA-A*24:78', 'HLA-A*24:79', 'HLA-A*24:80',
                                     'HLA-A*24:81', 'HLA-A*24:82', 'HLA-A*24:85',
                                     'HLA-A*24:87', 'HLA-A*24:88', 'HLA-A*24:89', 'HLA-A*24:91', 'HLA-A*24:92', 'HLA-A*24:93', 'HLA-A*24:94',
                                     'HLA-A*24:95', 'HLA-A*24:96', 'HLA-A*24:97',
                                     'HLA-A*24:98', 'HLA-A*24:99', 'HLA-A*24:100', 'HLA-A*24:101', 'HLA-A*24:102', 'HLA-A*24:103', 'HLA-A*24:104',
                                     'HLA-A*24:105', 'HLA-A*24:106',
                                     'HLA-A*24:107', 'HLA-A*24:108', 'HLA-A*24:109', 'HLA-A*24:110', 'HLA-A*24:111', 'HLA-A*24:112', 'HLA-A*24:113',
                                     'HLA-A*24:114', 'HLA-A*24:115',
                                     'HLA-A*24:116', 'HLA-A*24:117', 'HLA-A*24:118', 'HLA-A*24:119', 'HLA-A*24:120', 'HLA-A*24:121', 'HLA-A*24:122',
                                     'HLA-A*24:123', 'HLA-A*24:124',
                                     'HLA-A*24:125', 'HLA-A*24:126', 'HLA-A*24:127', 'HLA-A*24:128', 'HLA-A*24:129', 'HLA-A*24:130', 'HLA-A*24:131',
                                     'HLA-A*24:133', 'HLA-A*24:134',
                                     'HLA-A*24:135', 'HLA-A*24:136', 'HLA-A*24:137', 'HLA-A*24:138', 'HLA-A*24:139', 'HLA-A*24:140', 'HLA-A*24:141',
                                     'HLA-A*24:142', 'HLA-A*24:143',
                                     'HLA-A*24:144', 'HLA-A*25:01', 'HLA-A*25:02', 'HLA-A*25:03', 'HLA-A*25:04', 'HLA-A*25:05', 'HLA-A*25:06',
                                     'HLA-A*25:07', 'HLA-A*25:08', 'HLA-A*25:09',
                                     'HLA-A*25:10', 'HLA-A*25:11', 'HLA-A*25:13', 'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*26:04',
                                     'HLA-A*26:05', 'HLA-A*26:06', 'HLA-A*26:07',
                                     'HLA-A*26:08', 'HLA-A*26:09', 'HLA-A*26:10', 'HLA-A*26:12', 'HLA-A*26:13', 'HLA-A*26:14', 'HLA-A*26:15',
                                     'HLA-A*26:16', 'HLA-A*26:17', 'HLA-A*26:18',
                                     'HLA-A*26:19', 'HLA-A*26:20', 'HLA-A*26:21', 'HLA-A*26:22', 'HLA-A*26:23', 'HLA-A*26:24', 'HLA-A*26:26',
                                     'HLA-A*26:27', 'HLA-A*26:28', 'HLA-A*26:29',
                                     'HLA-A*26:30', 'HLA-A*26:31', 'HLA-A*26:32', 'HLA-A*26:33', 'HLA-A*26:34', 'HLA-A*26:35', 'HLA-A*26:36',
                                     'HLA-A*26:37', 'HLA-A*26:38', 'HLA-A*26:39',
                                     'HLA-A*26:40', 'HLA-A*26:41', 'HLA-A*26:42', 'HLA-A*26:43', 'HLA-A*26:45', 'HLA-A*26:46', 'HLA-A*26:47',
                                     'HLA-A*26:48', 'HLA-A*26:49', 'HLA-A*26:50',
                                     'HLA-A*29:01', 'HLA-A*29:02', 'HLA-A*29:03', 'HLA-A*29:04', 'HLA-A*29:05', 'HLA-A*29:06', 'HLA-A*29:07',
                                     'HLA-A*29:09', 'HLA-A*29:10', 'HLA-A*29:11',
                                     'HLA-A*29:12', 'HLA-A*29:13', 'HLA-A*29:14', 'HLA-A*29:15', 'HLA-A*29:16', 'HLA-A*29:17', 'HLA-A*29:18',
                                     'HLA-A*29:19', 'HLA-A*29:20', 'HLA-A*29:21',
                                     'HLA-A*29:22', 'HLA-A*30:01', 'HLA-A*30:02', 'HLA-A*30:03', 'HLA-A*30:04', 'HLA-A*30:06', 'HLA-A*30:07',
                                     'HLA-A*30:08', 'HLA-A*30:09', 'HLA-A*30:10',
                                     'HLA-A*30:11', 'HLA-A*30:12', 'HLA-A*30:13', 'HLA-A*30:15', 'HLA-A*30:16', 'HLA-A*30:17', 'HLA-A*30:18',
                                     'HLA-A*30:19', 'HLA-A*30:20', 'HLA-A*30:22',
                                     'HLA-A*30:23', 'HLA-A*30:24', 'HLA-A*30:25', 'HLA-A*30:26', 'HLA-A*30:28', 'HLA-A*30:29', 'HLA-A*30:30',
                                     'HLA-A*30:31', 'HLA-A*30:32', 'HLA-A*30:33',
                                     'HLA-A*30:34', 'HLA-A*30:35', 'HLA-A*30:36', 'HLA-A*30:37', 'HLA-A*30:38', 'HLA-A*30:39', 'HLA-A*30:40',
                                     'HLA-A*30:41', 'HLA-A*31:01', 'HLA-A*31:02',
                                     'HLA-A*31:03', 'HLA-A*31:04', 'HLA-A*31:05', 'HLA-A*31:06', 'HLA-A*31:07', 'HLA-A*31:08', 'HLA-A*31:09',
                                     'HLA-A*31:10', 'HLA-A*31:11', 'HLA-A*31:12',
                                     'HLA-A*31:13', 'HLA-A*31:15', 'HLA-A*31:16', 'HLA-A*31:17', 'HLA-A*31:18', 'HLA-A*31:19', 'HLA-A*31:20',
                                     'HLA-A*31:21', 'HLA-A*31:22', 'HLA-A*31:23',
                                     'HLA-A*31:24', 'HLA-A*31:25', 'HLA-A*31:26', 'HLA-A*31:27', 'HLA-A*31:28', 'HLA-A*31:29', 'HLA-A*31:30',
                                     'HLA-A*31:31', 'HLA-A*31:32', 'HLA-A*31:33',
                                     'HLA-A*31:34', 'HLA-A*31:35', 'HLA-A*31:36', 'HLA-A*31:37', 'HLA-A*32:01', 'HLA-A*32:02', 'HLA-A*32:03',
                                     'HLA-A*32:04', 'HLA-A*32:05', 'HLA-A*32:06',
                                     'HLA-A*32:07', 'HLA-A*32:08', 'HLA-A*32:09', 'HLA-A*32:10', 'HLA-A*32:12', 'HLA-A*32:13', 'HLA-A*32:14',
                                     'HLA-A*32:15', 'HLA-A*32:16', 'HLA-A*32:17',
                                     'HLA-A*32:18', 'HLA-A*32:20', 'HLA-A*32:21', 'HLA-A*32:22', 'HLA-A*32:23', 'HLA-A*32:24', 'HLA-A*32:25',
                                     'HLA-A*33:01', 'HLA-A*33:03', 'HLA-A*33:04',
                                     'HLA-A*33:05', 'HLA-A*33:06', 'HLA-A*33:07', 'HLA-A*33:08', 'HLA-A*33:09', 'HLA-A*33:10', 'HLA-A*33:11',
                                     'HLA-A*33:12', 'HLA-A*33:13', 'HLA-A*33:14',
                                     'HLA-A*33:15', 'HLA-A*33:16', 'HLA-A*33:17', 'HLA-A*33:18', 'HLA-A*33:19', 'HLA-A*33:20', 'HLA-A*33:21',
                                     'HLA-A*33:22', 'HLA-A*33:23', 'HLA-A*33:24',
                                     'HLA-A*33:25', 'HLA-A*33:26', 'HLA-A*33:27', 'HLA-A*33:28', 'HLA-A*33:29', 'HLA-A*33:30', 'HLA-A*33:31',
                                     'HLA-A*34:01', 'HLA-A*34:02', 'HLA-A*34:03',
                                     'HLA-A*34:04', 'HLA-A*34:05', 'HLA-A*34:06', 'HLA-A*34:07', 'HLA-A*34:08', 'HLA-A*36:01', 'HLA-A*36:02',
                                     'HLA-A*36:03', 'HLA-A*36:04', 'HLA-A*36:05',
                                     'HLA-A*43:01', 'HLA-A*66:01', 'HLA-A*66:02', 'HLA-A*66:03', 'HLA-A*66:04', 'HLA-A*66:05', 'HLA-A*66:06',
                                     'HLA-A*66:07', 'HLA-A*66:08', 'HLA-A*66:09',
                                     'HLA-A*66:10', 'HLA-A*66:11', 'HLA-A*66:12', 'HLA-A*66:13', 'HLA-A*66:14', 'HLA-A*66:15', 'HLA-A*68:01',
                                     'HLA-A*68:02', 'HLA-A*68:03', 'HLA-A*68:04',
                                     'HLA-A*68:05', 'HLA-A*68:06', 'HLA-A*68:07', 'HLA-A*68:08', 'HLA-A*68:09', 'HLA-A*68:10', 'HLA-A*68:12',
                                     'HLA-A*68:13', 'HLA-A*68:14', 'HLA-A*68:15',
                                     'HLA-A*68:16', 'HLA-A*68:17', 'HLA-A*68:19', 'HLA-A*68:20', 'HLA-A*68:21', 'HLA-A*68:22', 'HLA-A*68:23',
                                     'HLA-A*68:24', 'HLA-A*68:25', 'HLA-A*68:26',
                                     'HLA-A*68:27', 'HLA-A*68:28', 'HLA-A*68:29', 'HLA-A*68:30', 'HLA-A*68:31', 'HLA-A*68:32', 'HLA-A*68:33',
                                     'HLA-A*68:34', 'HLA-A*68:35', 'HLA-A*68:36',
                                     'HLA-A*68:37', 'HLA-A*68:38', 'HLA-A*68:39', 'HLA-A*68:40', 'HLA-A*68:41', 'HLA-A*68:42', 'HLA-A*68:43',
                                     'HLA-A*68:44', 'HLA-A*68:45', 'HLA-A*68:46',
                                     'HLA-A*68:47', 'HLA-A*68:48', 'HLA-A*68:50', 'HLA-A*68:51', 'HLA-A*68:52', 'HLA-A*68:53', 'HLA-A*68:54',
                                     'HLA-A*69:01', 'HLA-A*74:01', 'HLA-A*74:02',
                                     'HLA-A*74:03', 'HLA-A*74:04', 'HLA-A*74:05', 'HLA-A*74:06', 'HLA-A*74:07', 'HLA-A*74:08', 'HLA-A*74:09',
                                     'HLA-A*74:10', 'HLA-A*74:11', 'HLA-A*74:13',
                                     'HLA-A*80:01', 'HLA-A*80:02', 'HLA-B*07:02', 'HLA-B*07:03', 'HLA-B*07:04', 'HLA-B*07:05', 'HLA-B*07:06',
                                     'HLA-B*07:07', 'HLA-B*07:08', 'HLA-B*07:09',
                                     'HLA-B*07:10', 'HLA-B*07:11', 'HLA-B*07:12', 'HLA-B*07:13', 'HLA-B*07:14', 'HLA-B*07:15', 'HLA-B*07:16',
                                     'HLA-B*07:17', 'HLA-B*07:18', 'HLA-B*07:19',
                                     'HLA-B*07:20', 'HLA-B*07:21', 'HLA-B*07:22', 'HLA-B*07:23', 'HLA-B*07:24', 'HLA-B*07:25', 'HLA-B*07:26',
                                     'HLA-B*07:27', 'HLA-B*07:28', 'HLA-B*07:29',
                                     'HLA-B*07:30', 'HLA-B*07:31', 'HLA-B*07:32', 'HLA-B*07:33', 'HLA-B*07:34', 'HLA-B*07:35', 'HLA-B*07:36',
                                     'HLA-B*07:37', 'HLA-B*07:38', 'HLA-B*07:39',
                                     'HLA-B*07:40', 'HLA-B*07:41', 'HLA-B*07:42', 'HLA-B*07:43', 'HLA-B*07:44', 'HLA-B*07:45', 'HLA-B*07:46',
                                     'HLA-B*07:47', 'HLA-B*07:48', 'HLA-B*07:50',
                                     'HLA-B*07:51', 'HLA-B*07:52', 'HLA-B*07:53', 'HLA-B*07:54', 'HLA-B*07:55', 'HLA-B*07:56', 'HLA-B*07:57',
                                     'HLA-B*07:58', 'HLA-B*07:59', 'HLA-B*07:60',
                                     'HLA-B*07:61', 'HLA-B*07:62', 'HLA-B*07:63', 'HLA-B*07:64', 'HLA-B*07:65', 'HLA-B*07:66', 'HLA-B*07:68',
                                     'HLA-B*07:69', 'HLA-B*07:70', 'HLA-B*07:71',
                                     'HLA-B*07:72', 'HLA-B*07:73', 'HLA-B*07:74', 'HLA-B*07:75', 'HLA-B*07:76', 'HLA-B*07:77', 'HLA-B*07:78',
                                     'HLA-B*07:79', 'HLA-B*07:80', 'HLA-B*07:81',
                                     'HLA-B*07:82', 'HLA-B*07:83', 'HLA-B*07:84', 'HLA-B*07:85', 'HLA-B*07:86', 'HLA-B*07:87', 'HLA-B*07:88',
                                     'HLA-B*07:89', 'HLA-B*07:90', 'HLA-B*07:91',
                                     'HLA-B*07:92', 'HLA-B*07:93', 'HLA-B*07:94', 'HLA-B*07:95', 'HLA-B*07:96', 'HLA-B*07:97', 'HLA-B*07:98',
                                     'HLA-B*07:99', 'HLA-B*07:100', 'HLA-B*07:101',
                                     'HLA-B*07:102', 'HLA-B*07:103', 'HLA-B*07:104', 'HLA-B*07:105', 'HLA-B*07:106', 'HLA-B*07:107', 'HLA-B*07:108',
                                     'HLA-B*07:109', 'HLA-B*07:110',
                                     'HLA-B*07:112', 'HLA-B*07:113', 'HLA-B*07:114', 'HLA-B*07:115', 'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*08:03',
                                     'HLA-B*08:04', 'HLA-B*08:05',
                                     'HLA-B*08:07', 'HLA-B*08:09', 'HLA-B*08:10', 'HLA-B*08:11', 'HLA-B*08:12', 'HLA-B*08:13', 'HLA-B*08:14',
                                     'HLA-B*08:15', 'HLA-B*08:16', 'HLA-B*08:17',
                                     'HLA-B*08:18', 'HLA-B*08:20', 'HLA-B*08:21', 'HLA-B*08:22', 'HLA-B*08:23', 'HLA-B*08:24', 'HLA-B*08:25',
                                     'HLA-B*08:26', 'HLA-B*08:27', 'HLA-B*08:28',
                                     'HLA-B*08:29', 'HLA-B*08:31', 'HLA-B*08:32', 'HLA-B*08:33', 'HLA-B*08:34', 'HLA-B*08:35', 'HLA-B*08:36',
                                     'HLA-B*08:37', 'HLA-B*08:38', 'HLA-B*08:39',
                                     'HLA-B*08:40', 'HLA-B*08:41', 'HLA-B*08:42', 'HLA-B*08:43', 'HLA-B*08:44', 'HLA-B*08:45', 'HLA-B*08:46',
                                     'HLA-B*08:47', 'HLA-B*08:48', 'HLA-B*08:49',
                                     'HLA-B*08:50', 'HLA-B*08:51', 'HLA-B*08:52', 'HLA-B*08:53', 'HLA-B*08:54', 'HLA-B*08:55', 'HLA-B*08:56',
                                     'HLA-B*08:57', 'HLA-B*08:58', 'HLA-B*08:59',
                                     'HLA-B*08:60', 'HLA-B*08:61', 'HLA-B*08:62', 'HLA-B*13:01', 'HLA-B*13:02', 'HLA-B*13:03', 'HLA-B*13:04',
                                     'HLA-B*13:06', 'HLA-B*13:09', 'HLA-B*13:10',
                                     'HLA-B*13:11', 'HLA-B*13:12', 'HLA-B*13:13', 'HLA-B*13:14', 'HLA-B*13:15', 'HLA-B*13:16', 'HLA-B*13:17',
                                     'HLA-B*13:18', 'HLA-B*13:19', 'HLA-B*13:20',
                                     'HLA-B*13:21', 'HLA-B*13:22', 'HLA-B*13:23', 'HLA-B*13:25', 'HLA-B*13:26', 'HLA-B*13:27', 'HLA-B*13:28',
                                     'HLA-B*13:29', 'HLA-B*13:30', 'HLA-B*13:31',
                                     'HLA-B*13:32', 'HLA-B*13:33', 'HLA-B*13:34', 'HLA-B*13:35', 'HLA-B*13:36', 'HLA-B*13:37', 'HLA-B*13:38',
                                     'HLA-B*13:39', 'HLA-B*14:01', 'HLA-B*14:02',
                                     'HLA-B*14:03', 'HLA-B*14:04', 'HLA-B*14:05', 'HLA-B*14:06', 'HLA-B*14:08', 'HLA-B*14:09', 'HLA-B*14:10',
                                     'HLA-B*14:11', 'HLA-B*14:12', 'HLA-B*14:13',
                                     'HLA-B*14:14', 'HLA-B*14:15', 'HLA-B*14:16', 'HLA-B*14:17', 'HLA-B*14:18', 'HLA-B*15:01', 'HLA-B*15:02',
                                     'HLA-B*15:03', 'HLA-B*15:04', 'HLA-B*15:05',
                                     'HLA-B*15:06', 'HLA-B*15:07', 'HLA-B*15:08', 'HLA-B*15:09', 'HLA-B*15:10', 'HLA-B*15:11', 'HLA-B*15:12',
                                     'HLA-B*15:13', 'HLA-B*15:14', 'HLA-B*15:15',
                                     'HLA-B*15:16', 'HLA-B*15:17', 'HLA-B*15:18', 'HLA-B*15:19', 'HLA-B*15:20', 'HLA-B*15:21', 'HLA-B*15:23',
                                     'HLA-B*15:24', 'HLA-B*15:25', 'HLA-B*15:27',
                                     'HLA-B*15:28', 'HLA-B*15:29', 'HLA-B*15:30', 'HLA-B*15:31', 'HLA-B*15:32', 'HLA-B*15:33', 'HLA-B*15:34',
                                     'HLA-B*15:35', 'HLA-B*15:36', 'HLA-B*15:37',
                                     'HLA-B*15:38', 'HLA-B*15:39', 'HLA-B*15:40', 'HLA-B*15:42', 'HLA-B*15:43', 'HLA-B*15:44', 'HLA-B*15:45',
                                     'HLA-B*15:46', 'HLA-B*15:47', 'HLA-B*15:48',
                                     'HLA-B*15:49', 'HLA-B*15:50', 'HLA-B*15:51', 'HLA-B*15:52', 'HLA-B*15:53', 'HLA-B*15:54', 'HLA-B*15:55',
                                     'HLA-B*15:56', 'HLA-B*15:57', 'HLA-B*15:58',
                                     'HLA-B*15:60', 'HLA-B*15:61', 'HLA-B*15:62', 'HLA-B*15:63', 'HLA-B*15:64', 'HLA-B*15:65', 'HLA-B*15:66',
                                     'HLA-B*15:67', 'HLA-B*15:68', 'HLA-B*15:69',
                                     'HLA-B*15:70', 'HLA-B*15:71', 'HLA-B*15:72', 'HLA-B*15:73', 'HLA-B*15:74', 'HLA-B*15:75', 'HLA-B*15:76',
                                     'HLA-B*15:77', 'HLA-B*15:78', 'HLA-B*15:80',
                                     'HLA-B*15:81', 'HLA-B*15:82', 'HLA-B*15:83', 'HLA-B*15:84', 'HLA-B*15:85', 'HLA-B*15:86', 'HLA-B*15:87',
                                     'HLA-B*15:88', 'HLA-B*15:89', 'HLA-B*15:90',
                                     'HLA-B*15:91', 'HLA-B*15:92', 'HLA-B*15:93', 'HLA-B*15:95', 'HLA-B*15:96', 'HLA-B*15:97', 'HLA-B*15:98',
                                     'HLA-B*15:99', 'HLA-B*15:101', 'HLA-B*15:102',
                                     'HLA-B*15:103', 'HLA-B*15:104', 'HLA-B*15:105', 'HLA-B*15:106', 'HLA-B*15:107', 'HLA-B*15:108', 'HLA-B*15:109',
                                     'HLA-B*15:110', 'HLA-B*15:112',
                                     'HLA-B*15:113', 'HLA-B*15:114', 'HLA-B*15:115', 'HLA-B*15:116', 'HLA-B*15:117', 'HLA-B*15:118', 'HLA-B*15:119',
                                     'HLA-B*15:120', 'HLA-B*15:121',
                                     'HLA-B*15:122', 'HLA-B*15:123', 'HLA-B*15:124', 'HLA-B*15:125', 'HLA-B*15:126', 'HLA-B*15:127', 'HLA-B*15:128',
                                     'HLA-B*15:129', 'HLA-B*15:131',
                                     'HLA-B*15:132', 'HLA-B*15:133', 'HLA-B*15:134', 'HLA-B*15:135', 'HLA-B*15:136', 'HLA-B*15:137', 'HLA-B*15:138',
                                     'HLA-B*15:139', 'HLA-B*15:140',
                                     'HLA-B*15:141', 'HLA-B*15:142', 'HLA-B*15:143', 'HLA-B*15:144', 'HLA-B*15:145', 'HLA-B*15:146', 'HLA-B*15:147',
                                     'HLA-B*15:148', 'HLA-B*15:150',
                                     'HLA-B*15:151', 'HLA-B*15:152', 'HLA-B*15:153', 'HLA-B*15:154', 'HLA-B*15:155', 'HLA-B*15:156', 'HLA-B*15:157',
                                     'HLA-B*15:158', 'HLA-B*15:159',
                                     'HLA-B*15:160', 'HLA-B*15:161', 'HLA-B*15:162', 'HLA-B*15:163', 'HLA-B*15:164', 'HLA-B*15:165', 'HLA-B*15:166',
                                     'HLA-B*15:167', 'HLA-B*15:168',
                                     'HLA-B*15:169', 'HLA-B*15:170', 'HLA-B*15:171', 'HLA-B*15:172', 'HLA-B*15:173', 'HLA-B*15:174', 'HLA-B*15:175',
                                     'HLA-B*15:176', 'HLA-B*15:177',
                                     'HLA-B*15:178', 'HLA-B*15:179', 'HLA-B*15:180', 'HLA-B*15:183', 'HLA-B*15:184', 'HLA-B*15:185', 'HLA-B*15:186',
                                     'HLA-B*15:187', 'HLA-B*15:188',
                                     'HLA-B*15:189', 'HLA-B*15:191', 'HLA-B*15:192', 'HLA-B*15:193', 'HLA-B*15:194', 'HLA-B*15:195', 'HLA-B*15:196',
                                     'HLA-B*15:197', 'HLA-B*15:198',
                                     'HLA-B*15:199', 'HLA-B*15:200', 'HLA-B*15:201', 'HLA-B*15:202', 'HLA-B*18:01', 'HLA-B*18:02', 'HLA-B*18:03',
                                     'HLA-B*18:04', 'HLA-B*18:05',
                                     'HLA-B*18:06', 'HLA-B*18:07', 'HLA-B*18:08', 'HLA-B*18:09', 'HLA-B*18:10', 'HLA-B*18:11', 'HLA-B*18:12',
                                     'HLA-B*18:13', 'HLA-B*18:14', 'HLA-B*18:15',
                                     'HLA-B*18:18', 'HLA-B*18:19', 'HLA-B*18:20', 'HLA-B*18:21', 'HLA-B*18:22', 'HLA-B*18:24', 'HLA-B*18:25',
                                     'HLA-B*18:26', 'HLA-B*18:27', 'HLA-B*18:28',
                                     'HLA-B*18:29', 'HLA-B*18:30', 'HLA-B*18:31', 'HLA-B*18:32', 'HLA-B*18:33', 'HLA-B*18:34', 'HLA-B*18:35',
                                     'HLA-B*18:36', 'HLA-B*18:37', 'HLA-B*18:38',
                                     'HLA-B*18:39', 'HLA-B*18:40', 'HLA-B*18:41', 'HLA-B*18:42', 'HLA-B*18:43', 'HLA-B*18:44', 'HLA-B*18:45',
                                     'HLA-B*18:46', 'HLA-B*18:47', 'HLA-B*18:48',
                                     'HLA-B*18:49', 'HLA-B*18:50', 'HLA-B*27:01', 'HLA-B*27:02', 'HLA-B*27:03', 'HLA-B*27:04', 'HLA-B*27:05',
                                     'HLA-B*27:06', 'HLA-B*27:07', 'HLA-B*27:08',
                                     'HLA-B*27:09', 'HLA-B*27:10', 'HLA-B*27:11', 'HLA-B*27:12', 'HLA-B*27:13', 'HLA-B*27:14', 'HLA-B*27:15',
                                     'HLA-B*27:16', 'HLA-B*27:17', 'HLA-B*27:18',
                                     'HLA-B*27:19', 'HLA-B*27:20', 'HLA-B*27:21', 'HLA-B*27:23', 'HLA-B*27:24', 'HLA-B*27:25', 'HLA-B*27:26',
                                     'HLA-B*27:27', 'HLA-B*27:28', 'HLA-B*27:29',
                                     'HLA-B*27:30', 'HLA-B*27:31', 'HLA-B*27:32', 'HLA-B*27:33', 'HLA-B*27:34', 'HLA-B*27:35', 'HLA-B*27:36',
                                     'HLA-B*27:37', 'HLA-B*27:38', 'HLA-B*27:39',
                                     'HLA-B*27:40', 'HLA-B*27:41', 'HLA-B*27:42', 'HLA-B*27:43', 'HLA-B*27:44', 'HLA-B*27:45', 'HLA-B*27:46',
                                     'HLA-B*27:47', 'HLA-B*27:48', 'HLA-B*27:49',
                                     'HLA-B*27:50', 'HLA-B*27:51', 'HLA-B*27:52', 'HLA-B*27:53', 'HLA-B*27:54', 'HLA-B*27:55', 'HLA-B*27:56',
                                     'HLA-B*27:57', 'HLA-B*27:58', 'HLA-B*27:60',
                                     'HLA-B*27:61', 'HLA-B*27:62', 'HLA-B*27:63', 'HLA-B*27:67', 'HLA-B*27:68', 'HLA-B*27:69', 'HLA-B*35:01',
                                     'HLA-B*35:02', 'HLA-B*35:03', 'HLA-B*35:04',
                                     'HLA-B*35:05', 'HLA-B*35:06', 'HLA-B*35:07', 'HLA-B*35:08', 'HLA-B*35:09', 'HLA-B*35:10', 'HLA-B*35:11',
                                     'HLA-B*35:12', 'HLA-B*35:13', 'HLA-B*35:14',
                                     'HLA-B*35:15', 'HLA-B*35:16', 'HLA-B*35:17', 'HLA-B*35:18', 'HLA-B*35:19', 'HLA-B*35:20', 'HLA-B*35:21',
                                     'HLA-B*35:22', 'HLA-B*35:23', 'HLA-B*35:24',
                                     'HLA-B*35:25', 'HLA-B*35:26', 'HLA-B*35:27', 'HLA-B*35:28', 'HLA-B*35:29', 'HLA-B*35:30', 'HLA-B*35:31',
                                     'HLA-B*35:32', 'HLA-B*35:33', 'HLA-B*35:34',
                                     'HLA-B*35:35', 'HLA-B*35:36', 'HLA-B*35:37', 'HLA-B*35:38', 'HLA-B*35:39', 'HLA-B*35:41', 'HLA-B*35:42',
                                     'HLA-B*35:43', 'HLA-B*35:44', 'HLA-B*35:45',
                                     'HLA-B*35:46', 'HLA-B*35:47', 'HLA-B*35:48', 'HLA-B*35:49', 'HLA-B*35:50', 'HLA-B*35:51', 'HLA-B*35:52',
                                     'HLA-B*35:54', 'HLA-B*35:55', 'HLA-B*35:56',
                                     'HLA-B*35:57', 'HLA-B*35:58', 'HLA-B*35:59', 'HLA-B*35:60', 'HLA-B*35:61', 'HLA-B*35:62', 'HLA-B*35:63',
                                     'HLA-B*35:64', 'HLA-B*35:66', 'HLA-B*35:67',
                                     'HLA-B*35:68', 'HLA-B*35:69', 'HLA-B*35:70', 'HLA-B*35:71', 'HLA-B*35:72', 'HLA-B*35:74', 'HLA-B*35:75',
                                     'HLA-B*35:76', 'HLA-B*35:77', 'HLA-B*35:78',
                                     'HLA-B*35:79', 'HLA-B*35:80', 'HLA-B*35:81', 'HLA-B*35:82', 'HLA-B*35:83', 'HLA-B*35:84', 'HLA-B*35:85',
                                     'HLA-B*35:86', 'HLA-B*35:87', 'HLA-B*35:88',
                                     'HLA-B*35:89', 'HLA-B*35:90', 'HLA-B*35:91', 'HLA-B*35:92', 'HLA-B*35:93', 'HLA-B*35:94', 'HLA-B*35:95',
                                     'HLA-B*35:96', 'HLA-B*35:97', 'HLA-B*35:98',
                                     'HLA-B*35:99', 'HLA-B*35:100', 'HLA-B*35:101', 'HLA-B*35:102', 'HLA-B*35:103', 'HLA-B*35:104', 'HLA-B*35:105',
                                     'HLA-B*35:106', 'HLA-B*35:107',
                                     'HLA-B*35:108', 'HLA-B*35:109', 'HLA-B*35:110', 'HLA-B*35:111', 'HLA-B*35:112', 'HLA-B*35:113', 'HLA-B*35:114',
                                     'HLA-B*35:115', 'HLA-B*35:116',
                                     'HLA-B*35:117', 'HLA-B*35:118', 'HLA-B*35:119', 'HLA-B*35:120', 'HLA-B*35:121', 'HLA-B*35:122', 'HLA-B*35:123',
                                     'HLA-B*35:124', 'HLA-B*35:125',
                                     'HLA-B*35:126', 'HLA-B*35:127', 'HLA-B*35:128', 'HLA-B*35:131', 'HLA-B*35:132', 'HLA-B*35:133', 'HLA-B*35:135',
                                     'HLA-B*35:136', 'HLA-B*35:137',
                                     'HLA-B*35:138', 'HLA-B*35:139', 'HLA-B*35:140', 'HLA-B*35:141', 'HLA-B*35:142', 'HLA-B*35:143', 'HLA-B*35:144',
                                     'HLA-B*37:01', 'HLA-B*37:02',
                                     'HLA-B*37:04', 'HLA-B*37:05', 'HLA-B*37:06', 'HLA-B*37:07', 'HLA-B*37:08', 'HLA-B*37:09', 'HLA-B*37:10',
                                     'HLA-B*37:11', 'HLA-B*37:12', 'HLA-B*37:13',
                                     'HLA-B*37:14', 'HLA-B*37:15', 'HLA-B*37:17', 'HLA-B*37:18', 'HLA-B*37:19', 'HLA-B*37:20', 'HLA-B*37:21',
                                     'HLA-B*37:22', 'HLA-B*37:23', 'HLA-B*38:01',
                                     'HLA-B*38:02', 'HLA-B*38:03', 'HLA-B*38:04', 'HLA-B*38:05', 'HLA-B*38:06', 'HLA-B*38:07', 'HLA-B*38:08',
                                     'HLA-B*38:09', 'HLA-B*38:10', 'HLA-B*38:11',
                                     'HLA-B*38:12', 'HLA-B*38:13', 'HLA-B*38:14', 'HLA-B*38:15', 'HLA-B*38:16', 'HLA-B*38:17', 'HLA-B*38:18',
                                     'HLA-B*38:19', 'HLA-B*38:20', 'HLA-B*38:21',
                                     'HLA-B*38:22', 'HLA-B*38:23', 'HLA-B*39:01', 'HLA-B*39:02', 'HLA-B*39:03', 'HLA-B*39:04', 'HLA-B*39:05',
                                     'HLA-B*39:06', 'HLA-B*39:07', 'HLA-B*39:08',
                                     'HLA-B*39:09', 'HLA-B*39:10', 'HLA-B*39:11', 'HLA-B*39:12', 'HLA-B*39:13', 'HLA-B*39:14', 'HLA-B*39:15',
                                     'HLA-B*39:16', 'HLA-B*39:17', 'HLA-B*39:18',
                                     'HLA-B*39:19', 'HLA-B*39:20', 'HLA-B*39:22', 'HLA-B*39:23', 'HLA-B*39:24', 'HLA-B*39:26', 'HLA-B*39:27',
                                     'HLA-B*39:28', 'HLA-B*39:29', 'HLA-B*39:30',
                                     'HLA-B*39:31', 'HLA-B*39:32', 'HLA-B*39:33', 'HLA-B*39:34', 'HLA-B*39:35', 'HLA-B*39:36', 'HLA-B*39:37',
                                     'HLA-B*39:39', 'HLA-B*39:41', 'HLA-B*39:42',
                                     'HLA-B*39:43', 'HLA-B*39:44', 'HLA-B*39:45', 'HLA-B*39:46', 'HLA-B*39:47', 'HLA-B*39:48', 'HLA-B*39:49',
                                     'HLA-B*39:50', 'HLA-B*39:51', 'HLA-B*39:52',
                                     'HLA-B*39:53', 'HLA-B*39:54', 'HLA-B*39:55', 'HLA-B*39:56', 'HLA-B*39:57', 'HLA-B*39:58', 'HLA-B*39:59',
                                     'HLA-B*39:60', 'HLA-B*40:01', 'HLA-B*40:02',
                                     'HLA-B*40:03', 'HLA-B*40:04', 'HLA-B*40:05', 'HLA-B*40:06', 'HLA-B*40:07', 'HLA-B*40:08', 'HLA-B*40:09',
                                     'HLA-B*40:10', 'HLA-B*40:11', 'HLA-B*40:12',
                                     'HLA-B*40:13', 'HLA-B*40:14', 'HLA-B*40:15', 'HLA-B*40:16', 'HLA-B*40:18', 'HLA-B*40:19', 'HLA-B*40:20',
                                     'HLA-B*40:21', 'HLA-B*40:23', 'HLA-B*40:24',
                                     'HLA-B*40:25', 'HLA-B*40:26', 'HLA-B*40:27', 'HLA-B*40:28', 'HLA-B*40:29', 'HLA-B*40:30', 'HLA-B*40:31',
                                     'HLA-B*40:32', 'HLA-B*40:33', 'HLA-B*40:34',
                                     'HLA-B*40:35', 'HLA-B*40:36', 'HLA-B*40:37', 'HLA-B*40:38', 'HLA-B*40:39', 'HLA-B*40:40', 'HLA-B*40:42',
                                     'HLA-B*40:43', 'HLA-B*40:44', 'HLA-B*40:45',
                                     'HLA-B*40:46', 'HLA-B*40:47', 'HLA-B*40:48', 'HLA-B*40:49', 'HLA-B*40:50', 'HLA-B*40:51', 'HLA-B*40:52',
                                     'HLA-B*40:53', 'HLA-B*40:54', 'HLA-B*40:55',
                                     'HLA-B*40:56', 'HLA-B*40:57', 'HLA-B*40:58', 'HLA-B*40:59', 'HLA-B*40:60', 'HLA-B*40:61', 'HLA-B*40:62',
                                     'HLA-B*40:63', 'HLA-B*40:64', 'HLA-B*40:65',
                                     'HLA-B*40:66', 'HLA-B*40:67', 'HLA-B*40:68', 'HLA-B*40:69', 'HLA-B*40:70', 'HLA-B*40:71', 'HLA-B*40:72',
                                     'HLA-B*40:73', 'HLA-B*40:74', 'HLA-B*40:75',
                                     'HLA-B*40:76', 'HLA-B*40:77', 'HLA-B*40:78', 'HLA-B*40:79', 'HLA-B*40:80', 'HLA-B*40:81', 'HLA-B*40:82',
                                     'HLA-B*40:83', 'HLA-B*40:84', 'HLA-B*40:85',
                                     'HLA-B*40:86', 'HLA-B*40:87', 'HLA-B*40:88', 'HLA-B*40:89', 'HLA-B*40:90', 'HLA-B*40:91', 'HLA-B*40:92',
                                     'HLA-B*40:93', 'HLA-B*40:94', 'HLA-B*40:95',
                                     'HLA-B*40:96', 'HLA-B*40:97', 'HLA-B*40:98', 'HLA-B*40:99', 'HLA-B*40:100', 'HLA-B*40:101', 'HLA-B*40:102',
                                     'HLA-B*40:103', 'HLA-B*40:104',
                                     'HLA-B*40:105', 'HLA-B*40:106', 'HLA-B*40:107', 'HLA-B*40:108', 'HLA-B*40:109', 'HLA-B*40:110', 'HLA-B*40:111',
                                     'HLA-B*40:112', 'HLA-B*40:113',
                                     'HLA-B*40:114', 'HLA-B*40:115', 'HLA-B*40:116', 'HLA-B*40:117', 'HLA-B*40:119', 'HLA-B*40:120', 'HLA-B*40:121',
                                     'HLA-B*40:122', 'HLA-B*40:123',
                                     'HLA-B*40:124', 'HLA-B*40:125', 'HLA-B*40:126', 'HLA-B*40:127', 'HLA-B*40:128', 'HLA-B*40:129', 'HLA-B*40:130',
                                     'HLA-B*40:131', 'HLA-B*40:132',
                                     'HLA-B*40:134', 'HLA-B*40:135', 'HLA-B*40:136', 'HLA-B*40:137', 'HLA-B*40:138', 'HLA-B*40:139', 'HLA-B*40:140',
                                     'HLA-B*40:141', 'HLA-B*40:143',
                                     'HLA-B*40:145', 'HLA-B*40:146', 'HLA-B*40:147', 'HLA-B*41:01', 'HLA-B*41:02', 'HLA-B*41:03', 'HLA-B*41:04',
                                     'HLA-B*41:05', 'HLA-B*41:06', 'HLA-B*41:07',
                                     'HLA-B*41:08', 'HLA-B*41:09', 'HLA-B*41:10', 'HLA-B*41:11', 'HLA-B*41:12', 'HLA-B*42:01', 'HLA-B*42:02',
                                     'HLA-B*42:04', 'HLA-B*42:05', 'HLA-B*42:06',
                                     'HLA-B*42:07', 'HLA-B*42:08', 'HLA-B*42:09', 'HLA-B*42:10', 'HLA-B*42:11', 'HLA-B*42:12', 'HLA-B*42:13',
                                     'HLA-B*42:14', 'HLA-B*44:02', 'HLA-B*44:03',
                                     'HLA-B*44:04', 'HLA-B*44:05', 'HLA-B*44:06', 'HLA-B*44:07', 'HLA-B*44:08', 'HLA-B*44:09', 'HLA-B*44:10',
                                     'HLA-B*44:11', 'HLA-B*44:12', 'HLA-B*44:13',
                                     'HLA-B*44:14', 'HLA-B*44:15', 'HLA-B*44:16', 'HLA-B*44:17', 'HLA-B*44:18', 'HLA-B*44:20', 'HLA-B*44:21',
                                     'HLA-B*44:22', 'HLA-B*44:24', 'HLA-B*44:25',
                                     'HLA-B*44:26', 'HLA-B*44:27', 'HLA-B*44:28', 'HLA-B*44:29', 'HLA-B*44:30', 'HLA-B*44:31', 'HLA-B*44:32',
                                     'HLA-B*44:33', 'HLA-B*44:34', 'HLA-B*44:35',
                                     'HLA-B*44:36', 'HLA-B*44:37', 'HLA-B*44:38', 'HLA-B*44:39', 'HLA-B*44:40', 'HLA-B*44:41', 'HLA-B*44:42',
                                     'HLA-B*44:43', 'HLA-B*44:44', 'HLA-B*44:45',
                                     'HLA-B*44:46', 'HLA-B*44:47', 'HLA-B*44:48', 'HLA-B*44:49', 'HLA-B*44:50', 'HLA-B*44:51', 'HLA-B*44:53',
                                     'HLA-B*44:54', 'HLA-B*44:55', 'HLA-B*44:57',
                                     'HLA-B*44:59', 'HLA-B*44:60', 'HLA-B*44:62', 'HLA-B*44:63', 'HLA-B*44:64', 'HLA-B*44:65', 'HLA-B*44:66',
                                     'HLA-B*44:67', 'HLA-B*44:68', 'HLA-B*44:69',
                                     'HLA-B*44:70', 'HLA-B*44:71', 'HLA-B*44:72', 'HLA-B*44:73', 'HLA-B*44:74', 'HLA-B*44:75', 'HLA-B*44:76',
                                     'HLA-B*44:77', 'HLA-B*44:78', 'HLA-B*44:79',
                                     'HLA-B*44:80', 'HLA-B*44:81', 'HLA-B*44:82', 'HLA-B*44:83', 'HLA-B*44:84', 'HLA-B*44:85', 'HLA-B*44:86',
                                     'HLA-B*44:87', 'HLA-B*44:88', 'HLA-B*44:89',
                                     'HLA-B*44:90', 'HLA-B*44:91', 'HLA-B*44:92', 'HLA-B*44:93', 'HLA-B*44:94', 'HLA-B*44:95', 'HLA-B*44:96',
                                     'HLA-B*44:97', 'HLA-B*44:98', 'HLA-B*44:99',
                                     'HLA-B*44:100', 'HLA-B*44:101', 'HLA-B*44:102', 'HLA-B*44:103', 'HLA-B*44:104', 'HLA-B*44:105', 'HLA-B*44:106',
                                     'HLA-B*44:107', 'HLA-B*44:109',
                                     'HLA-B*44:110', 'HLA-B*45:01', 'HLA-B*45:02', 'HLA-B*45:03', 'HLA-B*45:04', 'HLA-B*45:05', 'HLA-B*45:06',
                                     'HLA-B*45:07', 'HLA-B*45:08', 'HLA-B*45:09',
                                     'HLA-B*45:10', 'HLA-B*45:11', 'HLA-B*45:12', 'HLA-B*46:01', 'HLA-B*46:02', 'HLA-B*46:03', 'HLA-B*46:04',
                                     'HLA-B*46:05', 'HLA-B*46:06', 'HLA-B*46:08',
                                     'HLA-B*46:09', 'HLA-B*46:10', 'HLA-B*46:11', 'HLA-B*46:12', 'HLA-B*46:13', 'HLA-B*46:14', 'HLA-B*46:16',
                                     'HLA-B*46:17', 'HLA-B*46:18', 'HLA-B*46:19',
                                     'HLA-B*46:20', 'HLA-B*46:21', 'HLA-B*46:22', 'HLA-B*46:23', 'HLA-B*46:24', 'HLA-B*47:01', 'HLA-B*47:02',
                                     'HLA-B*47:03', 'HLA-B*47:04', 'HLA-B*47:05',
                                     'HLA-B*47:06', 'HLA-B*47:07', 'HLA-B*48:01', 'HLA-B*48:02', 'HLA-B*48:03', 'HLA-B*48:04', 'HLA-B*48:05',
                                     'HLA-B*48:06', 'HLA-B*48:07', 'HLA-B*48:08',
                                     'HLA-B*48:09', 'HLA-B*48:10', 'HLA-B*48:11', 'HLA-B*48:12', 'HLA-B*48:13', 'HLA-B*48:14', 'HLA-B*48:15',
                                     'HLA-B*48:16', 'HLA-B*48:17', 'HLA-B*48:18',
                                     'HLA-B*48:19', 'HLA-B*48:20', 'HLA-B*48:21', 'HLA-B*48:22', 'HLA-B*48:23', 'HLA-B*49:01', 'HLA-B*49:02',
                                     'HLA-B*49:03', 'HLA-B*49:04', 'HLA-B*49:05',
                                     'HLA-B*49:06', 'HLA-B*49:07', 'HLA-B*49:08', 'HLA-B*49:09', 'HLA-B*49:10', 'HLA-B*50:01', 'HLA-B*50:02',
                                     'HLA-B*50:04', 'HLA-B*50:05', 'HLA-B*50:06',
                                     'HLA-B*50:07', 'HLA-B*50:08', 'HLA-B*50:09', 'HLA-B*51:01', 'HLA-B*51:02', 'HLA-B*51:03', 'HLA-B*51:04',
                                     'HLA-B*51:05', 'HLA-B*51:06', 'HLA-B*51:07',
                                     'HLA-B*51:08', 'HLA-B*51:09', 'HLA-B*51:12', 'HLA-B*51:13', 'HLA-B*51:14', 'HLA-B*51:15', 'HLA-B*51:16',
                                     'HLA-B*51:17', 'HLA-B*51:18', 'HLA-B*51:19',
                                     'HLA-B*51:20', 'HLA-B*51:21', 'HLA-B*51:22', 'HLA-B*51:23', 'HLA-B*51:24', 'HLA-B*51:26', 'HLA-B*51:28',
                                     'HLA-B*51:29', 'HLA-B*51:30', 'HLA-B*51:31',
                                     'HLA-B*51:32', 'HLA-B*51:33', 'HLA-B*51:34', 'HLA-B*51:35', 'HLA-B*51:36', 'HLA-B*51:37', 'HLA-B*51:38',
                                     'HLA-B*51:39', 'HLA-B*51:40', 'HLA-B*51:42',
                                     'HLA-B*51:43', 'HLA-B*51:45', 'HLA-B*51:46', 'HLA-B*51:48', 'HLA-B*51:49', 'HLA-B*51:50', 'HLA-B*51:51',
                                     'HLA-B*51:52', 'HLA-B*51:53', 'HLA-B*51:54',
                                     'HLA-B*51:55', 'HLA-B*51:56', 'HLA-B*51:57', 'HLA-B*51:58', 'HLA-B*51:59', 'HLA-B*51:60', 'HLA-B*51:61',
                                     'HLA-B*51:62', 'HLA-B*51:63', 'HLA-B*51:64',
                                     'HLA-B*51:65', 'HLA-B*51:66', 'HLA-B*51:67', 'HLA-B*51:68', 'HLA-B*51:69', 'HLA-B*51:70', 'HLA-B*51:71',
                                     'HLA-B*51:72', 'HLA-B*51:73', 'HLA-B*51:74',
                                     'HLA-B*51:75', 'HLA-B*51:76', 'HLA-B*51:77', 'HLA-B*51:78', 'HLA-B*51:79', 'HLA-B*51:80', 'HLA-B*51:81',
                                     'HLA-B*51:82', 'HLA-B*51:83', 'HLA-B*51:84',
                                     'HLA-B*51:85', 'HLA-B*51:86', 'HLA-B*51:87', 'HLA-B*51:88', 'HLA-B*51:89', 'HLA-B*51:90', 'HLA-B*51:91',
                                     'HLA-B*51:92', 'HLA-B*51:93', 'HLA-B*51:94',
                                     'HLA-B*51:95', 'HLA-B*51:96', 'HLA-B*52:01', 'HLA-B*52:02', 'HLA-B*52:03', 'HLA-B*52:04', 'HLA-B*52:05',
                                     'HLA-B*52:06', 'HLA-B*52:07', 'HLA-B*52:08',
                                     'HLA-B*52:09', 'HLA-B*52:10', 'HLA-B*52:11', 'HLA-B*52:12', 'HLA-B*52:13', 'HLA-B*52:14', 'HLA-B*52:15',
                                     'HLA-B*52:16', 'HLA-B*52:17', 'HLA-B*52:18',
                                     'HLA-B*52:19', 'HLA-B*52:20', 'HLA-B*52:21', 'HLA-B*53:01', 'HLA-B*53:02', 'HLA-B*53:03', 'HLA-B*53:04',
                                     'HLA-B*53:05', 'HLA-B*53:06', 'HLA-B*53:07',
                                     'HLA-B*53:08', 'HLA-B*53:09', 'HLA-B*53:10', 'HLA-B*53:11', 'HLA-B*53:12', 'HLA-B*53:13', 'HLA-B*53:14',
                                     'HLA-B*53:15', 'HLA-B*53:16', 'HLA-B*53:17',
                                     'HLA-B*53:18', 'HLA-B*53:19', 'HLA-B*53:20', 'HLA-B*53:21', 'HLA-B*53:22', 'HLA-B*53:23', 'HLA-B*54:01',
                                     'HLA-B*54:02', 'HLA-B*54:03', 'HLA-B*54:04',
                                     'HLA-B*54:06', 'HLA-B*54:07', 'HLA-B*54:09', 'HLA-B*54:10', 'HLA-B*54:11', 'HLA-B*54:12', 'HLA-B*54:13',
                                     'HLA-B*54:14', 'HLA-B*54:15', 'HLA-B*54:16',
                                     'HLA-B*54:17', 'HLA-B*54:18', 'HLA-B*54:19', 'HLA-B*54:20', 'HLA-B*54:21', 'HLA-B*54:22', 'HLA-B*54:23',
                                     'HLA-B*55:01', 'HLA-B*55:02', 'HLA-B*55:03',
                                     'HLA-B*55:04', 'HLA-B*55:05', 'HLA-B*55:07', 'HLA-B*55:08', 'HLA-B*55:09', 'HLA-B*55:10', 'HLA-B*55:11',
                                     'HLA-B*55:12', 'HLA-B*55:13', 'HLA-B*55:14',
                                     'HLA-B*55:15', 'HLA-B*55:16', 'HLA-B*55:17', 'HLA-B*55:18', 'HLA-B*55:19', 'HLA-B*55:20', 'HLA-B*55:21',
                                     'HLA-B*55:22', 'HLA-B*55:23', 'HLA-B*55:24',
                                     'HLA-B*55:25', 'HLA-B*55:26', 'HLA-B*55:27', 'HLA-B*55:28', 'HLA-B*55:29', 'HLA-B*55:30', 'HLA-B*55:31',
                                     'HLA-B*55:32', 'HLA-B*55:33', 'HLA-B*55:34',
                                     'HLA-B*55:35', 'HLA-B*55:36', 'HLA-B*55:37', 'HLA-B*55:38', 'HLA-B*55:39', 'HLA-B*55:40', 'HLA-B*55:41',
                                     'HLA-B*55:42', 'HLA-B*55:43', 'HLA-B*56:01',
                                     'HLA-B*56:02', 'HLA-B*56:03', 'HLA-B*56:04', 'HLA-B*56:05', 'HLA-B*56:06', 'HLA-B*56:07', 'HLA-B*56:08',
                                     'HLA-B*56:09', 'HLA-B*56:10', 'HLA-B*56:11',
                                     'HLA-B*56:12', 'HLA-B*56:13', 'HLA-B*56:14', 'HLA-B*56:15', 'HLA-B*56:16', 'HLA-B*56:17', 'HLA-B*56:18',
                                     'HLA-B*56:20', 'HLA-B*56:21', 'HLA-B*56:22',
                                     'HLA-B*56:23', 'HLA-B*56:24', 'HLA-B*56:25', 'HLA-B*56:26', 'HLA-B*56:27', 'HLA-B*56:29', 'HLA-B*57:01',
                                     'HLA-B*57:02', 'HLA-B*57:03', 'HLA-B*57:04',
                                     'HLA-B*57:05', 'HLA-B*57:06', 'HLA-B*57:07', 'HLA-B*57:08', 'HLA-B*57:09', 'HLA-B*57:10', 'HLA-B*57:11',
                                     'HLA-B*57:12', 'HLA-B*57:13', 'HLA-B*57:14',
                                     'HLA-B*57:15', 'HLA-B*57:16', 'HLA-B*57:17', 'HLA-B*57:18', 'HLA-B*57:19', 'HLA-B*57:20', 'HLA-B*57:21',
                                     'HLA-B*57:22', 'HLA-B*57:23', 'HLA-B*57:24',
                                     'HLA-B*57:25', 'HLA-B*57:26', 'HLA-B*57:27', 'HLA-B*57:29', 'HLA-B*57:30', 'HLA-B*57:31', 'HLA-B*57:32',
                                     'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*58:04',
                                     'HLA-B*58:05', 'HLA-B*58:06', 'HLA-B*58:07', 'HLA-B*58:08', 'HLA-B*58:09', 'HLA-B*58:11', 'HLA-B*58:12',
                                     'HLA-B*58:13', 'HLA-B*58:14', 'HLA-B*58:15',
                                     'HLA-B*58:16', 'HLA-B*58:18', 'HLA-B*58:19', 'HLA-B*58:20', 'HLA-B*58:21', 'HLA-B*58:22', 'HLA-B*58:23',
                                     'HLA-B*58:24', 'HLA-B*58:25', 'HLA-B*58:26',
                                     'HLA-B*58:27', 'HLA-B*58:28', 'HLA-B*58:29', 'HLA-B*58:30', 'HLA-B*59:01', 'HLA-B*59:02', 'HLA-B*59:03',
                                     'HLA-B*59:04', 'HLA-B*59:05', 'HLA-B*67:01',
                                     'HLA-B*67:02', 'HLA-B*73:01', 'HLA-B*73:02', 'HLA-B*78:01', 'HLA-B*78:02', 'HLA-B*78:03', 'HLA-B*78:04',
                                     'HLA-B*78:05', 'HLA-B*78:06', 'HLA-B*78:07',
                                     'HLA-B*81:01', 'HLA-B*81:02', 'HLA-B*81:03', 'HLA-B*81:05', 'HLA-B*82:01', 'HLA-B*82:02', 'HLA-B*82:03',
                                     'HLA-B*83:01', 'HLA-C*01:02', 'HLA-C*01:03',
                                     'HLA-C*01:04', 'HLA-C*01:05', 'HLA-C*01:06', 'HLA-C*01:07', 'HLA-C*01:08', 'HLA-C*01:09', 'HLA-C*01:10',
                                     'HLA-C*01:11', 'HLA-C*01:12', 'HLA-C*01:13',
                                     'HLA-C*01:14', 'HLA-C*01:15', 'HLA-C*01:16', 'HLA-C*01:17', 'HLA-C*01:18', 'HLA-C*01:19', 'HLA-C*01:20',
                                     'HLA-C*01:21', 'HLA-C*01:22', 'HLA-C*01:23',
                                     'HLA-C*01:24', 'HLA-C*01:25', 'HLA-C*01:26', 'HLA-C*01:27', 'HLA-C*01:28', 'HLA-C*01:29', 'HLA-C*01:30',
                                     'HLA-C*01:31', 'HLA-C*01:32', 'HLA-C*01:33',
                                     'HLA-C*01:34', 'HLA-C*01:35', 'HLA-C*01:36', 'HLA-C*01:38', 'HLA-C*01:39', 'HLA-C*01:40', 'HLA-C*02:02',
                                     'HLA-C*02:03', 'HLA-C*02:04', 'HLA-C*02:05',
                                     'HLA-C*02:06', 'HLA-C*02:07', 'HLA-C*02:08', 'HLA-C*02:09', 'HLA-C*02:10', 'HLA-C*02:11', 'HLA-C*02:12',
                                     'HLA-C*02:13', 'HLA-C*02:14', 'HLA-C*02:15',
                                     'HLA-C*02:16', 'HLA-C*02:17', 'HLA-C*02:18', 'HLA-C*02:19', 'HLA-C*02:20', 'HLA-C*02:21', 'HLA-C*02:22',
                                     'HLA-C*02:23', 'HLA-C*02:24', 'HLA-C*02:26',
                                     'HLA-C*02:27', 'HLA-C*02:28', 'HLA-C*02:29', 'HLA-C*02:30', 'HLA-C*02:31', 'HLA-C*02:32', 'HLA-C*02:33',
                                     'HLA-C*02:34', 'HLA-C*02:35', 'HLA-C*02:36',
                                     'HLA-C*02:37', 'HLA-C*02:39', 'HLA-C*02:40', 'HLA-C*03:01', 'HLA-C*03:02', 'HLA-C*03:03', 'HLA-C*03:04',
                                     'HLA-C*03:05', 'HLA-C*03:06', 'HLA-C*03:07',
                                     'HLA-C*03:08', 'HLA-C*03:09', 'HLA-C*03:10', 'HLA-C*03:11', 'HLA-C*03:12', 'HLA-C*03:13', 'HLA-C*03:14',
                                     'HLA-C*03:15', 'HLA-C*03:16', 'HLA-C*03:17',
                                     'HLA-C*03:18', 'HLA-C*03:19', 'HLA-C*03:21', 'HLA-C*03:23', 'HLA-C*03:24', 'HLA-C*03:25', 'HLA-C*03:26',
                                     'HLA-C*03:27', 'HLA-C*03:28', 'HLA-C*03:29',
                                     'HLA-C*03:30', 'HLA-C*03:31', 'HLA-C*03:32', 'HLA-C*03:33', 'HLA-C*03:34', 'HLA-C*03:35', 'HLA-C*03:36',
                                     'HLA-C*03:37', 'HLA-C*03:38', 'HLA-C*03:39',
                                     'HLA-C*03:40', 'HLA-C*03:41', 'HLA-C*03:42', 'HLA-C*03:43', 'HLA-C*03:44', 'HLA-C*03:45', 'HLA-C*03:46',
                                     'HLA-C*03:47', 'HLA-C*03:48', 'HLA-C*03:49',
                                     'HLA-C*03:50', 'HLA-C*03:51', 'HLA-C*03:52', 'HLA-C*03:53', 'HLA-C*03:54', 'HLA-C*03:55', 'HLA-C*03:56',
                                     'HLA-C*03:57', 'HLA-C*03:58', 'HLA-C*03:59',
                                     'HLA-C*03:60', 'HLA-C*03:61', 'HLA-C*03:62', 'HLA-C*03:63', 'HLA-C*03:64', 'HLA-C*03:65', 'HLA-C*03:66',
                                     'HLA-C*03:67', 'HLA-C*03:68', 'HLA-C*03:69',
                                     'HLA-C*03:70', 'HLA-C*03:71', 'HLA-C*03:72', 'HLA-C*03:73', 'HLA-C*03:74', 'HLA-C*03:75', 'HLA-C*03:76',
                                     'HLA-C*03:77', 'HLA-C*03:78', 'HLA-C*03:79',
                                     'HLA-C*03:80', 'HLA-C*03:81', 'HLA-C*03:82', 'HLA-C*03:83', 'HLA-C*03:84', 'HLA-C*03:85', 'HLA-C*03:86',
                                     'HLA-C*03:87', 'HLA-C*03:88', 'HLA-C*03:89',
                                     'HLA-C*03:90', 'HLA-C*03:91', 'HLA-C*03:92', 'HLA-C*03:93', 'HLA-C*03:94', 'HLA-C*04:01', 'HLA-C*04:03',
                                     'HLA-C*04:04', 'HLA-C*04:05', 'HLA-C*04:06',
                                     'HLA-C*04:07', 'HLA-C*04:08', 'HLA-C*04:10', 'HLA-C*04:11', 'HLA-C*04:12', 'HLA-C*04:13', 'HLA-C*04:14',
                                     'HLA-C*04:15', 'HLA-C*04:16', 'HLA-C*04:17',
                                     'HLA-C*04:18', 'HLA-C*04:19', 'HLA-C*04:20', 'HLA-C*04:23', 'HLA-C*04:24', 'HLA-C*04:25', 'HLA-C*04:26',
                                     'HLA-C*04:27', 'HLA-C*04:28', 'HLA-C*04:29',
                                     'HLA-C*04:30', 'HLA-C*04:31', 'HLA-C*04:32', 'HLA-C*04:33', 'HLA-C*04:34', 'HLA-C*04:35', 'HLA-C*04:36',
                                     'HLA-C*04:37', 'HLA-C*04:38', 'HLA-C*04:39',
                                     'HLA-C*04:40', 'HLA-C*04:41', 'HLA-C*04:42', 'HLA-C*04:43', 'HLA-C*04:44', 'HLA-C*04:45', 'HLA-C*04:46',
                                     'HLA-C*04:47', 'HLA-C*04:48', 'HLA-C*04:49',
                                     'HLA-C*04:50', 'HLA-C*04:51', 'HLA-C*04:52', 'HLA-C*04:53', 'HLA-C*04:54', 'HLA-C*04:55', 'HLA-C*04:56',
                                     'HLA-C*04:57', 'HLA-C*04:58', 'HLA-C*04:60',
                                     'HLA-C*04:61', 'HLA-C*04:62', 'HLA-C*04:63', 'HLA-C*04:64', 'HLA-C*04:65', 'HLA-C*04:66', 'HLA-C*04:67',
                                     'HLA-C*04:68', 'HLA-C*04:69', 'HLA-C*04:70',
                                     'HLA-C*05:01', 'HLA-C*05:03', 'HLA-C*05:04', 'HLA-C*05:05', 'HLA-C*05:06', 'HLA-C*05:08', 'HLA-C*05:09',
                                     'HLA-C*05:10', 'HLA-C*05:11', 'HLA-C*05:12',
                                     'HLA-C*05:13', 'HLA-C*05:14', 'HLA-C*05:15', 'HLA-C*05:16', 'HLA-C*05:17', 'HLA-C*05:18', 'HLA-C*05:19',
                                     'HLA-C*05:20', 'HLA-C*05:21', 'HLA-C*05:22',
                                     'HLA-C*05:23', 'HLA-C*05:24', 'HLA-C*05:25', 'HLA-C*05:26', 'HLA-C*05:27', 'HLA-C*05:28', 'HLA-C*05:29',
                                     'HLA-C*05:30', 'HLA-C*05:31', 'HLA-C*05:32',
                                     'HLA-C*05:33', 'HLA-C*05:34', 'HLA-C*05:35', 'HLA-C*05:36', 'HLA-C*05:37', 'HLA-C*05:38', 'HLA-C*05:39',
                                     'HLA-C*05:40', 'HLA-C*05:41', 'HLA-C*05:42',
                                     'HLA-C*05:43', 'HLA-C*05:44', 'HLA-C*05:45', 'HLA-C*06:02', 'HLA-C*06:03', 'HLA-C*06:04', 'HLA-C*06:05',
                                     'HLA-C*06:06', 'HLA-C*06:07', 'HLA-C*06:08',
                                     'HLA-C*06:09', 'HLA-C*06:10', 'HLA-C*06:11', 'HLA-C*06:12', 'HLA-C*06:13', 'HLA-C*06:14', 'HLA-C*06:15',
                                     'HLA-C*06:17', 'HLA-C*06:18', 'HLA-C*06:19',
                                     'HLA-C*06:20', 'HLA-C*06:21', 'HLA-C*06:22', 'HLA-C*06:23', 'HLA-C*06:24', 'HLA-C*06:25', 'HLA-C*06:26',
                                     'HLA-C*06:27', 'HLA-C*06:28', 'HLA-C*06:29',
                                     'HLA-C*06:30', 'HLA-C*06:31', 'HLA-C*06:32', 'HLA-C*06:33', 'HLA-C*06:34', 'HLA-C*06:35', 'HLA-C*06:36',
                                     'HLA-C*06:37', 'HLA-C*06:38', 'HLA-C*06:39',
                                     'HLA-C*06:40', 'HLA-C*06:41', 'HLA-C*06:42', 'HLA-C*06:43', 'HLA-C*06:44', 'HLA-C*06:45', 'HLA-C*07:01',
                                     'HLA-C*07:02', 'HLA-C*07:03', 'HLA-C*07:04',
                                     'HLA-C*07:05', 'HLA-C*07:06', 'HLA-C*07:07', 'HLA-C*07:08', 'HLA-C*07:09', 'HLA-C*07:10', 'HLA-C*07:11',
                                     'HLA-C*07:12', 'HLA-C*07:13', 'HLA-C*07:14',
                                     'HLA-C*07:15', 'HLA-C*07:16', 'HLA-C*07:17', 'HLA-C*07:18', 'HLA-C*07:19', 'HLA-C*07:20', 'HLA-C*07:21',
                                     'HLA-C*07:22', 'HLA-C*07:23', 'HLA-C*07:24',
                                     'HLA-C*07:25', 'HLA-C*07:26', 'HLA-C*07:27', 'HLA-C*07:28', 'HLA-C*07:29', 'HLA-C*07:30', 'HLA-C*07:31',
                                     'HLA-C*07:35', 'HLA-C*07:36', 'HLA-C*07:37',
                                     'HLA-C*07:38', 'HLA-C*07:39', 'HLA-C*07:40', 'HLA-C*07:41', 'HLA-C*07:42', 'HLA-C*07:43', 'HLA-C*07:44',
                                     'HLA-C*07:45', 'HLA-C*07:46', 'HLA-C*07:47',
                                     'HLA-C*07:48', 'HLA-C*07:49', 'HLA-C*07:50', 'HLA-C*07:51', 'HLA-C*07:52', 'HLA-C*07:53', 'HLA-C*07:54',
                                     'HLA-C*07:56', 'HLA-C*07:57', 'HLA-C*07:58',
                                     'HLA-C*07:59', 'HLA-C*07:60', 'HLA-C*07:62', 'HLA-C*07:63', 'HLA-C*07:64', 'HLA-C*07:65', 'HLA-C*07:66',
                                     'HLA-C*07:67', 'HLA-C*07:68', 'HLA-C*07:69',
                                     'HLA-C*07:70', 'HLA-C*07:71', 'HLA-C*07:72', 'HLA-C*07:73', 'HLA-C*07:74', 'HLA-C*07:75', 'HLA-C*07:76',
                                     'HLA-C*07:77', 'HLA-C*07:78', 'HLA-C*07:79',
                                     'HLA-C*07:80', 'HLA-C*07:81', 'HLA-C*07:82', 'HLA-C*07:83', 'HLA-C*07:84', 'HLA-C*07:85', 'HLA-C*07:86',
                                     'HLA-C*07:87', 'HLA-C*07:88', 'HLA-C*07:89',
                                     'HLA-C*07:90', 'HLA-C*07:91', 'HLA-C*07:92', 'HLA-C*07:93', 'HLA-C*07:94', 'HLA-C*07:95', 'HLA-C*07:96',
                                     'HLA-C*07:97', 'HLA-C*07:99', 'HLA-C*07:100',
                                     'HLA-C*07:101', 'HLA-C*07:102', 'HLA-C*07:103', 'HLA-C*07:105', 'HLA-C*07:106', 'HLA-C*07:107', 'HLA-C*07:108',
                                     'HLA-C*07:109', 'HLA-C*07:110',
                                     'HLA-C*07:111', 'HLA-C*07:112', 'HLA-C*07:113', 'HLA-C*07:114', 'HLA-C*07:115', 'HLA-C*07:116', 'HLA-C*07:117',
                                     'HLA-C*07:118', 'HLA-C*07:119',
                                     'HLA-C*07:120', 'HLA-C*07:122', 'HLA-C*07:123', 'HLA-C*07:124', 'HLA-C*07:125', 'HLA-C*07:126', 'HLA-C*07:127',
                                     'HLA-C*07:128', 'HLA-C*07:129',
                                     'HLA-C*07:130', 'HLA-C*07:131', 'HLA-C*07:132', 'HLA-C*07:133', 'HLA-C*07:134', 'HLA-C*07:135', 'HLA-C*07:136',
                                     'HLA-C*07:137', 'HLA-C*07:138',
                                     'HLA-C*07:139', 'HLA-C*07:140', 'HLA-C*07:141', 'HLA-C*07:142', 'HLA-C*07:143', 'HLA-C*07:144', 'HLA-C*07:145',
                                     'HLA-C*07:146', 'HLA-C*07:147',
                                     'HLA-C*07:148', 'HLA-C*07:149', 'HLA-C*08:01', 'HLA-C*08:02', 'HLA-C*08:03', 'HLA-C*08:04', 'HLA-C*08:05',
                                     'HLA-C*08:06', 'HLA-C*08:07', 'HLA-C*08:08',
                                     'HLA-C*08:09', 'HLA-C*08:10', 'HLA-C*08:11', 'HLA-C*08:12', 'HLA-C*08:13', 'HLA-C*08:14', 'HLA-C*08:15',
                                     'HLA-C*08:16', 'HLA-C*08:17', 'HLA-C*08:18',
                                     'HLA-C*08:19', 'HLA-C*08:20', 'HLA-C*08:21', 'HLA-C*08:22', 'HLA-C*08:23', 'HLA-C*08:24', 'HLA-C*08:25',
                                     'HLA-C*08:27', 'HLA-C*08:28', 'HLA-C*08:29',
                                     'HLA-C*08:30', 'HLA-C*08:31', 'HLA-C*08:32', 'HLA-C*08:33', 'HLA-C*08:34', 'HLA-C*08:35', 'HLA-C*12:02',
                                     'HLA-C*12:03', 'HLA-C*12:04', 'HLA-C*12:05',
                                     'HLA-C*12:06', 'HLA-C*12:07', 'HLA-C*12:08', 'HLA-C*12:09', 'HLA-C*12:10', 'HLA-C*12:11', 'HLA-C*12:12',
                                     'HLA-C*12:13', 'HLA-C*12:14', 'HLA-C*12:15',
                                     'HLA-C*12:16', 'HLA-C*12:17', 'HLA-C*12:18', 'HLA-C*12:19', 'HLA-C*12:20', 'HLA-C*12:21', 'HLA-C*12:22',
                                     'HLA-C*12:23', 'HLA-C*12:24', 'HLA-C*12:25',
                                     'HLA-C*12:26', 'HLA-C*12:27', 'HLA-C*12:28', 'HLA-C*12:29', 'HLA-C*12:30', 'HLA-C*12:31', 'HLA-C*12:32',
                                     'HLA-C*12:33', 'HLA-C*12:34', 'HLA-C*12:35',
                                     'HLA-C*12:36', 'HLA-C*12:37', 'HLA-C*12:38', 'HLA-C*12:40', 'HLA-C*12:41', 'HLA-C*12:43', 'HLA-C*12:44',
                                     'HLA-C*14:02', 'HLA-C*14:03', 'HLA-C*14:04',
                                     'HLA-C*14:05', 'HLA-C*14:06', 'HLA-C*14:08', 'HLA-C*14:09', 'HLA-C*14:10', 'HLA-C*14:11', 'HLA-C*14:12',
                                     'HLA-C*14:13', 'HLA-C*14:14', 'HLA-C*14:15',
                                     'HLA-C*14:16', 'HLA-C*14:17', 'HLA-C*14:18', 'HLA-C*14:19', 'HLA-C*14:20', 'HLA-C*15:02', 'HLA-C*15:03',
                                     'HLA-C*15:04', 'HLA-C*15:05', 'HLA-C*15:06',
                                     'HLA-C*15:07', 'HLA-C*15:08', 'HLA-C*15:09', 'HLA-C*15:10', 'HLA-C*15:11', 'HLA-C*15:12', 'HLA-C*15:13',
                                     'HLA-C*15:15', 'HLA-C*15:16', 'HLA-C*15:17',
                                     'HLA-C*15:18', 'HLA-C*15:19', 'HLA-C*15:20', 'HLA-C*15:21', 'HLA-C*15:22', 'HLA-C*15:23', 'HLA-C*15:24',
                                     'HLA-C*15:25', 'HLA-C*15:26', 'HLA-C*15:27',
                                     'HLA-C*15:28', 'HLA-C*15:29', 'HLA-C*15:30', 'HLA-C*15:31', 'HLA-C*15:33', 'HLA-C*15:34', 'HLA-C*15:35',
                                     'HLA-C*16:01', 'HLA-C*16:02', 'HLA-C*16:04',
                                     'HLA-C*16:06', 'HLA-C*16:07', 'HLA-C*16:08', 'HLA-C*16:09', 'HLA-C*16:10', 'HLA-C*16:11', 'HLA-C*16:12',
                                     'HLA-C*16:13', 'HLA-C*16:14', 'HLA-C*16:15',
                                     'HLA-C*16:17', 'HLA-C*16:18', 'HLA-C*16:19', 'HLA-C*16:20', 'HLA-C*16:21', 'HLA-C*16:22', 'HLA-C*16:23',
                                     'HLA-C*16:24', 'HLA-C*16:25', 'HLA-C*16:26',
                                     'HLA-C*17:01', 'HLA-C*17:02', 'HLA-C*17:03', 'HLA-C*17:04', 'HLA-C*17:05', 'HLA-C*17:06', 'HLA-C*17:07',
                                     'HLA-C*18:01', 'HLA-C*18:02', 'HLA-C*18:03',
                                     'HLA-G*01:01', 'HLA-G*01:02', 'HLA-G*01:03', 'HLA-G*01:04', 'HLA-G*01:06', 'HLA-G*01:07', 'HLA-G*01:08',
                                     'HLA-G*01:09', 'HLA-E*01:01',
                                     'H2-Db', 'H2-Dd', 'H2-Kb', 'H2-Kd', 'H2-Kk', 'H2-Ld'])
    __version = "1.1"

    @property
    def version(self):
        """The version of the predictor"""
        return self.__version

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__supported_alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    def _represent(self, allele):
        """
        Internal function transforming an allele object into its representative string
        :param allele: The :class:`~epytope.Core.Allele.Allele` for which the internal predictor representation is
                        needed
        :type alleles: :class:`~epytope.Core.Allele.Allele`
        :return: str
        """
        if isinstance(allele, MouseAllele):
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        scores = defaultdict(defaultdict)
        alleles = []
        with open(file, "r") as f:
            for row in f:
                if row[0] in ["#", "-"] or row.strip() == "" or "pos" in row:
                    continue
                else:
                    allele = row.split()[HLAIndex.PICKPOCKET_1_1].replace('*','')
                    pep = row.split()[PeptideIndex.PICKPOCKET_1_1]
                    score = float(row.split()[ScoreIndex.PICKPOCKET_1_1])
                    if allele not in alleles:
                        alleles.append(allele)

                    scores[allele][pep] = score

            result = {allele: {"Score": list(scores.values())[j]} for j, allele in enumerate(alleles)}

        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`elf.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into _file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(input))


class NetCTLpan_1_1(AExternalEpitopePrediction):
    """
    Interface for NetCTLpan 1.1.

    .. note::

        NetCTLpan - Pan-specific MHC class I epitope predictions Stranzl T., Larsen M. V., Lundegaard C., Nielsen M.
        Immunogenetics. 2010 Apr 9. [Epub ahead of print]
    """
    __name = "netctlpan"
    __command = "netCTLpan -f {peptides} -a {alleles} {options} > {out}"
    __supported_length = frozenset([8, 9, 10, 11])
    __alleles = frozenset(
        ['HLA-A*01:01', 'HLA-A*01:02', 'HLA-A*01:03', 'HLA-A*01:06', 'HLA-A*01:07', 'HLA-A*01:08', 'HLA-A*01:09', 'HLA-A*01:10', 'HLA-A*01:12',
         'HLA-A*01:13', 'HLA-A*01:14', 'HLA-A*01:17', 'HLA-A*01:19', 'HLA-A*01:20', 'HLA-A*01:21', 'HLA-A*01:23', 'HLA-A*01:24', 'HLA-A*01:25',
         'HLA-A*01:26', 'HLA-A*01:28', 'HLA-A*01:29', 'HLA-A*01:30', 'HLA-A*01:32', 'HLA-A*01:33', 'HLA-A*01:35', 'HLA-A*01:36', 'HLA-A*01:37',
         'HLA-A*01:38', 'HLA-A*01:39', 'HLA-A*01:40', 'HLA-A*01:41', 'HLA-A*01:42', 'HLA-A*01:43', 'HLA-A*01:44', 'HLA-A*01:45', 'HLA-A*01:46',
         'HLA-A*01:47', 'HLA-A*01:48', 'HLA-A*01:49', 'HLA-A*01:50', 'HLA-A*01:51', 'HLA-A*01:54', 'HLA-A*01:55', 'HLA-A*01:58', 'HLA-A*01:59',
         'HLA-A*01:60', 'HLA-A*01:61', 'HLA-A*01:62', 'HLA-A*01:63', 'HLA-A*01:64', 'HLA-A*01:65', 'HLA-A*01:66', 'HLA-A*02:01', 'HLA-A*02:02',
         'HLA-A*02:03', 'HLA-A*02:04', 'HLA-A*02:05', 'HLA-A*02:06', 'HLA-A*02:07', 'HLA-A*02:08', 'HLA-A*02:09', 'HLA-A*02:10', 'HLA-A*02:101',
         'HLA-A*02:102', 'HLA-A*02:103', 'HLA-A*02:104', 'HLA-A*02:105', 'HLA-A*02:106', 'HLA-A*02:107', 'HLA-A*02:108', 'HLA-A*02:109',
         'HLA-A*02:11', 'HLA-A*02:110', 'HLA-A*02:111', 'HLA-A*02:112', 'HLA-A*02:114', 'HLA-A*02:115', 'HLA-A*02:116', 'HLA-A*02:117',
         'HLA-A*02:118', 'HLA-A*02:119', 'HLA-A*02:12', 'HLA-A*02:120', 'HLA-A*02:121', 'HLA-A*02:122', 'HLA-A*02:123', 'HLA-A*02:124',
         'HLA-A*02:126', 'HLA-A*02:127', 'HLA-A*02:128', 'HLA-A*02:129', 'HLA-A*02:13', 'HLA-A*02:130', 'HLA-A*02:131', 'HLA-A*02:132',
         'HLA-A*02:133', 'HLA-A*02:134', 'HLA-A*02:135', 'HLA-A*02:136', 'HLA-A*02:137', 'HLA-A*02:138', 'HLA-A*02:139', 'HLA-A*02:14',
         'HLA-A*02:140', 'HLA-A*02:141', 'HLA-A*02:142', 'HLA-A*02:143', 'HLA-A*02:144', 'HLA-A*02:145', 'HLA-A*02:146', 'HLA-A*02:147',
         'HLA-A*02:148', 'HLA-A*02:149', 'HLA-A*02:150', 'HLA-A*02:151', 'HLA-A*02:152', 'HLA-A*02:153', 'HLA-A*02:154', 'HLA-A*02:155',
         'HLA-A*02:156', 'HLA-A*02:157', 'HLA-A*02:158', 'HLA-A*02:159', 'HLA-A*02:16', 'HLA-A*02:160', 'HLA-A*02:161', 'HLA-A*02:162',
         'HLA-A*02:163', 'HLA-A*02:164', 'HLA-A*02:165', 'HLA-A*02:166', 'HLA-A*02:167', 'HLA-A*02:168', 'HLA-A*02:169', 'HLA-A*02:17',
         'HLA-A*02:170', 'HLA-A*02:171', 'HLA-A*02:172', 'HLA-A*02:173', 'HLA-A*02:174', 'HLA-A*02:175', 'HLA-A*02:176', 'HLA-A*02:177',
         'HLA-A*02:178', 'HLA-A*02:179', 'HLA-A*02:18', 'HLA-A*02:180', 'HLA-A*02:181', 'HLA-A*02:182', 'HLA-A*02:183', 'HLA-A*02:184',
         'HLA-A*02:185', 'HLA-A*02:186', 'HLA-A*02:187', 'HLA-A*02:188', 'HLA-A*02:189', 'HLA-A*02:19', 'HLA-A*02:190', 'HLA-A*02:191',
         'HLA-A*02:192', 'HLA-A*02:193', 'HLA-A*02:194', 'HLA-A*02:195', 'HLA-A*02:196', 'HLA-A*02:197', 'HLA-A*02:198', 'HLA-A*02:199',
         'HLA-A*02:20', 'HLA-A*02:200', 'HLA-A*02:201', 'HLA-A*02:202', 'HLA-A*02:203', 'HLA-A*02:204', 'HLA-A*02:205', 'HLA-A*02:206',
         'HLA-A*02:207', 'HLA-A*02:208', 'HLA-A*02:209', 'HLA-A*02:21', 'HLA-A*02:210', 'HLA-A*02:211', 'HLA-A*02:212', 'HLA-A*02:213',
         'HLA-A*02:214', 'HLA-A*02:215', 'HLA-A*02:216', 'HLA-A*02:217', 'HLA-A*02:218', 'HLA-A*02:219', 'HLA-A*02:22', 'HLA-A*02:220',
         'HLA-A*02:221', 'HLA-A*02:224', 'HLA-A*02:228', 'HLA-A*02:229', 'HLA-A*02:230', 'HLA-A*02:231', 'HLA-A*02:232', 'HLA-A*02:233',
         'HLA-A*02:234', 'HLA-A*02:235', 'HLA-A*02:236', 'HLA-A*02:237', 'HLA-A*02:238', 'HLA-A*02:239', 'HLA-A*02:24', 'HLA-A*02:240',
         'HLA-A*02:241', 'HLA-A*02:242', 'HLA-A*02:243', 'HLA-A*02:244', 'HLA-A*02:245', 'HLA-A*02:246', 'HLA-A*02:247', 'HLA-A*02:248',
         'HLA-A*02:249', 'HLA-A*02:25', 'HLA-A*02:251', 'HLA-A*02:252', 'HLA-A*02:253', 'HLA-A*02:254', 'HLA-A*02:255', 'HLA-A*02:256',
         'HLA-A*02:257', 'HLA-A*02:258', 'HLA-A*02:259', 'HLA-A*02:26', 'HLA-A*02:260', 'HLA-A*02:261', 'HLA-A*02:262', 'HLA-A*02:263',
         'HLA-A*02:264', 'HLA-A*02:265', 'HLA-A*02:266', 'HLA-A*02:27', 'HLA-A*02:28', 'HLA-A*02:29', 'HLA-A*02:30', 'HLA-A*02:31', 'HLA-A*02:33',
         'HLA-A*02:34', 'HLA-A*02:35', 'HLA-A*02:36', 'HLA-A*02:37', 'HLA-A*02:38', 'HLA-A*02:39', 'HLA-A*02:40', 'HLA-A*02:41', 'HLA-A*02:42',
         'HLA-A*02:44', 'HLA-A*02:45', 'HLA-A*02:46', 'HLA-A*02:47', 'HLA-A*02:48', 'HLA-A*02:49', 'HLA-A*02:50', 'HLA-A*02:51', 'HLA-A*02:52',
         'HLA-A*02:54', 'HLA-A*02:55', 'HLA-A*02:56', 'HLA-A*02:57', 'HLA-A*02:58', 'HLA-A*02:59', 'HLA-A*02:60', 'HLA-A*02:61', 'HLA-A*02:62',
         'HLA-A*02:63', 'HLA-A*02:64', 'HLA-A*02:65', 'HLA-A*02:66', 'HLA-A*02:67', 'HLA-A*02:68', 'HLA-A*02:69', 'HLA-A*02:70', 'HLA-A*02:71',
         'HLA-A*02:72', 'HLA-A*02:73', 'HLA-A*02:74', 'HLA-A*02:75', 'HLA-A*02:76', 'HLA-A*02:77', 'HLA-A*02:78', 'HLA-A*02:79', 'HLA-A*02:80',
         'HLA-A*02:81', 'HLA-A*02:84', 'HLA-A*02:85', 'HLA-A*02:86', 'HLA-A*02:87', 'HLA-A*02:89', 'HLA-A*02:90', 'HLA-A*02:91', 'HLA-A*02:92',
         'HLA-A*02:93', 'HLA-A*02:95', 'HLA-A*02:96', 'HLA-A*02:97', 'HLA-A*02:99', 'HLA-A*03:01', 'HLA-A*03:02', 'HLA-A*03:04', 'HLA-A*03:05',
         'HLA-A*03:06', 'HLA-A*03:07', 'HLA-A*03:08', 'HLA-A*03:09', 'HLA-A*03:10', 'HLA-A*03:12', 'HLA-A*03:13', 'HLA-A*03:14', 'HLA-A*03:15',
         'HLA-A*03:16', 'HLA-A*03:17', 'HLA-A*03:18', 'HLA-A*03:19', 'HLA-A*03:20', 'HLA-A*03:22', 'HLA-A*03:23', 'HLA-A*03:24', 'HLA-A*03:25',
         'HLA-A*03:26', 'HLA-A*03:27', 'HLA-A*03:28', 'HLA-A*03:29', 'HLA-A*03:30', 'HLA-A*03:31', 'HLA-A*03:32', 'HLA-A*03:33', 'HLA-A*03:34',
         'HLA-A*03:35', 'HLA-A*03:37', 'HLA-A*03:38', 'HLA-A*03:39', 'HLA-A*03:40', 'HLA-A*03:41', 'HLA-A*03:42', 'HLA-A*03:43', 'HLA-A*03:44',
         'HLA-A*03:45', 'HLA-A*03:46', 'HLA-A*03:47', 'HLA-A*03:48', 'HLA-A*03:49', 'HLA-A*03:50', 'HLA-A*03:51', 'HLA-A*03:52', 'HLA-A*03:53',
         'HLA-A*03:54', 'HLA-A*03:55', 'HLA-A*03:56', 'HLA-A*03:57', 'HLA-A*03:58', 'HLA-A*03:59', 'HLA-A*03:60', 'HLA-A*03:61', 'HLA-A*03:62',
         'HLA-A*03:63', 'HLA-A*03:64', 'HLA-A*03:65', 'HLA-A*03:66', 'HLA-A*03:67', 'HLA-A*03:70', 'HLA-A*03:71', 'HLA-A*03:72', 'HLA-A*03:73',
         'HLA-A*03:74', 'HLA-A*03:75', 'HLA-A*03:76', 'HLA-A*03:77', 'HLA-A*03:78', 'HLA-A*03:79', 'HLA-A*03:80', 'HLA-A*03:81', 'HLA-A*03:82',
         'HLA-A*11:01', 'HLA-A*11:02', 'HLA-A*11:03', 'HLA-A*11:04', 'HLA-A*11:05', 'HLA-A*11:06', 'HLA-A*11:07', 'HLA-A*11:08', 'HLA-A*11:09',
         'HLA-A*11:10', 'HLA-A*11:11', 'HLA-A*11:12', 'HLA-A*11:13', 'HLA-A*11:14', 'HLA-A*11:15', 'HLA-A*11:16', 'HLA-A*11:17', 'HLA-A*11:18',
         'HLA-A*11:19', 'HLA-A*11:20', 'HLA-A*11:22', 'HLA-A*11:23', 'HLA-A*11:24', 'HLA-A*11:25', 'HLA-A*11:26', 'HLA-A*11:27', 'HLA-A*11:29',
         'HLA-A*11:30', 'HLA-A*11:31', 'HLA-A*11:32', 'HLA-A*11:33', 'HLA-A*11:34', 'HLA-A*11:35', 'HLA-A*11:36', 'HLA-A*11:37', 'HLA-A*11:38',
         'HLA-A*11:39', 'HLA-A*11:40', 'HLA-A*11:41', 'HLA-A*11:42', 'HLA-A*11:43', 'HLA-A*11:44', 'HLA-A*11:45', 'HLA-A*11:46', 'HLA-A*11:47',
         'HLA-A*11:48', 'HLA-A*11:49', 'HLA-A*11:51', 'HLA-A*11:53', 'HLA-A*11:54', 'HLA-A*11:55', 'HLA-A*11:56', 'HLA-A*11:57', 'HLA-A*11:58',
         'HLA-A*11:59', 'HLA-A*11:60', 'HLA-A*11:61', 'HLA-A*11:62', 'HLA-A*11:63', 'HLA-A*11:64', 'HLA-A*23:01', 'HLA-A*23:02', 'HLA-A*23:03',
         'HLA-A*23:04', 'HLA-A*23:05', 'HLA-A*23:06', 'HLA-A*23:09', 'HLA-A*23:10', 'HLA-A*23:12', 'HLA-A*23:13', 'HLA-A*23:14', 'HLA-A*23:15',
         'HLA-A*23:16', 'HLA-A*23:17', 'HLA-A*23:18', 'HLA-A*23:20', 'HLA-A*23:21', 'HLA-A*23:22', 'HLA-A*23:23', 'HLA-A*23:24', 'HLA-A*23:25',
         'HLA-A*23:26', 'HLA-A*24:02', 'HLA-A*24:03', 'HLA-A*24:04', 'HLA-A*24:05', 'HLA-A*24:06', 'HLA-A*24:07', 'HLA-A*24:08', 'HLA-A*24:10',
         'HLA-A*24:100', 'HLA-A*24:101', 'HLA-A*24:102', 'HLA-A*24:103', 'HLA-A*24:104', 'HLA-A*24:105', 'HLA-A*24:106', 'HLA-A*24:107',
         'HLA-A*24:108', 'HLA-A*24:109', 'HLA-A*24:110', 'HLA-A*24:111', 'HLA-A*24:112', 'HLA-A*24:113', 'HLA-A*24:114', 'HLA-A*24:115',
         'HLA-A*24:116', 'HLA-A*24:117', 'HLA-A*24:118', 'HLA-A*24:119', 'HLA-A*24:120', 'HLA-A*24:121', 'HLA-A*24:122', 'HLA-A*24:123',
         'HLA-A*24:124', 'HLA-A*24:125', 'HLA-A*24:126', 'HLA-A*24:127', 'HLA-A*24:128', 'HLA-A*24:129', 'HLA-A*24:13', 'HLA-A*24:130',
         'HLA-A*24:131', 'HLA-A*24:133', 'HLA-A*24:134', 'HLA-A*24:135', 'HLA-A*24:136', 'HLA-A*24:137', 'HLA-A*24:138', 'HLA-A*24:139',
         'HLA-A*24:14', 'HLA-A*24:140', 'HLA-A*24:141', 'HLA-A*24:142', 'HLA-A*24:143', 'HLA-A*24:144', 'HLA-A*24:15', 'HLA-A*24:17', 'HLA-A*24:18',
         'HLA-A*24:19', 'HLA-A*24:20', 'HLA-A*24:21', 'HLA-A*24:22', 'HLA-A*24:23', 'HLA-A*24:24', 'HLA-A*24:25', 'HLA-A*24:26', 'HLA-A*24:27',
         'HLA-A*24:28', 'HLA-A*24:29', 'HLA-A*24:30', 'HLA-A*24:31', 'HLA-A*24:32', 'HLA-A*24:33', 'HLA-A*24:34', 'HLA-A*24:35', 'HLA-A*24:37',
         'HLA-A*24:38', 'HLA-A*24:39', 'HLA-A*24:41', 'HLA-A*24:42', 'HLA-A*24:43', 'HLA-A*24:44', 'HLA-A*24:46', 'HLA-A*24:47', 'HLA-A*24:49',
         'HLA-A*24:50', 'HLA-A*24:51', 'HLA-A*24:52', 'HLA-A*24:53', 'HLA-A*24:54', 'HLA-A*24:55', 'HLA-A*24:56', 'HLA-A*24:57', 'HLA-A*24:58',
         'HLA-A*24:59', 'HLA-A*24:61', 'HLA-A*24:62', 'HLA-A*24:63', 'HLA-A*24:64', 'HLA-A*24:66', 'HLA-A*24:67', 'HLA-A*24:68', 'HLA-A*24:69',
         'HLA-A*24:70', 'HLA-A*24:71', 'HLA-A*24:72', 'HLA-A*24:73', 'HLA-A*24:74', 'HLA-A*24:75', 'HLA-A*24:76', 'HLA-A*24:77', 'HLA-A*24:78',
         'HLA-A*24:79', 'HLA-A*24:80', 'HLA-A*24:81', 'HLA-A*24:82', 'HLA-A*24:85', 'HLA-A*24:87', 'HLA-A*24:88', 'HLA-A*24:89', 'HLA-A*24:91',
         'HLA-A*24:92', 'HLA-A*24:93', 'HLA-A*24:94', 'HLA-A*24:95', 'HLA-A*24:96', 'HLA-A*24:97', 'HLA-A*24:98', 'HLA-A*24:99', 'HLA-A*25:01',
         'HLA-A*25:02', 'HLA-A*25:03', 'HLA-A*25:04', 'HLA-A*25:05', 'HLA-A*25:06', 'HLA-A*25:07', 'HLA-A*25:08', 'HLA-A*25:09', 'HLA-A*25:10',
         'HLA-A*25:11', 'HLA-A*25:13', 'HLA-A*26:01', 'HLA-A*26:02', 'HLA-A*26:03', 'HLA-A*26:04', 'HLA-A*26:05', 'HLA-A*26:06', 'HLA-A*26:07',
         'HLA-A*26:08', 'HLA-A*26:09', 'HLA-A*26:10', 'HLA-A*26:12', 'HLA-A*26:13', 'HLA-A*26:14', 'HLA-A*26:15', 'HLA-A*26:16', 'HLA-A*26:17',
         'HLA-A*26:18', 'HLA-A*26:19', 'HLA-A*26:20', 'HLA-A*26:21', 'HLA-A*26:22', 'HLA-A*26:23', 'HLA-A*26:24', 'HLA-A*26:26', 'HLA-A*26:27',
         'HLA-A*26:28', 'HLA-A*26:29', 'HLA-A*26:30', 'HLA-A*26:31', 'HLA-A*26:32', 'HLA-A*26:33', 'HLA-A*26:34', 'HLA-A*26:35', 'HLA-A*26:36',
         'HLA-A*26:37', 'HLA-A*26:38', 'HLA-A*26:39', 'HLA-A*26:40', 'HLA-A*26:41', 'HLA-A*26:42', 'HLA-A*26:43', 'HLA-A*26:45', 'HLA-A*26:46',
         'HLA-A*26:47', 'HLA-A*26:48', 'HLA-A*26:49', 'HLA-A*26:50', 'HLA-A*29:01', 'HLA-A*29:02', 'HLA-A*29:03', 'HLA-A*29:04', 'HLA-A*29:05',
         'HLA-A*29:06', 'HLA-A*29:07', 'HLA-A*29:09', 'HLA-A*29:10', 'HLA-A*29:11', 'HLA-A*29:12', 'HLA-A*29:13', 'HLA-A*29:14', 'HLA-A*29:15',
         'HLA-A*29:16', 'HLA-A*29:17', 'HLA-A*29:18', 'HLA-A*29:19', 'HLA-A*29:20', 'HLA-A*29:21', 'HLA-A*29:22', 'HLA-A*30:01', 'HLA-A*30:02',
         'HLA-A*30:03', 'HLA-A*30:04', 'HLA-A*30:06', 'HLA-A*30:07', 'HLA-A*30:08', 'HLA-A*30:09', 'HLA-A*30:10', 'HLA-A*30:11', 'HLA-A*30:12',
         'HLA-A*30:13', 'HLA-A*30:15', 'HLA-A*30:16', 'HLA-A*30:17', 'HLA-A*30:18', 'HLA-A*30:19', 'HLA-A*30:20', 'HLA-A*30:22', 'HLA-A*30:23',
         'HLA-A*30:24', 'HLA-A*30:25', 'HLA-A*30:26', 'HLA-A*30:28', 'HLA-A*30:29', 'HLA-A*30:30', 'HLA-A*30:31', 'HLA-A*30:32', 'HLA-A*30:33',
         'HLA-A*30:34', 'HLA-A*30:35', 'HLA-A*30:36', 'HLA-A*30:37', 'HLA-A*30:38', 'HLA-A*30:39', 'HLA-A*30:40', 'HLA-A*30:41', 'HLA-A*31:01',
         'HLA-A*31:02', 'HLA-A*31:03', 'HLA-A*31:04', 'HLA-A*31:05', 'HLA-A*31:06', 'HLA-A*31:07', 'HLA-A*31:08', 'HLA-A*31:09', 'HLA-A*31:10',
         'HLA-A*31:11', 'HLA-A*31:12', 'HLA-A*31:13', 'HLA-A*31:15', 'HLA-A*31:16', 'HLA-A*31:17', 'HLA-A*31:18', 'HLA-A*31:19', 'HLA-A*31:20',
         'HLA-A*31:21', 'HLA-A*31:22', 'HLA-A*31:23', 'HLA-A*31:24', 'HLA-A*31:25', 'HLA-A*31:26', 'HLA-A*31:27', 'HLA-A*31:28', 'HLA-A*31:29',
         'HLA-A*31:30', 'HLA-A*31:31', 'HLA-A*31:32', 'HLA-A*31:33', 'HLA-A*31:34', 'HLA-A*31:35', 'HLA-A*31:36', 'HLA-A*31:37', 'HLA-A*32:01',
         'HLA-A*32:02', 'HLA-A*32:03', 'HLA-A*32:04', 'HLA-A*32:05', 'HLA-A*32:06', 'HLA-A*32:07', 'HLA-A*32:08', 'HLA-A*32:09', 'HLA-A*32:10',
         'HLA-A*32:12', 'HLA-A*32:13', 'HLA-A*32:14', 'HLA-A*32:15', 'HLA-A*32:16', 'HLA-A*32:17', 'HLA-A*32:18', 'HLA-A*32:20', 'HLA-A*32:21',
         'HLA-A*32:22', 'HLA-A*32:23', 'HLA-A*32:24', 'HLA-A*32:25', 'HLA-A*33:01', 'HLA-A*33:03', 'HLA-A*33:04', 'HLA-A*33:05', 'HLA-A*33:06',
         'HLA-A*33:07', 'HLA-A*33:08', 'HLA-A*33:09', 'HLA-A*33:10', 'HLA-A*33:11', 'HLA-A*33:12', 'HLA-A*33:13', 'HLA-A*33:14', 'HLA-A*33:15',
         'HLA-A*33:16', 'HLA-A*33:17', 'HLA-A*33:18', 'HLA-A*33:19', 'HLA-A*33:20', 'HLA-A*33:21', 'HLA-A*33:22', 'HLA-A*33:23', 'HLA-A*33:24',
         'HLA-A*33:25', 'HLA-A*33:26', 'HLA-A*33:27', 'HLA-A*33:28', 'HLA-A*33:29', 'HLA-A*33:30', 'HLA-A*33:31', 'HLA-A*34:01', 'HLA-A*34:02',
         'HLA-A*34:03', 'HLA-A*34:04', 'HLA-A*34:05', 'HLA-A*34:06', 'HLA-A*34:07', 'HLA-A*34:08', 'HLA-A*36:01', 'HLA-A*36:02', 'HLA-A*36:03',
         'HLA-A*36:04', 'HLA-A*36:05', 'HLA-A*43:01', 'HLA-A*66:01', 'HLA-A*66:02', 'HLA-A*66:03', 'HLA-A*66:04', 'HLA-A*66:05', 'HLA-A*66:06',
         'HLA-A*66:07', 'HLA-A*66:08', 'HLA-A*66:09', 'HLA-A*66:10', 'HLA-A*66:11', 'HLA-A*66:12', 'HLA-A*66:13', 'HLA-A*66:14', 'HLA-A*66:15',
         'HLA-A*68:01', 'HLA-A*68:02', 'HLA-A*68:03', 'HLA-A*68:04', 'HLA-A*68:05', 'HLA-A*68:06', 'HLA-A*68:07', 'HLA-A*68:08', 'HLA-A*68:09',
         'HLA-A*68:10', 'HLA-A*68:12', 'HLA-A*68:13', 'HLA-A*68:14', 'HLA-A*68:15', 'HLA-A*68:16', 'HLA-A*68:17', 'HLA-A*68:19', 'HLA-A*68:20',
         'HLA-A*68:21', 'HLA-A*68:22', 'HLA-A*68:23', 'HLA-A*68:24', 'HLA-A*68:25', 'HLA-A*68:26', 'HLA-A*68:27', 'HLA-A*68:28', 'HLA-A*68:29',
         'HLA-A*68:30', 'HLA-A*68:31', 'HLA-A*68:32', 'HLA-A*68:33', 'HLA-A*68:34', 'HLA-A*68:35', 'HLA-A*68:36', 'HLA-A*68:37', 'HLA-A*68:38',
         'HLA-A*68:39', 'HLA-A*68:40', 'HLA-A*68:41', 'HLA-A*68:42', 'HLA-A*68:43', 'HLA-A*68:44', 'HLA-A*68:45', 'HLA-A*68:46', 'HLA-A*68:47',
         'HLA-A*68:48', 'HLA-A*68:50', 'HLA-A*68:51', 'HLA-A*68:52', 'HLA-A*68:53', 'HLA-A*68:54', 'HLA-A*69:01', 'HLA-A*74:01', 'HLA-A*74:02',
         'HLA-A*74:03', 'HLA-A*74:04', 'HLA-A*74:05', 'HLA-A*74:06', 'HLA-A*74:07', 'HLA-A*74:08', 'HLA-A*74:09', 'HLA-A*74:10', 'HLA-A*74:11',
         'HLA-A*74:13', 'HLA-A*80:01', 'HLA-A*80:02', 'HLA-B*07:02', 'HLA-B*07:03', 'HLA-B*07:04', 'HLA-B*07:05', 'HLA-B*07:06', 'HLA-B*07:07',
         'HLA-B*07:08', 'HLA-B*07:09', 'HLA-B*07:10', 'HLA-B*07:100', 'HLA-B*07:101', 'HLA-B*07:102', 'HLA-B*07:103', 'HLA-B*07:104',
         'HLA-B*07:105', 'HLA-B*07:106', 'HLA-B*07:107', 'HLA-B*07:108', 'HLA-B*07:109', 'HLA-B*07:11', 'HLA-B*07:110', 'HLA-B*07:112',
         'HLA-B*07:113', 'HLA-B*07:114', 'HLA-B*07:115', 'HLA-B*07:12', 'HLA-B*07:13', 'HLA-B*07:14', 'HLA-B*07:15', 'HLA-B*07:16', 'HLA-B*07:17',
         'HLA-B*07:18', 'HLA-B*07:19', 'HLA-B*07:20', 'HLA-B*07:21', 'HLA-B*07:22', 'HLA-B*07:23', 'HLA-B*07:24', 'HLA-B*07:25', 'HLA-B*07:26',
         'HLA-B*07:27', 'HLA-B*07:28', 'HLA-B*07:29', 'HLA-B*07:30', 'HLA-B*07:31', 'HLA-B*07:32', 'HLA-B*07:33', 'HLA-B*07:34', 'HLA-B*07:35',
         'HLA-B*07:36', 'HLA-B*07:37', 'HLA-B*07:38', 'HLA-B*07:39', 'HLA-B*07:40', 'HLA-B*07:41', 'HLA-B*07:42', 'HLA-B*07:43', 'HLA-B*07:44',
         'HLA-B*07:45', 'HLA-B*07:46', 'HLA-B*07:47', 'HLA-B*07:48', 'HLA-B*07:50', 'HLA-B*07:51', 'HLA-B*07:52', 'HLA-B*07:53', 'HLA-B*07:54',
         'HLA-B*07:55', 'HLA-B*07:56', 'HLA-B*07:57', 'HLA-B*07:58', 'HLA-B*07:59', 'HLA-B*07:60', 'HLA-B*07:61', 'HLA-B*07:62', 'HLA-B*07:63',
         'HLA-B*07:64', 'HLA-B*07:65', 'HLA-B*07:66', 'HLA-B*07:68', 'HLA-B*07:69', 'HLA-B*07:70', 'HLA-B*07:71', 'HLA-B*07:72', 'HLA-B*07:73',
         'HLA-B*07:74', 'HLA-B*07:75', 'HLA-B*07:76', 'HLA-B*07:77', 'HLA-B*07:78', 'HLA-B*07:79', 'HLA-B*07:80', 'HLA-B*07:81', 'HLA-B*07:82',
         'HLA-B*07:83', 'HLA-B*07:84', 'HLA-B*07:85', 'HLA-B*07:86', 'HLA-B*07:87', 'HLA-B*07:88', 'HLA-B*07:89', 'HLA-B*07:90', 'HLA-B*07:91',
         'HLA-B*07:92', 'HLA-B*07:93', 'HLA-B*07:94', 'HLA-B*07:95', 'HLA-B*07:96', 'HLA-B*07:97', 'HLA-B*07:98', 'HLA-B*07:99', 'HLA-B*08:01',
         'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*08:04', 'HLA-B*08:05', 'HLA-B*08:07', 'HLA-B*08:09', 'HLA-B*08:10', 'HLA-B*08:11', 'HLA-B*08:12',
         'HLA-B*08:13', 'HLA-B*08:14', 'HLA-B*08:15', 'HLA-B*08:16', 'HLA-B*08:17', 'HLA-B*08:18', 'HLA-B*08:20', 'HLA-B*08:21', 'HLA-B*08:22',
         'HLA-B*08:23', 'HLA-B*08:24', 'HLA-B*08:25', 'HLA-B*08:26', 'HLA-B*08:27', 'HLA-B*08:28', 'HLA-B*08:29', 'HLA-B*08:31', 'HLA-B*08:32',
         'HLA-B*08:33', 'HLA-B*08:34', 'HLA-B*08:35', 'HLA-B*08:36', 'HLA-B*08:37', 'HLA-B*08:38', 'HLA-B*08:39', 'HLA-B*08:40', 'HLA-B*08:41',
         'HLA-B*08:42', 'HLA-B*08:43', 'HLA-B*08:44', 'HLA-B*08:45', 'HLA-B*08:46', 'HLA-B*08:47', 'HLA-B*08:48', 'HLA-B*08:49', 'HLA-B*08:50',
         'HLA-B*08:51', 'HLA-B*08:52', 'HLA-B*08:53', 'HLA-B*08:54', 'HLA-B*08:55', 'HLA-B*08:56', 'HLA-B*08:57', 'HLA-B*08:58', 'HLA-B*08:59',
         'HLA-B*08:60', 'HLA-B*08:61', 'HLA-B*08:62', 'HLA-B*13:01', 'HLA-B*13:02', 'HLA-B*13:03', 'HLA-B*13:04', 'HLA-B*13:06', 'HLA-B*13:09',
         'HLA-B*13:10', 'HLA-B*13:11', 'HLA-B*13:12', 'HLA-B*13:13', 'HLA-B*13:14', 'HLA-B*13:15', 'HLA-B*13:16', 'HLA-B*13:17', 'HLA-B*13:18',
         'HLA-B*13:19', 'HLA-B*13:20', 'HLA-B*13:21', 'HLA-B*13:22', 'HLA-B*13:23', 'HLA-B*13:25', 'HLA-B*13:26', 'HLA-B*13:27', 'HLA-B*13:28',
         'HLA-B*13:29', 'HLA-B*13:30', 'HLA-B*13:31', 'HLA-B*13:32', 'HLA-B*13:33', 'HLA-B*13:34', 'HLA-B*13:35', 'HLA-B*13:36', 'HLA-B*13:37',
         'HLA-B*13:38', 'HLA-B*13:39', 'HLA-B*14:01', 'HLA-B*14:02', 'HLA-B*14:03', 'HLA-B*14:04', 'HLA-B*14:05', 'HLA-B*14:06', 'HLA-B*14:08',
         'HLA-B*14:09', 'HLA-B*14:10', 'HLA-B*14:11', 'HLA-B*14:12', 'HLA-B*14:13', 'HLA-B*14:14', 'HLA-B*14:15', 'HLA-B*14:16', 'HLA-B*14:17',
         'HLA-B*14:18', 'HLA-B*15:01', 'HLA-B*15:02', 'HLA-B*15:03', 'HLA-B*15:04', 'HLA-B*15:05', 'HLA-B*15:06', 'HLA-B*15:07', 'HLA-B*15:08',
         'HLA-B*15:09', 'HLA-B*15:10', 'HLA-B*15:101', 'HLA-B*15:102', 'HLA-B*15:103', 'HLA-B*15:104', 'HLA-B*15:105', 'HLA-B*15:106',
         'HLA-B*15:107', 'HLA-B*15:108', 'HLA-B*15:109', 'HLA-B*15:11', 'HLA-B*15:110', 'HLA-B*15:112', 'HLA-B*15:113', 'HLA-B*15:114',
         'HLA-B*15:115', 'HLA-B*15:116', 'HLA-B*15:117', 'HLA-B*15:118', 'HLA-B*15:119', 'HLA-B*15:12', 'HLA-B*15:120', 'HLA-B*15:121',
         'HLA-B*15:122', 'HLA-B*15:123', 'HLA-B*15:124', 'HLA-B*15:125', 'HLA-B*15:126', 'HLA-B*15:127', 'HLA-B*15:128', 'HLA-B*15:129',
         'HLA-B*15:13', 'HLA-B*15:131', 'HLA-B*15:132', 'HLA-B*15:133', 'HLA-B*15:134', 'HLA-B*15:135', 'HLA-B*15:136', 'HLA-B*15:137',
         'HLA-B*15:138', 'HLA-B*15:139', 'HLA-B*15:14', 'HLA-B*15:140', 'HLA-B*15:141', 'HLA-B*15:142', 'HLA-B*15:143', 'HLA-B*15:144',
         'HLA-B*15:145', 'HLA-B*15:146', 'HLA-B*15:147', 'HLA-B*15:148', 'HLA-B*15:15', 'HLA-B*15:150', 'HLA-B*15:151', 'HLA-B*15:152',
         'HLA-B*15:153', 'HLA-B*15:154', 'HLA-B*15:155', 'HLA-B*15:156', 'HLA-B*15:157', 'HLA-B*15:158', 'HLA-B*15:159', 'HLA-B*15:16',
         'HLA-B*15:160', 'HLA-B*15:161', 'HLA-B*15:162', 'HLA-B*15:163', 'HLA-B*15:164', 'HLA-B*15:165', 'HLA-B*15:166', 'HLA-B*15:167',
         'HLA-B*15:168', 'HLA-B*15:169', 'HLA-B*15:17', 'HLA-B*15:170', 'HLA-B*15:171', 'HLA-B*15:172', 'HLA-B*15:173', 'HLA-B*15:174',
         'HLA-B*15:175', 'HLA-B*15:176', 'HLA-B*15:177', 'HLA-B*15:178', 'HLA-B*15:179', 'HLA-B*15:18', 'HLA-B*15:180', 'HLA-B*15:183',
         'HLA-B*15:184', 'HLA-B*15:185', 'HLA-B*15:186', 'HLA-B*15:187', 'HLA-B*15:188', 'HLA-B*15:189', 'HLA-B*15:19', 'HLA-B*15:191',
         'HLA-B*15:192', 'HLA-B*15:193', 'HLA-B*15:194', 'HLA-B*15:195', 'HLA-B*15:196', 'HLA-B*15:197', 'HLA-B*15:198', 'HLA-B*15:199',
         'HLA-B*15:20', 'HLA-B*15:200', 'HLA-B*15:201', 'HLA-B*15:202', 'HLA-B*15:21', 'HLA-B*15:23', 'HLA-B*15:24', 'HLA-B*15:25', 'HLA-B*15:27',
         'HLA-B*15:28', 'HLA-B*15:29', 'HLA-B*15:30', 'HLA-B*15:31', 'HLA-B*15:32', 'HLA-B*15:33', 'HLA-B*15:34', 'HLA-B*15:35', 'HLA-B*15:36',
         'HLA-B*15:37', 'HLA-B*15:38', 'HLA-B*15:39', 'HLA-B*15:40', 'HLA-B*15:42', 'HLA-B*15:43', 'HLA-B*15:44', 'HLA-B*15:45', 'HLA-B*15:46',
         'HLA-B*15:47', 'HLA-B*15:48', 'HLA-B*15:49', 'HLA-B*15:50', 'HLA-B*15:51', 'HLA-B*15:52', 'HLA-B*15:53', 'HLA-B*15:54', 'HLA-B*15:55',
         'HLA-B*15:56', 'HLA-B*15:57', 'HLA-B*15:58', 'HLA-B*15:60', 'HLA-B*15:61', 'HLA-B*15:62', 'HLA-B*15:63', 'HLA-B*15:64', 'HLA-B*15:65',
         'HLA-B*15:66', 'HLA-B*15:67', 'HLA-B*15:68', 'HLA-B*15:69', 'HLA-B*15:70', 'HLA-B*15:71', 'HLA-B*15:72', 'HLA-B*15:73', 'HLA-B*15:74',
         'HLA-B*15:75', 'HLA-B*15:76', 'HLA-B*15:77', 'HLA-B*15:78', 'HLA-B*15:80', 'HLA-B*15:81', 'HLA-B*15:82', 'HLA-B*15:83', 'HLA-B*15:84',
         'HLA-B*15:85', 'HLA-B*15:86', 'HLA-B*15:87', 'HLA-B*15:88', 'HLA-B*15:89', 'HLA-B*15:90', 'HLA-B*15:91', 'HLA-B*15:92', 'HLA-B*15:93',
         'HLA-B*15:95', 'HLA-B*15:96', 'HLA-B*15:97', 'HLA-B*15:98', 'HLA-B*15:99', 'HLA-B*18:01', 'HLA-B*18:02', 'HLA-B*18:03', 'HLA-B*18:04',
         'HLA-B*18:05', 'HLA-B*18:06', 'HLA-B*18:07', 'HLA-B*18:08', 'HLA-B*18:09', 'HLA-B*18:10', 'HLA-B*18:11', 'HLA-B*18:12', 'HLA-B*18:13',
         'HLA-B*18:14', 'HLA-B*18:15', 'HLA-B*18:18', 'HLA-B*18:19', 'HLA-B*18:20', 'HLA-B*18:21', 'HLA-B*18:22', 'HLA-B*18:24', 'HLA-B*18:25',
         'HLA-B*18:26', 'HLA-B*18:27', 'HLA-B*18:28', 'HLA-B*18:29', 'HLA-B*18:30', 'HLA-B*18:31', 'HLA-B*18:32', 'HLA-B*18:33', 'HLA-B*18:34',
         'HLA-B*18:35', 'HLA-B*18:36', 'HLA-B*18:37', 'HLA-B*18:38', 'HLA-B*18:39', 'HLA-B*18:40', 'HLA-B*18:41', 'HLA-B*18:42', 'HLA-B*18:43',
         'HLA-B*18:44', 'HLA-B*18:45', 'HLA-B*18:46', 'HLA-B*18:47', 'HLA-B*18:48', 'HLA-B*18:49', 'HLA-B*18:50', 'HLA-B*27:01', 'HLA-B*27:02',
         'HLA-B*27:03', 'HLA-B*27:04', 'HLA-B*27:05', 'HLA-B*27:06', 'HLA-B*27:07', 'HLA-B*27:08', 'HLA-B*27:09', 'HLA-B*27:10', 'HLA-B*27:11',
         'HLA-B*27:12', 'HLA-B*27:13', 'HLA-B*27:14', 'HLA-B*27:15', 'HLA-B*27:16', 'HLA-B*27:17', 'HLA-B*27:18', 'HLA-B*27:19', 'HLA-B*27:20',
         'HLA-B*27:21', 'HLA-B*27:23', 'HLA-B*27:24', 'HLA-B*27:25', 'HLA-B*27:26', 'HLA-B*27:27', 'HLA-B*27:28', 'HLA-B*27:29', 'HLA-B*27:30',
         'HLA-B*27:31', 'HLA-B*27:32', 'HLA-B*27:33', 'HLA-B*27:34', 'HLA-B*27:35', 'HLA-B*27:36', 'HLA-B*27:37', 'HLA-B*27:38', 'HLA-B*27:39',
         'HLA-B*27:40', 'HLA-B*27:41', 'HLA-B*27:42', 'HLA-B*27:43', 'HLA-B*27:44', 'HLA-B*27:45', 'HLA-B*27:46', 'HLA-B*27:47', 'HLA-B*27:48',
         'HLA-B*27:49', 'HLA-B*27:50', 'HLA-B*27:51', 'HLA-B*27:52', 'HLA-B*27:53', 'HLA-B*27:54', 'HLA-B*27:55', 'HLA-B*27:56', 'HLA-B*27:57',
         'HLA-B*27:58', 'HLA-B*27:60', 'HLA-B*27:61', 'HLA-B*27:62', 'HLA-B*27:63', 'HLA-B*27:67', 'HLA-B*27:68', 'HLA-B*27:69', 'HLA-B*35:01',
         'HLA-B*35:02', 'HLA-B*35:03', 'HLA-B*35:04', 'HLA-B*35:05', 'HLA-B*35:06', 'HLA-B*35:07', 'HLA-B*35:08', 'HLA-B*35:09', 'HLA-B*35:10',
         'HLA-B*35:100', 'HLA-B*35:101', 'HLA-B*35:102', 'HLA-B*35:103', 'HLA-B*35:104', 'HLA-B*35:105', 'HLA-B*35:106', 'HLA-B*35:107',
         'HLA-B*35:108', 'HLA-B*35:109', 'HLA-B*35:11', 'HLA-B*35:110', 'HLA-B*35:111', 'HLA-B*35:112', 'HLA-B*35:113', 'HLA-B*35:114',
         'HLA-B*35:115', 'HLA-B*35:116', 'HLA-B*35:117', 'HLA-B*35:118', 'HLA-B*35:119', 'HLA-B*35:12', 'HLA-B*35:120', 'HLA-B*35:121',
         'HLA-B*35:122', 'HLA-B*35:123', 'HLA-B*35:124', 'HLA-B*35:125', 'HLA-B*35:126', 'HLA-B*35:127', 'HLA-B*35:128', 'HLA-B*35:13',
         'HLA-B*35:131', 'HLA-B*35:132', 'HLA-B*35:133', 'HLA-B*35:135', 'HLA-B*35:136', 'HLA-B*35:137', 'HLA-B*35:138', 'HLA-B*35:139',
         'HLA-B*35:14', 'HLA-B*35:140', 'HLA-B*35:141', 'HLA-B*35:142', 'HLA-B*35:143', 'HLA-B*35:144', 'HLA-B*35:15', 'HLA-B*35:16', 'HLA-B*35:17',
         'HLA-B*35:18', 'HLA-B*35:19', 'HLA-B*35:20', 'HLA-B*35:21', 'HLA-B*35:22', 'HLA-B*35:23', 'HLA-B*35:24', 'HLA-B*35:25', 'HLA-B*35:26',
         'HLA-B*35:27', 'HLA-B*35:28', 'HLA-B*35:29', 'HLA-B*35:30', 'HLA-B*35:31', 'HLA-B*35:32', 'HLA-B*35:33', 'HLA-B*35:34', 'HLA-B*35:35',
         'HLA-B*35:36', 'HLA-B*35:37', 'HLA-B*35:38', 'HLA-B*35:39', 'HLA-B*35:41', 'HLA-B*35:42', 'HLA-B*35:43', 'HLA-B*35:44', 'HLA-B*35:45',
         'HLA-B*35:46', 'HLA-B*35:47', 'HLA-B*35:48', 'HLA-B*35:49', 'HLA-B*35:50', 'HLA-B*35:51', 'HLA-B*35:52', 'HLA-B*35:54', 'HLA-B*35:55',
         'HLA-B*35:56', 'HLA-B*35:57', 'HLA-B*35:58', 'HLA-B*35:59', 'HLA-B*35:60', 'HLA-B*35:61', 'HLA-B*35:62', 'HLA-B*35:63', 'HLA-B*35:64',
         'HLA-B*35:66', 'HLA-B*35:67', 'HLA-B*35:68', 'HLA-B*35:69', 'HLA-B*35:70', 'HLA-B*35:71', 'HLA-B*35:72', 'HLA-B*35:74', 'HLA-B*35:75',
         'HLA-B*35:76', 'HLA-B*35:77', 'HLA-B*35:78', 'HLA-B*35:79', 'HLA-B*35:80', 'HLA-B*35:81', 'HLA-B*35:82', 'HLA-B*35:83', 'HLA-B*35:84',
         'HLA-B*35:85', 'HLA-B*35:86', 'HLA-B*35:87', 'HLA-B*35:88', 'HLA-B*35:89', 'HLA-B*35:90', 'HLA-B*35:91', 'HLA-B*35:92', 'HLA-B*35:93',
         'HLA-B*35:94', 'HLA-B*35:95', 'HLA-B*35:96', 'HLA-B*35:97', 'HLA-B*35:98', 'HLA-B*35:99', 'HLA-B*37:01', 'HLA-B*37:02', 'HLA-B*37:04',
         'HLA-B*37:05', 'HLA-B*37:06', 'HLA-B*37:07', 'HLA-B*37:08', 'HLA-B*37:09', 'HLA-B*37:10', 'HLA-B*37:11', 'HLA-B*37:12', 'HLA-B*37:13',
         'HLA-B*37:14', 'HLA-B*37:15', 'HLA-B*37:17', 'HLA-B*37:18', 'HLA-B*37:19', 'HLA-B*37:20', 'HLA-B*37:21', 'HLA-B*37:22', 'HLA-B*37:23',
         'HLA-B*38:01', 'HLA-B*38:02', 'HLA-B*38:03', 'HLA-B*38:04', 'HLA-B*38:05', 'HLA-B*38:06', 'HLA-B*38:07', 'HLA-B*38:08', 'HLA-B*38:09',
         'HLA-B*38:10', 'HLA-B*38:11', 'HLA-B*38:12', 'HLA-B*38:13', 'HLA-B*38:14', 'HLA-B*38:15', 'HLA-B*38:16', 'HLA-B*38:17', 'HLA-B*38:18',
         'HLA-B*38:19', 'HLA-B*38:20', 'HLA-B*38:21', 'HLA-B*38:22', 'HLA-B*38:23', 'HLA-B*39:01', 'HLA-B*39:02', 'HLA-B*39:03', 'HLA-B*39:04',
         'HLA-B*39:05', 'HLA-B*39:06', 'HLA-B*39:07', 'HLA-B*39:08', 'HLA-B*39:09', 'HLA-B*39:10', 'HLA-B*39:11', 'HLA-B*39:12', 'HLA-B*39:13',
         'HLA-B*39:14', 'HLA-B*39:15', 'HLA-B*39:16', 'HLA-B*39:17', 'HLA-B*39:18', 'HLA-B*39:19', 'HLA-B*39:20', 'HLA-B*39:22', 'HLA-B*39:23',
         'HLA-B*39:24', 'HLA-B*39:26', 'HLA-B*39:27', 'HLA-B*39:28', 'HLA-B*39:29', 'HLA-B*39:30', 'HLA-B*39:31', 'HLA-B*39:32', 'HLA-B*39:33',
         'HLA-B*39:34', 'HLA-B*39:35', 'HLA-B*39:36', 'HLA-B*39:37', 'HLA-B*39:39', 'HLA-B*39:41', 'HLA-B*39:42', 'HLA-B*39:43', 'HLA-B*39:44',
         'HLA-B*39:45', 'HLA-B*39:46', 'HLA-B*39:47', 'HLA-B*39:48', 'HLA-B*39:49', 'HLA-B*39:50', 'HLA-B*39:51', 'HLA-B*39:52', 'HLA-B*39:53',
         'HLA-B*39:54', 'HLA-B*39:55', 'HLA-B*39:56', 'HLA-B*39:57', 'HLA-B*39:58', 'HLA-B*39:59', 'HLA-B*39:60', 'HLA-B*40:01', 'HLA-B*40:02',
         'HLA-B*40:03', 'HLA-B*40:04', 'HLA-B*40:05', 'HLA-B*40:06', 'HLA-B*40:07', 'HLA-B*40:08', 'HLA-B*40:09', 'HLA-B*40:10', 'HLA-B*40:100',
         'HLA-B*40:101', 'HLA-B*40:102', 'HLA-B*40:103', 'HLA-B*40:104', 'HLA-B*40:105', 'HLA-B*40:106', 'HLA-B*40:107', 'HLA-B*40:108',
         'HLA-B*40:109', 'HLA-B*40:11', 'HLA-B*40:110', 'HLA-B*40:111', 'HLA-B*40:112', 'HLA-B*40:113', 'HLA-B*40:114', 'HLA-B*40:115',
         'HLA-B*40:116', 'HLA-B*40:117', 'HLA-B*40:119', 'HLA-B*40:12', 'HLA-B*40:120', 'HLA-B*40:121', 'HLA-B*40:122', 'HLA-B*40:123',
         'HLA-B*40:124', 'HLA-B*40:125', 'HLA-B*40:126', 'HLA-B*40:127', 'HLA-B*40:128', 'HLA-B*40:129', 'HLA-B*40:13', 'HLA-B*40:130',
         'HLA-B*40:131', 'HLA-B*40:132', 'HLA-B*40:134', 'HLA-B*40:135', 'HLA-B*40:136', 'HLA-B*40:137', 'HLA-B*40:138', 'HLA-B*40:139',
         'HLA-B*40:14', 'HLA-B*40:140', 'HLA-B*40:141', 'HLA-B*40:143', 'HLA-B*40:145', 'HLA-B*40:146', 'HLA-B*40:147', 'HLA-B*40:15',
         'HLA-B*40:16', 'HLA-B*40:18', 'HLA-B*40:19', 'HLA-B*40:20', 'HLA-B*40:21', 'HLA-B*40:23', 'HLA-B*40:24', 'HLA-B*40:25', 'HLA-B*40:26',
         'HLA-B*40:27', 'HLA-B*40:28', 'HLA-B*40:29', 'HLA-B*40:30', 'HLA-B*40:31', 'HLA-B*40:32', 'HLA-B*40:33', 'HLA-B*40:34', 'HLA-B*40:35',
         'HLA-B*40:36', 'HLA-B*40:37', 'HLA-B*40:38', 'HLA-B*40:39', 'HLA-B*40:40', 'HLA-B*40:42', 'HLA-B*40:43', 'HLA-B*40:44', 'HLA-B*40:45',
         'HLA-B*40:46', 'HLA-B*40:47', 'HLA-B*40:48', 'HLA-B*40:49', 'HLA-B*40:50', 'HLA-B*40:51', 'HLA-B*40:52', 'HLA-B*40:53', 'HLA-B*40:54',
         'HLA-B*40:55', 'HLA-B*40:56', 'HLA-B*40:57', 'HLA-B*40:58', 'HLA-B*40:59', 'HLA-B*40:60', 'HLA-B*40:61', 'HLA-B*40:62', 'HLA-B*40:63',
         'HLA-B*40:64', 'HLA-B*40:65', 'HLA-B*40:66', 'HLA-B*40:67', 'HLA-B*40:68', 'HLA-B*40:69', 'HLA-B*40:70', 'HLA-B*40:71', 'HLA-B*40:72',
         'HLA-B*40:73', 'HLA-B*40:74', 'HLA-B*40:75', 'HLA-B*40:76', 'HLA-B*40:77', 'HLA-B*40:78', 'HLA-B*40:79', 'HLA-B*40:80', 'HLA-B*40:81',
         'HLA-B*40:82', 'HLA-B*40:83', 'HLA-B*40:84', 'HLA-B*40:85', 'HLA-B*40:86', 'HLA-B*40:87', 'HLA-B*40:88', 'HLA-B*40:89', 'HLA-B*40:90',
         'HLA-B*40:91', 'HLA-B*40:92', 'HLA-B*40:93', 'HLA-B*40:94', 'HLA-B*40:95', 'HLA-B*40:96', 'HLA-B*40:97', 'HLA-B*40:98', 'HLA-B*40:99',
         'HLA-B*41:01', 'HLA-B*41:02', 'HLA-B*41:03', 'HLA-B*41:04', 'HLA-B*41:05', 'HLA-B*41:06', 'HLA-B*41:07', 'HLA-B*41:08', 'HLA-B*41:09',
         'HLA-B*41:10', 'HLA-B*41:11', 'HLA-B*41:12', 'HLA-B*42:01', 'HLA-B*42:02', 'HLA-B*42:04', 'HLA-B*42:05', 'HLA-B*42:06', 'HLA-B*42:07',
         'HLA-B*42:08', 'HLA-B*42:09', 'HLA-B*42:10', 'HLA-B*42:11', 'HLA-B*42:12', 'HLA-B*42:13', 'HLA-B*42:14', 'HLA-B*44:02', 'HLA-B*44:03',
         'HLA-B*44:04', 'HLA-B*44:05', 'HLA-B*44:06', 'HLA-B*44:07', 'HLA-B*44:08', 'HLA-B*44:09', 'HLA-B*44:10', 'HLA-B*44:100', 'HLA-B*44:101',
         'HLA-B*44:102', 'HLA-B*44:103', 'HLA-B*44:104', 'HLA-B*44:105', 'HLA-B*44:106', 'HLA-B*44:107', 'HLA-B*44:109', 'HLA-B*44:11',
         'HLA-B*44:110', 'HLA-B*44:12', 'HLA-B*44:13', 'HLA-B*44:14', 'HLA-B*44:15', 'HLA-B*44:16', 'HLA-B*44:17', 'HLA-B*44:18', 'HLA-B*44:20',
         'HLA-B*44:21', 'HLA-B*44:22', 'HLA-B*44:24', 'HLA-B*44:25', 'HLA-B*44:26', 'HLA-B*44:27', 'HLA-B*44:28', 'HLA-B*44:29', 'HLA-B*44:30',
         'HLA-B*44:31', 'HLA-B*44:32', 'HLA-B*44:33', 'HLA-B*44:34', 'HLA-B*44:35', 'HLA-B*44:36', 'HLA-B*44:37', 'HLA-B*44:38', 'HLA-B*44:39',
         'HLA-B*44:40', 'HLA-B*44:41', 'HLA-B*44:42', 'HLA-B*44:43', 'HLA-B*44:44', 'HLA-B*44:45', 'HLA-B*44:46', 'HLA-B*44:47', 'HLA-B*44:48',
         'HLA-B*44:49', 'HLA-B*44:50', 'HLA-B*44:51', 'HLA-B*44:53', 'HLA-B*44:54', 'HLA-B*44:55', 'HLA-B*44:57', 'HLA-B*44:59', 'HLA-B*44:60',
         'HLA-B*44:62', 'HLA-B*44:63', 'HLA-B*44:64', 'HLA-B*44:65', 'HLA-B*44:66', 'HLA-B*44:67', 'HLA-B*44:68', 'HLA-B*44:69', 'HLA-B*44:70',
         'HLA-B*44:71', 'HLA-B*44:72', 'HLA-B*44:73', 'HLA-B*44:74', 'HLA-B*44:75', 'HLA-B*44:76', 'HLA-B*44:77', 'HLA-B*44:78', 'HLA-B*44:79',
         'HLA-B*44:80', 'HLA-B*44:81', 'HLA-B*44:82', 'HLA-B*44:83', 'HLA-B*44:84', 'HLA-B*44:85', 'HLA-B*44:86', 'HLA-B*44:87', 'HLA-B*44:88',
         'HLA-B*44:89', 'HLA-B*44:90', 'HLA-B*44:91', 'HLA-B*44:92', 'HLA-B*44:93', 'HLA-B*44:94', 'HLA-B*44:95', 'HLA-B*44:96', 'HLA-B*44:97',
         'HLA-B*44:98', 'HLA-B*44:99', 'HLA-B*45:01', 'HLA-B*45:02', 'HLA-B*45:03', 'HLA-B*45:04', 'HLA-B*45:05', 'HLA-B*45:06', 'HLA-B*45:07',
         'HLA-B*45:08', 'HLA-B*45:09', 'HLA-B*45:10', 'HLA-B*45:11', 'HLA-B*45:12', 'HLA-B*46:01', 'HLA-B*46:02', 'HLA-B*46:03', 'HLA-B*46:04',
         'HLA-B*46:05', 'HLA-B*46:06', 'HLA-B*46:08', 'HLA-B*46:09', 'HLA-B*46:10', 'HLA-B*46:11', 'HLA-B*46:12', 'HLA-B*46:13', 'HLA-B*46:14',
         'HLA-B*46:16', 'HLA-B*46:17', 'HLA-B*46:18', 'HLA-B*46:19', 'HLA-B*46:20', 'HLA-B*46:21', 'HLA-B*46:22', 'HLA-B*46:23', 'HLA-B*46:24',
         'HLA-B*47:01', 'HLA-B*47:02', 'HLA-B*47:03', 'HLA-B*47:04', 'HLA-B*47:05', 'HLA-B*47:06', 'HLA-B*47:07', 'HLA-B*48:01', 'HLA-B*48:02',
         'HLA-B*48:03', 'HLA-B*48:04', 'HLA-B*48:05', 'HLA-B*48:06', 'HLA-B*48:07', 'HLA-B*48:08', 'HLA-B*48:09', 'HLA-B*48:10', 'HLA-B*48:11',
         'HLA-B*48:12', 'HLA-B*48:13', 'HLA-B*48:14', 'HLA-B*48:15', 'HLA-B*48:16', 'HLA-B*48:17', 'HLA-B*48:18', 'HLA-B*48:19', 'HLA-B*48:20',
         'HLA-B*48:21', 'HLA-B*48:22', 'HLA-B*48:23', 'HLA-B*49:01', 'HLA-B*49:02', 'HLA-B*49:03', 'HLA-B*49:04', 'HLA-B*49:05', 'HLA-B*49:06',
         'HLA-B*49:07', 'HLA-B*49:08', 'HLA-B*49:09', 'HLA-B*49:10', 'HLA-B*50:01', 'HLA-B*50:02', 'HLA-B*50:04', 'HLA-B*50:05', 'HLA-B*50:06',
         'HLA-B*50:07', 'HLA-B*50:08', 'HLA-B*50:09', 'HLA-B*51:01', 'HLA-B*51:02', 'HLA-B*51:03', 'HLA-B*51:04', 'HLA-B*51:05', 'HLA-B*51:06',
         'HLA-B*51:07', 'HLA-B*51:08', 'HLA-B*51:09', 'HLA-B*51:12', 'HLA-B*51:13', 'HLA-B*51:14', 'HLA-B*51:15', 'HLA-B*51:16', 'HLA-B*51:17',
         'HLA-B*51:18', 'HLA-B*51:19', 'HLA-B*51:20', 'HLA-B*51:21', 'HLA-B*51:22', 'HLA-B*51:23', 'HLA-B*51:24', 'HLA-B*51:26', 'HLA-B*51:28',
         'HLA-B*51:29', 'HLA-B*51:30', 'HLA-B*51:31', 'HLA-B*51:32', 'HLA-B*51:33', 'HLA-B*51:34', 'HLA-B*51:35', 'HLA-B*51:36', 'HLA-B*51:37',
         'HLA-B*51:38', 'HLA-B*51:39', 'HLA-B*51:40', 'HLA-B*51:42', 'HLA-B*51:43', 'HLA-B*51:45', 'HLA-B*51:46', 'HLA-B*51:48', 'HLA-B*51:49',
         'HLA-B*51:50', 'HLA-B*51:51', 'HLA-B*51:52', 'HLA-B*51:53', 'HLA-B*51:54', 'HLA-B*51:55', 'HLA-B*51:56', 'HLA-B*51:57', 'HLA-B*51:58',
         'HLA-B*51:59', 'HLA-B*51:60', 'HLA-B*51:61', 'HLA-B*51:62', 'HLA-B*51:63', 'HLA-B*51:64', 'HLA-B*51:65', 'HLA-B*51:66', 'HLA-B*51:67',
         'HLA-B*51:68', 'HLA-B*51:69', 'HLA-B*51:70', 'HLA-B*51:71', 'HLA-B*51:72', 'HLA-B*51:73', 'HLA-B*51:74', 'HLA-B*51:75', 'HLA-B*51:76',
         'HLA-B*51:77', 'HLA-B*51:78', 'HLA-B*51:79', 'HLA-B*51:80', 'HLA-B*51:81', 'HLA-B*51:82', 'HLA-B*51:83', 'HLA-B*51:84', 'HLA-B*51:85',
         'HLA-B*51:86', 'HLA-B*51:87', 'HLA-B*51:88', 'HLA-B*51:89', 'HLA-B*51:90', 'HLA-B*51:91', 'HLA-B*51:92', 'HLA-B*51:93', 'HLA-B*51:94',
         'HLA-B*51:95', 'HLA-B*51:96', 'HLA-B*52:01', 'HLA-B*52:02', 'HLA-B*52:03', 'HLA-B*52:04', 'HLA-B*52:05', 'HLA-B*52:06', 'HLA-B*52:07',
         'HLA-B*52:08', 'HLA-B*52:09', 'HLA-B*52:10', 'HLA-B*52:11', 'HLA-B*52:12', 'HLA-B*52:13', 'HLA-B*52:14', 'HLA-B*52:15', 'HLA-B*52:16',
         'HLA-B*52:17', 'HLA-B*52:18', 'HLA-B*52:19', 'HLA-B*52:20', 'HLA-B*52:21', 'HLA-B*53:01', 'HLA-B*53:02', 'HLA-B*53:03', 'HLA-B*53:04',
         'HLA-B*53:05', 'HLA-B*53:06', 'HLA-B*53:07', 'HLA-B*53:08', 'HLA-B*53:09', 'HLA-B*53:10', 'HLA-B*53:11', 'HLA-B*53:12', 'HLA-B*53:13',
         'HLA-B*53:14', 'HLA-B*53:15', 'HLA-B*53:16', 'HLA-B*53:17', 'HLA-B*53:18', 'HLA-B*53:19', 'HLA-B*53:20', 'HLA-B*53:21', 'HLA-B*53:22',
         'HLA-B*53:23', 'HLA-B*54:01', 'HLA-B*54:02', 'HLA-B*54:03', 'HLA-B*54:04', 'HLA-B*54:06', 'HLA-B*54:07', 'HLA-B*54:09', 'HLA-B*54:10',
         'HLA-B*54:11', 'HLA-B*54:12', 'HLA-B*54:13', 'HLA-B*54:14', 'HLA-B*54:15', 'HLA-B*54:16', 'HLA-B*54:17', 'HLA-B*54:18', 'HLA-B*54:19',
         'HLA-B*54:20', 'HLA-B*54:21', 'HLA-B*54:22', 'HLA-B*54:23', 'HLA-B*55:01', 'HLA-B*55:02', 'HLA-B*55:03', 'HLA-B*55:04', 'HLA-B*55:05',
         'HLA-B*55:07', 'HLA-B*55:08', 'HLA-B*55:09', 'HLA-B*55:10', 'HLA-B*55:11', 'HLA-B*55:12', 'HLA-B*55:13', 'HLA-B*55:14', 'HLA-B*55:15',
         'HLA-B*55:16', 'HLA-B*55:17', 'HLA-B*55:18', 'HLA-B*55:19', 'HLA-B*55:20', 'HLA-B*55:21', 'HLA-B*55:22', 'HLA-B*55:23', 'HLA-B*55:24',
         'HLA-B*55:25', 'HLA-B*55:26', 'HLA-B*55:27', 'HLA-B*55:28', 'HLA-B*55:29', 'HLA-B*55:30', 'HLA-B*55:31', 'HLA-B*55:32', 'HLA-B*55:33',
         'HLA-B*55:34', 'HLA-B*55:35', 'HLA-B*55:36', 'HLA-B*55:37', 'HLA-B*55:38', 'HLA-B*55:39', 'HLA-B*55:40', 'HLA-B*55:41', 'HLA-B*55:42',
         'HLA-B*55:43', 'HLA-B*56:01', 'HLA-B*56:02', 'HLA-B*56:03', 'HLA-B*56:04', 'HLA-B*56:05', 'HLA-B*56:06', 'HLA-B*56:07', 'HLA-B*56:08',
         'HLA-B*56:09', 'HLA-B*56:10', 'HLA-B*56:11', 'HLA-B*56:12', 'HLA-B*56:13', 'HLA-B*56:14', 'HLA-B*56:15', 'HLA-B*56:16', 'HLA-B*56:17',
         'HLA-B*56:18', 'HLA-B*56:20', 'HLA-B*56:21', 'HLA-B*56:22', 'HLA-B*56:23', 'HLA-B*56:24', 'HLA-B*56:25', 'HLA-B*56:26', 'HLA-B*56:27',
         'HLA-B*56:29', 'HLA-B*57:01', 'HLA-B*57:02', 'HLA-B*57:03', 'HLA-B*57:04', 'HLA-B*57:05', 'HLA-B*57:06', 'HLA-B*57:07', 'HLA-B*57:08',
         'HLA-B*57:09', 'HLA-B*57:10', 'HLA-B*57:11', 'HLA-B*57:12', 'HLA-B*57:13', 'HLA-B*57:14', 'HLA-B*57:15', 'HLA-B*57:16', 'HLA-B*57:17',
         'HLA-B*57:18', 'HLA-B*57:19', 'HLA-B*57:20', 'HLA-B*57:21', 'HLA-B*57:22', 'HLA-B*57:23', 'HLA-B*57:24', 'HLA-B*57:25', 'HLA-B*57:26',
         'HLA-B*57:27', 'HLA-B*57:29', 'HLA-B*57:30', 'HLA-B*57:31', 'HLA-B*57:32', 'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*58:04', 'HLA-B*58:05',
         'HLA-B*58:06', 'HLA-B*58:07', 'HLA-B*58:08', 'HLA-B*58:09', 'HLA-B*58:11', 'HLA-B*58:12', 'HLA-B*58:13', 'HLA-B*58:14', 'HLA-B*58:15',
         'HLA-B*58:16', 'HLA-B*58:18', 'HLA-B*58:19', 'HLA-B*58:20', 'HLA-B*58:21', 'HLA-B*58:22', 'HLA-B*58:23', 'HLA-B*58:24', 'HLA-B*58:25',
         'HLA-B*58:26', 'HLA-B*58:27', 'HLA-B*58:28', 'HLA-B*58:29', 'HLA-B*58:30', 'HLA-B*59:01', 'HLA-B*59:02', 'HLA-B*59:03', 'HLA-B*59:04',
         'HLA-B*59:05', 'HLA-B*67:01', 'HLA-B*67:02', 'HLA-B*73:01', 'HLA-B*73:02', 'HLA-B*78:01', 'HLA-B*78:02', 'HLA-B*78:03', 'HLA-B*78:04',
         'HLA-B*78:05', 'HLA-B*78:06', 'HLA-B*78:07', 'HLA-B*81:01', 'HLA-B*81:02', 'HLA-B*81:03', 'HLA-B*81:05', 'HLA-B*82:01', 'HLA-B*82:02',
         'HLA-B*82:03', 'HLA-B*83:01', 'HLA-C*01:02', 'HLA-C*01:03', 'HLA-C*01:04', 'HLA-C*01:05', 'HLA-C*01:06', 'HLA-C*01:07', 'HLA-C*01:08',
         'HLA-C*01:09', 'HLA-C*01:10', 'HLA-C*01:11', 'HLA-C*01:12', 'HLA-C*01:13', 'HLA-C*01:14', 'HLA-C*01:15', 'HLA-C*01:16', 'HLA-C*01:17',
         'HLA-C*01:18', 'HLA-C*01:19', 'HLA-C*01:20', 'HLA-C*01:21', 'HLA-C*01:22', 'HLA-C*01:23', 'HLA-C*01:24', 'HLA-C*01:25', 'HLA-C*01:26',
         'HLA-C*01:27', 'HLA-C*01:28', 'HLA-C*01:29', 'HLA-C*01:30', 'HLA-C*01:31', 'HLA-C*01:32', 'HLA-C*01:33', 'HLA-C*01:34', 'HLA-C*01:35',
         'HLA-C*01:36', 'HLA-C*01:38', 'HLA-C*01:39', 'HLA-C*01:40', 'HLA-C*02:02', 'HLA-C*02:03', 'HLA-C*02:04', 'HLA-C*02:05', 'HLA-C*02:06',
         'HLA-C*02:07', 'HLA-C*02:08', 'HLA-C*02:09', 'HLA-C*02:10', 'HLA-C*02:11', 'HLA-C*02:12', 'HLA-C*02:13', 'HLA-C*02:14', 'HLA-C*02:15',
         'HLA-C*02:16', 'HLA-C*02:17', 'HLA-C*02:18', 'HLA-C*02:19', 'HLA-C*02:20', 'HLA-C*02:21', 'HLA-C*02:22', 'HLA-C*02:23', 'HLA-C*02:24',
         'HLA-C*02:26', 'HLA-C*02:27', 'HLA-C*02:28', 'HLA-C*02:29', 'HLA-C*02:30', 'HLA-C*02:31', 'HLA-C*02:32', 'HLA-C*02:33', 'HLA-C*02:34',
         'HLA-C*02:35', 'HLA-C*02:36', 'HLA-C*02:37', 'HLA-C*02:39', 'HLA-C*02:40', 'HLA-C*03:01', 'HLA-C*03:02', 'HLA-C*03:03', 'HLA-C*03:04',
         'HLA-C*03:05', 'HLA-C*03:06', 'HLA-C*03:07', 'HLA-C*03:08', 'HLA-C*03:09', 'HLA-C*03:10', 'HLA-C*03:11', 'HLA-C*03:12', 'HLA-C*03:13',
         'HLA-C*03:14', 'HLA-C*03:15', 'HLA-C*03:16', 'HLA-C*03:17', 'HLA-C*03:18', 'HLA-C*03:19', 'HLA-C*03:21', 'HLA-C*03:23', 'HLA-C*03:24',
         'HLA-C*03:25', 'HLA-C*03:26', 'HLA-C*03:27', 'HLA-C*03:28', 'HLA-C*03:29', 'HLA-C*03:30', 'HLA-C*03:31', 'HLA-C*03:32', 'HLA-C*03:33',
         'HLA-C*03:34', 'HLA-C*03:35', 'HLA-C*03:36', 'HLA-C*03:37', 'HLA-C*03:38', 'HLA-C*03:39', 'HLA-C*03:40', 'HLA-C*03:41', 'HLA-C*03:42',
         'HLA-C*03:43', 'HLA-C*03:44', 'HLA-C*03:45', 'HLA-C*03:46', 'HLA-C*03:47', 'HLA-C*03:48', 'HLA-C*03:49', 'HLA-C*03:50', 'HLA-C*03:51',
         'HLA-C*03:52', 'HLA-C*03:53', 'HLA-C*03:54', 'HLA-C*03:55', 'HLA-C*03:56', 'HLA-C*03:57', 'HLA-C*03:58', 'HLA-C*03:59', 'HLA-C*03:60',
         'HLA-C*03:61', 'HLA-C*03:62', 'HLA-C*03:63', 'HLA-C*03:64', 'HLA-C*03:65', 'HLA-C*03:66', 'HLA-C*03:67', 'HLA-C*03:68', 'HLA-C*03:69',
         'HLA-C*03:70', 'HLA-C*03:71', 'HLA-C*03:72', 'HLA-C*03:73', 'HLA-C*03:74', 'HLA-C*03:75', 'HLA-C*03:76', 'HLA-C*03:77', 'HLA-C*03:78',
         'HLA-C*03:79', 'HLA-C*03:80', 'HLA-C*03:81', 'HLA-C*03:82', 'HLA-C*03:83', 'HLA-C*03:84', 'HLA-C*03:85', 'HLA-C*03:86', 'HLA-C*03:87',
         'HLA-C*03:88', 'HLA-C*03:89', 'HLA-C*03:90', 'HLA-C*03:91', 'HLA-C*03:92', 'HLA-C*03:93', 'HLA-C*03:94', 'HLA-C*04:01', 'HLA-C*04:03',
         'HLA-C*04:04', 'HLA-C*04:05', 'HLA-C*04:06', 'HLA-C*04:07', 'HLA-C*04:08', 'HLA-C*04:10', 'HLA-C*04:11', 'HLA-C*04:12', 'HLA-C*04:13',
         'HLA-C*04:14', 'HLA-C*04:15', 'HLA-C*04:16', 'HLA-C*04:17', 'HLA-C*04:18', 'HLA-C*04:19', 'HLA-C*04:20', 'HLA-C*04:23', 'HLA-C*04:24',
         'HLA-C*04:25', 'HLA-C*04:26', 'HLA-C*04:27', 'HLA-C*04:28', 'HLA-C*04:29', 'HLA-C*04:30', 'HLA-C*04:31', 'HLA-C*04:32', 'HLA-C*04:33',
         'HLA-C*04:34', 'HLA-C*04:35', 'HLA-C*04:36', 'HLA-C*04:37', 'HLA-C*04:38', 'HLA-C*04:39', 'HLA-C*04:40', 'HLA-C*04:41', 'HLA-C*04:42',
         'HLA-C*04:43', 'HLA-C*04:44', 'HLA-C*04:45', 'HLA-C*04:46', 'HLA-C*04:47', 'HLA-C*04:48', 'HLA-C*04:49', 'HLA-C*04:50', 'HLA-C*04:51',
         'HLA-C*04:52', 'HLA-C*04:53', 'HLA-C*04:54', 'HLA-C*04:55', 'HLA-C*04:56', 'HLA-C*04:57', 'HLA-C*04:58', 'HLA-C*04:60', 'HLA-C*04:61',
         'HLA-C*04:62', 'HLA-C*04:63', 'HLA-C*04:64', 'HLA-C*04:65', 'HLA-C*04:66', 'HLA-C*04:67', 'HLA-C*04:68', 'HLA-C*04:69', 'HLA-C*04:70',
         'HLA-C*05:01', 'HLA-C*05:03', 'HLA-C*05:04', 'HLA-C*05:05', 'HLA-C*05:06', 'HLA-C*05:08', 'HLA-C*05:09', 'HLA-C*05:10', 'HLA-C*05:11',
         'HLA-C*05:12', 'HLA-C*05:13', 'HLA-C*05:14', 'HLA-C*05:15', 'HLA-C*05:16', 'HLA-C*05:17', 'HLA-C*05:18', 'HLA-C*05:19', 'HLA-C*05:20',
         'HLA-C*05:21', 'HLA-C*05:22', 'HLA-C*05:23', 'HLA-C*05:24', 'HLA-C*05:25', 'HLA-C*05:26', 'HLA-C*05:27', 'HLA-C*05:28', 'HLA-C*05:29',
         'HLA-C*05:30', 'HLA-C*05:31', 'HLA-C*05:32', 'HLA-C*05:33', 'HLA-C*05:34', 'HLA-C*05:35', 'HLA-C*05:36', 'HLA-C*05:37', 'HLA-C*05:38',
         'HLA-C*05:39', 'HLA-C*05:40', 'HLA-C*05:41', 'HLA-C*05:42', 'HLA-C*05:43', 'HLA-C*05:44', 'HLA-C*05:45', 'HLA-C*06:02', 'HLA-C*06:03',
         'HLA-C*06:04', 'HLA-C*06:05', 'HLA-C*06:06', 'HLA-C*06:07', 'HLA-C*06:08', 'HLA-C*06:09', 'HLA-C*06:10', 'HLA-C*06:11', 'HLA-C*06:12',
         'HLA-C*06:13', 'HLA-C*06:14', 'HLA-C*06:15', 'HLA-C*06:17', 'HLA-C*06:18', 'HLA-C*06:19', 'HLA-C*06:20', 'HLA-C*06:21', 'HLA-C*06:22',
         'HLA-C*06:23', 'HLA-C*06:24', 'HLA-C*06:25', 'HLA-C*06:26', 'HLA-C*06:27', 'HLA-C*06:28', 'HLA-C*06:29', 'HLA-C*06:30', 'HLA-C*06:31',
         'HLA-C*06:32', 'HLA-C*06:33', 'HLA-C*06:34', 'HLA-C*06:35', 'HLA-C*06:36', 'HLA-C*06:37', 'HLA-C*06:38', 'HLA-C*06:39', 'HLA-C*06:40',
         'HLA-C*06:41', 'HLA-C*06:42', 'HLA-C*06:43', 'HLA-C*06:44', 'HLA-C*06:45', 'HLA-C*07:01', 'HLA-C*07:02', 'HLA-C*07:03', 'HLA-C*07:04',
         'HLA-C*07:05', 'HLA-C*07:06', 'HLA-C*07:07', 'HLA-C*07:08', 'HLA-C*07:09', 'HLA-C*07:10', 'HLA-C*07:100', 'HLA-C*07:101', 'HLA-C*07:102',
         'HLA-C*07:103', 'HLA-C*07:105', 'HLA-C*07:106', 'HLA-C*07:107', 'HLA-C*07:108', 'HLA-C*07:109', 'HLA-C*07:11', 'HLA-C*07:110',
         'HLA-C*07:111', 'HLA-C*07:112', 'HLA-C*07:113', 'HLA-C*07:114', 'HLA-C*07:115', 'HLA-C*07:116', 'HLA-C*07:117', 'HLA-C*07:118',
         'HLA-C*07:119', 'HLA-C*07:12', 'HLA-C*07:120', 'HLA-C*07:122', 'HLA-C*07:123', 'HLA-C*07:124', 'HLA-C*07:125', 'HLA-C*07:126',
         'HLA-C*07:127', 'HLA-C*07:128', 'HLA-C*07:129', 'HLA-C*07:13', 'HLA-C*07:130', 'HLA-C*07:131', 'HLA-C*07:132', 'HLA-C*07:133',
         'HLA-C*07:134', 'HLA-C*07:135', 'HLA-C*07:136', 'HLA-C*07:137', 'HLA-C*07:138', 'HLA-C*07:139', 'HLA-C*07:14', 'HLA-C*07:140',
         'HLA-C*07:141', 'HLA-C*07:142', 'HLA-C*07:143', 'HLA-C*07:144', 'HLA-C*07:145', 'HLA-C*07:146', 'HLA-C*07:147', 'HLA-C*07:148',
         'HLA-C*07:149', 'HLA-C*07:15', 'HLA-C*07:16', 'HLA-C*07:17', 'HLA-C*07:18', 'HLA-C*07:19', 'HLA-C*07:20', 'HLA-C*07:21', 'HLA-C*07:22',
         'HLA-C*07:23', 'HLA-C*07:24', 'HLA-C*07:25', 'HLA-C*07:26', 'HLA-C*07:27', 'HLA-C*07:28', 'HLA-C*07:29', 'HLA-C*07:30', 'HLA-C*07:31',
         'HLA-C*07:35', 'HLA-C*07:36', 'HLA-C*07:37', 'HLA-C*07:38', 'HLA-C*07:39', 'HLA-C*07:40', 'HLA-C*07:41', 'HLA-C*07:42', 'HLA-C*07:43',
         'HLA-C*07:44', 'HLA-C*07:45', 'HLA-C*07:46', 'HLA-C*07:47', 'HLA-C*07:48', 'HLA-C*07:49', 'HLA-C*07:50', 'HLA-C*07:51', 'HLA-C*07:52',
         'HLA-C*07:53', 'HLA-C*07:54', 'HLA-C*07:56', 'HLA-C*07:57', 'HLA-C*07:58', 'HLA-C*07:59', 'HLA-C*07:60', 'HLA-C*07:62', 'HLA-C*07:63',
         'HLA-C*07:64', 'HLA-C*07:65', 'HLA-C*07:66', 'HLA-C*07:67', 'HLA-C*07:68', 'HLA-C*07:69', 'HLA-C*07:70', 'HLA-C*07:71', 'HLA-C*07:72',
         'HLA-C*07:73', 'HLA-C*07:74', 'HLA-C*07:75', 'HLA-C*07:76', 'HLA-C*07:77', 'HLA-C*07:78', 'HLA-C*07:79', 'HLA-C*07:80', 'HLA-C*07:81',
         'HLA-C*07:82', 'HLA-C*07:83', 'HLA-C*07:84', 'HLA-C*07:85', 'HLA-C*07:86', 'HLA-C*07:87', 'HLA-C*07:88', 'HLA-C*07:89', 'HLA-C*07:90',
         'HLA-C*07:91', 'HLA-C*07:92', 'HLA-C*07:93', 'HLA-C*07:94', 'HLA-C*07:95', 'HLA-C*07:96', 'HLA-C*07:97', 'HLA-C*07:99', 'HLA-C*08:01',
         'HLA-C*08:02', 'HLA-C*08:03', 'HLA-C*08:04', 'HLA-C*08:05', 'HLA-C*08:06', 'HLA-C*08:07', 'HLA-C*08:08', 'HLA-C*08:09', 'HLA-C*08:10',
         'HLA-C*08:11', 'HLA-C*08:12', 'HLA-C*08:13', 'HLA-C*08:14', 'HLA-C*08:15', 'HLA-C*08:16', 'HLA-C*08:17', 'HLA-C*08:18', 'HLA-C*08:19',
         'HLA-C*08:20', 'HLA-C*08:21', 'HLA-C*08:22', 'HLA-C*08:23', 'HLA-C*08:24', 'HLA-C*08:25', 'HLA-C*08:27', 'HLA-C*08:28', 'HLA-C*08:29',
         'HLA-C*08:30', 'HLA-C*08:31', 'HLA-C*08:32', 'HLA-C*08:33', 'HLA-C*08:34', 'HLA-C*08:35', 'HLA-C*12:02', 'HLA-C*12:03', 'HLA-C*12:04',
         'HLA-C*12:05', 'HLA-C*12:06', 'HLA-C*12:07', 'HLA-C*12:08', 'HLA-C*12:09', 'HLA-C*12:10', 'HLA-C*12:11', 'HLA-C*12:12', 'HLA-C*12:13',
         'HLA-C*12:14', 'HLA-C*12:15', 'HLA-C*12:16', 'HLA-C*12:17', 'HLA-C*12:18', 'HLA-C*12:19', 'HLA-C*12:20', 'HLA-C*12:21', 'HLA-C*12:22',
         'HLA-C*12:23', 'HLA-C*12:24', 'HLA-C*12:25', 'HLA-C*12:26', 'HLA-C*12:27', 'HLA-C*12:28', 'HLA-C*12:29', 'HLA-C*12:30', 'HLA-C*12:31',
         'HLA-C*12:32', 'HLA-C*12:33', 'HLA-C*12:34', 'HLA-C*12:35', 'HLA-C*12:36', 'HLA-C*12:37', 'HLA-C*12:38', 'HLA-C*12:40', 'HLA-C*12:41',
         'HLA-C*12:43', 'HLA-C*12:44', 'HLA-C*14:02', 'HLA-C*14:03', 'HLA-C*14:04', 'HLA-C*14:05', 'HLA-C*14:06', 'HLA-C*14:08', 'HLA-C*14:09',
         'HLA-C*14:10', 'HLA-C*14:11', 'HLA-C*14:12', 'HLA-C*14:13', 'HLA-C*14:14', 'HLA-C*14:15', 'HLA-C*14:16', 'HLA-C*14:17', 'HLA-C*14:18',
         'HLA-C*14:19', 'HLA-C*14:20', 'HLA-C*15:02', 'HLA-C*15:03', 'HLA-C*15:04', 'HLA-C*15:05', 'HLA-C*15:06', 'HLA-C*15:07', 'HLA-C*15:08',
         'HLA-C*15:09', 'HLA-C*15:10', 'HLA-C*15:11', 'HLA-C*15:12', 'HLA-C*15:13', 'HLA-C*15:15', 'HLA-C*15:16', 'HLA-C*15:17', 'HLA-C*15:18',
         'HLA-C*15:19', 'HLA-C*15:20', 'HLA-C*15:21', 'HLA-C*15:22', 'HLA-C*15:23', 'HLA-C*15:24', 'HLA-C*15:25', 'HLA-C*15:26', 'HLA-C*15:27',
         'HLA-C*15:28', 'HLA-C*15:29', 'HLA-C*15:30', 'HLA-C*15:31', 'HLA-C*15:33', 'HLA-C*15:34', 'HLA-C*15:35', 'HLA-C*16:01', 'HLA-C*16:02',
         'HLA-C*16:04', 'HLA-C*16:06', 'HLA-C*16:07', 'HLA-C*16:08', 'HLA-C*16:09', 'HLA-C*16:10', 'HLA-C*16:11', 'HLA-C*16:12', 'HLA-C*16:13',
         'HLA-C*16:14', 'HLA-C*16:15', 'HLA-C*16:17', 'HLA-C*16:18', 'HLA-C*16:19', 'HLA-C*16:20', 'HLA-C*16:21', 'HLA-C*16:22', 'HLA-C*16:23',
         'HLA-C*16:24', 'HLA-C*16:25', 'HLA-C*16:26', 'HLA-C*17:01', 'HLA-C*17:02', 'HLA-C*17:03', 'HLA-C*17:04', 'HLA-C*17:05', 'HLA-C*17:06',
         'HLA-C*17:07', 'HLA-C*18:01', 'HLA-C*18:02', 'HLA-C*18:03', 'HLA-E*01:01', 'HLA-G*01:01', 'HLA-G*01:02', 'HLA-G*01:03', 'HLA-G*01:04',
         'HLA-G*01:06', 'HLA-G*01:07', 'HLA-G*01:08', 'HLA-G*01:09',
         'H2-Db', 'H2-Dd', 'H2-Kb', 'H2-Kd', 'H2-Kk', 'H2-Ld'])
    __version = "1.1"

    @property
    def version(self):
        """The version of the predictor"""
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
            return "H-2-%s%s%s" % (allele.locus, allele.supertype, allele.subtype)
        else:
            return "HLA-%s%s:%s" % (allele.locus, allele.supertype, allele.subtype)

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

    @property
    def supportedAlleles(self):
        """
        A list of supported :class:`~epytope.Core.Allele.Allele`
        """
        return self.__alleles

    @property
    def name(self):
        """The name of the predictor"""
        return self.__name

    @property
    def command(self):
        """
        Defines the commandline call for external tool
        """
        return self.__command

    @property
    def supportedLength(self):
        """
        A list of supported :class:`~epytope.Core.Peptide.Peptide` lengths
        """
        return self.__supported_length

    def parse_external_result(self, file):
        """
        Parses external results and returns the result containing the predictors string representation
        of alleles and peptides.

        :param str file: The file path or the external prediction results
        :return: A dictionary containing the prediction results
        :rtype: dict
        """
        scores = defaultdict(defaultdict)
        alleles = []
        with open(file, "r") as f:
            for l in f:
                if l.startswith("#") or l.startswith("-") or l.strip() == "":
                    continue
                row = l.strip().split()
                if not row[0].isdigit():
                    continue
            
                epitope = row[PeptideIndex.NETCTLPAN_1_1]
                # Allele input representation differs from output representation. Needs to be in input representation to parse the output properly
                allele = row[HLAIndex.NETCTLPAN_1_1].replace('*','')
                comb_score = float(row[ScoreIndex.NETCTLPAN_1_1])
                if allele not in alleles:
                    alleles.append(allele)

                scores[allele][epitope] = comb_score

        result = {allele: {"Score": list(scores.values())[j]} for j, allele in enumerate(alleles)}
        
        return result

    def get_external_version(self, path=None):
        """
        Returns the external version of the tool by executing
        >{command} --version

        might be dependent on the method and has to be overwritten
        therefore it is declared abstract to enforce the user to
        overwrite the method. The function in the base class can be called
        with super()

        :param str path: Optional specification of executable path if deviant from :attr:`self.__command`
        :return: The external version of the tool or None if tool does not support versioning
        :rtype: str
        """
        return None

    def prepare_input(self, input, file):
        """
        Prepares input for external tools and writes them to file in the specific format

        No return value!

        :param: list(str) input: The :class:`~epytope.Core.Peptide.Peptide` sequences to write into file
        :param File file: File-handler to input file for external tool
        """
        file.write("\n".join(">pepe_%i\n%s" % (i, p) for i, p in enumerate(input)))


class PeptideIndex(IntEnum):
    """
    Specifies the index of the peptide sequence from the parsed output format
    """
    NETMHC_3_0 = 2
    NETMHC_3_4 = 2
    NETMHC_4_0 = 1
    NETMHCPAN_2_4 = 1
    NETMHCPAN_2_8 = 1
    NETMHCPAN_3_0 = 1
    NETMHCPAN_4_0 = 1
    NETMHCPAN_4_1 = 1
    NETMHCSTABPAN_1_0 = 1
    NETMHCII_2_2 = 2
    NETMHCII_2_3 = 2
    NETMHCIIPAN_3_0 = 1
    NETMHCIIPAN_3_1 = 1
    NETMHCIIPAN_4_0 = 1
    PICKPOCKET_1_1 = 2
    NETCTLPAN_1_1 = 3

class ScoreIndex(IntEnum):
    """
    Specifies the score index from the parsed output format
    """
    NETMHC_3_0 = 2
    NETMHC_3_4 = 3
    NETMHCPAN_2_4 = 3
    NETMHCPAN_2_8 = 3
    NETMHCPAN_3_0 = 4
    NETMHCPAN_4_0 = 5
    NETMHCPAN_4_1 = 5
    NETMHCSTABPAN_1_0 = 6
    NETMHCII_2_2 = 4
    NETMHCII_2_3 = 5
    NETMHCIIPAN_3_0 = 3
    NETMHCIIPAN_3_1 = 3
    NETMHCIIPAN_4_0 = 3
    PICKPOCKET_1_1 = 4
    NETCTLPAN_1_1 = 7

class RankIndex(IntEnum):
    """
    Specifies the rank index from the parsed output format if there is a rank score provided by the predictor
    """
    NETMHCPAN_2_8 = 5
    NETMHCPAN_3_0 = 6
    NETMHCPAN_4_0 = 7
    NETMHCPAN_4_1 = 6
    NETMHCSTABPAN_1_0 = 5
    NETMHCII_2_3 = 7
    NETMHCIIPAN_3_0 = 5
    NETMHCIIPAN_3_1 = 5
    NETMHCIIPAN_4_0 = 5

class Offset(IntEnum):
    """
    Specifies the offset of columns for multiple predicted HLA-alleles in the given predictors in order to
    correctly access score and rank per HLA-allele
    """
    NETMHC_4_0 = 3
    NETMHCPAN_2_8 = 3
    NETMHCPAN_3_0 = 4
    NETMHCPAN_4_0 = 5
    NETMHCPAN_4_1 = 4
    NETMHCSTABPAN_1_0_W_SCORE = 8
    NETMHCSTABPAN_1_0_WO_SCORE = 3
    NETMHCIIPAN_3_0 = 3
    NETMHCIIPAN_3_1 = 3
    NETMHCIIPAN_4_0 = 1

class HLAIndex(IntEnum):
    """
    Specifies the HLA-allele index in the parsed output of the predictor
    """
    NETMHCII_2_2 = 0
    NETMHCII_2_3 = 0
    PICKPOCKET_1_1 = 1
    NETCTLPAN_1_1 = 2
