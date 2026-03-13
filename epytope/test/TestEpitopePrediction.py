"""
Unittest for PSSM predictors
"""
__author__ = 'schubert'


import unittest

# Variants and Generator
from epytope.Core import Allele
from epytope.Core import Peptide

# Predictions
from epytope.EpitopePrediction import EpitopePredictorFactory, AExternalEpitopePrediction
from epytope.EpitopePrediction.ANN import AANNEpitopePrediction


def _mhcflurry_available():
    try:
        import mhcflurry
        return True
    except ImportError:
        return False


class TestCaseEpitopePrediction(unittest.TestCase):

    def setUp(self):
        #Peptides of different length 9,10,11,12,13,14,15
        self.peptides_mhcI = [Peptide("SYFPEITHI"), Peptide("IHTIEPFYS")]
        self.peptides_mhcII = [Peptide("SYFPEITHI"), Peptide("IHTIEPFYSAAAAAA")]
        self.mhcI = [Allele("HLA-B*15:01"), Allele("HLA-A*02:01")]
        self.mhcII = [Allele("HLA-DRB1*07:01"), Allele("HLA-DRB1*15:01")]
        self.mouse = [Allele("H2-Kd"), Allele("H2-Kb")]

    def _should_skip_model(self, model):
        """Skip external predictors and ANN predictors when mhcflurry is not installed."""
        if isinstance(model, AExternalEpitopePrediction):
            return True
        if isinstance(model, AANNEpitopePrediction) and not _mhcflurry_available():
            return True
        return False

    def test_multiple_peptide_input_mhcI(self):
            for m in EpitopePredictorFactory.available_methods():
                model = EpitopePredictorFactory(m)
                if not self._should_skip_model(model):
                    if all(a in model.supportedAlleles for a in self.mhcI):
                        res = model.predict(self.peptides_mhcI, alleles=self.mhcI)

    def test_single_peptide_input_mhcI(self):
            for m in EpitopePredictorFactory.available_methods():
                model = EpitopePredictorFactory(m)
                if not self._should_skip_model(model):
                    if all(a in model.supportedAlleles for a in self.mhcI):
                        res = model.predict(self.peptides_mhcI, alleles=self.mhcI)

    def test_multiple_peptide_input_mhcII(self):
            for m in EpitopePredictorFactory.available_methods():
                model = EpitopePredictorFactory(m)
                if not self._should_skip_model(model):
                    if all(a in model.supportedAlleles for a in self.mhcII) and m != "MHCIIMulti":
                        res = model.predict(self.peptides_mhcII, alleles=self.mhcII)

    def test_single_peptide_input_mhcII(self):
            for m in EpitopePredictorFactory.available_methods():
                model = EpitopePredictorFactory(m)
                if not self._should_skip_model(model):
                    if all(a in model.supportedAlleles for a in self.mhcII):
                        res = model.predict(self.peptides_mhcII, alleles=self.mhcII)
    
    def test_prediction_for_mouse(self):
        syfpeithi = EpitopePredictorFactory("Syfpeithi")
        res = syfpeithi.predict(self.peptides_mhcI, alleles=self.mouse)

        netmhc = EpitopePredictorFactory("netmhc")
        netmhcpan = EpitopePredictorFactory("netmhcpan")

        for allele in self.mouse:
            for m in [netmhc, netmhcpan]:
                self.assertTrue(allele in m.supportedAlleles)

    @unittest.skipUnless(
        _mhcflurry_available(),
        "mhcflurry not installed"
    )
    def test_mhcflurry_mouse_alleles(self):
        mhcflurry = EpitopePredictorFactory("mhcflurry")
        for allele in self.mouse:
            self.assertTrue(allele in mhcflurry.supportedAlleles)

if __name__ == '__main__':
    unittest.main()
