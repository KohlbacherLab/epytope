# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: IO.PyEnsemblAdapter
   :synopsis: ADBAdapter implementation using PyEnsembl for offline Ensembl queries
.. moduleauthor:: jonasscheid
"""

import logging

import pandas as pd

from epytope.IO.ADBAdapter import ADBAdapter, EAdapterFields, EIdentifierTypes

try:
    from pyensembl import EnsemblRelease
    HAS_PYENSEMBL = True
except ImportError:
    HAS_PYENSEMBL = False


class PyEnsemblAdapter(ADBAdapter):
    """
    Adapter using PyEnsembl for offline access to Ensembl gene/transcript/protein data.

    PyEnsembl downloads GTF + FASTA files from Ensembl FTP and indexes them into a local
    SQLite database. All queries after initial setup are fully offline.

    Only supports Ensembl identifiers (ENST, ENSP, ENSG). RefSeq and UniProt lookups
    are not available — use MartsAdapter or EnsemblRESTAdapter for those.

    :param int release: Ensembl release number (e.g. 75 for GRCh37, 112 for GRCh38)
    :param str species: Species name (default: 'human')
    :param bool auto_download: If True, download and index data on init (default: True)
    """

    def __init__(self, release=112, species='human', auto_download=True):
        if not HAS_PYENSEMBL:
            raise ImportError(
                "pyensembl is required for PyEnsemblAdapter. "
                "Install it with: pip install pyensembl"
            )
        self._genome = EnsemblRelease(release=release, species=species)
        if auto_download:
            self._genome.download()
            self._genome.index()

        self._ids_cache = {}
        self._sequence_cache = {}
        self._gene_cache = {}

    @staticmethod
    def _strip_version(id_str):
        """Strip version suffix from Ensembl ID (e.g. ENST00000361221.8 -> ENST00000361221)."""
        return id_str.split('.')[0]

    def _check_ensembl_only(self, id_type, id_str):
        """Return True if id_type is ENSEMBL, else warn and return False."""
        if id_type != EIdentifierTypes.ENSEMBL:
            logging.warning(
                "PyEnsemblAdapter only supports Ensembl IDs, got type %s for %s",
                id_type, id_str
            )
            return False
        return True

    def get_product_sequence(self, product_id, **kwargs):
        """
        Fetches protein sequence for the given Ensembl protein ID.

        :param str product_id: Ensembl protein ID (e.g. ENSP00000369497)
        :keyword type: Only EIdentifierTypes.ENSEMBL is supported
        :return: Protein sequence string, or None if not found
        :rtype: str
        """
        if not self._check_ensembl_only(kwargs.get("type", EIdentifierTypes.ENSEMBL), product_id):
            return None

        if product_id in self._sequence_cache:
            return self._sequence_cache[product_id]

        try:
            seq = self._genome.protein_sequence(self._strip_version(product_id))
        except (ValueError, KeyError):
            logging.warning("There seems to be no Protein sequence for %s", product_id)
            return None

        if seq and seq.endswith('*'):
            seq = seq[:-1]

        self._sequence_cache[product_id] = seq
        return seq

    def get_transcript_sequence(self, transcript_id, **kwargs):
        """
        Fetches coding DNA sequence for the given Ensembl transcript ID.

        :param str transcript_id: Ensembl transcript ID (e.g. ENST00000361221)
        :keyword type: Only EIdentifierTypes.ENSEMBL is supported
        :return: Coding sequence string, or None if not found
        :rtype: str
        """
        if not self._check_ensembl_only(kwargs.get("type", EIdentifierTypes.ENSEMBL), transcript_id):
            return None

        if transcript_id in self._sequence_cache:
            return self._sequence_cache[transcript_id]

        try:
            cds = self._genome.transcript_by_id(self._strip_version(transcript_id)).coding_sequence
        except (ValueError, KeyError):
            logging.warning("No transcript sequence available for %s", transcript_id)
            return None

        if not cds:
            logging.warning("No transcript sequence available for %s", transcript_id)
            return None

        self._sequence_cache[transcript_id] = cds
        return cds

    def get_transcript_information(self, transcript_id, **kwargs):
        """
        Fetches transcript sequence, gene name, and strand for the given Ensembl transcript ID.

        :param str transcript_id: Ensembl transcript ID
        :keyword type: Only EIdentifierTypes.ENSEMBL is supported
        :return: dict with EAdapterFields keys {SEQ, GENE, STRAND}, or None
        :rtype: dict
        """
        if not self._check_ensembl_only(kwargs.get("type", EIdentifierTypes.ENSEMBL), transcript_id):
            return None

        if transcript_id in self._ids_cache:
            return self._ids_cache[transcript_id]

        try:
            t = self._genome.transcript_by_id(self._strip_version(transcript_id))
            cds = t.coding_sequence
            gene = self._genome.gene_by_id(t.gene_id)
        except (ValueError, KeyError):
            logging.warning("No information available on transcript %s", transcript_id)
            return None

        if not cds:
            logging.warning("No information available on transcript %s", transcript_id)
            return None

        info = {
            EAdapterFields.SEQ: cds,
            EAdapterFields.GENE: gene.name,
            EAdapterFields.STRAND: t.strand
        }
        self._ids_cache[transcript_id] = info
        return info

    def get_transcript_information_from_protein_id(self, protein_id, **kwargs):
        """
        Fetches transcript information (CDS, gene, strand) from an Ensembl protein ID.

        :param str protein_id: Ensembl protein ID (e.g. ENSP00000355060)
        :return: dict with EAdapterFields keys, or None
        :rtype: dict
        """
        if protein_id in self._ids_cache:
            return self._ids_cache[protein_id]

        try:
            t = self._genome.transcript_by_protein_id(self._strip_version(protein_id))
        except (ValueError, KeyError):
            logging.warning("No entry found for id %s", protein_id)
            return None

        info = self.get_transcript_information(t.id, **kwargs)
        if info:
            self._ids_cache[protein_id] = info
        return info

    def get_gene_by_position(self, chromosome, start, stop, **kwargs):
        """
        Fetches gene name at the given chromosomal position.

        :param chromosome: Chromosome name (e.g. '17' or 17)
        :param start: Start position
        :param stop: Stop position
        :return: Gene name string, or None
        :rtype: str
        """
        cache_key = f"{chromosome}:{start}:{stop}"
        if cache_key in self._gene_cache:
            return self._gene_cache[cache_key]

        try:
            names = self._genome.gene_names_at_locus(
                contig=str(chromosome), position=int(start), end=int(stop)
            )
        except Exception:
            logging.warning("%s does not denote a known gene location",
                            f"{chromosome},{start},{stop}")
            return None

        if not names:
            logging.warning("%s does not denote a known gene location",
                            f"{chromosome},{start},{stop}")
            return None

        self._gene_cache[cache_key] = names[0]
        return names[0]

    def get_ensembl_ids_from_gene(self, gene_id, **kwargs):
        """
        Fetches transcript and protein IDs for a given gene name or Ensembl gene ID.

        :param str gene_id: Gene name (e.g. 'TP53') or Ensembl gene ID
        :keyword type: EIdentifierTypes.GENENAME, HGNC, or ENSEMBL
        :return: List of dicts with PROTID, GENE, TRANSID, STRAND keys
        :rtype: list(dict)
        """
        id_type = kwargs.get("type", EIdentifierTypes.GENENAME)

        if gene_id in self._ids_cache:
            return self._ids_cache[gene_id]

        try:
            if id_type == EIdentifierTypes.ENSEMBL:
                stripped = self._strip_version(gene_id)
                gene = self._genome.gene_by_id(stripped)
                gene_name = gene.name
                t_ids = self._genome.transcript_ids_of_gene_id(stripped)
            elif id_type in (EIdentifierTypes.GENENAME, EIdentifierTypes.HGNC):
                gene_name = gene_id
                t_ids = self._genome.transcript_ids_of_gene_name(gene_id)
            else:
                logging.warning("Could not infer the origin of gene id %s", gene_id)
                return None
        except (ValueError, KeyError):
            logging.warning("No entry found for id %s", gene_id)
            return None

        if not t_ids:
            logging.warning("No entry found for id %s", gene_id)
            return None

        # MartsAdapter overwrites cache per iteration (line 764-770), keeping only
        # the last transcript as a single-element list. Replicated for compatibility.
        for t_id in t_ids:
            t = self._genome.transcript_by_id(t_id)
            # GENE field: MartsAdapter uses query_filter value (ENSG ID for ENSEMBL
            # type, gene name for GENENAME), matching that convention here.
            self._ids_cache[gene_id] = [{
                EAdapterFields.PROTID: t.protein_id or '',
                EAdapterFields.GENE: gene_name if id_type == EIdentifierTypes.ENSEMBL else t.gene_id,
                EAdapterFields.TRANSID: t.id,
                EAdapterFields.STRAND: t.strand
            }]

        return self._ids_cache[gene_id]

    def get_genes_from_location(self, chromosome, start, stop, **kwargs):
        """
        Fetches genes in a chromosomal region.

        :return: DataFrame with ensembl_gene_id, uniprot_gn_symbol, external_gene_name
        :rtype: pandas.DataFrame
        """
        try:
            genes = self._genome.genes_at_locus(
                contig=str(chromosome), position=int(start), end=int(stop)
            )
        except Exception:
            logging.warning("No identifiers found for specified region %s:%s:%s",
                            chromosome, start, stop)
            return None

        if not genes:
            logging.warning("No identifiers found for specified region %s:%s:%s",
                            chromosome, start, stop)
            return None

        return pd.DataFrame([
            {"ensembl_gene_id": g.id, "uniprot_gn_symbol": "", "external_gene_name": g.name}
            for g in genes
        ])

    def get_gene_names_from_ids(self, gene_ids, **kwargs):
        """
        Fetches gene names for given Ensembl gene IDs.

        :param list gene_ids: List of Ensembl gene IDs
        :return: DataFrame with gene_name, gene_id columns
        :rtype: pandas.DataFrame
        """
        rows = []
        for gid in gene_ids:
            try:
                gene = self._genome.gene_by_id(self._strip_version(gid))
                rows.append({"gene_name": gene.name, "gene_id": gid})
            except (ValueError, KeyError):
                logging.warning("No gene found for id %s", gid)

        return pd.DataFrame(rows) if rows else None

    def get_protein_ids_from_transcripts(self, transcripts, **kwargs):
        """Not supported by PyEnsembl (no RefSeq/UniProt cross-references)."""
        logging.warning(
            "PyEnsemblAdapter does not support cross-reference lookups "
            "(RefSeq/UniProt). Use MartsAdapter or EnsemblRESTAdapter instead."
        )
        return None

    def get_variants_from_transcript_id(self, transcript_id, **kwargs):
        """Not supported by PyEnsembl (no variant data)."""
        logging.warning(
            "PyEnsemblAdapter does not support variant lookups. "
            "Use MartsAdapter or EnsemblRESTAdapter instead."
        )
        return None
