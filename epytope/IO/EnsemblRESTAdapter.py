# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: IO.EnsemblRESTAdapter
   :synopsis: ADBAdapter implementation using the Ensembl REST API
.. moduleauthor:: jonasscheid
"""

import logging
import time
from collections import deque

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from epytope.IO.ADBAdapter import ADBAdapter, EAdapterFields, EIdentifierTypes


class EnsemblRESTError(Exception):
    """Definitive failure talking to the Ensembl REST API (not a 'not found')."""


class EnsemblRateLimitError(EnsemblRESTError):
    """429 rate-limit retries exhausted."""


class EnsemblConnectionError(EnsemblRESTError):
    """Connection error or timeout talking to the Ensembl REST API."""


class _RateLimiter:
    """Sliding-window limiter enforcing several (max_calls, period_seconds) caps.

    Blocks in ``acquire()`` until making a call would not breach any window.
    Not thread-safe (the adapter is used single-threaded).
    """

    def __init__(self, limits):
        # limits: iterable of (max_calls, period_seconds)
        self._windows = [(n, p, deque()) for n, p in limits]

    def acquire(self):
        while True:
            now = time.monotonic()
            wait = 0.0
            for n, p, hits in self._windows:
                while hits and hits[0] <= now - p:
                    hits.popleft()
                if len(hits) >= n:
                    wait = max(wait, hits[0] + p - now)
            if wait <= 0:
                break
            time.sleep(wait)
        now = time.monotonic()
        for _, _, hits in self._windows:
            hits.append(now)


_DB_TO_SPECIES = {
    "hsapiens_gene_ensembl": "homo_sapiens",
    "mmusculus_gene_ensembl": "mus_musculus",
    "rnorvegicus_gene_ensembl": "rattus_norvegicus",
    "cfamiliaris_gene_ensembl": "canis_lupus_familiaris",
    "drerio_gene_ensembl": "danio_rerio",
    "ggallus_gene_ensembl": "gallus_gallus",
    "sscrofa_gene_ensembl": "sus_scrofa",
}


class EnsemblRESTAdapter(ADBAdapter):
    """
    Adapter using the Ensembl REST API (https://rest.ensembl.org) for
    gene/transcript/protein queries.

    More stable than BioMart, returns structured JSON, and supports
    both Ensembl and RefSeq identifiers.

    :param str server: Ensembl REST server URL. Use 'https://grch37.rest.ensembl.org'
                       for GRCh37 queries.
    :param str species: Species name (default: 'homo_sapiens')
    """

    def __init__(self, server='https://rest.ensembl.org', species='homo_sapiens',
                 max_requests_per_second=13, max_requests_per_hour=50000,
                 max_rate_limit_retries=3):
        self._server = server.rstrip('/')
        self._species = species
        self._max_rate_limit_retries = max_rate_limit_retries

        self._session = requests.Session()
        # urllib3 Retry handles 5xx + connection only; 429 is owned by the
        # manual loop in _request so it can honor Retry-After and raise a
        # precisely-typed exception on exhaustion.
        retry = Retry(
            total=3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            backoff_factor=3
        )
        self._session.mount('https://', HTTPAdapter(max_retries=retry))
        self._session.mount('http://', HTTPAdapter(max_retries=retry))

        self._rate_limiter = _RateLimiter([
            (max_requests_per_second, 1.0),
            (max_requests_per_hour, 3600.0),
        ])

        self._ids_cache = {}
        self._sequence_cache = {}
        self._gene_cache = {}

    @staticmethod
    def _strip_version(id_str):
        """Strip version suffix from Ensembl ID (e.g. ENST00000361221.8 -> ENST00000361221)."""
        return id_str.split('.')[0]

    def _resolve_species(self, kwargs):
        """Resolve species from _db kwarg or fall back to default."""
        _db = kwargs.get("_db")
        if _db and _db in _DB_TO_SPECIES:
            return _DB_TO_SPECIES[_db]
        return self._species

    def _resolve_refseq_to_ensembl(self, refseq_id, expected_type, species):
        """Resolve a RefSeq ID to an Ensembl ID via xrefs.

        :param str refseq_id: RefSeq ID (e.g. NP_001005353, NM_001005353)
        :param str expected_type: Expected xref type ('translation' for protein, 'transcript' for transcript)
        :param str species: Species name for xrefs endpoint
        :return: Ensembl ID string, or None
        """
        xrefs = self._request(f"/xrefs/symbol/{species}/{refseq_id}")
        if not xrefs:
            return None
        for x in xrefs:
            if x.get("type") == expected_type:
                return x.get("id")
        return None

    def _request(self, endpoint, params=None, content_type='application/json',
                 method='GET', data=None):
        """
        Make a rate-limited request to the Ensembl REST API.

        :return: parsed JSON (dict/list) or raw text for 2xx; None for a genuine
                 'not found' (HTTP 400/404).
        :raises EnsemblRateLimitError: 429 rate-limit retries exhausted.
        :raises EnsemblConnectionError: connection error or timeout.
        :raises EnsemblRESTError: other definitive HTTP failure (e.g. 5xx exhausted).
        """
        url = self._server + endpoint
        headers = {"Content-Type": content_type, "Accept": content_type}

        for attempt in range(self._max_rate_limit_retries + 1):
            self._rate_limiter.acquire()
            try:
                if method == 'POST':
                    resp = self._session.post(
                        url, headers={**headers, "Content-Type": "application/json"},
                        json=data, timeout=30)
                else:
                    resp = self._session.get(
                        url, headers=headers, params=params, timeout=30)
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                raise EnsemblConnectionError(
                    f"Ensembl REST request to {endpoint} failed: {e}") from e
            except requests.exceptions.RequestException as e:
                # Includes RetryError raised when urllib3 exhausts 5xx retries.
                raise EnsemblRESTError(
                    f"Ensembl REST request to {endpoint} failed: {e}") from e

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 1.0))
                logging.warning(
                    "Rate limited by Ensembl REST API, sleeping %.1fs", retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code in (400, 404):
                logging.warning("Ensembl REST API: no entry for %s (HTTP %d)",
                                endpoint, resp.status_code)
                return None

            if not resp.ok:
                raise EnsemblRESTError(
                    f"Ensembl REST API error {resp.status_code} for {endpoint}")

            if content_type == 'text/plain':
                return resp.text
            return resp.json()

        raise EnsemblRateLimitError(
            f"Ensembl REST API rate limit retries exhausted for {endpoint}")

    def get_product_sequence(self, product_id, **kwargs):
        """
        Fetches protein sequence for the given protein ID.

        :param str product_id: Ensembl protein ID (ENSP) or RefSeq protein ID (NP_)
        :keyword type: EIdentifierTypes (ENSEMBL, REFSEQ supported)
        :return: Protein sequence string, or None
        :rtype: str
        """
        id_type = kwargs.get("type", EIdentifierTypes.ENSEMBL)

        if id_type == EIdentifierTypes.UNIPROT:
            logging.warning("Could not infer the origin of product id %s", product_id)
            return None

        if product_id in self._sequence_cache:
            return self._sequence_cache[product_id]

        if id_type == EIdentifierTypes.REFSEQ:
            ensembl_id = self._resolve_refseq_to_ensembl(
                product_id, "translation", self._resolve_species(kwargs))
            if not ensembl_id:
                logging.warning("There seems to be no Protein sequence for %s", product_id)
                return None
            seq = self._request(f"/sequence/id/{ensembl_id}?type=protein",
                                content_type='text/plain')
        else:
            seq = self._request(f"/sequence/id/{self._strip_version(product_id)}?type=protein",
                                content_type='text/plain')

        if not seq:
            logging.warning("There seems to be no Protein sequence for %s", product_id)
            return None

        if seq.endswith('*'):
            seq = seq[:-1]

        self._sequence_cache[product_id] = seq
        return seq

    def get_transcript_sequence(self, transcript_id, **kwargs):
        """
        Fetches coding DNA sequence for the given transcript ID.

        :param str transcript_id: Ensembl (ENST) or RefSeq (NM_) transcript ID
        :keyword type: EIdentifierTypes (ENSEMBL, REFSEQ supported)
        :return: Coding sequence string, or None
        :rtype: str
        """
        id_type = kwargs.get("type", EIdentifierTypes.ENSEMBL)

        if transcript_id in self._sequence_cache:
            return self._sequence_cache[transcript_id]

        if id_type == EIdentifierTypes.REFSEQ:
            ensembl_id = self._resolve_refseq_to_ensembl(
                transcript_id, "transcript", self._resolve_species(kwargs))
            if not ensembl_id:
                logging.warning("No transcript sequence available for %s", transcript_id)
                return None
            seq = self._request(f"/sequence/id/{ensembl_id}?type=cds",
                                content_type='text/plain')
        else:
            seq = self._request(f"/sequence/id/{self._strip_version(transcript_id)}?type=cds",
                                content_type='text/plain')

        if not seq or 'Sequence unavailable' in seq:
            logging.warning("No transcript sequence available for %s", transcript_id)
            return None

        self._sequence_cache[transcript_id] = seq
        return seq

    def get_transcript_information(self, transcript_id, **kwargs):
        """
        Fetches transcript CDS, gene name, and strand.

        :param str transcript_id: Transcript ID
        :keyword type: EIdentifierTypes
        :return: dict {SEQ, GENE, STRAND} or None
        :rtype: dict
        """
        id_type = kwargs.get("type", EIdentifierTypes.ENSEMBL)

        if transcript_id in self._ids_cache:
            return self._ids_cache[transcript_id]

        # Get CDS
        cds = self.get_transcript_sequence(transcript_id, **kwargs)
        if not cds:
            logging.warning("No information available on transcript %s", transcript_id)
            return None

        # Get metadata (gene name, strand) via lookup
        if id_type == EIdentifierTypes.REFSEQ:
            ensembl_id = self._resolve_refseq_to_ensembl(
                transcript_id, "transcript", self._resolve_species(kwargs))
            if not ensembl_id:
                logging.warning("No information available on transcript %s", transcript_id)
                return None
            lookup = self._request(f"/lookup/id/{ensembl_id}?expand=0")
        else:
            lookup = self._request(f"/lookup/id/{self._strip_version(transcript_id)}?expand=0")

        if not lookup:
            logging.warning("No information available on transcript %s", transcript_id)
            return None

        gene_name = lookup.get("display_name", "")
        # display_name for transcripts includes the transcript name, not gene name
        # Use the Parent gene's display_name instead
        parent_gene_id = lookup.get("Parent")
        if parent_gene_id:
            gene_lookup = self._request(f"/lookup/id/{parent_gene_id}")
            if gene_lookup:
                gene_name = gene_lookup.get("display_name", gene_name)

        strand_int = lookup.get("strand", 1)
        strand = "-" if int(strand_int) < 0 else "+"

        info = {
            EAdapterFields.SEQ: cds,
            EAdapterFields.GENE: gene_name,
            EAdapterFields.STRAND: strand
        }
        self._ids_cache[transcript_id] = info
        return info

    def get_transcript_information_from_protein_id(self, protein_id, **kwargs):
        """
        Fetches transcript info from an Ensembl protein ID.

        :param str protein_id: Ensembl protein ID
        :return: dict {SEQ, GENE, STRAND} or None
        :rtype: dict
        """
        if protein_id in self._ids_cache:
            return self._ids_cache[protein_id]

        lookup = self._request(f"/lookup/id/{self._strip_version(protein_id)}")
        if not lookup:
            logging.warning("No entry found for id %s", protein_id)
            return None

        # Get parent transcript
        parent_id = lookup.get("Parent")
        if not parent_id:
            logging.warning("No entry found for id %s", protein_id)
            return None

        info = self.get_transcript_information(parent_id, **kwargs)
        if info:
            self._ids_cache[protein_id] = info
        return info

    def get_gene_by_position(self, chromosome, start, stop, **kwargs):
        """
        Fetches gene name at chromosomal position.

        :return: Gene name string, or None
        :rtype: str
        """
        cache_key = f"{chromosome}:{start}:{stop}"
        if cache_key in self._gene_cache:
            return self._gene_cache[cache_key]

        species = self._resolve_species(kwargs)
        result = self._request(
            f"/overlap/region/{species}/{chromosome}:{start}-{stop}?feature=gene"
        )

        if not result:
            logging.warning(
                "%s does not denote a known gene location",
                ','.join([str(chromosome), str(start), str(stop)])
            )
            return None

        gene_name = result[0].get("external_name", "")
        self._gene_cache[cache_key] = gene_name
        return gene_name

    def get_ensembl_ids_from_gene(self, gene_id, **kwargs):
        """
        Fetches transcript and protein IDs for a gene.

        :param str gene_id: Gene name or Ensembl gene ID
        :keyword type: EIdentifierTypes
        :return: List of dicts with PROTID, GENE, TRANSID, STRAND
        :rtype: list(dict)
        """
        id_type = kwargs.get("type", EIdentifierTypes.GENENAME)

        if gene_id in self._ids_cache:
            return self._ids_cache[gene_id]

        species = self._resolve_species(kwargs)

        if id_type == EIdentifierTypes.ENSEMBL:
            lookup = self._request(f"/lookup/id/{self._strip_version(gene_id)}?expand=1")
        elif id_type in (EIdentifierTypes.GENENAME, EIdentifierTypes.HGNC):
            lookup = self._request(f"/lookup/symbol/{species}/{gene_id}?expand=1")
        else:
            logging.warning("Could not infer the origin of gene id %s", gene_id)
            return None

        if not lookup:
            logging.warning("No entry found for id %s", gene_id)
            return None

        transcripts = lookup.get("Transcript", [])
        if not transcripts:
            logging.warning("No entry found for id %s", gene_id)
            return None

        gene_display = lookup.get("display_name", gene_id)
        strand_int = lookup.get("strand", 1)
        strand = "-" if int(strand_int) < 0 else "+"

        # MartsAdapter overwrites cache per row, keeping single-element list
        for t in transcripts:
            prot_id = ""
            translation = t.get("Translation")
            if translation:
                prot_id = translation.get("id", "")

            self._ids_cache[gene_id] = [{
                EAdapterFields.PROTID: prot_id,
                EAdapterFields.GENE: lookup.get("id", gene_display),
                EAdapterFields.TRANSID: t.get("id", ""),
                EAdapterFields.STRAND: strand
            }]

        return self._ids_cache[gene_id]

    def get_genes_from_location(self, chromosome, start, stop, **kwargs):
        """
        Fetches genes in a chromosomal region.

        :return: DataFrame with ensembl_gene_id, uniprot_gn_symbol, external_gene_name
        :rtype: pandas.DataFrame
        """
        species = self._resolve_species(kwargs)
        result = self._request(
            f"/overlap/region/{species}/{chromosome}:{start}-{stop}?feature=gene"
        )

        if not result:
            logging.warning("No identifiers found for specified region %s:%s:%s",
                            chromosome, start, stop)
            return None

        return pd.DataFrame([
            {"ensembl_gene_id": g.get("gene_id", g.get("id", "")),
             "uniprot_gn_symbol": "",
             "external_gene_name": g.get("external_name", "")}
            for g in result
        ])

    def get_protein_ids_from_transcripts(self, transcripts, **kwargs):
        """
        Fetches protein identifiers (Ensembl, RefSeq, UniProt) for transcript IDs.

        :param list transcripts: List of Ensembl transcript IDs
        :return: DataFrame with ensembl_id, refseq_id, uniprot_id, transcript_id
        :rtype: pandas.DataFrame
        """
        max_batch = kwargs.get("_max_request_length", 300)
        rows = []

        # Batch lookup transcripts (with expand=1 to get Translation)
        for i in range(0, len(transcripts), max_batch):
            chunk = transcripts[i:i + max_batch]
            stripped = [self._strip_version(t) for t in chunk]

            batch_result = self._request(
                "/lookup/id", method='POST',
                data={"ids": stripped, "expand": 1},
                content_type='application/json'
            )
            if not batch_result:
                continue

            for orig_id, stripped_id in zip(chunk, stripped):
                info = batch_result.get(stripped_id)
                if not info or info.get("object_type") != "Transcript":
                    continue

                translation = info.get("Translation")
                ensembl_prot_id = translation.get("id", "") if translation else ""

                refseq_id = ""
                uniprot_id = ""

                if ensembl_prot_id:
                    xrefs = self._request(f"/xrefs/id/{ensembl_prot_id}")
                    if xrefs:
                        for x in xrefs:
                            db = x.get("dbname", "")
                            if db == "RefSeq_peptide" and not refseq_id:
                                refseq_id = x.get("primary_id", "")
                            elif db in ("Uniprot/SWISSPROT", "UniProtKB/Swiss-Prot") and not uniprot_id:
                                uniprot_id = x.get("primary_id", "")

                rows.append({
                    "ensembl_id": ensembl_prot_id,
                    "refseq_id": refseq_id,
                    "uniprot_id": uniprot_id,
                    "transcript_id": stripped_id
                })

        if not rows:
            return None

        return pd.DataFrame(rows)

    def get_gene_names_from_ids(self, gene_ids, **kwargs):
        """
        Fetches gene names for Ensembl gene IDs.

        :param list gene_ids: List of gene IDs
        :return: DataFrame with gene_name, gene_id columns
        :rtype: pandas.DataFrame
        """
        stripped = [self._strip_version(g) for g in gene_ids]
        batch_result = self._request(
            "/lookup/id", method='POST',
            data={"ids": stripped},
            content_type='application/json'
        )

        if not batch_result:
            return None

        rows = []
        for orig_id, s_id in zip(gene_ids, stripped):
            info = batch_result.get(s_id)
            if info:
                rows.append({
                    "gene_name": info.get("display_name", ""),
                    "gene_id": orig_id
                })

        if not rows:
            return None

        return pd.DataFrame(rows)

    def get_variants_from_transcript_id(self, transcript_id, **kwargs):
        """
        Fetches variants overlapping a transcript.

        Note: The REST API returns a different schema than BioMart.
        This is a best-effort implementation.

        :return: DataFrame or None
        :rtype: pandas.DataFrame
        """
        result = self._request(f"/overlap/id/{self._strip_version(transcript_id)}?feature=variation")

        if not result:
            logging.warning("No variants found for %s", transcript_id)
            return None

        rows = []
        for var in result:
            rows.append({
                "variation_name": var.get("id", ""),
                "chromosome_name": var.get("seq_region_name", ""),
                "chromosome_start": var.get("start", ""),
                "chromosome_end": var.get("end", ""),
                "allele": var.get("alleles", ""),
                "snp_chromosome_strand": var.get("strand", ""),
            })

        if not rows:
            return None

        return pd.DataFrame(rows)
