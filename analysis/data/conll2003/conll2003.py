""" NER dataset compiled by T-NER library https://github.com/asahi417/tner/tree/master/tner """
import json
from itertools import chain
import datasets

logger = datasets.logging.get_logger(__name__)
_DESCRIPTION = """[CoNLL 2003 NER dataset](https://aclanthology.org/W03-0419/)"""
_NAME = "conll2003"
_VERSION = "1.0.0"
_CITATION = """
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.  and De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    url = "https://www.aclweb.org/anthology/W03-0419",
    pages = "142--147",
}
"""

_HOME_PAGE = "https://github.com/asahi417/tner"
_URL = f'https://huggingface.co/datasets/tner/{_NAME}/raw/main/dataset'
_URLS = {
    str(datasets.Split.TEST): [f'{_URL}/test.json'],
    str(datasets.Split.TRAIN): [f'{_URL}/train.json'],
    str(datasets.Split.VALIDATION): [f'{_URL}/valid.json'],
}


class Conll2003Config(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """BuilderConfig.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Conll2003Config, self).__init__(**kwargs)


class Conll2003(datasets.GeneratorBasedBuilder):
    """Dataset."""

    BUILDER_CONFIGS = [
        Conll2003Config(name=_NAME, version=datasets.Version(_VERSION), description=_DESCRIPTION),
    ]

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download_and_extract(_URLS)
        return [datasets.SplitGenerator(name=i, gen_kwargs={"filepaths": downloaded_file[str(i)]})
                for i in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]]

    def _generate_examples(self, filepaths):
        _key = 0
        for filepath in filepaths:
            logger.info(f"generating examples from = {filepath}")
            with open(filepath, encoding="utf-8") as f:
                _list = [i for i in f.read().split('\n') if len(i) > 0]
                for i in _list:
                    data = json.loads(i)
                    yield _key, data
                    _key += 1

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
        )
