from .builder import ROTATED_DATASETS
from .dota import DOTADataset


@ROTATED_DATASETS.register_module()
class BrickKilnDataset(DOTADataset):
    CLASSES = ('CFCBK', 'FCBK', 'Zigzag')
    PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
