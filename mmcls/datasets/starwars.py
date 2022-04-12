import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class SWdataset(BaseDataset):
    CLASSES = ["Admiral Ackbar", "Admiral Piett", "Anakin Skywalker", "Bail Organa", "BB-8", "Bib Fortuna", "Boba Fett", "Bodhi Rook", "C-3PO", "Captain Phasma", "Cassian Andor", "Chewbacca", "Dark Sidious", "Darth Maul", "Darth Vader", "Finn (FN-2187)", "General Grievous", "General Hux", "Grand Moff Tarkin", "Greedo", "Han Solo", "Jabba the Hutt", "Jango Fett", "Jar Jar Binks", "Jyn Erso", "K-2SO", "Kenobi", "Kylo Ren", "Lando Calrissian", "Luke Skywalker", "Mace Windu", "Maz Kanata", "Nien Nunb", "Obi-Wan", "Orson Krennic", "PadmÃ© Amidala", "Poe Dameron", "Princess Leia Organa", "Qi'ra", "Qui-Gon Jinn", "R2-D2", "Rey", "Rose Tico", "Saw Gerrera", "Supreme Leader Snoke", "Tobias Beckett", "Vice-Admiral Holdo", "Watto", "Wedge Antilles", "Wicket W. Warrick", "Yoda"]

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(':') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos
