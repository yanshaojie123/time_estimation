import pandas as pd
from gensim.models import Word2Vec
import numpy as np


class SkipGram_model:
    def __init__(self):

        # list of lists of regions (sequences)
        self.sentences = None

        self.w2v_model = None

    def get_travel_tracks(self, path):
        data = np.load(path)
        sentences = list(data[:, 2])
        sentences = [[str(j) for j in i]  for i in sentences]
        # sentences = []

        self.sentences = sentences

    def build_and_train(self, embed_size=32, window_size=5, workers=1, iter=50, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning skip gram models...")
        model = Word2Vec(**kwargs)
        print("Learning skip gram models done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, region_ids):
        """

        :param region_ids: list, ids of embedding regions (the type of id is str)
        :return:
        """
        if self.w2v_model is None:
            print("model not train")
            return {}

        embeddings = {}
        for region_id in region_ids:
            if region_id in self.w2v_model.wv:
                embeddings[region_id] = self.w2v_model.wv[region_id]

        return embeddings
