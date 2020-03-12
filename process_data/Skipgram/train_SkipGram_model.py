from .SkipGram_model import SkipGram_model
# from utils.load_config import get_attribute
# from preprocess_data.get_travel_tracks import get_unique_regions_id

import os
import json
import sys


if __name__ == "__main__":

    data_path = "../../data/porto/portotrain/train.npy"
    embed_size = 32
    window_size = 5
    iteration = 50

    skip_gram_model = SkipGram_model()
    skip_gram_model.get_travel_tracks(path=data_path)
    # skip_gram_model.build_and_train(embed_size=get_attribute("region_embed_dim"), window_size=5, iter=50)
    skip_gram_model.build_and_train(embed_size=embed_size, window_size=window_size, iter=iteration)

    ids = [str(i) for i in range(1, 11343)]

    embeddings = skip_gram_model.get_embeddings(region_ids=ids)
    for key, value in embeddings.items():
        embeddings[int(key)] = value.tolist()

    import pickle
    with open('./edgesgdict.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    sys.exit(0)
