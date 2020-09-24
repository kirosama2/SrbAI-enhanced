from model import train
from text_preprocessing import preprocess_corpus
from globals import GlobalConfig
from io_utils import serialize_object, deserialize_object
from similarity import word_similarity
from embedding import embed_word, embed_word_map


def model_training():
    """
    Example function showing how to train the model for a given language
    """
    # Step 1. Load a configuration with which you would like to train the model
    # To load a config just type the folder name of the model and language
    cfg = GlobalConfig("5gram_5sctx_300vs", "serbian")

    # Step 2. Preprocess the corpus and prepare it for the NN
    _, _, ngram_vectors, word_vectors, word_map = preprocess_corpus(
        cfg.language_config, cfg.model_config
    )

    # Optional Step. Serialize the ngram_vectors and word_map so that they can be used later
    # Keep in mind that the save_dir from the config must exist, the serialization won't check for that itself
    serialize_object(ngram_vectors, cfg.language_config.get_ngram_vectors_file_path())
    serialize_object(word_map, cfg.language_config.get_word_map_file_path())

    # Step 3. Train the NN
    # The result of training will be the weights that lead to the embedding layer
    # If plot graph is true, it will plot the loss graph as well
    vector_space = train(word_map, ngram_vectors, word_vector