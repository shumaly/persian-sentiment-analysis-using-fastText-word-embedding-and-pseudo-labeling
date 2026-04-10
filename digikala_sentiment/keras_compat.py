try:
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
except ModuleNotFoundError:
    try:
        from keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D
        from keras.models import Sequential, load_model
        from keras.src.legacy.preprocessing.text import Tokenizer, tokenizer_from_json
        from keras.src.utils.sequence_utils import pad_sequences
    except ModuleNotFoundError:
        from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.saving.save import load_model
        from keras.src.legacy.preprocessing.text import Tokenizer, tokenizer_from_json
        from keras.src.utils.sequence_utils import pad_sequences
