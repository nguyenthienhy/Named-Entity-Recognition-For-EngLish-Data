from src.data_loader.data_loader import *
from src.model.bilstm import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.generate_data_batch import *
from src.utils.evaluate_model import *
from path import *
import constant
import json

loader = DataLoader()

# train_sentences, train_entities, train_unique_entities = loader.load_and_clean_data(get_path(constant.ROOT_DIR +
#                                                                                               "/data/train.txt"))

# num_classes = len(train_unique_entities)
# entity_to_index = {t: i for i, t in enumerate(train_unique_entities)}

with open(get_path(constant.ROOT_DIR + '/data/entity_to_index.json'), 'r') as f:
    entity_to_index = json.load(f)

with open(get_path(constant.ROOT_DIR + '/data/num_classes.json'), 'r') as f:
    num_classes = json.load(f)

test_sentences, test_entities, _ = loader.load_and_clean_data(get_path(constant.ROOT_DIR + "/data/test.txt"))

test_sentences = pad_sequences(
            sequences=[sentence.split() for sentence in test_sentences],
            maxlen=constant.MAX_WORD_NUMBER,
            dtype=object,
            padding='post',
            truncating='post',
            value=constant.INPUT_PADDING
)

test_entities = pad_sequences(
    sequences=[entity.split() for entity in test_entities],
    maxlen=constant.MAX_WORD_NUMBER,
    dtype=object,
    padding='post',
    truncating='post',
    value=constant.TARGET_PADDING
)


test_generator = generate_data_batch(
    X=test_sentences,
    y=test_entities,
    entity_to_index=entity_to_index,
    num_classes=num_classes,
    batch_size=constant.BATCH_SIZE
)

feature_test = get_data_feature(test_sentences[10])
print(test_entities[10])

model = BiLSTM((constant.MAX_WORD_NUMBER, constant.WORD_VECTOR_SIZE),
                hidden_dim=constant.HIDDEN_DIM,
                num_classes=num_classes)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy", custom_f1, custom_precision, custom_recall])
model.load_weights(get_path(constant.ROOT_DIR + '/save_model/weights.04-0.10.h5'))

model.evaluate_generator(
    test_generator,
    verbose=1,
    steps=test_sentences.shape[0] // constant.BATCH_SIZE
)

pred = model.predict(feature_test)
y_pred = []
for i in range(52):
    y_pred.append(np.argmax(pred[0][i]))
print(y_pred)