from src.data_loader.data_loader import *
from src.model.bilstm import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.generate_data_batch import *
from src.utils.evaluate_model import *
from path import *
import constant

loader = DataLoader()

train_sentences, train_entities, train_unique_entities = loader.load_and_clean_data(get_path(constant.ROOT_DIR +
                                                                                              "/data/train.txt"))

num_classes = len(train_unique_entities)
entity_to_index = {t: i for i, t in enumerate(train_unique_entities)}

val_sentences, val_entities, _ = loader.load_and_clean_data(get_path(constant.ROOT_DIR + "/data/valid.txt"))

train_sentences = pad_sequences(
            sequences=[sentence.split() for sentence in train_sentences],
            maxlen=constant.MAX_WORD_NUMBER,
            dtype=object,
            padding='post',
            truncating='post',
            value=constant.INPUT_PADDING
)
train_entities = pad_sequences(
    sequences=[entity.split() for entity in train_entities],
    maxlen=constant.MAX_WORD_NUMBER,
    dtype=object,
    padding='post',
    truncating='post',
    value=constant.TARGET_PADDING
)

val_sentences = pad_sequences(
            sequences=[sentence.split() for sentence in val_sentences],
            maxlen=constant.MAX_WORD_NUMBER,
            dtype=object,
            padding='post',
            truncating='post',
            value=constant.INPUT_PADDING
)
val_entities = pad_sequences(
    sequences=[entity.split() for entity in val_entities],
    maxlen=constant.MAX_WORD_NUMBER,
    dtype=object,
    padding='post',
    truncating='post',
    value=constant.TARGET_PADDING
)

train_generator = generate_data_batch(
    X=train_sentences,
    y=train_entities,
    entity_to_index=entity_to_index,
    num_classes=num_classes,
    batch_size=constant.BATCH_SIZE
)

val_generator = generate_data_batch(
    X=val_sentences,
    y=val_entities,
    entity_to_index=entity_to_index,
    num_classes=num_classes,
    batch_size=constant.BATCH_SIZE
)

model = BiLSTM((constant.MAX_WORD_NUMBER, constant.WORD_VECTOR_SIZE),
                hidden_dim=constant.HIDDEN_DIM,
                num_classes=num_classes)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy", custom_f1, custom_precision, custom_recall])
checkpoint_cb = ModelCheckpoint(
        constant.ROOT_DIR + '/save_model/weights.{epoch:02d}-{val_loss:.2f}.h5',
        save_weights_only=True, period=1)
model.fit_generator(
    train_generator,
    steps_per_epoch=train_sentences.shape[0] // constant.BATCH_SIZE,
    epochs=constant.NUM_EPOCH,
    callbacks=[checkpoint_cb],
    verbose=1,
    validation_data=val_generator,
    validation_steps=val_sentences.shape[0] // constant.BATCH_SIZE
)