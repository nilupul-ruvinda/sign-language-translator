import os
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

LETTERS = list(string.ascii_uppercase)
SAMPLES_PER_LETTER = 100
DATA_PATH = os.path.join('letter_data')

label_map = {letter: idx for idx, letter in enumerate(LETTERS)}
X, y = [], []

for letter in LETTERS:
    for sample_idx in range(SAMPLES_PER_LETTER):
        data = np.load(os.path.join(DATA_PATH, letter, f"{sample_idx}.npy"))
        X.append(data)
        y.append(label_map[letter])

X = np.array(X)
y = to_categorical(y).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

os.makedirs('models', exist_ok=True)

model = Sequential([
    Dense(256, activation='relu', input_shape=(67,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(LETTERS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# Print per-letter accuracy report
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\n--- Per-Letter Classification Report ---")
print(classification_report(y_true, y_pred, target_names=LETTERS))

model.save(os.path.join('models', 'letter_model.h5'))
print("Letter model saved to models/letter_model.h5")