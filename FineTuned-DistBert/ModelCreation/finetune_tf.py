import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Load data
df = pd.read_csv("youtube-comments-sentiment.csv")

# Map sentiment to numeric labels
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df = df[df['Sentiment'].isin(label_map.keys())]
df['label'] = df['Sentiment'].map(label_map)

# Split data
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def encode_texts(texts, tokenizer, max_len=128):
    return tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )

# Encode datasets
train_encodings = encode_texts(train_df['CommentText'], tokenizer)
val_encodings = encode_texts(val_df['CommentText'], tokenizer)

train_labels = tf.convert_to_tensor(train_df['label'].values)
val_labels = tf.convert_to_tensor(val_df['label'].values)

# Build TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(1000).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
)).batch(32)

# Load model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2
)

# Save model and tokenizer
model.save_pretrained('./ModelCreation/distilbert-finetuned-youtube-tf')
tokenizer.save_pretrained('./ModelCreation/distilbert-finetuned-youtube-tf')