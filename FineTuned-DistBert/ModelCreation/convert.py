from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-finetuned-youtube-tf",
    from_tf=True
)
model.save_pretrained("distilbert-finetuned-youtube-tf")  # This will now save pytorch_model.bin