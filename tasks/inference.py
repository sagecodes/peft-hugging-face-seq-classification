
from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, pipeline)


def model_predict(
    trained_model_dir: str, texts: list[str]
) -> list[dict]:
    
    # Download and load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(trained_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)

    # Initialize the pipeline for sentiment analysis
    nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Perform batch prediction
    predictions = nlp_pipeline(texts, batch_size=8)

    return predictions
