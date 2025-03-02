
from tasks.inference import model_predict

model_dir = "model_lora" 

preds = model_predict(model_dir, ["I love this movie!",
                                     "I hate this movie!"])
print(preds)

# python -m workflows.inference model_predict
