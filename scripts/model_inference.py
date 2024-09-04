from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
import transformers
import json
import pandas as pd
from datasets import Dataset
import argparse
import time
from tqdm import tqdm


transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)

class ModelInference:
    def __init__(self, dataset_path, model_name, prompt_path, output_path) -> None:
        self.access_token = "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"
        self.data_filepath = dataset_path
        self.base_model_id = model_name
        self.prompt_filepath = prompt_path
        self.output_path = output_path

    def get_data(self):
        df = pd.read_json(self.data_filepath)
        return df

    def get_model_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, model_max_length=720, padding_side="left", add_eos_token=True, 
            # add_bos_token=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, device_map="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", token=self.access_token)
        return pipe

    def get_prompt(self):
        with open(self.prompt_filepath) as f:
            prompt = f.read()
        return prompt

    def generate_output(self, text_generation_pipeline, dataset):
        predictions = []
        texts = []
        labels = []
        for i, out in tqdm(enumerate(
            text_generation_pipeline(
                KeyDataset(dataset, "abuse_prompt"),
                batch_size=16,
                max_new_tokens=5,
                temperature=0.01,
                top_k=50,
                top_p=0.95,
                return_full_text=False,
                do_sample=True,
            )
        )):
            response = out[0]["generated_text"]
            predictions.append(response)
            texts.append(dataset[i]["text"])
            labels.append(dataset[i]["label"])
        out_df = pd.DataFrame()
        out_df["text"] = texts
        out_df["label"] = labels
        out_df["prediction"] = predictions
        return out_df

    def predict_labels(self):
        print("Run Started")
        df = self.get_data()
        print("Fetched Data")
        # df = df[:100]
        prompt = self.get_prompt()
        print("Prompt template fetched")
        df["abuse_prompt"] = df["text"].apply(lambda x: prompt.format(input_sentence=x))
        print("Prompt formatted")
        text_generation_pipeline = self.get_model_pipeline()
        print("Model Fetched")
        dataset = Dataset.from_pandas(df)
        print("Run started")
        out_df = self.generate_output(text_generation_pipeline, dataset)
        print("Model Run finished")
        return out_df

def post_process(generated_text:str):
    generated_text = generated_text.lower()
    if "non-toxic" in generated_text:
        return "Non-Toxic"
    elif "toxic" in generated_text:
        return "Toxic"
    else:
        return generated_text
    
if __name__ == "__main__":
    DATASET_PATH = "../data/processed_data/"
    PROMPTS_PATH = "../prompts/"
    OUTPUT_PATH = "../outputs/"
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp","--dataset_path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("-mn","--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("-pp","--prompt_path", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("-op","--output_path", type=str, required=True, help="Path to save the output file")
    args = parser.parse_args()

    model_pipeline = ModelInference(DATASET_PATH+args.dataset_path, args.model_name, PROMPTS_PATH+args.prompt_path, OUTPUT_PATH+args.output_path)
    out_df = model_pipeline.predict_labels()
    out_df["prediction"] = out_df["prediction"].apply(post_process)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    out_df.to_csv(OUTPUT_PATH+args.output_path+timestr+'.csv', index=False)