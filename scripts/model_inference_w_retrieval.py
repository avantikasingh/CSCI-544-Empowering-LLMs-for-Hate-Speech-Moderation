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

import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch, os

tqdm.pandas()
transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_embeddings(train_data_file_name):
    # Load the JSON dataset
    with open(train_data_file_name, 'r') as file:
        train_data_json = json.load(file)
    
    # Extract texts and their labels
    train_data_texts = [item['text'].strip() for item in train_data_json]
    train_data_labels = [item['label'] for item in train_data_json]
    
    embeddings_file = os.path.basename(train_data_file_name) + "_embeddings.npy"
    
    # Embedding
    
    embeddings = model.encode(train_data_texts, convert_to_tensor=True)
    return embeddings, train_data_texts, train_data_labels
    
def few_shot_prompting_with_retrieval_as_string(prompt, embeddings, train_data_texts, train_data_labels, n=3):

    # Encode the prompt
    prompt_embedding = model.encode(prompt, convert_to_tensor=True).to(embeddings.device)
    
    # Find the most similar texts in the training data
    cos_scores = util.pytorch_cos_sim(prompt_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=n)
    
    # Retrieve the most similar texts, their labels, and their cosine similarity scores
    similar_texts_and_labels_and_scores = [
        (train_data_texts[index], train_data_labels[index], top_results.values[i].item()) 
        for i, index in enumerate(top_results.indices.cpu().numpy())
    ]
    
    # Assemble the output string
    output_str = "Given the definition of hate speech as any form of communication in speech, writing, or behavior that attacks or\n"
    output_str += "uses pejorative or discriminatory language with reference to a person or a group based on who they are—specifically\n"
    output_str += "their religion, ethnicity, nationality, race, color, descent, gender, or other identity factor—classify the following\n"
    output_str += "sentences as \"Toxic\" or \"Non-Toxic\":\n\n"
    output_str += "Examples\n"
    for text, label, _ in similar_texts_and_labels_and_scores:
        output_str += f"- {text} -> {label}\n"
    output_str += "\nPlease use the above knowledge to classify the below sentence as \"Toxic\" or \"Non-Toxic\". Your response must be only 'Toxic' or 'Non-Toxic'.\n\n"
    output_str += f"Input: {prompt}\n"
    output_str += "Response:"
    
    return output_str


class ModelInference:
    def __init__(self, train_dataset_path, test_dataset_path, model_name, prompt_path, output_path) -> None:
        self.access_token = "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"
        self.train_data_filepath = train_dataset_path
        self.test_data_filepath = test_dataset_path
        self.base_model_id = model_name
        self.prompt_filepath = prompt_path
        self.output_path = output_path

    def get_data(self):
        df = pd.read_json(self.test_data_filepath)
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


    def generate_output(self, text_generation_pipeline, dataset):
        predictions = []
        texts = []
        labels = []
        for i, out in enumerate(tqdm(
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
        # df = df[:150]
        embeddings, train_data_texts, train_data_labels = load_embeddings(self.train_data_filepath)
        df["abuse_prompt"] = df["text"].progress_apply(lambda x: few_shot_prompting_with_retrieval_as_string(x, embeddings, train_data_texts, train_data_labels, n=3))
        print(df.iloc[0]["abuse_prompt"])
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
    parser.add_argument("-train","--train_dataset_path", type=str, required=True, help="Path to the train dataset file")
    parser.add_argument("-test","--test_dataset_path", type=str, required=True, help="Path to the test dataset file")
    parser.add_argument("-mn","--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("-pp","--prompt_path", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("-op","--output_path", type=str, required=True, help="Path to save the output file")
    args = parser.parse_args()

    train_data_file_name = DATASET_PATH +args.train_dataset_path
    test_data_file_name = DATASET_PATH +args.test_dataset_path
    model_pipeline = ModelInference(train_data_file_name, test_data_file_name, args.model_name, PROMPTS_PATH+args.prompt_path, OUTPUT_PATH+args.output_path)
    out_df = model_pipeline.predict_labels()
    out_df["prediction"] = out_df["prediction"].apply(post_process)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    out_df.to_csv(OUTPUT_PATH+args.output_path+timestr+'.csv', index=False)
    