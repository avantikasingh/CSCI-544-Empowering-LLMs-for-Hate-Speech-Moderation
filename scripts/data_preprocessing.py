import pandas as pd
import re
CAD = 'cad'
HATEEXPLAIN = 'hatexplain'
JIGSAW = 'jigsaw'
ORIGNAL_DATASET_PATH = './../data/original_data/' 
OUTPUT_FILE_PATH = './../data/processed_data/'
class DataPreprocessor:
    def __init__(self,filepath):
        self.filepath  = filepath
    
    def read_json_data(self):
        colnames = ['text', 'label']
        train = pd.read_json(self.filepath + 'train.json')
        test = pd.read_json(self.filepath + 'test.json')
        val = pd.read_json(self.filepath + 'val.json')
        print(train.shape, val.shape, test.shape)
        df = pd.concat([train, test, val])
        df.columns = colnames
        df.reset_index(inplace=True)
        return df
    
    def remove_punctuation_except_periods(self,text):
        # Replace all punctuation except periods with nothing
        return re.sub(r'[^\w\s]', '', text)
    
    def process_text(self,text):
        # Remove leading and trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Join lines with a period, adding one if not already present
        joined_text = '. '.join(line if line.endswith('.') else line + '.' for line in lines if line)
        
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', joined_text)

        return cleaned_text
    
    def remove_special_characters(self,text):
        # Replace any character that is not a letter or number with nothing
        return re.sub(r'[^A-Za-z\s\.]', '', text)
    
    
    
    def preprocess_data(self):
        df = self.read_json_data()
        df = df[["text","label"]]
        df['text'] = df['text'].apply(self.remove_punctuation_except_periods)
        df['text'] = df['text'].apply(self.process_text)
        df['text'] = df['text'].apply(self.remove_special_characters)
        df["label"] = df["label"].apply(lambda x: 'Non-Toxic' if x=='normal' else 'Toxic')
        cnt_dict = df["label"].value_counts().to_dict()
        print(cnt_dict)
        out_df = pd.concat([
            df[df["label"]=="Toxic"],
            df[df["label"]=="Non-Toxic"].sample(n=cnt_dict["Toxic"] if cnt_dict["Toxic"]<cnt_dict["Non-Toxic"] else cnt_dict["Non-Toxic"])])
        
        return out_df
    
if __name__=="__main__":
    file_type = HATEEXPLAIN
    dp = DataPreprocessor(ORIGNAL_DATASET_PATH + file_type + '/')
    out_df = dp.preprocess_data()
    out_df.to_json(OUTPUT_FILE_PATH + file_type + '.json', orient='records',index=False)

    print("Successfully processed data")
    