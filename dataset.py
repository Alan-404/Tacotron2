from torch.utils.data import Dataset
import pandas as pd
from processor import Tacotron2Processor
from tqdm import tqdm

class Tacotron2Dataset(Dataset):
    def __init__(self, manifest_path: str, processor: Tacotron2Processor, num_examples: int = None, make_grapheme: bool = False) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]
        self.processor = processor

        if 'graphemes' not in self.prompts.columns or make_grapheme:
            print("Converting Text to Graphemes")
            graphemes = []
            sentences = self.prompts['text'].to_list()
            for sentence in tqdm(sentences):
                graphemes_ = self.processor.sentence2graphemes(sentence)
                graphemes.append(" ".join(graphemes_))

            self.prompts['graphemes'] = graphemes

            self.prompts[['path', 'text', 'graphememes']].to_csv(manifest_path, sep="\t", index=False)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        prompt = self.prompts.iloc[index]

        graphemes = self.processor(prompt['graphemes'])
        signal = self.processor.load_audio(prompt['path'])

        return graphemes, signal