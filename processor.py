import os
import numpy as np
import json
import librosa
from typing import Optional, List, Tuple
import re
from torchaudio.transforms import MelSpectrogram
import torch
import torch.nn.functional as F
from torchtext.vocab import Vocab, vocab as create_vocab

class Tacotron2Processor:
    def __init__(self, vocab_path: str, bos_token: str = "<s>", eos_token: str = "</s>", unk_token: str = "<unk>", pad_token: str = "<pad>", word_delim_token: str = "|", sampling_rate: int = 22050, num_mels: int = 80, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, fmin: float = 0.0, fmax: float = 8000.0, puncs: str = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\-\\])") -> None:
        # Text
        self.replace_dict = dict()
        self.dictionary = None

        self.word_delim_item = word_delim_token
        self.pad_item = pad_token
        self.unk_item = unk_token
        self.bos_item = bos_token
        self.eos_item = eos_token

        self.create_vocab(vocab_path)

        self.unk_token = self.find_token(unk_token)
        self.pad_token = self.find_token(pad_token)
        self.word_delim_token = self.find_token(word_delim_token)
        self.bos_token = self.find_token(bos_token)
        self.eos_token = self.find_token(eos_token)

        self.puncs = puncs

        # Audio
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.hop_length = hop_length

        self.mel_transform = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels
        )

    def create_vocab(self, vocab_path: SyntaxWarning) -> Vocab:
        data = json.load(open(vocab_path, encoding='utf-8'))

        assert "vocab" in data.keys() and "replace" in data.keys()

        vocabs = data['vocab']
        self.replace_dict = data['replace']
        
        dictionary = dict()
        count = 0
        for item in vocabs:
            count += 1
            dictionary[item] = count

        self.dictionary = Vocab(
            vocab=create_vocab(
                dictionary,
                specials=[self.pad_item]
            ))
        
        self.dictionary.insert_token(self.word_delim_item, index=len(self.dictionary))
        self.dictionary.insert_token(self.bos_item, index=len(self.dictionary))
        self.dictionary.insert_token(self.eos_item, index=len(self.dictionary))
        self.dictionary.insert_token(self.unk_item, index=len(self.dictionary))
    
    def read_audio(self, path: str) -> np.ndarray:
        signal, _ = librosa.load(path, sr=self.sampling_rate, mono=True)

        return signal

    def load_audio(self, path: str, start: Optional[float] = None, end: Optional[float] = None) -> torch.Tensor:
        signal = self.read_audio(path)

        if start is not None and end is not None:
            signal = signal[int(start * self.sampling_rate) : int(end * self.sampling_rate)]

        signal = torch.FloatTensor(signal)

        return signal
    
    def find_token(self, char: str) -> int:
        if char in self.dictionary:
            return self.dictionary.__getitem__(char)
        return self.unk_token

    def clean_text(self, sentence: str) -> str:
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip().lower()

        return sentence
    
    def spec_replace(self, word: str):
        for key in self.replace_dict:
            word = word.replace(key, self.replace_dict[key])
        
        return word
    
    def word2graphemes(self, text: str,  n_grams: int = 3):
        if len(text) == 1:
            if text in self.dictionary:
                return [text]
            return [self.unk_item]
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]

            if num_steps == 2:
                item = self.spec_replace(item)
            
            if item in self.dictionary:
                graphemes.append(item)
            elif num_steps == 1:
                graphemes.append(self.unk_item)
            else:
                found = False

            if found:
                start += num_steps
                if len(text[start:]) < n_grams:
                    num_steps = len(text[start:])
                else:
                    num_steps = n_grams
            else:
                num_steps -= 1

        return graphemes
    
    def sentence2graphemes(self, sentence: str):
        sentence = self.clean_text(sentence)
        words = sentence.split(' ')
        graphemes = []
        graphemes.append(self.bos_item)
        for index, word in enumerate(words):
            graphemes += self.word2graphemes(word)
            if index != len(words) -1:
                graphemes.append("|")
        graphemes.append(self.eos_item)
        return graphemes
    
    def generate_mask(self, lengths: List[int], max_len: Optional[int] = None) -> torch.Tensor:
        masks = []

        if max_len is None:
            max_len = np.max(lengths)

        for length in lengths:
            masks.append(torch.tensor(np.array([True] * length + [False] * (max_len - length), dtype=bool)))
        
        return torch.stack(masks)
    
    def spectral_normalize(self, x: torch.Tensor, C: int = 1, clip_val: float = 1e-5) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(signal)
        log_mel = self.spectral_normalize(mel_spec)
        return log_mel
    
    def mel_spectrogize(self, signals: List[torch.Tensor], max_len: Optional[int] = None, return_length: bool = False) -> torch.Tensor:
        if max_len is None:
            max_len = np.max([len(signal) for signal in signals])

        mels = []
        mel_lengths = []

        for signal in signals:
            signal_length = len(signal)
            padded_signal = F.pad(signal, (0, max_len - signal_length), mode='constant', value=0.0)
            mels.append(self.mel_spectrogram(padded_signal))
            mel_lengths.append((signal_length // self.hop_length) + 1)

        mels = torch.stack(mels).type(torch.FloatTensor)

        if return_length:
            return mels, torch.tensor(mel_lengths)
        
        return mels
    
    def __call__(self, graphemes: List[List[str]], max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lengths = []
        for item in graphemes:
            if item != ['']:
                token = torch.tensor(np.array(self.dictionary(item)))
            else:
                token = torch.tensor(np.array([]))
            lengths.append(len(token))
            tokens.append(token)

        if max_len is None:
            max_len = np.max(lengths)

        padded_tokens = []
    
        for index, token in enumerate(tokens):
            padded_tokens.append(F.pad(token, (0, max_len - lengths[index]), mode='constant', value=self.pad_token))

        return torch.stack(padded_tokens), torch.tensor(lengths)