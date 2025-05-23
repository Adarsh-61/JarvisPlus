import os, sys
import re
import platform
import subprocess
from sys import modules
from typing import List, Tuple, Type, Dict

from IPython.display import display, Audio
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile

from sources.utility import pretty_print, animate_thinking

class Speech():
    """
    Speech is a class for generating speech from text.
    """
    def __init__(self, enable: bool = True, language: str = "en", voice_idx: int = 0) -> None:
        self.lang_map = {
            "en": 'a',
            "zh": 'z',
            "fr": 'f'
        }
        self.voice_map = {
            "en": ['af_kore', 'af_bella', 'af_alloy', 'af_nicole', 'af_nova', 'af_sky', 'am_echo', 'am_michael', 'am_puck'],
            "zh": ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'],
            "fr": ['ff_siwis']
        }
        self.language = language
        self.voice = self.voice_map[language][voice_idx]
        self.speed = 1.2
        self.voice_folder = ".voices"
        self.create_voice_folder(self.voice_folder)
        self.enable = enable
        if enable:
            if language == "en":
                self.model_id = "facebook/mms-tts-eng"
            else:
                raise ValueError("Currently only the English model (facebook/mms-tts-eng) is supported.")
            self.model = VitsModel.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.device = self.get_device()
            self.model.to(self.device)

    def get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"

    def create_voice_folder(self, path: str = ".voices") -> None:
        """
        Create a folder to store the voices.
        Args:
            path (str): The path to the folder.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def speak(self, sentence: str, voice_idx: int = 1):
        if not self.enable:
            return
        if voice_idx >= len(self.voice_map[self.language]):
            pretty_print("Invalid voice number, using default voice", color="error")
            voice_idx = 0
        sentence = self.clean_sentence(sentence)
        audio_file = f"{self.voice_folder}/sample_{self.voice_map[self.language][voice_idx]}.wav"
        self.voice = self.voice_map[self.language][voice_idx]

        # Tokenize and prepare inputs
        inputs = self.tokenizer(sentence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Ensure indices are integer type for embedding
        if 'input_ids' in inputs:
            inputs['input_ids'] = inputs['input_ids'].long()
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'].long()

        # Generate waveform
        with torch.no_grad():
            output = self.model(**inputs).waveform

        sampling_rate = self.model.config.sampling_rate
        scipy.io.wavfile.write(audio_file, rate=sampling_rate, data=output.squeeze().cpu().numpy())

        # Play audio depending on platform
        if platform.system().lower() == "windows":
            import winsound
            winsound.PlaySound(audio_file, winsound.SND_FILENAME)
        elif platform.system().lower() == "darwin":  # macOS
            subprocess.call(["afplay", audio_file])
        else:  # linux or other.
            subprocess.call(["aplay", audio_file])

        # If running in notebook, display audio widget
        if 'ipykernel' in modules:
            display(Audio(data=output.squeeze().cpu().numpy(), rate=sampling_rate, autoplay=True))

    def replace_url(self, url: re.Match) -> str:
        """
        Replace URL with domain name or empty string if IP address.
        Args:
            url (re.Match): Match object containing the URL pattern match
        Returns:
            str: The domain name from the URL, or empty string if IP address
        """
        domain = url.group(1)
        if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain):
            return ''
        return domain

    def extract_filename(self, m: re.Match) -> str:
        """
        Extract filename from path.
        Args:
            m (re.Match): Match object containing the path pattern match
        Returns:
            str: The filename from the path
        """
        path = m.group()
        parts = re.split(r'/|\\', path)
        return parts[-1] if parts else path
    
    def shorten_paragraph(self, sentence):
        #TODO find a better way, we would like to have the TTS not be annoying, speak only useful informations
        """
        Find long paragraph like **explaination**: <long text> by keeping only the first sentence.
        Args:
            sentence (str): The sentence to shorten
        Returns:
            str: The shortened sentence
        """
        lines = sentence.split('\n')
        lines_edited = []
        for line in lines:
            if line.startswith('**'):
                lines_edited.append(line.split('.')[0])
            else:
                lines_edited.append(line)
        return '\n'.join(lines_edited)

    def clean_sentence(self, sentence):
        """
        Clean and normalize text for speech synthesis by removing technical elements.
        Args:
            sentence (str): The input text to clean
        Returns:
            str: The cleaned text with URLs replaced by domain names, code blocks removed, etc..
        """
        lines = sentence.split('\n')
        filtered_lines = [line for line in lines if re.match(r'^\s*[a-zA-Z]', line)]
        sentence = ' '.join(filtered_lines)
        sentence = re.sub(r'`.*?`', '', sentence)
        sentence = re.sub(r'https?://(?:www\.)?([^\s/]+)(?:/[^\s]*)?', self.replace_url, sentence)
        sentence = re.sub(r'\b[\w./\\-]+\b', self.extract_filename, sentence)
        sentence = re.sub(r'\b-\w+\b', '', sentence)
        sentence = re.sub(r'[^a-zA-Z0-9.,!? _ -]+', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        sentence = sentence.replace('.com', '')
        return sentence

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    speech = Speech()
    tosay_en = """
    I looked up recent news using the website https://www.theguardian.com/world
    """
    tosay_zh = """
    我使用网站 https://www.theguardian.com/world 查阅了最近的新闻。
    """
    tosay_fr = """
    J'ai consulté les dernières nouvelles sur le site https://www.theguardian.com/world
    """
    spk = Speech(enable=True, language="en", voice_idx=0)
    spk.speak(tosay_en, voice_idx=0)
    spk = Speech(enable=True, language="fr", voice_idx=0)
    spk.speak(tosay_fr)
    #spk = Speech(enable=True, language="zh", voice_idx=0)
    #spk.speak(tosay_zh)
