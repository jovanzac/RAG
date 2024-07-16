import os
import torch
import torchaudio
import numpy as np
import random
from ..VoiceCraft.data.tokenizer import AudioTokenizer, TextTokenizer
from ..VoiceCraft.models import voicecraft
from ..VoiceCraft.inference_tts_scale import inference_one_sample
import importlib

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'VoiceCraft')))

class EnvironmentSetup:
    @staticmethod
    def setup_environment(cuda_devices="0", username="your_username"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ["USER"] = username


class ModelLoader:
    def __init__(self, voicecraft_name, encodec_fn, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.voicecraft_name = voicecraft_name
        self.ckpt_fn = f"./VoiceCraft/pretrained_models/{voicecraft_name}"
        self.encodec_fn = encodec_fn
        self._download_pretrained_models()
        self.ckpt = torch.load(self.ckpt_fn, map_location="cpu")
        self.model = voicecraft.VoiceCraft(self.ckpt["config"])
        self.model.load_state_dict(self.ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        self.phn2num = self.ckpt['phn2num']
        self.text_tokenizer = TextTokenizer(backend="es2peak")
        self.audio_tokenizer = AudioTokenizer(signature=self.encodec_fn, device=self.device)

    def _download_pretrained_models(self):
        if not os.path.exists(self.ckpt_fn):
            os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{self.voicecraft_name}?download=true")
            os.system(f"mv {self.voicecraft_name}?download=true ./VoiceCraft/pretrained_models/{self.voicecraft_name}")
        if not os.path.exists(self.encodec_fn):
            os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
            os.system(f"mv encodec_4cb2048_giga.th ./VoiceCraft/pretrained_models/encodec_4cb2048_giga.th")


class AudioProcessor:
    def __init__(self, temp_folder="./VoiceCraft/demo/temp"):
        self.temp_folder = temp_folder
        os.makedirs(self.temp_folder, exist_ok=True)

    def prepare_audio(self, orig_audio, orig_transcript):
        os.system(f"cp {orig_audio} {self.temp_folder}")
        filename = os.path.splitext(orig_audio.split("/")[-1])[0]
        print("*"*50)
        print(f"file loc: {self.temp_folder}/{filename}.txt")
        with open(f"{self.temp_folder}/{filename}.txt", "w") as f:
            f.write(orig_transcript)
        align_temp = f"{self.temp_folder}/mfa_alignments"
        os.system(f"mfa align -j 1 --clean --output_format csv {self.temp_folder} english_us_arpa english_us_arpa {align_temp}")

    def validate_audio_duration(self, audio_fn, cut_off_sec):
        info = torchaudio.info(audio_fn)
        audio_dur = info.num_frames / info.sample_rate
        assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
        return int(cut_off_sec * info.sample_rate)


class VoiceCloner:
    def __init__(self, model_loader, decode_config=None):
        self.model = model_loader.model
        self.config = model_loader.ckpt["config"]
        self.phn2num = model_loader.phn2num
        self.text_tokenizer = model_loader.text_tokenizer
        self.audio_tokenizer = model_loader.audio_tokenizer
        self.device = model_loader.device
        self.decode_config = decode_config if decode_config else self.default_decode_config()

    def default_decode_config(self):
        return {
            'top_k': 0,
            'top_p': 0.8,
            'temperature': 1,
            'stop_repetition': 3,
            'kvcache': 1,
            'codec_audio_sr': 16000,
            'codec_sr': 50,
            'silence_tokens': [1388, 1898, 131],
            'sample_batch_size': 2
        }

    def seed_everything(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def clone_voice(self, audio_fn, target_transcript, prompt_end_frame, seed=1):
        self.seed_everything(seed)
        concated_audio, gen_audio = inference_one_sample(
            self.model,
            self.config,
            self.phn2num,
            self.text_tokenizer,
            self.audio_tokenizer,
            audio_fn,
            target_transcript,
            self.device,
            self.decode_config,
            prompt_end_frame
        )
        return concated_audio[0].cpu(), gen_audio[0].cpu()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'phn2num': self.phn2num,
            'text_tokenizer': self.text_tokenizer,
            'audio_tokenizer': self.audio_tokenizer
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.phn2num = checkpoint['phn2num']
        self.text_tokenizer = checkpoint['text_tokenizer']
        self.audio_tokenizer = checkpoint['audio_tokenizer']
        self.model.to(self.device)
        self.model.eval()


if __name__ == "__main__" :
    print(f"current working directory is: {os.getcwd()}")
    env_setup = EnvironmentSetup()
    env_setup.setup_environment(cuda_devices="0", username="your_username")

    print(f"current working directory is: {os.getcwd()}")
    model_loader = ModelLoader(
        voicecraft_name="gigaHalfLibri330M_TTSEnhanced_max16s.pth",
        encodec_fn="./VoiceCraft/pretrained_models/encodec_4cb2048_giga.th"
    )


    audio_processor = AudioProcessor(temp_folder="./VoiceCraft/demo/temp")
    audio_processor.prepare_audio(
        orig_audio="./VoiceCraft/demo/enhanced_jovan2.mp3",
        orig_transcript="Once upon a time, there was an old mother pig, who had three little pigs, and not enough food to feed them."
    )
    prompt_end_frame = audio_processor.validate_audio_duration(
        audio_fn="./VoiceCraft/demo/temp/enhanced_jovan2.mp3",
        cut_off_sec=6.55
    )
    print("after prompt_end_frame")
    print(f"current working directory is: {os.getcwd()}")
    
    voice_cloner = VoiceCloner(model_loader)
    concated_audio, gen_audio = voice_cloner.clone_voice(
        audio_fn="./VoiceCraft/demo/temp/enhanced_jovan2.mp3",
        target_transcript="Once upon a time, there was an old mother pig, who had three little pigs, and not enough food to feed them. So she got to work, doing three jobs a day to feed her kids until they were old enough to get their own jobs.",
        prompt_end_frame=prompt_end_frame,
        seed=1
    )
    print("after clone_voice")

    # save the audio
    audio_fn="./VoiceCraft/demo/temp/enhanced_jovan2.mp3"
    seed = 1
    output_dir = "./demo/output"
    codec_audio_sr = 16000
    os.makedirs(output_dir, exist_ok=True)
    seg_save_fn_gen = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_gen_seed{seed}.mp3"
    seg_save_fn_concat = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_concat_seed{seed}.mp3"

    torchaudio.save(seg_save_fn_gen, gen_audio, codec_audio_sr)
    torchaudio.save(seg_save_fn_concat, concated_audio, codec_audio_sr)