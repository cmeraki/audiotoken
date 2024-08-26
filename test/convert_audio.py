import torch
import unittest

from encodec.utils import convert_audio as convert_audio_encodec
from audiotoken.utils import convert_audio as convert_audio_audiotoken

length = 160000

class TestAudioConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_rates = [44100, 48000, 24000, 16000]
        cls.target_sample_rates = [16000, 22050, 24000, 48000]
        cls.length = 16000  # Define a standard length for test audio

    def test_mono_audio_same_sample_rate(self):
        for sr in self.sample_rates:
            audio = torch.randn(1, self.length)
            
            result1 = convert_audio_audiotoken(audio, sr, sr)
            result2 = convert_audio_encodec(audio, sr, sr, 1)
            
            self.assertTrue(torch.allclose(result1, result2))

    def test_stereo_audio_same_sample_rate(self):
        for sr in self.sample_rates:
            audio = torch.randn(2, self.length)
            
            result1 = convert_audio_audiotoken(audio, sr, sr)
            result2 = convert_audio_encodec(audio, sr, sr, 1)
            
            self.assertTrue(torch.allclose(result1, result2))

    def test_mono_audio_resample(self):
        for sr in self.sample_rates:
            for target_sr in self.target_sample_rates:
                if sr != target_sr:
                    audio = torch.randn(1, self.length)
                    
                    result1 = convert_audio_audiotoken(audio, sr, target_sr)
                    result2 = convert_audio_encodec(audio, sr, target_sr, 1)
                    
                    self.assertEqual(result1.shape, result2.shape)
                    self.assertTrue(torch.allclose(result1, result2))

    def test_stereo_audio_resample(self):
        for sr in self.sample_rates:
            for target_sr in self.target_sample_rates:
                if sr != target_sr:
                    audio = torch.randn(2, self.length)
                    
                    result1 = convert_audio_audiotoken(audio, sr, target_sr)
                    result2 = convert_audio_encodec(audio, sr, target_sr, 1)
                    
                    self.assertEqual(result1.shape, result2.shape)
                    self.assertTrue(torch.allclose(result1, result2))

    def test_invalid_channel_count(self):
        audio = torch.randn(3, self.length)
        sr = 44100
        target_sr = 22050
        
        with self.assertRaises(RuntimeError):
            convert_audio_audiotoken(audio, sr, target_sr)
        with self.assertRaises(AssertionError):
            convert_audio_encodec(audio, sr, target_sr, 1)

if __name__ == '__main__':
    unittest.main()