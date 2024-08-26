import torch
import unittest

class AudioBatchPreparationTest(unittest.TestCase):
    def setUp(self):
        self.segment_length = 240000
        self.stride = 8000  # 0.5 second stride

    def original_prepare_batch(self, audio: torch.Tensor):
        _, length = audio.shape
        segments = []

        for i in range(0, length, self.stride):
            segment = audio[:, i:i+self.segment_length]
            if segment.shape[1] < self.segment_length:
                segment = torch.nn.functional.pad(segment, (0, self.segment_length - segment.shape[1]), value=0)

            segments.append(segment)

        return torch.vstack(segments).unsqueeze(1), len(segments)

    def optimized_prepare_batch(self, audio: torch.Tensor):
        _, length = audio.shape
        num_segments = (length - 1) // self.stride + 1
        
        batch = torch.zeros(num_segments, 1, self.segment_length, device=audio.device)
        
        for i in range(num_segments):
            start = i * self.stride
            end = start + self.segment_length
            segment = audio[:, start:end]
            
            if segment.shape[1] < self.segment_length:
                pad_size = self.segment_length - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, pad_size), value=0)
            
            batch[i, 0] = segment

        return batch, num_segments

    def test_exact_multiple(self):
        # Test with audio length exactly divisible by stride
        audio = torch.randn(1, self.segment_length * 2)
        original_output, original_count = self.original_prepare_batch(audio)
        optimized_output, optimized_count = self.optimized_prepare_batch(audio)

        self.assertTrue(torch.allclose(original_output, optimized_output))
        self.assertEqual(original_count, optimized_count)

    def test_shorter_than_segment(self):
        # Test with audio shorter than segment_length
        audio = torch.randn(1, self.segment_length // 2)
        original_output, original_count = self.original_prepare_batch(audio)
        optimized_output, optimized_count = self.optimized_prepare_batch(audio)

        self.assertTrue(torch.allclose(original_output, optimized_output))
        self.assertEqual(original_count, optimized_count)

    def test_longer_than_segment(self):
        # Test with audio longer than segment_length but not exact multiple
        audio = torch.randn(1, int(self.segment_length * 2.5))
        original_output, original_count = self.original_prepare_batch(audio)
        optimized_output, optimized_count = self.optimized_prepare_batch(audio)

        self.assertTrue(torch.allclose(original_output, optimized_output))
        self.assertEqual(original_count, optimized_count)

    def test_very_long_audio(self):
        # Test with very long audio
        audio = torch.randn(1, self.segment_length * 10)
        original_output, original_count = self.original_prepare_batch(audio)
        optimized_output, optimized_count = self.optimized_prepare_batch(audio)

        self.assertTrue(torch.allclose(original_output, optimized_output))
        self.assertEqual(original_count, optimized_count)

if __name__ == '__main__':
    unittest.main()