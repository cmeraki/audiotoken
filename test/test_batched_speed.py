import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

class AudioBatchPreparation:
    def __init__(self, segment_length, stride):
        self.segment_length = segment_length
        self.stride = stride

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
        
        segments = []
        
        for i in range(num_segments):
            start = i * self.stride
            end = start + self.segment_length
            segment = audio[:, start:end]
            
            if segment.shape[1] < self.segment_length:
                pad_size = self.segment_length - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, pad_size), value=0)
            
            segments.append(segment)

        return torch.stack(segments, dim=0).unsqueeze(1), num_segments

def benchmark(func, audio, num_runs=100):
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        _ = func(audio)
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / num_runs

def profile_func(func, audio):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            _ = func(audio)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this on a machine with a GPU.")
        return

    # Parameters
    segment_length = 16000*10  # 1 second at 16kHz
    stride = 16000  # 0.5 second stride
    audio_length = 16000*5  # 10 seconds at 16kHz
    num_runs = 1000

    # Create a random audio tensor on GPU
    audio = torch.randn(1, audio_length, device='cuda')

    audio_processor = AudioBatchPreparation(segment_length, stride)

    # Warm-up run
    _ = audio_processor.original_prepare_batch(audio)
    _ = audio_processor.optimized_prepare_batch(audio)

    # Benchmark
    original_time = benchmark(audio_processor.original_prepare_batch, audio, num_runs)
    optimized_time = benchmark(audio_processor.optimized_prepare_batch, audio, num_runs)

    print(f"Original implementation average time: {original_time*1000:.4f} ms")
    print(f"Optimized implementation average time: {optimized_time*1000:.4f} ms")
    print(f"Speedup: {original_time/optimized_time:.2f}x")

    print("\nDetailed profiling for original implementation:")
    profile_func(audio_processor.original_prepare_batch, audio)

    print("\nDetailed profiling for optimized implementation:")
    profile_func(audio_processor.optimized_prepare_batch, audio)

if __name__ == "__main__":
    main()