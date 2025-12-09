# Z-Image-Turbo Configuration

This node is pre-configured with defaults optimized for **Z-Image-Turbo** model.

## Default Configuration

Based on [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) scheduler config:

```json
{
  "num_train_timesteps": 1000,
  "use_dynamic_shifting": false,
  "shift": 3.0,
  "steps": 9
}
```

### Key Z-Image-Turbo Settings

- **steps**: `9` (actually results in 8 DiT forward passes)
- **shift**: `3.0` (optimized for Turbo model)
- **use_dynamic_shifting**: `disable` (disabled for consistent Turbo performance)
- **base_shift**: `0.5` (default diffusers value)
- **max_shift**: `1.15` (default diffusers value)
- **num_train_timesteps**: `1000`

### Usage with Z-Image-Turbo

```
Z-Image-Turbo Model -> SamplerCustom
                       ↑
                       |
                       FlowMatch Euler Scheduler (use defaults)
```

**Important**: For Z-Image-Turbo, use **guidance_scale=0.0** in your sampler/pipeline as Turbo models are guidance-free.

## Adjusting Parameters

While the defaults are optimized for Z-Image-Turbo, you can experiment:

- **More quality**: Increase `steps` to 15-20 (slower but potentially better)
- **More speed**: Reduce `steps` to 4-6 (faster but lower quality)
- **Different shift**: Adjust `shift` parameter (3.0 is optimal for Turbo)

## Reference

Official Z-Image-Turbo example:
```python
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # 9 steps = 8 DiT forwards
    guidance_scale=0.0,     # No guidance for Turbo
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
```

See full documentation: [https://huggingface.co/Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
