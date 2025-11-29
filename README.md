# FlowMatch Euler Discrete Scheduler for ComfyUI

Custom node that exposes all parameters of the FlowMatchEulerDiscreteScheduler. Outputs SIGMAS for use with **SamplerCustom** node.

## Usage

1. Add **FlowMatch Euler Discrete Scheduler (Custom)** node to your workflow
2. Connect its SIGMAS output to **SamplerCustom** node's sigmas input
3. Adjust parameters to control the sampling behavior

## Parameters

### steps
**Type:** Integer (1-10000, default: 20)  
**Description:** Number of diffusion steps during sampling.  
**Example:** Use 20-30 for fast previews, 40-50 for quality results.

### base_image_seq_len
**Type:** Integer (default: 256)  
**Description:** Base image sequence length for dynamic shifting calculations.  
**Example:** For 512x512 images, use 256. Matches model's training resolution.

### base_shift
**Type:** Float (default: 1.0986 ≈ log(3))  
**Description:** Stabilizes generation by reducing variation. Higher = more stable/consistent.  
**Example:** Increase to 1.5 for more controlled, predictable outputs.

### max_shift
**Type:** Float (default: 1.0986 ≈ log(3))  
**Description:** Maximum variation allowed. Higher = more exaggerated/stylized results.  
**Example:** Increase to 1.5 for more creative/varied outputs.

### shift
**Type:** Float (default: 1.0)  
**Description:** Global timestep schedule shift value. Affects overall sampling behavior.  
**Example:** Keep at 1.0 unless you understand timestep shifting theory.

### shift_terminal
**Type:** Float (default: 0.0, which means None)  
**Description:** End value for shifted timestep schedule. Set to 0.0 to disable.  
**Example:** Use 0.5 to modify how the schedule ends (advanced).

### use_dynamic_shifting
**Type:** Boolean (default: True)  
**Description:** Automatically adjust timesteps based on image resolution.  
**Example:** Keep True for better results with varying image sizes.

### time_shift_type
**Type:** Choice: "exponential" or "linear" (default: "exponential")  
**Description:** Method for resolution-dependent timestep shifting.  
**Example:** Use "exponential" for most cases, "linear" for experimental control.

### use_karras_sigmas
**Type:** Boolean (default: False)  
**Description:** Use Karras noise schedule (typically gives smoother results).  
**Example:** Enable for potentially higher quality, similar to DPM++ samplers.

### use_exponential_sigmas
**Type:** Boolean (default: False)  
**Description:** Use exponential sigma spacing instead of linear.  
**Example:** Try True for different noise distribution characteristics.

### use_beta_sigmas
**Type:** Boolean (default: False)  
**Description:** Use beta distribution for sigma values.  
**Example:** Enable for alternative noise scheduling (experimental).

### invert_sigmas
**Type:** Boolean (default: False)  
**Description:** Reverse the sigma schedule direction.  
**Example:** Keep False unless you have specific experimental needs.

### stochastic_sampling
**Type:** Boolean (default: False)  
**Description:** Add controlled randomness to each sampling step.  
**Example:** Enable for more varied outputs, similar to ancestral samplers.

### num_train_timesteps
**Type:** Integer (default: 1000)  
**Description:** Number of timesteps the model was trained with.  
**Example:** Match your model's training config (usually 1000).

## Tips

- **Start simple:** Use defaults, only adjust `steps` initially
- **For quality:** Try enabling `use_karras_sigmas` with 30-40 steps
- **For variation:** Enable `stochastic_sampling` or increase `max_shift`
- **For stability:** Increase `base_shift` to reduce randomness

## Example Workflow

```
Model -> SamplerCustom
         ^ 
         |
         FlowMatch Euler Scheduler (steps=30, use_karras_sigmas=True)
```
