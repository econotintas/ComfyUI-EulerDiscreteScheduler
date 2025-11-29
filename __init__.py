# ComfyUI custom node for FlowMatch Euler Scheduler
#
# This node creates a FlowMatchEulerDiscreteScheduler with configurable parameters
# so it can be used with compatible sampler nodes.
#
# Also registers the scheduler in ComfyUI's scheduler list with default config.
#
# Place this file into: ComfyUI/custom_nodes/
# Then restart ComfyUI. It will show up as "FlowMatch Euler Discrete Scheduler (Custom)"

import math
import torch

try:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers"])
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES

# Default config for registering in ComfyUI
default_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

def flow_match_euler_scheduler_handler(model_sampling, steps):
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(default_config)
    scheduler.set_timesteps(steps, device=model_sampling.device if hasattr(model_sampling, 'device') else 'cpu', mu=0.0)
    sigmas = scheduler.sigmas
    return sigmas

# Register the scheduler in ComfyUI
if "FlowMatchEulerDiscreteScheduler" not in SCHEDULER_HANDLERS:
    handler = SchedulerHandler(handler=flow_match_euler_scheduler_handler, use_ms=True)
    SCHEDULER_HANDLERS["FlowMatchEulerDiscreteScheduler"] = handler
    SCHEDULER_NAMES.append("FlowMatchEulerDiscreteScheduler")

class FlowMatchEulerSchedulerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 10000,
                    "tooltip": "Number of diffusion steps. Higher = better quality but slower. Try 20-30 for previews, 40-50 for final results."
                }),
                "base_image_seq_len": ("INT", {
                    "default": 256,
                    "tooltip": "Base sequence length for dynamic shifting. Should match model's training resolution (e.g., 256 for 512x512 images)."
                }),
                "base_shift": ("FLOAT", {
                    "default": math.log(3),
                    "tooltip": "Stabilizes generation. Higher values = more consistent/predictable outputs. Default: log(3) ≈ 1.099"
                }),
                "invert_sigmas": ("BOOL", {
                    "default": False,
                    "tooltip": "Reverses the sigma schedule. Keep False unless experimenting with advanced techniques."
                }),
                "max_image_seq_len": ("INT", {
                    "default": 8192,
                    "tooltip": "Maximum sequence length for dynamic shifting. Affects how the scheduler adapts to large images."
                }),
                "max_shift": ("FLOAT", {
                    "default": math.log(3),
                    "tooltip": "Maximum variation allowed. Higher = more exaggerated/stylized results. Default: log(3) ≈ 1.099"
                }),
                "num_train_timesteps": ("INT", {
                    "default": 1000,
                    "tooltip": "Timesteps the model was trained with. Should match your model's config (typically 1000)."
                }),
                "shift": ("FLOAT", {
                    "default": 1.0,
                    "tooltip": "Global timestep schedule shift. Affects overall sampling behavior. Keep at 1.0 unless you understand timestep theory."
                }),
                "shift_terminal": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "End value for shifted schedule. Set to 0.0 to disable. Advanced parameter for timestep schedule control."
                }),
                "stochastic_sampling": ("BOOL", {
                    "default": False,
                    "tooltip": "Adds controlled randomness to each step. Enable for more varied outputs (similar to ancestral samplers)."
                }),
                "time_shift_type": (["exponential", "linear"], {
                    "default": "exponential",
                    "tooltip": "Method for resolution-dependent shifting. Use 'exponential' for most cases, 'linear' for experiments."
                }),
                "use_beta_sigmas": ("BOOL", {
                    "default": False,
                    "tooltip": "Uses beta distribution for sigmas. Experimental alternative noise schedule."
                }),
                "use_dynamic_shifting": ("BOOL", {
                    "default": True,
                    "tooltip": "Auto-adjusts timesteps based on image resolution. Recommended: keep enabled for better multi-resolution results."
                }),
                "use_exponential_sigmas": ("BOOL", {
                    "default": False,
                    "tooltip": "Uses exponential sigma spacing. Try enabling for different noise distribution characteristics."
                }),
                "use_karras_sigmas": ("BOOL", {
                    "default": False,
                    "tooltip": "Uses Karras noise schedule for smoother results. Similar to DPM++ samplers, often improves quality."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "create"
    CATEGORY = "sampling/schedulers"
    DESCRIPTION = "FlowMatch Euler Discrete Scheduler with full parameter control. Outputs SIGMAS for use with SamplerCustom. Supports Karras sigmas, dynamic shifting, and stochastic sampling for advanced control over the diffusion process."

    def create(
        self,
        steps,
        base_image_seq_len,
        base_shift,
        invert_sigmas,
        max_image_seq_len,
        max_shift,
        num_train_timesteps,
        shift,
        shift_terminal,
        stochastic_sampling,
        time_shift_type,
        use_beta_sigmas,
        use_dynamic_shifting,
        use_exponential_sigmas,
        use_karras_sigmas,
    ):
        config = {
            "base_image_seq_len": base_image_seq_len,
            "base_shift": base_shift,
            "invert_sigmas": invert_sigmas,
            "max_image_seq_len": max_image_seq_len,
            "max_shift": max_shift,
            "num_train_timesteps": num_train_timesteps,
            "shift": shift,
            "shift_terminal": shift_terminal if shift_terminal != 0.0 else None,
            "stochastic_sampling": stochastic_sampling,
            "time_shift_type": time_shift_type,
            "use_beta_sigmas": use_beta_sigmas,
            "use_dynamic_shifting": use_dynamic_shifting,
            "use_exponential_sigmas": use_exponential_sigmas,
            "use_karras_sigmas": use_karras_sigmas,
        }

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
        
        # Set timesteps and get sigmas for the specified number of steps
        scheduler.set_timesteps(steps, device="cpu", mu=0.0)
        sigmas = scheduler.sigmas

        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "FlowMatchEulerDiscreteScheduler (Custom)": FlowMatchEulerSchedulerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowMatchEulerDiscreteScheduler (Custom)": "FlowMatch Euler Discrete Scheduler (Custom)"
}
