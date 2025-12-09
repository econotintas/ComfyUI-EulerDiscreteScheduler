
try:
    from diffusers import VQDiffusionScheduler
    import torch
    
    scheduler = VQDiffusionScheduler(num_vec_classes=4096, num_train_timesteps=100)
    print("Scheduler created successfully.")
    
    print(f"Has sigmas attribute? {hasattr(scheduler, 'sigmas')}")
    
    scheduler.set_timesteps(10)
    print("Timesteps set to 10.")
    print(f"Timesteps: {scheduler.timesteps}")
    
    if hasattr(scheduler, 'sigmas'):
        print(f"Sigmas: {scheduler.sigmas}")
    else:
        print("No 'sigmas' attribute found. This scheduler might not be compatible with ComfyUI's standard sampler loop which expects sigmas.")

except Exception as e:
    print(f"Error: {e}")
