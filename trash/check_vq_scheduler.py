
try:
    from diffusers import VQDiffusionScheduler
    import inspect
    
    print("VQDiffusionScheduler found!")
    print("Init signature:")
    print(inspect.signature(VQDiffusionScheduler.__init__))
    
    # Also check config defaults if possible
    scheduler = VQDiffusionScheduler()
    print("\nDefault config:")
    print(scheduler.config)

except ImportError:
    print("VQDiffusionScheduler not found in diffusers.")
except Exception as e:
    print(f"Error: {e}")
