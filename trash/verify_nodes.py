
import sys
import os
import importlib.util

# Add the parent directory to sys.path so we can import EulerDiscrete as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    # Import EulerDiscrete as a module
    import EulerDiscrete
    
    print("Successfully imported EulerDiscrete package.")
    
    mappings = EulerDiscrete.NODE_CLASS_MAPPINGS
    
    print("\nChecking NODE_CLASS_MAPPINGS:")
    
    has_flow_match = "FlowMatchEulerDiscreteScheduler (Custom)" in mappings
    has_vq = "VQDiffusionScheduler" in mappings
    
    if has_flow_match:
        print("✅ FlowMatchEulerDiscreteScheduler (Custom) found.")
    else:
        print("❌ FlowMatchEulerDiscreteScheduler (Custom) NOT found!")
        
    if has_vq:
        print("✅ VQDiffusionScheduler found.")
    else:
        print("❌ VQDiffusionScheduler NOT found!")
        
    if has_flow_match and has_vq:
        print("\nSUCCESS: Both schedulers are present.")
    else:
        print("\nFAILURE: Missing schedulers.")
        sys.exit(1)

except Exception as e:
    print(f"\nERROR: Import failed: {e}")
    # Print traceback for more details
    import traceback
    traceback.print_exc()
    sys.exit(1)
