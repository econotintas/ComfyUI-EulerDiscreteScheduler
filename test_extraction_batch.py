import os
import random
import sys

# Add custom node directory to path so we can import the class
sys.path.append(os.path.dirname(__file__))

# Mock folder_paths for the node to work
import folder_paths
folder_paths.get_annotated_filepath = lambda x: x # Just return the path as is

from extract_metadata_node import ImageMetadataExtractor

def main():
    output_dir = r"D:\ComfyUI7\ComfyUI\output"
    output_file = "batch_test_results.txt"
    
    # Get all image files
    all_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not all_files:
        print(f"No images found in {output_dir}")
        return

    # Select 20 random images
    num_samples = min(20, len(all_files))
    selected_files = random.sample(all_files, num_samples)
    
    extractor = ImageMetadataExtractor()
    
    print(f"Testing on {num_samples} images...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, file_path in enumerate(selected_files):
            try:
                # We need to bypass the folder_paths.get_annotated_filepath call inside the node 
                # by mocking it, or just passing the absolute path if our mock above works.
                # The node calls folder_paths.get_annotated_filepath(image)
                # Our mock returns x, so we pass the full path.
                
                prompt, width, height = extractor.extract_metadata(file_path)
                
                separator = "=" * 80
                entry = f"{separator}\nImage: {os.path.basename(file_path)}\nDimensions: {width}x{height}\nPrompt:\n{prompt}\n"
                
                f.write(entry + "\n")
                print(f"Processed {i+1}/{num_samples}: {os.path.basename(file_path)}")
                
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(file_path)}: {e}\n"
                f.write(error_msg)
                print(error_msg)

    print(f"Done. Results saved to {output_file}")

if __name__ == "__main__":
    main()
