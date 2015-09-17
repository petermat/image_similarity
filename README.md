# Image Similarity
Python image similarity comparison using several techniques

## Requirements:

  - `sudo apt-get install -y python-pip`

  - `sudo pip install PIL numpy`


## Usage
  - save script to same folder as your main script

  - import funtions from script

    - `from image_similarity import similarity_bands_via_numpy` 
    
    - `from image_similarity import similarity_histogram_via_pil` 
    
    - `from image_similarity import similarity_vectors_via_numpy` 
    
    - `from image_similarity import similarity_greyscale_hash_code`
    
  - call funtions from main script
  
  `similarity_bands_via_numpy, similarity_histogram_via_pil, similarity_vectors_via_numpy, similarity_greyscale_hash_code = image_similarity1(path_filename, path_previous_filename)`
    
