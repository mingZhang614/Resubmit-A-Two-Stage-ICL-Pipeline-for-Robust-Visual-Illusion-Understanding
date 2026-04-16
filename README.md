# Resubmit: A Two-Stage ICL Pipeline for Robust Visual Illusion Understanding

## Project Overview

This project is designed for A Two-Stage ICL Pipeline for Robust Visual Illusion Understanding. It provides the required script and data paths for the experiment.
Paper ID: 34

## Main modifications include
- **1. Fixed the output parsing bug:** We corrected the output parsing bug so that the system now outputs "None" when no valid option can be reliably extracted, rather than assigning a default option.
- **2. Fixed the random seed and temperature settings:** We fixed the random seed and explicitly set the temperature to improve reproducibility and reduce output variance.
- **3. Original logs and outputs:** We have included the original outputs from our initial submission. Other than these, 
no additional log files were generated in our pipeline. The original output file is: [Original_Outputs.txt](Original_Outputs.txt)


## Executable Script
- Due to the fact that, in our initial submission, the API was accessed through a non-official channel, 
we suspect that this may have caused instability in the API performance. 
However, we are uncertain whether we are allowed at the current stage to submit a version that uses the official API. 
In light of this, we provide both versions for your consideration, and kindly ask you to select the appropriate one according to the competition rules.

- For the version using the non-official API, the main executable script is located at: `Experiment/Task_2/Test_Gemini.py`

- For the version using the official API, the main executable script is located at: `Experiment/Task_2/Test_Gemini2.py`

## Image Format Conversion

- The images provided by this project have undergone a uniform format conversion. 
- Put your image data in `Task2_test_data/image` and run the `[image_preprocess.py](Experiment/Task_2/image_preprocess.py)`
- The processed images will be saved in the `Task2_test_data/processed_pngs_mix` folder.

## Result Storage Path

- The test results will be saved in the directory `Task2_test_data/2Stage_ICL_results`.

## ICL Reference Images

- The reference images for ICL are stored in the `Task2_test_data/ICL_images2` folder for use during testing.


## Usage Instructions
**1. Image Processing**:
   If you need to process the image first by [image_preprocess.py](Experiment/Task_2/image_preprocess.py), 
and the processed images can be found in `Task2_test_data/processed_pngs_mix`.

**2. Run the Experiment**:
   Run the `Test_Gemini.py` script after ensuring all dependencies are installed and the data paths are configured properly.
   
**3. Viewing Results**:
   After the testing is complete, check the output results in the `Task2_test_data/2Stage_ICL_results` folder.



## Dependencies

- Please ensure the required Python packages are installed by running:
    ```bash
    pip install -r requirements.txt
    ```

## Contact Information

If you have any questions, please contact [ming_zhang_sjtu@sjtu.edu.cn] or submit an issue through GitHub.