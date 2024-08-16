# An Investigation of Depression on a Language Moderation Model Using Concept Activation Vectors

## Getting Started

To get started with this project, follow the steps below:

### Prerequisites

You will need an Anaconda environment for a faster setup.

### Installation

1. **Download and Unzip Environment**
    - Download and unzip the provided [environment file](https://drive.google.com/file/d/1YqCbeX5evygBsHekhdWY1VxnmBcti_UK/view?usp=sharing)

2. **Clone the Repository**
    - Clone the project and checkout the main branch

3. **Add Conda Environment**
    - Add the conda environment to the project as the interpreter

4. **Install Dependencies**
    - Navigate to the directory of the [https://github.com/elenagaz/lit_bachelor/blob/lit_ba/lit_nlp/package.json](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/blob/main/lit_nlp/package.json) file:https://github.com/elenagaz/lit_bachelor/tree/lit_ba/lit_nlp
    - Run the following commands to install dependencies and build the project:
      ```
      yarn install
      yarn build
      ```
    

### Running the Demo

To run the demo I have created, execute the main method of the specified file. Navigate to the path: [my_model_moderation](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation) and run the [demo](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/blob/main/lit_nlp/my_model_moderation/moderation_demo.py)

To reproduce the TCAV scores for the depression concept, run the demo and only choose the first 148 examples labeled as OK (image 1) and add them as a slice (image 2), then navigate to the TCAV tab (image 3) and run TCAV with the selected classes (image 4) - this takes a while as there are predictions that must be rerun.

image 1:
![image](https://github.com/user-attachments/assets/ef5e8b67-4d49-4e8d-965e-4109f200cdcf)

image 2:
![image](https://github.com/user-attachments/assets/cb1250d6-185b-420c-9525-341a35710ef1)

image 3:
![image](https://github.com/user-attachments/assets/6673e665-c949-4df2-884d-2db11a8cabf8)

image 4:
![image](https://github.com/user-attachments/assets/0f4bd076-325a-4281-bf21-da40bdf67533)



Moreover, for each of the symptoms, each symptom file from the [directory](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation/TCAV_evaluation_files) must be added as the file path on [line](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/blob/main/lit_nlp/my_model_moderation/moderation_demo.py#L27) and the demo must be rerun.

### All results 

All results with the corrected p-value of 0.00256 can be found in this [directory](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation/results/p_value_0.00256)

However, the results with the initial p-value of 0.05 can be found in this [directory](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation/results/p_value_0.05)

### Additional Information regarding the paper

All protocols of data generation for depression and random data can be found [here](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation/protocols_of_data_generation)

All files used for the generation of TCAV scores can be found in this [directory](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation/TCAV_evaluation_files)

Additionally, all files for validation, including the files used as training data can be found in these directories [1](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation/model_validation/KoalaAI_Text-Moderation-v2-small) and [2](https://github.com/elenagaz/An-Investigation-of-Depression-on-a-Language-Moderation-Model-Using-Concept-Activation-Vectors/tree/main/lit_nlp/my_model_moderation/model_validation/moderation_api_release)

## Note: 
Some edits have been made to ensure compatibility with Windows.

