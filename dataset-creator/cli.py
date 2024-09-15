import os
import argparse
import pandas as pd
import json
import time
import glob
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
import vertexai.generative_models as generative_models

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
GENERATIVE_MODEL = "gemini-1.5-flash-001"
OUTPUT_FOLDER = "data"
# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 8192,  # Maximum number of tokens for output
    "temperature": 1,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}

# Safety settings to filter out harmful content
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    )
]

# System Prompt
SYSTEM_INSTRUCTION = """Generate a set of question-answer pairs about cheese in English, adopting the tone and perspective of an experienced Italian cheese expert. Adhere to the following guidelines:

1. Expert Perspective:
  - Embody the voice of a seasoned Italian cheese expert with deep knowledge of both Italian and international cheeses
  - Infuse responses with passion for cheese craftsmanship and Italian cheese-making traditions
  - Reference Italian cheese-making regions, techniques, and historical anecdotes where relevant

2. Content Coverage:
  - Traditional and modern Italian cheese production methods
  - Diverse Italian cheese types, their characteristics, and regional significance
  - Comparison of Italian cheeses with international varieties
  - Cheese aging processes, with emphasis on Italian techniques
  - Pairing Italian cheeses with wines, foods, and in cooking
  - Cultural importance of cheese in Italian cuisine and society
  - Artisanal cheese production in Italy and its global influence
  - DOP (Protected Designation of Origin) and IGP (Protected Geographical Indication) certifications for Italian cheeses
  - Scientific aspects of cheese, viewed through an Italian expert's lens

3. Tone and Style:
  - Use a passionate, authoritative tone that conveys years of expertise
  - Incorporate Italian terms where appropriate, always providing English translations or brief explanations
  - Balance technical knowledge with accessible explanations
  - Express pride in Italian cheese-making traditions while acknowledging global contributions

4. Complexity and Depth:
  - Provide a mix of basic information and advanced insights
  - Include lesser-known facts and expert observations
  - Offer nuanced explanations that reflect deep understanding of cheese science and art

5. Question Types:
  - Include both factual questions and those requiring expert analysis
  - Formulate questions that an enthusiast might ask an Italian cheese expert

6. Answer Format:
  - Give comprehensive answers that showcase expertise
  - Include relevant anecdotes or historical context where appropriate
  - Ensure answers are informative for both novices and cheese aficionados

7. Cultural Context:
  - Highlight the role of cheese in Italian culture and cuisine
  - Discuss regional variations and their historical or geographical reasons

8. Accuracy and Relevance:
  - Ensure all information is factually correct and up-to-date
  - Focus on widely accepted information in the field of Italian cheese expertise

9. Language:
  - Use English throughout, but feel free to include Italian terms (with translations) where they add authenticity or specificity

Output Format:
Provide the Q&A pairs in JSON format, with each pair as an object containing 'question' and 'answer' fields, within a JSON array.
Follow these strict guidelines:
1. Use double quotes for JSON keys and string values.
2. For any quotation marks within the text content, use single quotes (') instead of double quotes.
3. If a single quote (apostrophe) appears in the text, escape it with a backslash (\'). 
4. Ensure there are no unescaped special characters that could break the JSON structure.

Here's an example of the expected format:

[
  {
    "question": "What is the difference between 'mozzarella' and 'burrata'?",
    "answer": "While both are fresh Italian cheeses, 'mozzarella' is a solid cheese made from buffalo or cow\'s milk. 'Burrata', on the other hand, has an outer shell of mozzarella, but is filled with a mixture of cream and soft cheese curds, giving it a much creamier texture and richer flavor."
  }
]
"""


def generate():
    print("generate()")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Initialize Vertex AI project and location
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    
    # Initialize the GenerativeModel with specific system instructions
    model = GenerativeModel(
        GENERATIVE_MODEL,
        system_instruction=[SYSTEM_INSTRUCTION]
    )

    INPUT_PROMPT = """Generate 20 diverse, informative, and engaging question-answer pairs about cheese following these guidelines, embodying the passionate and knowledgeable tone of an Italian cheese expert, while keeping all content in English."""
    NUM_ITERATIONS = 5

    # Loop to generate and save the content
    for i in range(0, NUM_ITERATIONS):
        try:
          responses = model.generate_content(
            [INPUT_PROMPT],  # Input prompt
            generation_config=generation_config,  # Configuration settings
            safety_settings=safety_settings,  # Safety settings
            stream=False,  # Enable streaming for responses
          )
          generated_text = responses.text

          # Create a unique filename for each iteration
          file_name = f"{OUTPUT_FOLDER}/cheese_qa_{i}.txt"
          # Save
          with open(file_name, "w") as file:
            file.write(generated_text)
        except Exception as e:
          print(f"Error occurred while generating content: {e}")


def prepare():
    print("prepare()")

    # Get the generated files
    output_files = glob.glob(os.path.join(OUTPUT_FOLDER, "cheese_qa_*.txt"))
    output_files.sort()

    # Consolidate the data
    output = []
    for output_file in output_files:
        print("Processing file:", output_file)
        with open(output_file, "r") as read_file:
            text_response = read_file.read()
        
        text_response = text_response.replace("```json","").replace("```","")
    
        json_response = json.loads(text_response)
        output.extend(json_response)
    
    # Save the dataset
    output_df = pd.DataFrame(output)
    print("Shape:", output_df.shape)
    print(output_df.head())
    filename = os.path.join(OUTPUT_FOLDER, "instruct-dataset.csv")
    output_df.to_csv(filename, index=False)


def upload():
    print("upload()")
    

def main(args=None):
    print("CLI Arguments:", args)

    if args.generate:
        generate()

    if args.prepare:
        prepare()
      
    if args.upload:
        upload()

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate data",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare data",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload data to bucket",
    )

    args = parser.parse_args()

    main(args)