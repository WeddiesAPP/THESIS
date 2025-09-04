import openai
import os
import json
import time
import random
from datasets import load_dataset, Dataset, load_from_disk


openai.api_key = 'CLASSIFIED'
# GPT Assistant ID
ASSISTANT_ID = "CLASSIFIED"
ds = load_dataset("michelecafagna26/hl")["test"]


BATCH_SIZE = 1  
SAVE_INTERVAL = 20 

OUTPUT_PATH = "modified_hl_final_test"

# Function to format input for Assistant
def build_bulk_prompt_for_image(example):
    pairs = {"scene": [], "action": [], "rationale": []}
    for axis in ["scene", "action", "rationale"]:
        if axis in example["captions"] and axis in example["confidence"]:
            for cap, conf in zip(example["captions"][axis], example["confidence"][axis]):
                pairs[axis].append({"caption": cap, "confidence": conf})

    input_data = {
        "captions": pairs,
        "instructions": "Rewrite captions to accurately reflect certainty, purity, and diversity while preserving meaning."
    }
    print(input_data)
    return json.dumps(input_data, indent=2)


# Function to call GPT Assistant
def rewrite_captions_for_batch(examples):
    user_prompts = [build_bulk_prompt_for_image(example) for example in examples]
    modified_captions_list = []

    for prompt in user_prompts:
        retry_attempts = 3
        while retry_attempts > 0:
            try:
                thread = openai.beta.threads.create()
                openai.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)

                # Run Assistant and wait
                run = openai.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=ASSISTANT_ID)

                # Retrieve response
                messages = openai.beta.threads.messages.list(thread_id=thread.id)
                last_message = messages.data[0].content[0].text.value

                # Parse JSON response
                response_json = json.loads(last_message)
                modified_captions_list.append(response_json["modified_captions"])
                break  

            except Exception as e:
                print(f"API call failed: {e}")
                retry_attempts -= 1
                time.sleep(random.uniform(2, 5))  # Exponential backoff

        if retry_attempts == 0:
            print(" Failed after multiple attempts. Skipping...")
            modified_captions_list.append({"scene": [], "action": [], "rationale": []})

    return modified_captions_list


# Function to process dataset in batches
def process_and_save_dataset():
    modified_data = []
    total_images = len(ds)
    
    # Check if dataset already exists
    try:
        existing_ds = load_from_disk(OUTPUT_PATH)
        start_index = len(existing_ds)
        modified_data.extend(existing_ds)
        print(f"Resuming from {start_index}/{total_images}...")
    except:
        start_index = 0
    
    # Process dataset in batches
    for i in range(start_index, total_images, BATCH_SIZE):
        batch = ds.select(range(i, min(i + BATCH_SIZE, total_images)))
        print(f"\nProcessing images {i+1} to {min(i + BATCH_SIZE, total_images)}...")

        # Rewrite captions in batch
        modified_captions = rewrite_captions_for_batch(batch)

        # Store new captions
        for j, example in enumerate(batch):
            example["modified_captions"] = modified_captions[j]
            modified_data.append(example)

        # Save progress every X images
        if (i + BATCH_SIZE) % SAVE_INTERVAL == 0:
            new_dataset = Dataset.from_list(modified_data)
            new_dataset.save_to_disk(OUTPUT_PATH)
            print(f"Progress saved at {i+1} images.")

    # Final save
    new_dataset = Dataset.from_list(modified_data)
    new_dataset.save_to_disk(OUTPUT_PATH)
    print("\Full dataset successfully saved!")


# Run dataset processing
process_and_save_dataset()