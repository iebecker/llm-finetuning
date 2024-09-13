import time
import vertexai
from vertexai.preview.tuning import sft

# Constants (replace with your actual values)
TRAIN_DATASET = "gs://gemini-training-ac215/pavlos/train.jsonl"
VALIDATION_DATASET = "gs://gemini-training-ac215/pavlos/test.jsonl"

GCP_PROJECT = "mlproject01-207413"

LOCATION = "us-central1"

def train_and_monitor_tuning_job():
    vertexai.init(project=GCP_PROJECT, location=LOCATION)

    sft_tuning_job = sft.train(
        source_model="gemini-1.0-pro-002",
        train_dataset=TRAIN_DATASET,
        validation_dataset=VALIDATION_DATASET,
        epochs=3,
        adapter_size=4,
        learning_rate_multiplier=1.0,
        tuned_model_display_name="The cheese model",
    )

    print("Training job started. Monitoring progress...\n\n")
    
    print("Detaling tuning job:")
    print(sft_tuning_job)
    while not sft_tuning_job.has_ended:
        time.sleep(60)
        sft_tuning_job.refresh()
        print("Job still in progress...")

    print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
    print(f"Tuned model endpoint name: {sft_tuning_job.tuned_model_endpoint_name}")
    print(f"Experiment: {sft_tuning_job.experiment}")

def cancel_tuning_job():
    job_id = input("Enter the tuning job ID to cancel: ")
    job = sft.SupervisedTuningJob(
        f"projects/{GCP_PROJECT}/locations/{LOCATION}/tuningJobs/{job_id}"
    )
    job.cancel()
    print(f"Cancelled tuning job: {job_id}")

def main():
    while True:
        choice = input("Enter 'train' to start a new training job, 'cancel' to cancel a job, or 'quit' to exit: ").lower()
        
        if choice == 'train':
            train_and_monitor_tuning_job()
        elif choice == 'cancel':
            cancel_tuning_job()
        elif choice == 'quit':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 'train', 'cancel', or 'quit'.")

if __name__ == "__main__":
    main()