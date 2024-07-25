import boto3
import time
import json

# AWS configuration
ACCESS_KEY = ''
SECRET_KEY = ''
REGION = ''
BUCKET_NAME = ''
AUDIO_FILE_NAME = 'Arthur.mp3'
TRANSCRIPT_FILE_NAME = 'transcribed.txt'
SUMMARIZED_FILE_NAME = 'summarized.txt'

# Generate a unique job name with a timestamp
transcription_job_name = f'transcription_job_{int(time.time())}'

# Create AWS clients
transcribe_client = boto3.client('transcribe', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
comprehend_client = boto3.client('comprehend', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

def start_transcription_job():
    transcribe_client.start_transcription_job(
        TranscriptionJobName=transcription_job_name,
        Media={'MediaFileUri': f's3://{BUCKET_NAME}/{AUDIO_FILE_NAME}'},
        MediaFormat='mp3',
        LanguageCode='en-US',
        OutputBucketName=BUCKET_NAME,
        OutputKey=TRANSCRIPT_FILE_NAME
    )

def wait_for_transcription_job():
    while True:
        job = transcribe_client.get_transcription_job(TranscriptionJobName=transcription_job_name)
        if job['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            return job['TranscriptionJob']['TranscriptionJobStatus']
        time.sleep(10)

def get_transcribed_text():
    response = s3.get_object(Bucket=BUCKET_NAME, Key=TRANSCRIPT_FILE_NAME)
    content = json.loads(response['Body'].read().decode('utf-8'))
    return content['results']['transcripts'][0]['transcript']

def process_with_comprehend(text):
    # Detect dominant language
    language_response = comprehend_client.detect_dominant_language(Text=text)
    language_code = language_response['Languages'][0]['LanguageCode']

    # Generate summary
    summary_response = comprehend_client.detect_key_phrases(Text=text, LanguageCode=language_code)
    key_phrases = [phrase['Text'] for phrase in summary_response['KeyPhrases']]
    summary = ' '.join(key_phrases[:5])  # Use top 5 key phrases as a simple summary

    # Detect entities
    entities_response = comprehend_client.detect_entities(Text=text, LanguageCode=language_code)
    entities = [(entity['Text'], entity['Type']) for entity in entities_response['Entities']]

    # Detect sentiment
    sentiment_response = comprehend_client.detect_sentiment(Text=text, LanguageCode=language_code)
    sentiment = sentiment_response['Sentiment']

    # Detect PII entities
    pii_response = comprehend_client.detect_pii_entities(Text=text, LanguageCode=language_code)
    pii_entities = [entity['Type'] for entity in pii_response['Entities']]

    return {
        'summary': summary,
        'entities': entities,
        'sentiment': sentiment,
        'pii_entities': list(set(pii_entities)),
        'language': language_code
    }

def main():
    print("Starting transcription job...")
    start_transcription_job()
    
    print("Waiting for transcription job to complete...")
    job_status = wait_for_transcription_job()
    
    if job_status == 'COMPLETED':
        print("Transcription completed. Processing results...")
        transcribed_text = get_transcribed_text()
        
        # Process with Comprehend
        comprehend_results = process_with_comprehend(transcribed_text)
        
        # Prepare the summarized content
        summarized_content = f"Summary: {comprehend_results['summary']}\n\n"
        summarized_content += f"Entities: {comprehend_results['entities'][:10]}\n\n"
        summarized_content += f"Overall Sentiment: {comprehend_results['sentiment']}\n"
        summarized_content += f"PII Entities Detected: {comprehend_results['pii_entities']}\n"
        summarized_content += f"Detected Language: {comprehend_results['language']}\n"
        
        # Save summarized content to S3
        s3.put_object(Bucket=BUCKET_NAME, Key=SUMMARIZED_FILE_NAME, Body=summarized_content, ContentType='text/plain')
        
        print(f"Processing complete. Summary saved to {SUMMARIZED_FILE_NAME} in the S3 bucket.")
    else:
        print(f"Transcription job failed with status: {job_status}")

if __name__ == "__main__":
    main()
