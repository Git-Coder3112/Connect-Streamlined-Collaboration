import boto3
import time
import textwrap
import spacy
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Replace with your AWS access key and secret access key (or use AWS CLI/configuration for credentials)
# REGION can also be specified in the AWS configuration
ACCESS_KEY = ''
SECRET_KEY = ''

# Replace with your AWS region and S3 bucket name
REGION = ''
BUCKET_NAME = ''

# Replace with the name of the input audio file in your S3 bucket
AUDIO_FILE_NAME = 'Arthur.mp3'

# Replace with the name of the output transcript file in your S3 bucket
TRANSCRIPT_FILE_NAME = 'transcribed.txt'
SUMMARIZED_FILE_NAME = 'summarized.txt'  # Assuming this file is present in your S3 bucket

# Generate a unique job name with a timestamp
transcription_job_name = f'transcription_job_{int(time.time())}'

# Create a Transcribe client
transcribe_client = boto3.client('transcribe', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

# Start the transcription job
transcription_job = transcribe_client.start_transcription_job(
    TranscriptionJobName=transcription_job_name,
    Media={'MediaFileUri': f's3://{BUCKET_NAME}/{AUDIO_FILE_NAME}'},
    MediaFormat='mp3',
    LanguageCode='en-US',
    OutputBucketName=BUCKET_NAME,
    OutputKey=TRANSCRIPT_FILE_NAME
)

# Wait for the transcription job to complete
while True:
    job = transcribe_client.get_transcription_job(TranscriptionJobName=transcription_job_name)
    if job['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    time.sleep(10)

# Check if transcription job was successful
if job['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
    # Retrieve the transcribed text from the S3 bucket
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
    response = s3.get_object(Bucket=BUCKET_NAME, Key=TRANSCRIPT_FILE_NAME)
    transcribed_text = response['Body'].read().decode('utf-8')

    # Process the text with spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(transcribed_text)

    # Extract entities
    entities_list = []
    for ent in doc.ents:
        entities_list.append((ent.text, ent.label_))

    # Write entities to summarized file in S3
    summarized_text = "\nEntities:\n" + "\n".join(map(str, entities_list)) + "\n\n"
    s3.put_object(Bucket=BUCKET_NAME, Key=SUMMARIZED_FILE_NAME, Body=summarized_text, ContentType='text/plain')

    # Split the transcribed text into chunks
    chunk_size = 5000  # Set an appropriate chunk size
    text_chunks = textwrap.wrap(transcribed_text, chunk_size)

    # Initialize translated text
    translated_text = ""
    key_phrases_list = []
    language_list = []
    targeted_sentiments_list = []
    pii_entities_list = []
    keyphrase_extraction_list = []
    syntax_analysis_list = []
    # Create a Comprehend client
    comprehend_client = boto3.client('comprehend', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

    # Process each chunk for various Comprehend services
    for chunk in text_chunks:
        try:
            # Detect key phrases
            key_phrases_response = comprehend_client.detect_key_phrases(Text=chunk, LanguageCode='en')
            key_phrases_list.extend([phrase['Text'] for phrase in key_phrases_response['KeyPhrases']])

            # Detect language
            language_response = comprehend_client.detect_dominant_language(Text=chunk)
            language_list.extend([language['LanguageCode'] for language in language_response['Languages']])

            # Detect targeted sentiment
            targeted_sentiment_response = comprehend_client.detect_sentiment(Text=chunk, LanguageCode='en')
            targeted_sentiments_list.append(targeted_sentiment_response['Sentiment'])

            # Detect PII entities
            pii_entities_response = comprehend_client.detect_pii_entities(Text=chunk, LanguageCode='en')
            pii_entities_list.extend([pii_entity.get('Text', '') for pii_entity in pii_entities_response.get('Entities', [])])

            # Keyphrase extraction
            keyphrase_extraction_response = comprehend_client.detect_key_phrases(Text=chunk, LanguageCode='en')
            keyphrase_extraction_list.extend(keyphrase_extraction_response['KeyPhrases'])

            # Syntax analysis
            syntax_analysis_response = comprehend_client.detect_syntax(Text=chunk, LanguageCode='en')
            syntax_analysis_list.extend(syntax_analysis_response['SyntaxTokens'])
        except boto3.exceptions.botocore.exceptions.EndpointConnectionError:
            print("Error connecting to Comprehend endpoint. Please check your internet connection.")
            sys.exit(1)
        except boto3.exceptions.botocore.exceptions.BotoCoreError as e:
            print(f"Error calling Comprehend: {e}")

    # Write additional results to summarized file in S3
    summarized_text += f"\nKey Phrases:{key_phrases_list}\n"
    summarized_text += f"Languages:{language_list}\n"
    summarized_text += f"Targeted Sentiments:{targeted_sentiments_list}\n"
    summarized_text += f"PII Entities:{pii_entities_list}\n"
    summarized_text += f"Keyphrase Extraction:{keyphrase_extraction_list}\n"
    summarized_text += f"Syntax Analysis:{syntax_analysis_list}\n"

    s3.put_object(Bucket=BUCKET_NAME, Key=SUMMARIZED_FILE_NAME, Body=summarized_text, ContentType='text/plain')

    # Use Amazon Translate to translate the transcribed text
    translate_client = boto3.client('translate', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

    # Translate each chunk and append to the result
    for chunk in text_chunks:
        try:
            translate_response = translate_client.translate_text(
                Text=chunk,
                SourceLanguageCode='en',
                TargetLanguageCode='hi'  # Change this to 'hi' for Hindi
            )
            translated_text += translate_response['TranslatedText']
        except boto3.exceptions.botocore.exceptions.EndpointConnectionError:
            print("Error connecting to Translate endpoint. Please check your internet connection.")
            sys.exit(1)
        except boto3.exceptions.botocore.exceptions.BotoCoreError as e:
            print(f"Error calling Translate: {e}")

    # Use Amazon Polly to convert the translated text back to speech
    polly_client = boto3.client('polly', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

    # Split the translated text into smaller chunks
    translated_text_chunks = textwrap.wrap(translated_text, 1000)  # Set an appropriate chunk size

    for idx, chunk in enumerate(translated_text_chunks):
        try:
            polly_response = polly_client.synthesize_speech(
                Text=chunk,
                OutputFormat='mp3',
                VoiceId='Aditi'  # Change this to 'Aditi' for Hindi voice
            )

            # Save each synthesized chunk to a file in S3
            chunk_mp3_key = f'translated_chunk_{idx}.mp3'
            s3.put_object(Bucket=BUCKET_NAME, Key=chunk_mp3_key, Body=polly_response['AudioStream'].read())
        except boto3.exceptions.botocore.exceptions.EndpointConnectionError:
            print("Error connecting to Polly endpoint. Please check your internet connection.")
            sys.exit(1)
        except boto3.exceptions.botocore.exceptions.BotoCoreError as e:
            print(f"Error calling Polly: {e}")

    # Rest of your code to further process and summarize the results...

else:
    print(f"Transcription job failed with status: {job['TranscriptionJob']['TranscriptionJobStatus']}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Entity Distribution Bar Chart
entity_types, entity_counts = zip(*entities_list)
sns.barplot(ax=axes[0, 0], x=entity_counts, y=entity_types, palette="viridis")
axes[0, 0].set_title("Entity Distribution")
axes[0, 0].set_xlabel("Count")
axes[0, 0].set_ylabel("Entity Types")

# Sentiment Analysis Pie Chart
sentiment_counts = sns.countplot(ax=axes[0, 1], x=targeted_sentiments_list, palette="muted")
axes[0, 1].set_title("Sentiment Analysis")
axes[0, 1].set_xlabel("Sentiment")
axes[0, 1].set_ylabel("Count")

# Key Phrase Word Cloud
key_phrases_text = ' '.join(key_phrases_list)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(key_phrases_text)
axes[1, 0].imshow(wordcloud, interpolation='bilinear')
axes[1, 0].set_title("Key Phrase Word Cloud")
axes[1, 0].axis('off')

# Language Distribution Bar Chart
sns.countplot(ax=axes[1, 1], x=language_list, palette="pastel")
axes[1, 1].set_title("Language Distribution")
axes[1, 1].set_xlabel("Language")
axes[1, 1].set_ylabel("Count")

# Adjust layout for better visualization
plt.tight_layout()

# Save the figure if needed
plt.savefig("visualization.png")

# Show the plots
plt.show()

