import boto3
import time
import textwrap
import spacy
import json
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# AWS configuration
ACCESS_KEY = ''
SECRET_KEY = ''
REGION = ''
BUCKET_NAME = ''  # Removed trailing space
AUDIO_FILE_NAME = 'Arthur.mp3'
TRANSCRIPT_FILE_NAME = 't.txt'
SUMMARIZED_FILE_NAME = 's.txt'

# Generate a unique job name with a timestamp
transcription_job_name = f'transcription_job_{int(time.time())}'

# Create AWS clients
transcribe_client = boto3.client('transcribe', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
comprehend_client = boto3.client('comprehend', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
translate_client = boto3.client('translate', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
polly_client = boto3.client('polly', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

def generate_direct_summary(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])

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
    response = s3.get_object(Bucket=BUCKET_NAME, Key=TRANSCRIPT_FILE_NAME)
    content = json.loads(response['Body'].read().decode('utf-8'))
    transcribed_text = content['results']['transcripts'][0]['transcript']

    # Generate direct summary
    direct_summary = generate_direct_summary(transcribed_text)

    # Process the text with spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(transcribed_text)

    # Extract entities
    entities_list = []
    for ent in doc.ents:
        entities_list.append((ent.text, ent.label_))

    # Initialize summarized text
    summarized_text = f"Direct Summary:\n{direct_summary}\n\n"
    summarized_text += "Entities:\n" + "\n".join(map(str, entities_list)) + "\n\n"

    # Split the transcribed text into chunks
    chunk_size = 5000  # Set an appropriate chunk size
    text_chunks = textwrap.wrap(transcribed_text, chunk_size)

    # Initialize lists to store results from each chunk
    key_phrases_list = []
    language_list = []
    targeted_sentiments_list = []
    pii_entities_list = []
    keyphrase_extraction_list = []
    syntax_analysis_list = []

    # Process each chunk for various Comprehend services
    for chunk in text_chunks:
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
        pii_entities_list.extend([pii_entity.get('Type', '') for pii_entity in pii_entities_response.get('Entities', [])])

        # Keyphrase extraction
        keyphrase_extraction_response = comprehend_client.detect_key_phrases(Text=chunk, LanguageCode='en')
        keyphrase_extraction_list.extend(keyphrase_extraction_response['KeyPhrases'])

        # Syntax analysis
        syntax_analysis_response = comprehend_client.detect_syntax(Text=chunk, LanguageCode='en')
        syntax_analysis_list.extend(syntax_analysis_response['SyntaxTokens'])

    # Write additional results to summarized text
    summarized_text += f"Key Phrases: {key_phrases_list[:10]}\n"
    summarized_text += f"Languages: {list(set(language_list))}\n"
    summarized_text += f"Overall Sentiment: {max(set(targeted_sentiments_list), key=targeted_sentiments_list.count)}\n"
    summarized_text += f"PII Entities: {list(set(pii_entities_list))}\n"
    summarized_text += f"Top Keyphrases: {[kp['Text'] for kp in keyphrase_extraction_list[:5]]}\n"
    summarized_text += f"Syntax Analysis Sample: {syntax_analysis_list[:5]}\n"

    # Save summarized content to S3
    s3.put_object(Bucket=BUCKET_NAME, Key=SUMMARIZED_FILE_NAME, Body=summarized_text, ContentType='text/plain')

    # Translate the transcribed text
    translated_text = ""
    for chunk in text_chunks:
        translate_response = translate_client.translate_text(
            Text=chunk,
            SourceLanguageCode='en',
            TargetLanguageCode='hi'  # Hindi
        )
        translated_text += translate_response['TranslatedText']

    # Use Amazon Polly to convert the translated text back to speech
    translated_text_chunks = textwrap.wrap(translated_text, 1000)  # Set an appropriate chunk size

    for idx, chunk in enumerate(translated_text_chunks):
        polly_response = polly_client.synthesize_speech(
            Text=chunk,
            OutputFormat='mp3',
            VoiceId='Aditi'  # Hindi voice
        )

        # Save each synthesized chunk to a file in S3
        chunk_mp3_key = f'translated_chunk_{idx}.mp3'
        s3.put_object(Bucket=BUCKET_NAME, Key=chunk_mp3_key, Body=polly_response['AudioStream'].read())

    print("Processing completed successfully.")
else:
    print(f"Transcription job failed with status: {job['TranscriptionJob']['TranscriptionJobStatus']}")
