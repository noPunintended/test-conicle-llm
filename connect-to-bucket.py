from oauth2client.service_account import ServiceAccountCredentials
from gcloud import storage

SERVICE_ACCOUNT_FILE = 'config/conicle-ai.json'

credentials = ServiceAccountCredentials.from_json_keyfile_name(
    SERVICE_ACCOUNT_FILE)
storage_client = storage.Client(credentials=credentials, project='conicle-ai')
bucket = storage_client.get_bucket('conicle-ai-conicle-x-audio')
