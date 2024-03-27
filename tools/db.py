import os
from supabase import create_client, Client

# Initialize Supabase client
from dotenv import load_dotenv
load_dotenv()

url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

# Function to retrieve first_name by phone
def get_user_details_by_phone(phone: str) -> dict:
    data = supabase.table("users").select("first_name, language").eq("phone", phone).execute()
    if data.data and len(data.data) > 0:
        return {
            'first_name': data.data[0]['first_name'],
            'language': data.data[0]['language']
        }
    else:
        return {'first_name': "", 'language': ""}
