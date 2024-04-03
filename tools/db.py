import os
from supabase import create_client, Client

# Initialize Supabase client
from dotenv import load_dotenv
load_dotenv()

url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

# Function to retrieve user details by phone
def get_user_details_by_phone(phone: str) -> dict:
    data = supabase.table("users").select("first_name, language, system_prompt, intro_message").eq("phone", phone).execute()
    if data.data and len(data.data) > 0:
        return {
            'first_name': data.data[0]['first_name'],
            'language': data.data[0]['language'],
            'system_prompt': data.data[0]['system_prompt'],
            'intro_message': data.data[0]['intro_message']
        }
    else:
        return {'first_name': "", 'language': "", 'system_prompt': "", 'intro_message': ""}

async def create_conversation(user_id: str, conversation_text: str, user: str) -> dict:
    try:
        insert_response = await supabase.table("conversations").insert({
            "user_id": user_id,
            "conversation_text": conversation_text,
            "user": user
        }).execute()

        if insert_response.error:
            return {"error": str(insert_response.error)}
        else:
            return {"data": "Conversation created successfully."}
    except Exception as e:
        return {"error": str(e)}

# Function to retrieve a Slack thread ID by user ID
async def get_slack_thread_id(user_id: str) -> str:
    data = supabase.table("slack_threads").select("thread_id").eq("user_id", user_id).execute()
    if data.data and len(data.data) > 0:
        return data.data[0]['thread_id']
    else:
        return ""

# Function to store a new Slack thread ID
async def store_slack_thread_id(user_id: str, thread_id: str) -> dict:
    try:
        insert_response = supabase.table("slack_threads").insert({
            "user_id": user_id,
            "thread_id": thread_id
        }).execute()

        # Check if the operation was successful
        if insert_response.status_code in [200, 201]:  # 200 OK or 201 Created
            return {"data": "Thread ID stored successfully."}
        else:
            # Handle unsuccessful operation
            return {"error": f"Failed to store thread ID. Status code: {insert_response.status_code}"}
    except Exception as e:
        # Handle any exceptions that occurred during the operation
        return {"error": str(e)}
