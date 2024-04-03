import os
import slack_sdk
import dotenv
import asyncio
from tools.db import get_slack_thread_id, store_slack_thread_id 

dotenv.load_dotenv()

# Initialize the Slack client with the bot token from the environment variables
slack_client = slack_sdk.WebClient(token=os.getenv('SLACK_BOT_TOKEN'))

# Updated function to use Supabase for thread timestamps
async def send_slack_message(user_id: str, user: str, conversation_text: str):
    channel = "#bamboo-notifier"
    try:
        # Retrieve the thread ID from Supabase
        thread_id = await get_slack_thread_id(user_id)
        if thread_id:
            print("threeeaaaddd: ", thread_id)
            # Send the message in the thread corresponding to the user_id
            response = slack_client.chat_postMessage(
                channel=channel,
                text=f"User: {user}\nMessage: {conversation_text}",
                thread_ts=str(thread_id)
            )
        else:
            # Send the first message to the channel and store the thread timestamp
            response = slack_client.chat_postMessage(
                channel=channel,
                text=f"User ID: {user_id}\nUser: {user}\nMessage: {conversation_text}"
            )
            # Store the thread timestamp using the user_id as the key in Supabase
            await store_slack_thread_id(user_id, response['ts'])
    except Exception as e:
        print(f"Failed to send message to Slack: {e}")



"""#test
async def test():
    await send_slack_message("1", "agent", "hello yea!")



# Run the main function using asyncio.run() which handles creating an event loop
if __name__ == "__main__":
    asyncio.run(test())"""