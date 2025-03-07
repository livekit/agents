# Building a Dentist Scheduler Assistant
Build an AI-powered dentist assistant that manages appointments, takes messages, and answers inquiries

This example integrates the Cal.com and Supabase APIs with LiveKit and exhibits the functionality of a multi-agent framework. To curate a complete assistant, this recipe includes a Recepionist, Scheduler, and Messenger.

# Prerequisites
For this example, you need to create an VoiceAgent using the Voice Agent quickstart.

## Setting up Cal.com
You need to create an account and enroll in a plan.The .env.local file should have your API key and Cal.com username:
CAL_API_KEY="<cal_api_key>"
CAL_API_USERNAME="<cal_api_username>"

We'll set up the details for this example for you in-house with setup_event_types(). This will create a schedule and necessary event types.

## Setting up Supabase
1. Create an account and a new project for this example. 
2. After configuring your RLS settings, create a table in the public schema. 

Your table should have these text columns: "name", "message", "phone_number". You can optionally add a datetime column that automatically records the time the message was received.

Here is what your table should look like:


3. Add your project API key and URL into the .env.local file:
SUPABASE_API_KEY="<supabase_api_key>"
SUPABASE_URL="<supabase_project_url>"

# Understanding AgentTask
AgentTask allows you to establish a coherent workflow and execute actions appropriately.

## Building the Receptionist 
The Receptionist handles simple inquiries about the office and transfers you to either the Scheduler or Messenger upon request.

## Building the Scheduler
The Scheduler is able to schedule, reschedule, or cancel appointments. 

## Building the Messenger
The Messenger records user messages and sends it to Supabase. 