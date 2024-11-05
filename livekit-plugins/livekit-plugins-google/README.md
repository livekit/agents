# LiveKit Plugins Google

Agent Framework plugin for services from Google Cloud. Currently supporting Google's [Speech-to-Text](https://cloud.google.com/speech-to-text) API.

## Installation

```bash
pip install livekit-plugins-google
```

## Pre-requisites

For credentials, you'll need a Google Cloud account and obtain the correct credentials. Credentials can be passed directly or via Application Default Credentials as specified in [How Application Default Credentials works](https://cloud.google.com/docs/authentication/application-default-credentials). For more information see [Authentication Guide](#authentication-guide).

## Vertex AI pre-requisites
- `GOOGLE_APPLICATION_CREDENTIALS` variable should be set to the path of the service account key file.
- project: The Google Cloud project ID to use.
- location: The location of the Vertex AI API endpoint. i.e. 'us-central1'.



## Authentication Guide

#### **1. Create a Service Account and Download the Key File**

A **service account** is a special kind of account used by applications and virtual machines to make authorized API calls.

- **a. Go to the [Google Cloud Console](https://console.cloud.google.com/).**
  
- **b. Navigate to:**  
  `IAM & Admin` > `Service Accounts`

- **c. Create a New Service Account:**
  - Click on `+ CREATE SERVICE ACCOUNT`.
  - Provide a **Service account name** and **ID**.
  - Click `CREATE AND CONTINUE`.

- **d. Assign Roles:**
  - Assign the necessary roles to the service account (e.g., `AI Platform Admin`, `Viewer`, etc., depending on your needs).
  - Click `CONTINUE`.

- **e. Create Key:**
  - After creating the service account, click on it to open details.
  - Navigate to the `KEYS` tab.
  - Click `ADD KEY` > `Create new key`.
  - Choose `JSON` as the key type and click `CREATE`.
  - **Download and securely store** the JSON key file.

#### **2. Set the `GOOGLE_APPLICATION_CREDENTIALS` Environment Variable**

Your application needs to know where to find the service account key file.

- **a. Locate the Path to Your JSON Key File:**
  - For example: `C:\Users\Jp\path\to\your\service-account-key.json`

- **b. Set the Environment Variable:**

  **On Windows:**

  - **Using Command Prompt:**
    ```cmd
    set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Jp\path\to\your\service-account-key.json
    ```
  
  - **Permanently via System Settings:**
    1. Open **Start Menu** and search for `Environment Variables`.
    2. Click `Edit the system environment variables`.
    3. In the **System Properties** window, click `Environment Variables...`.
    4. Under **User variables** or **System variables**, click `New...`.
    5. Set:
       - **Variable name:** `GOOGLE_APPLICATION_CREDENTIALS`
       - **Variable value:** `C:\Users\Jp\path\to\your\service-account-key.json`
    6. Click `OK` to save.

  **On macOS/Linux:**

  - **Using Terminal:**
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    ```
  
  - **Permanently via Shell Configuration:**
    - Add the export command to your shell's configuration file (e.g., `.bashrc`, `.zshrc`).





### **Summary**

1. **Create a Service Account** with the necessary roles in the Google Cloud Console.
2. **Download the Service Account JSON Key File.**
3. **Set the `GOOGLE_APPLICATION_CREDENTIALS` Environment Variable** to point to the JSON key file.
4. **Ensure the Service Account Has Appropriate Permissions** for the operations your application needs.


### **Useful Resources**

- **Setting Up Application Default Credentials (ADC):**
  - [Google Cloud Documentation](https://cloud.google.com/docs/authentication/production)

- **Creating and Managing Service Accounts:**
  - [Google Cloud Service Accounts](https://cloud.google.com/iam/docs/understanding-service-accounts)

- **Best Practices for Managing Service Account Keys:**
  - [Google Cloud IAM Best Practices](https://cloud.google.com/iam/docs/best-practices)
