# Email Notification Setup Guide

Project Alpha uses standard SMTP to send daily reports. While Gmail has increased security for third-party apps, you can still use it securely via **App Passwords**.

## Option 1: Gmail (Recommended for Personal Use)

To use your Gmail account to send reports:

1.  **Enable 2-Step Verification** on your Google Account:
    *   Go to [Google Account Security](https://myaccount.google.com/security).
    *   Under "Signing in to Google," turn on **2-Step Verification**.

2.  **Generate an App Password**:
    *   Go to [App Passwords](https://myaccount.google.com/apppasswords).
    *   For "Select app," choose **Mail**.
    *   For "Select device," choose **Other (Custom name)** and enter `Project Alpha`.
    *   Click **Generate**.
    *   Copy the 16-character password shown.

3.  **Configure `email_config.json`**:
    Create a file named `email_config.json` in the project root:

    ```json
    {
        "email_credentials": {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "your_email@gmail.com",
            "smtp_password": "xxxx xxxx xxxx xxxx",  // Paste the 16-char App Password here
            "sender_id": "Project Alpha <your_email@gmail.com>"
        },
        "recipient_list": [
            "your_email@gmail.com"
        ]
    }
    ```

---

## Option 2: Transactional Email Service (Recommended for Reliability)

For better deliverability or if you don't want to use your personal Gmail, use a free tier from a provider like Brevo (formerly Sendinblue), SendGrid, or Mailgun.

### Example: Brevo (Free Plan: 300 emails/day)
1.  Sign up at [Brevo](https://www.brevo.com/).
2.  Go to **SMTP & API** settings.
3.  Generate a new SMTP Key.
4.  Configure `email_config.json`:

    ```json
    {
        "email_credentials": {
            "smtp_host": "smtp-relay.brevo.com",
            "smtp_port": 587,
            "smtp_user": "your_brevo_login_email@example.com",
            "smtp_password": "your_generated_smtp_key",
            "sender_id": "Project Alpha <no-reply@yourdomain.com>"
        },
        "recipient_list": [
            "recipient@example.com"
        ]
    }
    ```

---

## Testing Configuration

To verify your email setup works without running the full analysis:

```bash
# Create a test script
cat <<EOF > test_email.py
from classes.output.email import EmailConfig, EmailServer

config = EmailConfig.from_json("email_config.json")
server = EmailServer(config)

server.send(
    subject="Test Email from Project Alpha",
    message="<h1>It works!</h1><p>Your email configuration is correct.</p>",
    mock=False  # Set to False to actually send
)
EOF

# Run test
poetry run python test_email.py
```
