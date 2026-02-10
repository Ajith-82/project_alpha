from classes.output.email import EmailConfig, EmailServer

config = EmailConfig.from_json("email_config.json")
server = EmailServer(config)

server.send(
    subject="Test Email from Project Alpha",
    message="<h1>It works!</h1><p>Your email configuration is correct.</p>",
    mock=False  # Set to False to actually send
)