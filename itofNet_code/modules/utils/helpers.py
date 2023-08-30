import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def format_time(s_time: float, f_time: float):
    """
    Function used to format the time in a human readable format
        param:
            - s_time: start time
            - f_time: finish time
        return:
            - string containing the time in a human readable format
    """

    minutes, seconds = divmod(f_time - s_time, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 24:
        days, hours = divmod(hours, 24)
        return "%d:%02d:%02d:%02d" % (days, hours, minutes, seconds)
    return "%d:%02d:%02d" % (hours, minutes, seconds)


def send_email(receiver_email: str, subject: str, body: str):
    """
    Function used to send an email
        param:
            - receiver_email: email address of the receiver
            - subject: subject of the email
            - body: body of the email
    """

    email = "py.script.notifier@gmail.com"
    password = "sxruxiufydfhknov"

    message = MIMEMultipart()
    message["To"] = receiver_email
    message["From"] = "Python Notifier"
    message["Subject"] = subject

    messageText = MIMEText(body, "html")
    message.attach(messageText)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo("Gmail")
    server.starttls()
    server.login(email, password)
    server.sendmail(email, receiver_email, message.as_string())

    server.quit()
