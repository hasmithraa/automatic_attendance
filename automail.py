import os
import yagmail

receiver = "preethikahari2004@gmail.com"  # receiver email address
body = "Attendence File"  # email body
filename = "Attendance"+os.sep+"Attendance_2019-08-29_13-09-07.csv"  # attach the file

# mail information
yag = yagmail.SMTP("preethehari2004@gmail.com", "hari@123")

# sent the mail
yag.send(
    to=receiver,
    subject="Attendance Report",  # email subject
    contents=body,  # email body
    attachments=filename,  # file attached
)
