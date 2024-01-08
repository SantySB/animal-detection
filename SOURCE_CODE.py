import cv2
import pyttsx3


def mail_alert():
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    import smtplib
    from email.mime.base import MIMEBase
    from email import encoders

    strFrom = 'eshwarem26@gmail.com'
    strTo = 'sureshharisureshhari75@gmail.com'

    # Create the root message and fill in the from, to, and subject headers
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'alert from software'
    msgRoot['From'] = strFrom
    msgRoot['To'] = strTo
    msgRoot.preamble = 'This is a multi-part message in MIME format.'

    msgAlternative = MIMEMultipart('alternative')

    msgRoot.attach(msgAlternative)

    mail_message_Text = MIMEText('Bending detected')

    msgAlternative.attach(mail_message_Text)

    sending_image = open('animal.jpg', 'rb')

    msgImage = MIMEImage(sending_image.read())

    sending_image.close()

    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<image1>')

    msgRoot.attach(msgImage)

    smtp = smtplib.SMTP('smtp.gmail.com', 587)

    smtp.starttls()

    smtp.login('eshwarem26@gmail.com', 'cvdhhnzfmbtoeacx')

    print("mail id and password correct")

    smtp.sendmail(strFrom, strTo, msgRoot.as_string())

    print("mail send")

engine=pyttsx3.init()
thres = 0.45 # Threshold to detect object
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.data'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'MODEL_TRAINNED.pbtxt'
weightsPath = 'ANIMALS_DATASET.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)





def outdoor():
    font = cv2.FONT_HERSHEY_COMPLEX
    # org
    org = (20, 100)  # coloum-row
    org1 = (10, 50)  # 10,50
    org2 = (420, 370)
    org3 = (250, 440)
    org4 = (290, 470)
    # fontScale
    fontScale = 0.9
    fontScale1 = 1.0
    # Blue color in BGR
    color = (25, 255, 50)
    color1 = (255, 191, 0)  # sky blue deep
    color2 = (255, 255, 255)
    # Line thickness of 2 px
    thickness1 = 4
    thickness = 2



    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=0.45)


        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):

                object_name=(classNames[classId-1])
                object_id=[classId-1]
                print(object_id)

                if object_id == [20]:
                     print(" COW detected  ")
                     cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                     cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)
                     cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                     cv2.imwrite("animal.jpg",img)
                     cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                     mail_alert()

                if object_id == [21]:
                    print(" ELEPHANT detected  ")
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite("animal.jpg", img)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    mail_alert()

                if object_id == [19]:  #
                    print(" ANIMAL detected  ")
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite("animal.jpg", img)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    mail_alert()

        cv2.putText(img, 'SSEC COLLEGE', (00, 370), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1,cv2.LINE_AA)
        cv2.imshow("",img)
        cv2.waitKey(1)

        k = cv2.waitKey(27) & 0xff
        if k == 27:
            break

outdoor()