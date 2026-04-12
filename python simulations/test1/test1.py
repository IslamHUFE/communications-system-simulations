import cv2

# رقم 0 يعبر عن الكاميرا الأولى، إذا لم تعمل جرب 1 أو 2
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("خطأ: لا يمكن الوصول للكاميرا. تأكد من الوصلات.")
else:
    print("تم الاتصال بالكاميرا بنجاح! اضغط 'q' للخروج.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # عرض البث
    cv2.imshow("USB over Ethernet Cam", frame)

    # الخروج عند الضغط على حرف q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
