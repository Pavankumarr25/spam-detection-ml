from src.predict import predict_message

print("=== Spam Detection System ===")

while True:
    msg = input("\nEnter message (or type 'exit'): ")

    if msg.lower() == "exit":
        break

    result = predict_message(msg)
    print("Prediction:", result)