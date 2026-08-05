from retinaface import RetinaFaceDetector

detector = RetinaFaceDetector()

# Your image
image, faces = detector.detect('withoutmask.jpeg')

if faces:
    print(f"✓ Found {len(faces)} faces!")
    for i, face in enumerate(faces):
        print(f"  Face {i+1}: confidence={face.confidence:.3f}")
else:
    print("No faces detected")