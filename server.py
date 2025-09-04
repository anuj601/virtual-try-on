from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import mediapipe as mp
import uuid
import os
import uvicorn 

app = FastAPI(title="Virtual Try-On API")

# Configure CORS to allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Pose (do this once when the server starts)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Ensure a directory exists for uploaded/processed images
os.makedirs("static", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Virtual Try-On API is ready for action!"}

@app.post("/detect-pose/")
async def detect_pose(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image, runs pose detection on it,
    and returns the processed image with landmarks drawn.
    """
    # Generate a unique filename to avoid overwrites
    file_id = str(uuid.uuid4())
    input_path = f"static/{file_id}_input.jpg"
    output_path = f"static/{file_id}_output.jpg"

    # 1. Save the uploaded file
    with open(input_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # 2. Read the image and process it with MediaPipe
    image = cv2.imread(input_path)
    if image is None:
        return {"error": "Could not read the uploaded image."}

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 3. Draw the pose annotations on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 4. Save the processed image
    cv2.imwrite(output_path, image)

    # 5. Return the path to the processed image
    return {"processed_image_url": f"/static/{file_id}_output.jpg"}

# This endpoint serves the processed image file
@app.get("/static/{file_name}")
async def get_image(file_name: str):
    return FileResponse(f"static/{file_name}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)