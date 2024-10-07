Floorplan Detection API

Steps Performed to Create the API (testing.ipynb):

Text Detection:
Used EasyOCR to detect text in the blueprint and removed regions with unwanted text and white spaces.
Model Loading:
Loaded a pretrained model trained on the CubiCasa5k dataset.
Handling High-Resolution Blueprints:
As the input blueprints are large and high resolution, the model initially failed to detect walls and rooms. To solve this, the image was split into four quadrants, and each quadrant was fed to the model individually.
Merging Results:
The segmentation results from each quadrant were merged to obtain the full floorplan segmentation.
Wall and Room Detection:
After segmentation, the number of walls and rooms were detected.
FastAPI Creation:
A FastAPI application was created using the above steps:
Cropping, splitting into quadrants, feeding the quadrants to the model.
Docker Setup:
A Dockerfile was created to containerize the application.
How to Run the API and Test it Locally:

Run Locally:
Start the FastAPI server:

bash
Copy code
uvicorn app:app --host 0.0.0.0 --port 8000
Test the API using curl:

bash
Copy code
curl -X POST "http://localhost:8000/predict_image/" -F "file=@image4.png"
How to Run the API Using Docker Image:

Run with Docker:
Run the Docker image:

bash
Copy code
docker run -d -p 8000:8000 floorplan-fastapi
Test the API using curl:

bash
Copy code
curl -X POST "http://localhost:8000/predict_image/" -F "file=@image4.png"






