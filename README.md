# **Floorplan Detection API**

## **Steps Performed to Create the API (testing.ipynb):**
1. **Text Detection:**  
   Used EasyOCR to detect text in the blueprint and removed regions with unwanted text and white spaces.

2. **Model Loading:**  
   Loaded a pretrained model trained on the **CubiCasa5k** dataset.

3. **Handling High-Resolution Blueprints:**  
   As the input blueprints are large and high resolution, the model initially failed to detect walls and rooms. To solve this, the image was split into **four quadrants**, and each quadrant was fed to the model individually.

4. **Merging Results:**  
   The segmentation results from each quadrant were merged to obtain the full floorplan segmentation.

5. **Wall and Room Detection:**  
   After segmentation, the number of walls and rooms were detected.

6. **FastAPI Creation:**  
   A FastAPI application was created using the above steps:  
   **Cropping, splitting into quadrants, feeding the quadrants to the model.**

7. **Docker Setup:**  
   A **Dockerfile** was created to containerize the application.

---

## **How to Run the API and Test it:**

### **Run Locally:**
Start the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Testing the FastAPI server:
```bash
curl -X POST "http://localhost:8000/predict_image/" -F "file=@image4.png"
```


### **Run Using Docker Image:**
Start the FastAPI docker image:
```bash
docker run -d -p 8000:8000 floorplan-fastapi
```
Testing the FastAPI server:
```bash
curl -X POST "http://localhost:8000/predict_image/" -F "file=@image4.png"
```







