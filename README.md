Steps I performed to create this api:
1. Used easyocr to detect text in the blueprint, and removed the regions with unwanted text and white spaces.
2. Loaded a pretrained model that is trained on CubiCasa5k dataset.
3. Since the input blueprints are large and of high resolution, the model failed to detect walls and rooms. So, I split the image into four quadrants, and fed each quadrant to the model.
4. The resultant segmentation of each quadrant is merged to get full floorplan segmentation.
5. The number of walls and rooms are then found out from segmentation result.
6. A fast api is created using the above steps: cropping, splitting into quadrants, feeding the quadrants to model
7. A dockerfile is created for the same
