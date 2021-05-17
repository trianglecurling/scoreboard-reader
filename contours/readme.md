# Contour-based digit detection

Based on: https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python

Uses contours (edge-detection) to find regions that look like numbers. Compares them to the training set
using K-Nearest algorithm. Instead of manually inputting the training set, the labels are cached in 
responses.data. 

# Usage
```/>dotnet run```

This trains the k-nearest model and passes the scoreboard image as a test.
