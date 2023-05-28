## Text-to-Image Generation Website

This repository contains a Flask-based website that generates images from text using a pre-trained model. The generated image is displayed on the website for the given input text.

### Folder Structure
- website/
    - main.py
    - requirements.txt
    - templates/
    - static/
    - models/

### Usage
1. Install the required packages using `pip install -r requirements.txt`.

### Usage

1. Start the Flask server by running the following command:

```bash
python main.py --path_save_vae {path_model_vae.pt} --path_save_base {path_model_base.pt}
```

2. Navigate to `http://localhost:8000` in your web browser to access the app.
3. Enter a text input and click the "Generate Image" button.
4. The app will generate an image corresponding to the text input, and display it on the webpage.


### API Endpoints

- GET `/`: Renders the home page of the website.

- POST `/image`: Accepts a JSON payload containing the input text and returns the generated image URL in the JSON response.

### Code Explanation
`main.py`
This file contains the Flask application code. It defines the routes and handles the requests for generating images.

- `index()`: Renders the home page of the website.

- `generate_image_endpoint()`: Accepts a POST request with JSON payload containing the input text. It calls the inference() function from the models.backend_model module to generate an image based on the input text and returns the image URL in the JSON response.

`models/backend_model.py`
This module contains the backend logic for generating images from text.

- `inference(text)`: Takes an input text as a parameter and generates an image based on the text using a pre-trained model. The generated image is saved in the static/images/gen_image.png file and the file path is returned.

### Additional Notes
When running in debug mode (`is_debug = True`), a fake image is returned for any input text. This is useful for testing the website without loading the actual model.