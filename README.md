# City-Segmentation-Inpainting

## An image segmentation and replacement model for landscapes
City-Segmentation-Inpainting is an image segmentation model that allows you to remove specific classes within an image and then replace them with diffusion based in fill. 

## Dataset
The data used to train this model can be found [here](https://www.coursera.org/learn/convolutional-neural-networks). A more detailed discussion of its limitations and potential pitfalls can be found in the [Ethics statement](Ethics_statement.md).

## Scripts

### Naïve model
in the [Naïve model](scripts-final/Naive.py) there are very simple functions for converting to greyscale, segmenting by a set amount and then reconstructing the image. An example of usage is given in the 'Example use' function.

### Classical model

In the [Classical model](scripts-final/ML.py) there is extensive functions for testing and evaluation of the segmentation technique. A demonstration of these evaluation techniques is shown in the eval_on_dataset function. The segment_Kmeans function provides the core functionality for the segmentation, it should be passed an image and returns the segmented version.

### Deep Learning model

The [Deep Learning Model](scripts-final/Deep-Learning.py) model is a U-Net model trained from scratch, the training code for this is included in this script. This is by far our most accurate model.

## Setup and running of the main model

### Setting Up a Virtual Environment

To manage dependencies, it's recommended to use a virtual environment.

#### 1. Create a Virtual Environment

```bash
python -m venv venv
```

#### 2. Activate the Virtual Environment

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows (Command Prompt):**

  ```bash
  venv\Scripts\activate
  ```

- **On Windows (PowerShell):**

  ```bash
  venv\Scripts\Activate.ps1
  ```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run [main](main.py)

## Models

There are no models stored directly in this repo, this is because K-Means is not stored as a model and the U-Net is too large, and thus hosted externally.

## Hosted model

You can find our model hosted [here](https://huggingface.co/vihaannnn/City_Segmentation_UNet) on huggingface

## Notebooks

Notebooks folder includes the notebook for training the U-net. 

## Web App

The web app is included [here](https://city-segmentation-inpainting-gcpa28fv6enxw9ategareh.streamlit.app/), this demonstrates the entire project as an interactive experience. A video demonstration is shown [here](https://youtu.be/NKNGBUchROA).