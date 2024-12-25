## Description

[SDXL](https://arxiv.org/abs/2307.01952) consists of an [ensemble of experts](https://arxiv.org/abs/2211.01324) pipeline for latent diffusion: In a first step, the base model (available here: [https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)) is used to generate (noisy) latents, which are then further processed with a refinement model specialized for the final denoising steps. Note that the base model can be used as a standalone module.

Alternatively, we can use a two-stage pipeline as follows: First, the base model is used to generate latents of the desired output size. In the second step, we use a specialized high-resolution model and apply a technique called SDEdit ([https://arxiv.org/abs/2108.01073](https://arxiv.org/abs/2108.01073), also known as "img2img") to the latents generated in the first step, using the same prompt. This technique is slightly slower than the first one, as it requires more function evaluations.

The model is intended for research purposes only. Possible research areas and tasks include

- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.

# Evaluation Results

This [chart](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/comparison.png) evaluates user preference for SDXL (with and without refinement) over SDXL 0.9 and Stable Diffusion 1.5 and 2.1. The SDXL base model performs significantly better than the previous variants, and the model combined with the refinement module achieves the best overall performance.

# Limitations and Biases

## Limitations

- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.

## Bias

While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.

## Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

# License

CreativeML Open RAIL++-M License

# Inference Samples

> Note: The inferencing script of this model is optimized for high-throughput, low latency using [Deepspedd-mii](https://github.com/microsoft/DeepSpeed-MII) library. Please use `version 4` of this model for inferencing using default (FP32) diffusion pipeline implementation.

|Inference type|Python sample (Notebook)|CLI with YAML|
|---|---|---|
|Real time|[text-to-image-online-endpoint.ipynb](https://aka.ms/azureml-infer-sdk-text-to-image)|[text-to-image-online-endpoint.sh](https://aka.ms/azureml-infer-cli-text-to-image)|
|Batch|[text-to-image-batch-endpoint.ipynb](https://aka.ms/azureml-infer-batch-sdk-text-to-image)|[text-to-image-batch-endpoint.sh](https://aka.ms/azureml-infer-batch-cli-text-to-image)|

### Inference with [Azure AI Content Safety (AACS)](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart) samples

|Inference type|Python sample (Notebook)|
|---|---|
|Real time|[safe-text-to-image-online-deployment.ipynb](https://aka.ms/azureml-infer-sdk-safe-text-to-image)|
|Batch|[safe-text-to-image-batch-endpoint.ipynb](https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image)|

# Sample input and output

### Sample input

```json
{
   "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0],
    }
}
```

### Sample output

```json
[
    {
        "prompt": "a photograph of an astronaut riding a horse",
        "generated_image": "image",
        "nsfw_content_detected": null
    }
]
```

> Note:
> - "image" string is in base64 format.
> - The `stabilityai-stable-diffusion-xl-base-1-0` model doesn't check for the NSFW content in generated image. We highly recommend to use the model with [Azure AI Content Safety (AACS)](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart). Please refer sample [online](https://aka.ms/azureml-infer-sdk-safe-text-to-image) and [batch](https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image) notebooks for AACS integrated deployments.

### Azure AI Connection Info:

##### Target:
https://azureaihubeast8881743221.cognitiveservices.azure.com/
##### API Key:
92b61586478844e2bfabfe9cdc1a4c88
### Basic Consumption Info:
##### REST Endpoint:
https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/score
##### Swagger URI:
https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/swagger.json
##### Primary Key:
cTyAeKyjpc01eyhZCjLy8v2DMCynGaR2
##### Secondary Key:
yvrJZ8Vur2qCI3ieBZGLIvEqwFAXe775

##### Consumption Options:

```javascript
// Request data goes here

// The example below assumes JSON formatting which may be updated

// depending on the format your endpoint expects.

// More information can be found here:

// https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

const requestBody = `{}`;

  

const requestHeaders = new Headers({"Content-Type" : "application/json"});

  

// Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint

const apiKey = "";

if (!apiKey)

{

    throw new Error("A key should be provided to invoke the endpoint");

}

requestHeaders.append("Authorization", "Bearer " + apiKey)

  

// This header will force the request to go to a specific deployment.

// Remove this line to have the request observe the endpoint traffic rules

requestHeaders.append("azureml-model-deployment", "stabilityai-stable-diffusion--5");

  

const url = "https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/score";

  

fetch(url, {

  method: "POST",

  body: requestBody,

  headers: requestHeaders

})

    .then((response) => {

    if (response.ok) {

        return response.json();

    } else {

        // Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure

        console.debug(...response.headers);

        console.debug(response.body)

        throw new Error("Request failed with status code" + response.status);

    }

    })

    .then((json) => console.log(json))

    .catch((error) => {

        console.error(error)

    });
```

```python
import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):

    # bypass the server certificate verification on client side

    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):

        ssl._create_default_https_context = ssl._create_unverified_context

  

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here

# The example below assumes JSON formatting which may be updated

# depending on the format your endpoint expects.

# More information can be found here:

# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

data = {}

body = str.encode(json.dumps(data))

url = 'https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/score'

# Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint

api_key = ''
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")
  

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}  

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure

    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```

---

With the information above, walk me through being able to connect to this new stable diffusion XL model I have deployed so I can test it with prompts



To connect to and test the Stable Diffusion XL (SDXL) model you’ve deployed, we'll go through the following steps:

1. **Understand the Deployment Information**: Gather all necessary information from the provided deployment details.
2. **Set Up Authentication**: Use the API key to authenticate your requests.
3. **Formulate a Request**: Prepare a JSON payload that includes your prompt.
4. **Send a Request to the Endpoint**: Use a programming language like Python or JavaScript to send an HTTP POST request.
5. **Handle the Response**: Process and display the generated image or any errors returned by the service.

### Step 1: Understand Deployment Information

From your deployment details, here’s what we have:

- **REST Endpoint**: `https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/score`
- **Primary Key**: `cTyAeKyjpc01eyhZCjLy8v2DMCynGaR2`
- **Secondary Key**: `yvrJZ8Vur2qCI3ieBZGLIvEqwFAXe775`
- **API Endpoint Target URL** (for Azure AI Connection): `https://azureaihubeast8881743221.cognitiveservices.azure.com/`

### Step 2: Set Up Authentication

You’ll use one of the provided keys (`Primary Key` or `Secondary Key`) to authenticate your requests.

### Step 3: Formulate a Request

The request body should be in JSON format and include your input prompt. For example:

```json
{
   "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0]
    }
}
```

### Step 4: Send a Request to the Endpoint

#### Option 1: Using Python

Here’s how you can set up and send a request using Python:

```python
import urllib.request
import json
import ssl
import os

# Allow self-signed HTTPS certificates if needed
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', ''):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

# Define your API endpoint URL and API key
url = 'https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/score'
api_key = 'cTyAeKyjpc01eyhZCjLy8v2DMCynGaR2'

# Define your input data (the prompt)
data = {
    "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0]
    }
}

# Convert data to JSON string format for sending in HTTP POST request
body = str.encode(json.dumps(data))

# Set up HTTP headers, including authorization with API key
headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

# Create request object with URL, body, and headers
req = urllib.request.Request(url, body, headers)

try:
    # Send request and read response data
    response = urllib.request.urlopen(req)
    result = response.read()
    
    # Print out result (base64 encoded image string)
    print(result.decode('utf-8'))
except urllib.error.HTTPError as error:
    print("The request failed with status code:", error.code)
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```

#### Option 2: Using JavaScript (Fetch API)

Here’s how you can set up and send a request using JavaScript:

```javascript
const url = "https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/score";
const apiKey = "cTyAeKyjpc01eyhZCjLy8v2DMCynGaR2";

const data = {
   "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0],
   }
};

const requestHeaders = new Headers({
   "Content-Type" : "application/json",
   "Authorization" : `Bearer ${apiKey}`
});

fetch(url, {
  method: "POST",
  body: JSON.stringify(data),
  headers: requestHeaders,
})
.then((response) => {
  if (response.ok) {
      return response.json();
  } else {
      console.debug(...response.headers);
      console.debug(response.body);
      throw new Error("Request failed with status code" + response.status);
  }
})
.then((json) => console.log(json))
.catch((error) => console.error(error));
```

### Step 5: Handle the Response

The service will return a JSON object where one of its fields is `"generated_image"` containing the base64 encoded image string.

For example:

```json
[
    {
        "prompt": "a photograph of an astronaut riding a horse",
        "generated_image": "/9j/4AAQSkZJRgABAQEAAAAAAAD..."
    }
]
```

You can then decode this base64 string back into an image using appropriate tools in Python or JavaScript.

#### Example in Python:

To save this base64 encoded image as an actual image file:

```python
import base64

image_base64_string = "/9j/4AAQSkZJRgABAQEAAAAAAAD..."

with open("output_image.png", "wb") as file:
    file.write(base64.b64decode(image_base64_string))
```

This will create an `output_image.png` file on disk containing your generated image.

---

By following these steps, you should be able to successfully connect to your deployed SDXL model endpoint, send prompts for generation, and handle responses effectively!

Provide a more complete, in-depth walk through of the process from steps 3, 4 and 5 for Python. When faced with a significant technical decision, make your choice based on the platform and model we are using



Certainly! Let's dive deeper into steps 3, 4, and 5 for connecting to your deployed Stable Diffusion XL (SDXL) model using Python. We'll cover formulating the request, sending it to the endpoint, and handling the response in a detailed manner.

### Step 3: Formulate a Request

#### Understanding Your Input Data

The input data for the SDXL model consists of JSON formatted data where you specify prompts that guide the image generation process. Here's how you can structure this data:

- **Columns**: Define the structure of your input data. In this case, it will be `["prompt"]`.
- **Data**: Contains an array with your prompt text.
- **Index**: Optional but useful if you're tracking multiple requests or need consistent indexing.

#### Example Input JSON

```json
{
   "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0]
    }
}
```

This JSON object is what you'll send as part of your HTTP POST request body.

### Step 4: Send a Request to the Endpoint

#### Setting Up Your Python Environment

Ensure you have necessary libraries installed. You'll need `urllib` (part of Python's standard library), and optionally `ssl` if you're dealing with self-signed certificates.

#### Writing Your Python Code

1. **Import Necessary Modules**:
   - `urllib.request`: For making HTTP requests.
   - `json`: For handling JSON data.
   - `ssl`: To handle SSL certificate verifications if needed.

2. **Configure SSL (Optional)**:
   If your deployment uses self-signed certificates, configure SSL to allow these:

   ```python
   import ssl
   import os

   def allowSelfSignedHttps(allowed):
       if allowed and not os.environ.get('PYTHONHTTPSVERIFY', ''):
           ssl._create_default_https_context = ssl._create_unverified_context
   
   allowSelfSignedHttps(True)
   ```

3. **Define API Endpoint and Key**:
   Set up variables for your API endpoint URL and authentication key:

   ```python
   url = 'https://azureaiprojecteastus-ucvzk.eastus.inference.ml.azure.com/score'
   api_key = 'cTyAeKyjpc01eyhZCjLy8v2DMCynGaR2'
   ```

4. **Prepare Your Request Data**:
   
```python
data = {
    "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0]
    }
}
body = str.encode(json.dumps(data))
```

5. **Set Up Headers**:
   
```python
headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
```

6. **Create and Send Request**:

```python
import urllib.request

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)
    result = response.read()
    print(result.decode('utf-8'))
except urllib.error.HTTPError as error:
    print("The request failed with status code:", error.code)
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```

### Step 5: Handle the Response

After sending your request, you'll receive a response from the server containing either success or error information.

#### Processing Successful Responses

If successful, you'll get a JSON response which includes:

- The original prompt.
- The generated image encoded in base64 format.

Example Response:

```json
[
    {
        "prompt": "a photograph of an astronaut riding a horse",
        "generated_image": "/9j/4AAQSkZJRgABAQEAAAAAAAD..."
    }
]
```

#### Decoding Base64 Image in Python

To convert this base64 string back into an image file:

1. Import `base64` module.
2. Decode base64 string.
3. Save it to disk as an image file.

Example Code:

```python
import base64

# Extract base64 string from response (assuming result is already parsed JSON)
image_base64_string = "/9j/4AAQSkZJRgABAQEAAAAAAAD..."

# Decode and write to file
with open("output_image.png", "wb") as file:
    file.write(base64.b64decode(image_base64_string))
```

This will create an image file (`output_image.png`) that you can view using any standard image viewer.

### Conclusion & Technical Decisions 

In this walkthrough, we chose Python due to its robust libraries for HTTP requests (`urllib`) and native support for handling JSON data (`json`). This choice aligns well with Azure’s deployment environment because Python scripts are easy to deploy on cloud platforms like Azure ML services.

By following these steps thoroughly, you should be able to test out different prompts using your deployed SDXL model effectively!