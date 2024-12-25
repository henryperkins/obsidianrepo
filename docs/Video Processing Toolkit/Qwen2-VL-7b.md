<https://www.alibabacloud.com/help/en/model-studio/developer-reference/qwen-vl-quick-start>

# **Qwen-VL-Max**

## Documentation from Alibiba Cloud

Qwen's Most Capable Large Visual Language Model. Compared to the enhanced version, further improvements have been made to visual reasoning and instruction-following capabilities, offering a higher level of visual perception and cognitive understanding. It delivers optimal performance on an even broader range of complex tasks.

Model: qwen-vl-max

To manage model access and cost, the sub-workspaces require your permissions to call, train, and deploy models. Click the button below to grant the permissions.

```python
# Refer to the document for workspace information: https://www.alibabacloud.com/help/en/model-studio/developer-reference/model-calling-in-sub-workspace    
        
from dashscope import MultiModalConversation
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'


def simple_multimodal_conversation_call():
    """Simple single round multimodal conversation call.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
                {"text": "这是什么?"}
            ]
        }
    ]
    responses = MultiModalConversation.call(model='qwen-vl-max',
                                           messages=messages,
                                           stream=True)
    for response in responses:
        print(response)


if __name__ == '__main__':
    simple_multimodal_conversation_call()
```

# **Brief Introduction - From Alibaba**

**Qwen-VL** is a Large Vision Language Model (LVLM) developed by Alibaba Cloud. Qwen-VL can take images, text, and detection boxes as inputs, and text and detection boxes as outputs. Features of the Qwen-VL series models include:

- **Powerful performance**: In the standard English evaluation of four types of multi-modal tasks (Zero-shot Caption/VQA/DocVQA/Grounding), the best results are achieved under the same general model size;
- **Multilingual dialogue model**: Naturally supports multilingual dialogue, end-to-end support for long text recognition in Chinese and English.
- **Multi-picture interleaved dialogue**: support multi-picture input and comparison, designated picture Q&A, multi-picture literary creation, etc.;
- The first general model that supports Chinese open domain positioning: detection box annotation through Chinese open domain language expression;
- Fine-grained recognition and understanding: Qwen-VL is the first open-source 448-resolution LVLM model compared to the 224 resolution currently used by other open source LVLMs. Higher resolution can improve fine-grained text recognition, document Q&A, and detection box annotation.

# **Model Training - From Alibaba**

## **Training resources**

The minimum GPU memory required for LoRA training is 25G

If you're using a larger `batch_size`, `num_train_epochs`, or fine-tuning with video data, training requires a larger amount of video memory.

## **Training Data format**

Sample data Meta file download Image data download

Specify `train_data_dir` as the path of the image
Specify `train_meta_file` as the path where the meta file is located.

The meta file accepts input in JSON format, supports multiple rounds of dialogue, supports multiple images or no pictures in each round of dialogue, and the image needs to be the file name.

```json
[
    {
        "conversations": [
            {
                "from": "user",
                "value": "<img>two_col_103562.png</img>\nAs of 2021, how many championship titles had Ferrari won? Answer the question using a single word or phrase."
            },
            {
                "from": "assistant",
                "value": "16"
            },
            {
                "from": "user",
                "value": "As of 2021, how many championship titles had Ferrari won?"
            },
            {
                "from": "assistant",
                "value": "As of 2021, Ferrari had won 16 championship titles."
            }
        ]
    },
    {
        "conversations": [
            {
                "from": "user",
                "value": "<img>two_col_2954.png</img>\nWhat game topped the charts with 512.3 million hours watched on Twitch in the first half of 2019? Answer the question using a single word or phrase."
            },
            {
                "from": "assistant",
                "value": "League of Legends"
            },
            {
                "from": "user",
                "value": "What game topped the charts with 512.3 million hours watched on Twitch in the first half of 2019?"
            },
            {
                "from": "assistant",
                "value": "The game that topped the charts with 512.3 million hours watched on Twitch in the first half of 2019 is League of Legends."
            },
            {
                "from": "user",
                "value": "How many hours did League of Legends watch on Twitch in the first half of 2019? Answer the question using a single word or phrase."
            },
            {
                "from": "assistant",
                "value": "512.3"
            },
            {
                "from": "user",
                "value": "How many hours did League of Legends watch on Twitch in the first half of 2019?"
            },
            {
                "from": "assistant",
                "value": "In the first half of 2019, League of Legends was watched for a total of 512.3 hours on Twitch."
            }
        ]
    }
]
```

---

PAI SDK for Python provides a set of easy-to-use HighLevel APIs to simplify your use of PAI services by using SDKs.

## **Install SDKs**

You can install and configure the PAI Python SDK using the following commands:

```bash
# Install the PAI Python SDK.
pip install alipai>=0.4.5

# Configure the Alibaba Cloud service information.
python -m pai.toolkit.config
```

## **Fine-Tune**

Uses the default data or your own data to tune the model to customized scenarios.

```python
import pprint
from pai.model import RegisteredModel

# Get the latest version of qwen-vl-chat.
m = RegisteredModel(
    model_name="qwen-vl-chat",
    model_provider="pai",
)

# Generate AlgorithmEstimator from RegisteredModel.
# RegisteredModel will provide default configurations for AlgorithmEstimator.
# Users can modify the instance type, instance quantity, etc. according to their needs.
est = m.get_estimator()
print("Instance type:")
print(est.instance_type)

# Generate the default input for AlgorithmEstimator from RegisteredModel.
# Users can modify the training dataset, validation dataset, etc. according to their needs.
inputs = m.get_estimator_inputs()
print("Input information:")
pprint.pprint(inputs)

# Check the default hyperparameters for training. Users can modify them as needed.
print("Hyperparameters:")
pprint.pprint(est.hyperparameters)

# Submit the training job to the PAI platform using the default input.
# Please note that submitting the training job will incur DLC billing.
est.fit(
    inputs=inputs
)

# Check the output model address after training.
pprint.pprint(est.model_data())

# Check all output information (if there are multiple output channels)
pprint.pprint(est.get_outputs_data())
```

## **Deploy**

Deploy the model as an online service for real-time inference.

```python
import pprint
from pai.model import RegisteredModel

# Get the latest version of qwen-vl-chat.
m = RegisteredModel(
    model_name="qwen-vl-chat",
    model_provider="pai",
)

# Check the inference configuration.
pprint.pprint(m.inference_spec)

# Use the default configuration to deploy the service.
# Users can modify the service name, instance type, etc. according to their needs.
# Please note that deploying the service will incur EAS billing.
p = m.deploy(
    # service_name='xxx',
    # instance_type='xxx',
    # ... extra configurations
)

# When the service is successfully deployed, users can perform inference through the
# `Predictor.predict()` method.
p.predict(
    # input data
)

# After using the service, users can delete the service through the
# `Predictor.delete_service()` method to release resources.
```

---
# **More Documentation from Alibaba Cloud**
## **Install And Configure PAI SDK for Python**

`Updated: Jul 3, 2024, 05:42:06`

PAI SDK for Python provides easy-to-use high-level APIs and enables Artificial Intelligence (AI) and machine learning engineers to train and deploy models in Platform for AI (PAI) in an efficient manner. PAI SDK for Python is used in the entire AI and machine learning process.

## **Prerequisites**
- An AccessKey pair is obtained. For more information, see Obtain an AccessKey pair.
- A workspace is created. For more information, see Create a workspace.
- An OSS bucket is created. For more information, see Get started by using the OSS console.
- Python 3.7 or later is prepared.
- If you use the SDK as a RAM user, make sure that you have related permissions on the workspace and Object Storage Service (OSS) buckets.

To submit a training job in a workspace, a RAM user must be assigned a developer or administrator role of the workspace. For information about how to manage workspace members, see Manage members of the workspace.

To store assets such as code and models in OSS buckets, a RAM user must have read and write permissions on the OSS buckets. For more information, see the "Grant your RAM user or RAM role the permissions to access OSS" section in the Grant the permissions that are required to use Machine Learning Designer topic.

## **Install PAI SDK for Python**

Run the following command to install PAI SDK for Python. Make sure that your Python is of 3.7 or later.

```bash
pip install "alipai>=0.4.0"
```

## **Initialize The SDK**

```bash
python3 -m pai.toolkit.config
```

The following figure shows an example of running the command to grant permissions.python -m pai.toolkit.config

---

# **ModelScope Documentation**

You can download the model in the following ways. For details, see DownloadInstructions

## **SDK Download**

```html
[[Model]] Download
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2-VL-7B-Instruct')
```

## **Git Download**

Please ensure that LFS has been installed correctly.

```bash
git lfs install
git clone https://www.modelscope.cn/qwen/Qwen2-VL-7B-Instruct.git
```

## **Command Line Download**

Install ModelScope first by using the following command

```html
pip install modelscope
```

## **Download The Full Model repo**

```html
modelscope download --model qwen/Qwen2-VL-7B-Instruct
```

Download a single file (using README.md as an example)

```html
modelscope download --model qwen/Qwen2-VL-7B-Instruct README.md
```

The complete ModelScope command line download options are available in the specific documentation

<https://www.modelscope.cn/docs/Download%20Model>

<https://modelscope.cn/docs/Environment%20Setup>

## **How To Use the Model with Transformers and qwen_vl_utils**

### **Requirements**

The code of Qwen2-VL has been in the latest Hugging face transformers and we advise you to build from source with command `pip install git+https://github.com/huggingface/transformers`, or you might encounter the following error:

`KeyError: 'qwen2_vl'`

### **Quickstart**

We offer a toolkit to help you handle various types of visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved images and videos. You can install it using the following command:

```bash
pip install qwen-vl-utils
```

Here we show a code snippet to show you how to use the chat model with transformers and qwen_vl_utils:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_dir,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_dir)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## **Limitations**

While Qwen2-VL are applicable to a wide range of visual tasks, it is equally important to understand its limitations. Here are some known restrictions:

- **Lack of Audio Support:** The current model does not comprehend audio information within videos.
- **Data timeliness:** Our image dataset is updated until June 2023, and information subsequent to this date may not be covered.
- **Constraints in Individuals and Intellectual Property (IP)**: The model's capacity to recognize specific individuals or IPs is limited, potentially failing to comprehensively cover all well-known personalities or brands.
- **Limited Capacity for Complex Instruction:** When faced with intricate multi-step instructions, the model's understanding and execution capabilities require enhancement.
- **Insufficient Counting Accuracy**: Particularly in complex scenes, the accuracy of object counting is not high, necessitating further improvements.
- **Weak Spatial Reasoning Skills:** Especially in 3D spaces, the model's inference of object positional relationships is inadequate, making it difficult to precisely judge the relative positions of objects.

These limitations serve as ongoing directions for model optimization and improvement, and we are committed to continually enhancing the model's performance and scope of application.

---

Started to setup the environment and load/deploy the model here:
<https://chatgpt.com/share/de7458f2-affd-4f52-a247-6a0cb34aed54>

---

## **Step 1: Set Up Python Environment**

First, ensure that you have Python 3.7 or later installed. It's recommended to create a virtual environment to keep dependencies isolated.

### **1.1. Create a Virtual Environment (Optional but Recommended)**

- **On Linux/Mac**:

	```bash
    python3 -m venv qwen2-vl-7b-env
    source qwen2-vl-7b-env/bin/activate
    ```

- **On Windows**:

	```bash
    python -m venv qwen2-vl-7b-env
    qwen2-vl-7b-env\Scripts\activate
    ```

### **1.2. Install Required Python Packages**

Inside the activated virtual environment, install the following packages using `pip`:

```bash
pip install transformers qwen-vl-utils torch modelscope accelerate
```

These packages include tools required to download, load, and run the Qwen-VL model.

---

## **Step 2: Download and Load the Model**

Now that your environment is ready, you can proceed to download and load the model. The model will be saved to `/mnt/model/Qwen2-VL-7B-Instruct/`.

### **2.1. Create the Python Script**

Create a new Python script, for example named `load_qwen_vl.py`, and paste the following code into it:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
import os

# Set the path where you want to store the model
model_dir = '/mnt/model/Qwen2-VL-7B-Instruct'

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# Download and prepare the model into the specified directory
model_path = snapshot_download("qwen/Qwen2-VL-7B-Instruct", cache_dir=model_dir)

print(f"Model downloaded to: {model_path}")

# Ensure that the config.json file exists after download
config_path = os.path.join(model_dir, "config.json")
if not os.path.isfile(config_path):
    raise EnvironmentError(f"config.json not found in {model_dir}. Check the model directory structure.")

# Load the model into memory
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained(model_dir)

print("Model and processor have been successfully loaded.")
```

### **2.2. Run the Python Script**

Execute the script to download the model, save it to `/mnt/model/Qwen2-VL-7B-Instruct/`, and load it into memory.

```bash
python3 load_qwen_vl.py
```

#### **Expected Output:**
- The script will confirm that the model and processor have been successfully loaded without any issues.

---

## **Step 3: Running Inference**

With the model loaded, you can now run inference on some input data to verify its functionality.

### **3.1. Prepare an Input**

You'll need an image and some text. For example:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

This input contains an image and a request to describe the image.

Here is an example of input for **Video Inference**

```json
# Messages containing a images list as a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                ],
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]
# Messages containing a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### **Batch Inference**

```json
# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Combine messages for batch processing
messages = [messages1, messages1]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
```

---

### **3.2. Prepare the Input for Processing**

Prepare the image and text data based on the model's requirements:

```python
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
```

### **3.3. Run Inference**

Use the model to predict the output for the given input:

```python
generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Generated Output:", output_text)
```

**Expected Result:**
- The model will provide a text description of the image that was input.

---

## **Step 4: Review and Use the Output**

- Review the generated output to ensure it matches what you expect.
- Continue using the model for other tasks or develop additional scripts for more complex workflows.

---

## **FIrst Attempt to Setup Model Successfully

Here's how you can prepare your environment and codebase to perform inference with video inputs.

### Step 1: Setup and Initialization

If you haven't already set up your environment with the required packages, follow these instructions:

1) **Create a Virtual Environment** (Optional but recommended):

   - **Linux/Mac:**

	 ```bash
     python3 -m venv qwen-vl-env
     source qwen-vl-env/bin/activate
     ```

   - **Windows:**

	 ```bash
     python -m venv qwen-vl-env
     qwen-vl-env\Scripts\activate
     ```

2) **Install Required Python Packages**:
   Run the following command in your terminal to install the necessary Python packages:

   ```bash
   pip install transformers qwen-vl-utils torch modelscope accelerate
   ```

### Step 2: Modify and Execute the Script for Video Input

Now, let's update the basic inference script to handle video input. Instead of an image, we'll prepare a video file, set it up for processing, and pass it to the model.

1) **Prepare the Input with Video**:
   - Replace the image input with a video file's URL or path.
   - Ensure your video is accessible and consider using a public URL or a local file.

2) **Example Code**:
   Here's how the modified script might look:

   ```python
   from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
   from qwen_vl_utils import process_vision_info
   from modelscope import snapshot_download
   import torch

   # Download the model if not already done
   model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")

   # Load the model
   model = Qwen2VLForConditionalGeneration.from_pretrained(
       model_dir, torch_dtype="auto", device_map="auto"
   )

   # Load the processor
   processor = AutoProcessor.from_pretrained(model_dir)

   # Define your video input and prompt
   messages = [
       {
           "role": "user",
           "content": [
               {
                   "type": "video",
                   "video": "https://path-to-your-video-file.mp4", # Or a local file path
               },
               {"type": "text", "text": "Give a summary of the events depicted in this video."},
           ],
       }
   ]

   # Prepare the input
   text = processor.apply_chat_template(
       messages, tokenize=False, add_generation_prompt=True
   )
   image_inputs, video_inputs = process_vision_info(messages) # Expect video_inputs to be populated

   inputs = processor(
       text=[text],
       images=image_inputs,
       videos=video_inputs,
       padding=True,
       return_tensors="pt",
   )
   inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

   # Run inference
   generated_ids = model.generate(**inputs, max_new_tokens=200)
   output_text = processor.batch_decode(
       generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
   )
   print("Generated Output:", output_text)
   ```

3) **Run the Script**:
   Execute the Python script to perform video-based inference. The model will process the video input and generate a text output based on the prompt.

   ```bash
   python3 load_qwen_vl_with_video.py
   ```

### Step 3: Review and Fine-Tune

- **Check the Output**:
   After running the script, review the generated text output in your terminal or console. The model should provide a description or narrative based on the content of the video.

- **Fine-Tuning Inputs**:
   Depending on the video's complexity, you might need to fine-tune prompts or provide additional context to get the most accurate and relevant output.

### Additional Insights

1) **Video Limitations**:
   - The model has some limitations with video processing, such as no support for audio and reduced accuracy in tasks needing high levels of spatial or temporal reasoning.
   - Due to the model's inability to comprehend audio, ensure your task relies solely on visual information.

2) **Performance Considerations**:
   - When working with video inputs, consider the model's memory usage, especially with higher-resolution or long-duration videos. You may need to adjust `min_pixels` and `max_pixels` based on your hardware capabilities.
   - Flash Attention 2 might be beneficial. If applicable, you can enable this by setting `attn_implementation="flash_attention_2"` while loading the model for better performance.

---

### Conclusion

You're now equipped to perform video inference utilizing the Qwen-VL model. You can extend this workflow to more sophisticated tasks involving visual media by tuning prompts, expanding the processing pipeline, or combining the model's output with downstream applications.

Feel free to ask more questions if anything is unclear or if you need additional help!
