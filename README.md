<h1 style="text-align: center;">Region-Level Context-Aware Multimodal Understanding</h1>  

**Arxiv**: [📑Region-Level Context-Aware Multimodal Understanding](https://arxiv.org/abs/2508.12263)  
**Models**：[🤗RC-Qwen2VL-2b](https://huggingface.co/weihongliang/RC-Qwen2VL-2b/blob/main/README.md) | [🤗RC-Qwen2VL-7b](https://huggingface.co/weihongliang/RC-Qwen2VL-7b/blob/main/README.md)  
**Demos**:
[🚀Personalized Conversations About Images](https://huggingface.co/spaces/weihongliang/Personalized-VQA) | [🚀Celebrity Recognition and VQA](https://huggingface.co/spaces/weihongliang/RCMLLM)  

## Abstrct
Despite significant progress, existing research on Multimodal Large Language Models (MLLMs) mainly focuses on general visual understanding, overlooking the ability to integrate textual context associated with objects for a more context-aware multimodal understanding — an ability we refer to as Region-level Context-aware Multimodal Understanding (RCMU). To address this limitation, we first formulate the RCMU task, which requires models to respond to user instructions by integrating both image content and textual information of regions or objects. To equip MLLMs with RCMU capabilities, we propose Region-level Context-aware Visual Instruction Tuning (RCVIT), which incorporates object information into the model input and enables the model to utilize bounding box coordinates to effectively associate objects’ visual content with their textual information. To address the lack of datasets, we introduce the RCMU dataset, a large-scale visual instruction tuning dataset that covers multiple RCMU tasks. We also propose RC&P-Bench, a comprehensive benchmark that can evaluate the performance of MLLMs in RCMU and multimodal personalized understanding tasks. Additionally, we propose a reference-free evaluation metric to perform a comprehensive and fine-grained evaluation of the region-level context-aware image descriptions. By performing RCVIT on Qwen2-VL models with the RCMU dataset, we developed RC-Qwen2-VL models. Experimental results indicate that RC-Qwen2-VL models not only achieve outstanding performance on multiple RCMU tasks but also demonstrate successful applications in multimodal RAG and personalized conversation.  

## RCMU Tasks
![RCMU Tasks](/figs/RCMU.jpg)
![RCMU Tasks](/figs/qwen-exam.jpg)  

## RCMU Dataset
![RCMU Dataset](/figs/autopipline-2.jpg)  

## RC&P-Bench
![RC&P-Bench](/figs/rcmllm-bench.jpg)  

## Prompt format:  

```json
 messages = [
     {
         "role": "user",
         "content": [
             {
                 "type": "image",
                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
             },
             {
                 "type": "text",
                 "text": "[1]: The information of the {obj_1} located at <|box_start|>(x1,y1),(x2,y2)<|box_end|> in the image: {info_1}.\n[2]: The information of the {obj_2} located at <|box_start|>(x1,y1),(x2,y2)<|box_end|> in the image: {info_2}......"
             }
     }
 ]
```
Where the bbox coordinates range from 0 to 1000.    


Here we show a code snippet:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "weihongliang/RC-Qwen2VL-2b", torch_dtype="auto", device_map="auto"
)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "weihongliang/RC-Qwen2VL-2b",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text",
                "text": "[1]: The information of the man located at <|box_start|>(755,417),(991,885)<|box_end|> in the image: Lily, a 34-year-old marketing executive, resides in downtown Chicago. She completed her MBA at the University of Chicago in 2012 and has a passion for technology and innovation. In her spare time, she enjoys hiking and photography, often capturing stunning landscapes during her outdoor adventures. Currently, Lily is working on a project that focuses on digital marketing strategies, with a vision to integrate artificial intelligence. She also volunteers at a local animal shelter on weekends, promoting animal welfare and seeking to help abandoned pets find their forever homes.\n[2]: The information of the dog located at <|box_start|>(214,424),(583 884)<|box_end|> in the image: Bella is a 2-year-old dog owned by Lily, who emphasizes the joy pets bring to daily life. Bella was brought into the family on her adoption day in April 2022 from a nearby rescue organization. Lily enjoys spending quality time with her, and together they play a variety of games like hide and seek, where Bella loves to dart around furniture and pounce unexpectedly. Bella's preferred activities include watching birds from the window and exploring new boxes or bags that find their way into the house, always keeping Lily entertained with her curious antics.\nAnswer the following question based on the information above and the given image, and provide citations for your response.\nDescribe the image."
            }
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
