from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch
import inspect

# Load model and processor
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-Omni-7B', 
    torch_dtype='auto', 
    device_map='auto', 
    attn_implementation='eager'
)
processor = Qwen2_5OmniProcessor.from_pretrained('Qwen/Qwen2.5-Omni-7B')

print("Model class:", type(model))
print("Model forward method:", model.forward)
print("Forward method signature:", inspect.signature(model.forward))

# Check if model has a specific forward method
if hasattr(model, 'forward'):
    print("Model has forward method")
    print("Forward method source:")
    try:
        print(inspect.getsource(model.forward))
    except:
        print("Could not get source")

# Check the base class
print("\nBase classes:")
for base in type(model).__mro__:
    print(f"  {base}")

# Try to find the actual forward implementation
print("\nLooking for forward implementation...")
for attr_name in dir(model):
    if 'forward' in attr_name.lower():
        attr = getattr(model, attr_name)
        if callable(attr):
            print(f"  {attr_name}: {attr}")
            try:
                print(f"    Signature: {inspect.signature(attr)}")
            except:
                print(f"    Could not get signature")

# Create simple conversation
conversation = [
    {
        'role': 'user', 
        'content': [{'type': 'text', 'text': 'Hello'}]
    }
]

# Process inputs
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
inputs = processor(
    text=text, 
    audio=audios, 
    images=images, 
    videos=videos, 
    return_tensors='pt', 
    padding=True, 
    use_audio_in_video=True
)
inputs = inputs.to(model.device).to(model.dtype)

print('\nInput keys:', list(inputs.keys()))
print('Input shapes:')
for k, v in inputs.items():
    print(f'{k}: {v.shape}')

# Try calling the model directly with positional arguments
print('\n=== Testing direct model call ===')
try:
    # Try with just input_ids as first positional argument
    outputs = model(inputs['input_ids'])
    print('Success with input_ids only!')
    print('Output type:', type(outputs))
    if hasattr(outputs, 'logits'):
        print('Logits shape:', outputs.logits.shape)
except Exception as e:
    print('Error with input_ids only:', e)

try:
    # Try with input_ids and attention_mask as positional arguments
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    print('Success with input_ids and attention_mask!')
    print('Output type:', type(outputs))
    if hasattr(outputs, 'logits'):
        print('Logits shape:', outputs.logits.shape)
except Exception as e:
    print('Error with input_ids and attention_mask:', e) 