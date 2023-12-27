import json
from tqdm import tqdm
import os

def read_json(path):
    with open(path) as f:
        return json.load(f)
    
def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)
    
src_path = '/data/CourseProject/data/deepfashion/train_captions.json'
tgt_path = '/data/CourseProject/data/deepfashion/train_data_llava.json'


def process(src, tgt):
    data = read_json(src)
    
    tgt_format_data = []
    for image, caption in tqdm(data.items()):
        basename = os.path.basename(image)
        image_id = basename.replace('.jpg', '')
        tgt_format_data.append({
            'id': image_id,
            'image': basename,
            'conversations': [
                {
                    'from': 'human',
                    'value': 'Please describe this image in deepfashion dataset style.'
                },
                {
                    'from': 'gpt',
                    'value': caption
                }
            ]
        })
    save_json(tgt_format_data, tgt)

process(src_path, tgt_path)
