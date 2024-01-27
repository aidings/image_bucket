import json
import torch
from typing import List
from image_bucket import ImageBuckets, BuckNode
from PIL import Image
from torchvision import transforms

class SDDataset(ImageBuckets):
    def __init__(self, json_path, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data = self.__load(json_path)


        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
    
    def __load(self, json_path):
        data = []
        with open(json_path, 'r') as f:
            idx = 0
            for line in f:
                jdict = json.loads(line.strip())
                data.append(jdict)
                file_path = jdict['file_path']
                image = Image.open(file_path) 
                w, h = image.size
                self.inject(BuckNode(w, h, idx))
                idx += 1
        return data
    
    def __get_input_ids(self, caption):
        input_ids = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        ).input_ids

        return input_ids

    def transforms(self, bidxs: List[int], bucket, resolution):
        image_tensors = []
        input_ids = []
        for idx in bidxs:
            jdict = self.data[idx]
            file_path = jdict['file_path']
            tokens = self.__get_input_ids(jdict['caption'])
            image = Image.open(file_path).convert('RGB')
            image = image.resize(resolution)
            image_tensor = self.trans(image)
            image_tensors.append(image_tensor)
            input_ids.append(tokens)

        return torch.stack(image_tensors), torch.stack(input_ids)
    
    def collate_fn(self, batch):
        image_tensors, input_ids = batch[0]
        return image_tensors, input_ids

    

if __name__ == '__main__':
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    sd = SDDataset('./samples.jsonl', tokenizer)
    sd.make(4, shuffle=True)

    dataloader = torch.utils.data.DataLoader(sd, batch_size=1, shuffle=True, num_workers=4, collate_fn = sd.collate_fn)

    for data in dataloader:
        print(data[0].shape, data[1].shape)
