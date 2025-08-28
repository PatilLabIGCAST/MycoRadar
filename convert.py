import os
import json
from labelme import utils
import PIL.Image

def convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)

            with open(json_path) as f:
                data = json.load(f)

            imageData = data.get("imageData")
            if imageData is None:
                imagePath = os.path.join(input_dir, data["imagePath"])
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = utils.img_b64encode(imageData).decode("utf-8")

            label_name_to_value = {"_background_": 0}
            for shape in data["shapes"]:
                label_name = shape["label"]
                if label_name not in label_name_to_value:
                    label_name_to_value[label_name] = len(label_name_to_value)
            img_shape = (data["imageHeight"], data["imageWidth"], 3)

            lbl, _ = utils.shapes_to_label(
                img_shape, data["shapes"], label_name_to_value
            )

            if isinstance(lbl, tuple):
                lbl = lbl[0]

            out_path = os.path.join(output_dir, filename.replace(".json", ".png"))
            PIL.Image.fromarray(lbl).save(out_path)

            print(f"Converted {filename} to {out_path}")
            os.remove(json_path)
